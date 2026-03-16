import os, json, uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from pyenergyplus.api import EnergyPlusAPI
from andruix_policy_model import Actor
import torch
import torch.nn as nn
from typing import Sequence, Optional
from andruix_policy_model import TorchPolicyModel
from eplus_runperiod_utils import rewrite_first_runperiod

# ---------- metrics ----------
@dataclass
class EvalMetrics:
    energy_kwh: float = 0.0
    uncomfort_min: float = 0.0
    cooling_deg_c_sum: float = 0.0   # accumulate a "cooling degree" proxy
    kwh_per_cooling_deg_c: float = 0.0

def safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return a / (b if abs(b) > eps else eps)

# ---------- controller ----------
class EvalController:
    def __init__(self, api: EnergyPlusAPI, *,
                 zones: list[str],
                 energy_meter: str,
                 start_mmdd: str,
                 end_mmdd: str,
                 reward_scale: float = 1.0,
                 uncomfort_min_weight: float = 0.1,
                 # comfort band for counting discomfort
                 comfort_low: float = 21.0,
                 comfort_high: float = 25.0,
                 # for kWh/degC proxy (choose something consistent)
                 cooling_base_c: float = 18.0,
                 # policy mode knobs
                 mode: str = "baseline",
                 include_trend_60m: bool = True,
                 include_trend_15m: bool = True,
                 occupied_start_minute: int = 8 * 60,
                 occupied_end_minute: int = 18 * 60,
                 policy: TorchPolicyModel=None):
        self.api = api
        self.zones = zones
        self.energy_meter = energy_meter
        self.mode = mode
        self.policy = policy

        self.reward_scale = float(reward_scale)
        self.uncomfort_min_weight = float(uncomfort_min_weight)
        self.comfort_low = float(os.environ.get("ANDRUIX_COMFORT_LOW", comfort_low))
        self.comfort_high = float(os.environ.get("ANDRUIX_COMFORT_HIGH", comfort_high))
        self.cooling_base_c = float(cooling_base_c)

        # Observation flags (match training worker env toggles)
        self.include_occ_flag = bool(int(os.environ.get("ANDRUIX_OBS_OCC", "1")))
        self.include_doy_features = not bool(int(os.environ.get("ANDRUIX_OBS_NO_DOY", "0")))
        self.include_trend_60m = not bool(int(os.environ.get("ANDRUIX_OBS_NO_TREND_60M", "0"))) if "ANDRUIX_OBS_NO_TREND_60M" in os.environ else bool(include_trend_60m)
        self.include_trend_15m = not bool(int(os.environ.get("ANDRUIX_OBS_NO_TREND_15M", "0"))) if "ANDRUIX_OBS_NO_TREND_15M" in os.environ else bool(include_trend_15m)

        # Internal history buffer for trend features (same shape as train worker expects)
        self._hist: list[tuple[int, float, float]] = []
        self.step_count: int = 0

        # -- Action → setpoint mapping config (match train worker defaults + env overrides) ---
        self.occupied_start_minute = int(occupied_start_minute)
        self.occupied_end_minute = int(occupied_end_minute)

        self.sp_center_min_occ = float(os.environ.get("ANDRUIX_SP_CENTER_MIN_OCC", "21.0"))
        self.sp_center_max_occ = float(os.environ.get("ANDRUIX_SP_CENTER_MAX_OCC", "24.0"))
        self.sp_center_min_unocc = float(os.environ.get("ANDRUIX_SP_CENTER_MIN_UNOCC", "18.0"))
        self.sp_center_max_unocc = float(os.environ.get("ANDRUIX_SP_CENTER_MAX_UNOCC", "28.0"))
        self.deadband_occ = float(os.environ.get("ANDRUIX_DEADBAND_OCC", "1.0"))
        self.deadband_unocc = float(os.environ.get("ANDRUIX_DEADBAND_UNOCC", "2.0"))

        self.min_heat_sp = float(os.environ.get("ANDRUIX_MIN_HEAT_SP", "15.0"))
        self.max_cool_sp = float(os.environ.get("ANDRUIX_MAX_COOL_SP", "30.0"))
        self.min_deadband = float(os.environ.get("ANDRUIX_MIN_DEADBAND", "0.5"))
        self.control_minutes = int(os.environ.get("ANDRUIX_CONTROL_MINUTES", "15"))

        # handles
        self._meter_h = None
        self.room_temp_handles = {} 
        self.heat_sp_handles = {}
        self.cool_sp_handles = {}
        self._oat_h = None
        self.handles_ready = False

        # time tracking
        self._last_doy = None
        self._last_mod = None
        self._last_minute_seen = None

        # bucket accumulators (15-min control buckets)
        self._bucket_energy_kwh = 0.0
        self._bucket_uncomfort_min = 0.0
        self._bucket_cooling_deg_c = 0.0
        self._bucket_sys_steps = 0

        # episode totals
        self.metrics = EvalMetrics()

        # for policy mode
        self._prev_obs = None
        self._prev_act = None
        self._bucket_id = None

    # ---- required: build handles after warmup ----
    def _try_init_handles(self, state):
        if self.handles_ready:
            return
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if self.api.exchange.warmup_flag(state):
            return

        ex = self.api.exchange

        if self._meter_h is None:
            self._meter_h = ex.get_meter_handle(state, self.energy_meter)

        if self._oat_h is None:
            # example: if you already use an OAT variable handle, reuse that exact call
            self._oat_h = ex.get_variable_handle(state, "Site Outdoor Air Drybulb Temperature", "Environment")

        # Per-zone handles
        for zone in self.zones:
            self.heat_sp_handles[zone] = self.api.exchange.get_actuator_handle(state, 'Zone Temperature Control', 'Heating Setpoint', zone)
            self.cool_sp_handles[zone] = self.api.exchange.get_actuator_handle(state, 'Zone Temperature Control', 'Cooling Setpoint', zone)
            self.room_temp_handles[zone] = self.api.exchange.get_variable_handle(state, 'Zone Mean Air Temperature', zone)

        self.handles_ready = True

    # ---- helpers you already have (plug in your existing ones) ----
    def _read_oat_c(self, state) -> float:
        v = self.api.exchange.get_variable_value(state, self._oat_h)
        return float(v) if v is not None else float("nan")

    def _read_zone_temps_c(self, state) -> dict[str, float]:
        d: dict[str, float] = {}
        for z in self.zones:
            h = self.room_temp_handles.get(z, -1)
            if h == -1:
                d[z] = float("nan")
                continue
            v = self.api.exchange.get_variable_value(state, h)
            d[z] = float(v) if v is not None else float("nan")
        return d

    def _get_time(self, state) -> tuple[int, int]:
        """Return (day_of_year, minute_of_day, abs_minute) from EnergyPlus simulation clock."""
        try:
            doy = int(self.api.exchange.day_of_year(state))
            hr = int(self.api.exchange.hour(state))
            mn = int(self.api.exchange.minutes(state))
            if hr >= 24:
                hr = 23
                mn = 59
            mod = max(0, min(1439, hr * 60 + mn))
            abs_minute = (max(1, doy) - 1) * 1440 + mod
            return doy, mod, abs_minute
        except Exception:
            return None, None, None

    def _is_occupied(self, minute_of_day: int) -> bool:
        if minute_of_day is None:
            return True
        start = int(self.occupied_start_minute)
        end = int(self.occupied_end_minute)
        m = int(minute_of_day)
        if start == end:
            return True
        if start < end:
            return start <= m < end
        return m >= start or m < end

    def _bucket_from_time(self, doy: int, mod: int) -> int:
        # 15-min buckets, unique across year
        return doy * 96 + (mod // 15)
    
    def _coerce_action_vec(self, act, n: int) -> np.ndarray:
        """Minimal contract enforcement for actions.
        - ensures shape (n,)
        - replaces NaN/Inf with 0
        - clips to [-1, 1] because _map_action_to_setpoints expects normalized actions
        """
        try:
            a = np.asarray(act, dtype=np.float32).reshape(-1)
        except Exception:
            a = np.zeros((n,), dtype=np.float32)

        if a.size == 1 and n > 1:
            a = np.full((n,), float(a[0]), dtype=np.float32)
        elif a.size < n:
            pad = np.zeros((n - a.size,), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        elif a.size > n:
            a = a[:n]

        a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    
    def _coerce_action(self, act) -> float:
        """Back-compat scalar coercion."""
        return float(self._coerce_action_vec(act, 1)[0])

    def _select_action_vec(self, obs: np.ndarray) -> np.ndarray:
        a_pol = np.asarray(self.policy.get_action(obs), dtype=np.float32).reshape(-1)

        a_norm = self._coerce_action_vec(a_pol, n=len(self.zones))

        # (optional) saturation log
        if self.step_count % 10 == 0:
            sat = float(np.mean(np.abs(a_norm) >= 0.98))
            print(f"[a_ctl] step={self.step_count} sat_frac={sat:.2f} a_norm={np.round(a_norm,3)}")

        return a_norm

    # ---- callbacks ----
    def begin_cb(self, state):
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if self._meter_h is None or self._oat_h is None or not self.room_temp_handles:
            return  # handles not ready yet

        doy, mod, abs_minute = self._get_time(state)
        if abs_minute is None or doy is None or mod is None:
            return
        
        bucket_id = self._bucket_from_time(doy, mod)

        # on bucket boundary crossing, finalize previous bucket totals into episode totals
        if self._bucket_id is None:
            self._bucket_id = bucket_id

        crossed = (bucket_id != self._bucket_id)
        if crossed:
            # finalize bucket → episode totals
            self.metrics.energy_kwh += self._bucket_energy_kwh
            self.metrics.uncomfort_min += self._bucket_uncomfort_min
            self.metrics.cooling_deg_c_sum += self._bucket_cooling_deg_c

            # reset for new bucket
            self._bucket_energy_kwh = 0.0
            self._bucket_uncomfort_min = 0.0
            self._bucket_cooling_deg_c = 0.0
            self._bucket_sys_steps = 0
            self._bucket_id = bucket_id

        # Policy mode: on boundary, compute obs/action and apply.
        if self.mode == "policy":
            # only act on boundaries (when crossed OR first bucket)
            if crossed or self._prev_obs is None:
                zone_temps = self._read_zone_temps_c(state)
                outside_temp = self._read_oat_c(state)
                
                # build obs exactly like training
                obs, occ = self._make_obs_multi_zone(zone_temps, outside_temp, doy, mod, abs_minute)  # you already have this logic
                a_norm = self._select_action_vec(obs)

                self._apply_action(state, a_norm, occ)        # your existing actuator writes

                self._prev_obs = obs
                self._prev_act = a_norm

    def end_cb(self, state):
        if self._meter_h is None or self._oat_h is None or not self.room_temp_handles:
            return

        doy, mod, abs_minute = self._get_time(state)

        # Energy meter is Joules per system timestep in your setup → convert to kWh
        raw_j = float(self.api.exchange.get_meter_value(state, self._meter_h))
        kwh = raw_j / 3_600_000.0
        self._bucket_energy_kwh += kwh
        self._bucket_sys_steps += 1

        # Cooling degree proxy (bucket-based accumulation, simplest: add per system step)
        oat = self._read_oat_c(state)
        self._bucket_cooling_deg_c += max(0.0, oat - self.cooling_base_c) * (1.0 / max(1, self._bucket_sys_steps))

        # Discomfort minutes: only count when minute advances (avoid sub-minute repeats)
        if self._last_minute_seen is None:
            self._last_minute_seen = mod
            self._last_doy = doy
            self._last_mod = mod
            return

        # minute delta with midnight wrap
        prev = self._last_minute_seen
        cur = mod
        delta = cur - prev
        if delta < 0:
            delta += 1440  # wrap

        if delta > 0:
            # update history for trend features (once per minute advance)
            oat_now = self._read_oat_c(state)
            zone_temps_now = self._read_zone_temps_c(state)
            vals_now = np.asarray([zone_temps_now.get(z, np.nan) for z in self.zones], dtype=np.float32)
            avg_tz_now = float(np.nanmean(vals_now)) if np.isfinite(np.nanmean(vals_now)) else float('nan')
            self._update_history(abs_minute, avg_tz_now, oat_now)

            occ = self._is_occupied(prev)
            if occ:
                zone_temps = self._read_zone_temps_c(state)
                vals = np.asarray([zone_temps.get(z, np.nan) for z in self.zones], dtype=np.float32)
                too_cold = np.any(vals < self.comfort_low)
                too_hot  = np.any(vals > self.comfort_high)
                if too_cold or too_hot:
                    self._bucket_uncomfort_min += float(delta)

        self._last_minute_seen = mod
        self._last_doy = doy
        self._last_mod = mod
    

    
    def _update_history(self, abs_minute: int | None, avg_zone_temp: float, outside_temp: float) -> None:
        """Store (abs_minute, avg_zone_temp, outside_temp) for trend features."""
        if abs_minute is None:
            return
        try:
            tz = float(avg_zone_temp)
            toa = float(outside_temp)
        except Exception:
            return
        if not (np.isfinite(tz) and np.isfinite(toa)):
            return
        self._hist.append((int(abs_minute), tz, toa))
        # Keep buffer bounded (enough for several days of minute-resolution history)
        if len(self._hist) > 5000:
            self._hist = self._hist[-4000:]

    def _lookup_at_or_before(self, target_abs_minute: int):
        for t, tz, toa in reversed(self._hist):
            if t <= target_abs_minute and np.isfinite(tz) and np.isfinite(toa):
                return float(tz), float(toa)
        return None

    def _trend_deltas(self, abs_minute: int | None, room_temp: float, outside_temp: float) -> list[float]:
        feats: list[float] = []

        # Not enough info yet
        if abs_minute is None or not self._hist:
            if self.include_trend_60m:
                feats += [0.0, 0.0]
            if self.include_trend_15m:
                feats += [0.0, 0.0]
            return feats

        cur_tz = float(room_temp)
        cur_toa = float(outside_temp)

        # If current values are bad, don't emit NaNs
        if not (np.isfinite(cur_tz) and np.isfinite(cur_toa)):
            if self.include_trend_60m:
                feats += [0.0, 0.0]
            if self.include_trend_15m:
                feats += [0.0, 0.0]
            return feats

        if self.include_trend_60m:
            past = self._lookup_at_or_before(int(abs_minute) - 60)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                if not (np.isfinite(p_tz) and np.isfinite(p_toa)):
                    feats += [0.0, 0.0]
                else:
                    feats += [cur_tz - p_tz, cur_toa - p_toa]

        if self.include_trend_15m:
            past = self._lookup_at_or_before(int(abs_minute) - 15)
            if past is None:
                feats += [0.0, 0.0]
            else:
                p_tz, p_toa = past
                if not (np.isfinite(p_tz) and np.isfinite(p_toa)):
                    feats += [0.0, 0.0]
                else:
                    feats += [cur_tz - p_tz, cur_toa - p_toa]

        return feats

    # ---- placeholders you should point at your existing training logic ----
    def _make_obs_multi_zone(self, zone_temps, oat, doy, mod, abs_minute):
        """Observation layout (Path A):
        [Tz_1..Tz_N] + [Toa] + [sin_tod, cos_tod] +
        [dTzAvg_60, dToa_60] + [dTzAvg_15, dToa_15] +
        [sin_doy, cos_doy] + [occ_flag]
        """
        obs_parts: list[float] = []

        # 1) Zone temps in the provided zone order
        tz_list: list[float] = []
        for z in self.zones:
            v = zone_temps.get(z, None)
            if v is None:
                tz_list.append(float('nan'))
                print(f"[make_obs_bad] step={self.step_count} | zone={z} | missing value → NaN")
                continue

            try:
                tz = float(v)
                if not np.isfinite(tz):
                    raise ValueError(f"non-finite value: {tz}")
                tz_list.append(tz)
            except (ValueError, TypeError) as e:
                tz_list.append(float('nan'))
                print(f"[make_obs_bad] step={self.step_count} | zone={z} | value={v!r} → NaN | error={str(e)}")

        obs_parts.extend(tz_list)

        # 2) Outside air temp
        try:
            toa = float(oat)
            if not np.isfinite(toa):
                raise ValueError("non-finite Toa")
        except (ValueError, TypeError) as e:
            print(f"[make_obs_bad] Toa conversion failed  value={oat!r}  error={str(e)}")
            toa = float('nan')
        obs_parts.append(toa)

        # 3) Time-of-day sin/cos (always included)
        if mod is None:
            sin_tod, cos_tod = 0.0, 0.0
        else:
            tod = float(mod) / 1440.0
            sin_tod = float(np.sin(2.0 * np.pi * tod))
            cos_tod = float(np.cos(2.0 * np.pi * tod))
        obs_parts.extend([sin_tod, cos_tod])

        # 4) Shared trend deltas using avg zone temp + outside temp
        if len(tz_list) > 0:
            tz_avg = float(np.nanmean(tz_list))
        else:
            tz_avg = float('nan')
            print(f"[make_obs_bad] tz_list was empty → returning NaN")

        obs_parts.extend(self._trend_deltas(abs_minute, tz_avg, toa))

        # 5) Day-of-year sin/cos (optional)
        if self.include_doy_features:
            if doy is None:
                obs_parts.extend([0.0, 0.0])
            else:
                frac = float(doy) / 365.0
                obs_parts.extend([float(np.sin(2.0 * np.pi * frac)), float(np.cos(2.0 * np.pi * frac))])

        # 6) Occupancy flag (optional)
        occupied = self._is_occupied(mod)
        if self.include_occ_flag:
            obs_parts.append(1.0 if occupied else 0.0)

        return np.asarray(obs_parts, dtype=np.float32), occupied
    
    def _map_action_to_setpoints(self, a: float, occupied: bool) -> tuple[float, float, float]:
        if occupied:
            cmin, cmax = self.sp_center_min_occ, self.sp_center_max_occ
            deadband = self.deadband_occ
        else:
            cmin, cmax = self.sp_center_min_unocc, self.sp_center_max_unocc
            deadband = self.deadband_unocc

        center = float(cmin + 0.5 * (a + 1.0) * (cmax - cmin))
        heat = center - 0.5 * float(deadband)
        cool = center + 0.5 * float(deadband)

        heat = max(heat, self.min_heat_sp)
        cool = min(cool, self.max_cool_sp)

        if (cool - heat) < self.min_deadband:
            mid = 0.5 * (heat + cool)
            heat = mid - 0.5 * self.min_deadband
            cool = mid + 0.5 * self.min_deadband
            if heat < self.min_heat_sp:
                heat = self.min_heat_sp
                cool = heat + self.min_deadband
            if cool > self.max_cool_sp:
                cool = self.max_cool_sp
                heat = cool - self.min_deadband

        center = 0.5 * (heat + cool)
        return float(heat), float(cool), float(center)

    def _apply_action(self, state, a_norm: np.ndarray, occupied: bool):
                # Apply setpoints + update held setpoints for re-assertion
        for i, zone in enumerate(self.zones):
            heat_sp, cool_sp, center_sp = self._map_action_to_setpoints(float(a_norm[i]), occupied)

            self.api.exchange.set_actuator_value(state, self.heat_sp_handles[zone], heat_sp)
            self.api.exchange.set_actuator_value(state, self.cool_sp_handles[zone], cool_sp)



    def finalize(self) -> EvalMetrics:
        # flush last partial bucket
        self.metrics.energy_kwh += self._bucket_energy_kwh
        self.metrics.uncomfort_min += self._bucket_uncomfort_min
        self.metrics.cooling_deg_c_sum += self._bucket_cooling_deg_c

        self.metrics.kwh_per_cooling_deg_c = safe_div(self.metrics.energy_kwh, self.metrics.cooling_deg_c_sum)
        return self.metrics


def run_one(api: EnergyPlusAPI, *, mode: str, idf: str, epw: str, outdir: str,
            zones: list[str], energy_meter: str, start_mmdd: str, end_mmdd: str,
            policy=None) -> EvalMetrics:

    state = api.state_manager.new_state()

    ctrl = EvalController(
        api,
        zones=zones,
        energy_meter=energy_meter,
        start_mmdd=start_mmdd,
        end_mmdd=end_mmdd,
        mode=mode,
        policy=policy,
        # match your run settings
        uncomfort_min_weight=float(os.environ.get("ANDRUIX_UNCOMFORT_MIN_WEIGHT", "0.1")),
        comfort_low=float(os.environ.get("ANDRUIX_COMFORT_LOW", "21.0")),
        comfort_high=float(os.environ.get("ANDRUIX_COMFORT_HIGH", "25.0")),
    )

    api.runtime.callback_after_new_environment_warmup_complete(state, ctrl._try_init_handles)
    api.runtime.callback_begin_system_timestep_before_predictor(state, ctrl.begin_cb)
    api.runtime.callback_end_system_timestep_after_hvac_reporting(state, ctrl.end_cb)

    eplus_args = ["-w", str(epw), "-d", str(outdir), str(idf)]
    api.runtime.run_energyplus(state, eplus_args)

    metrics = ctrl.finalize()
    api.state_manager.delete_state(state)
    return metrics


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--idf", required=True)
    p.add_argument("--epw", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--zones", required=True)  # comma-separated
    p.add_argument("--energy-meter", default="Electricity:HVAC")
    p.add_argument("--start-date", required=True)  # MM/DD
    p.add_argument("--end-date", required=True)    # MM/DD
    p.add_argument("--policy-path", required=True)
    p.add_argument("--eval-root", default="/shared/evals/inbox")
    p.add_argument("--eval-id", default=None)
    args = p.parse_args()

    eval_id = args.eval_id or uuid.uuid4().hex[:12]
    zones = [z.strip() for z in args.zones.split(",") if z.strip()]

    api = EnergyPlusAPI()

    base_out = str(Path(args.outdir) / eval_id / "baseline")
    pol_out  = str(Path(args.outdir) / eval_id / "policy")

    idf_path = Path(args.idf)
    out_dir = Path(args.eval_root) / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)


    if args.start_date and args.end_date:
        idf_path = out_dir / f"runperiod_{args.start_date.replace('/', '-')}_{args.end_date.replace('/', '-')}.idf"
        rewrite_first_runperiod(Path(args.idf), idf_path, args.start_date, args.end_date)

    # TODO: load your Torch policy exactly like training worker does
    policy = load_policy(args.policy_path)

    m_base = run_one(api, mode="baseline", idf=idf_path, epw=args.epw, outdir=base_out,
                     zones=zones, energy_meter=args.energy_meter,
                     start_mmdd=args.start_date, end_mmdd=args.end_date, policy=None)

    m_pol  = run_one(api, mode="policy", idf=idf_path, epw=args.epw, outdir=pol_out,
                     zones=zones, energy_meter=args.energy_meter,
                     start_mmdd=args.start_date, end_mmdd=args.end_date, policy=policy)

    summary = {
        "eval_id": eval_id,
        "idf": args.idf,
        "epw": args.epw,
        "window": {"start": args.start_date, "end": args.end_date},
        "baseline": asdict(m_base),
        "policy": asdict(m_pol),
        "delta": {
            "energy_kwh": m_pol.energy_kwh - m_base.energy_kwh,
            "uncomfort_min": m_pol.uncomfort_min - m_base.uncomfort_min,
            "kwh_per_cooling_deg_c": m_pol.kwh_per_cooling_deg_c - m_base.kwh_per_cooling_deg_c,
        },
        "savings_pct": safe_div((m_base.energy_kwh - m_pol.energy_kwh), max(1e-6, m_base.energy_kwh)) * 100.0,
    }

    
    out_path = out_dir / "eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote {out_path}")

def load_policy(policy_path: str) -> TorchPolicyModel:
    return TorchPolicyModel(policy_path)



if __name__ == "__main__":
    main()