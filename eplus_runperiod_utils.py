"""Small utilities for patching EnergyPlus RunPeriod in an IDF.

Keep these helpers separate so your sim runner file stays readable as it grows.
"""

from __future__ import annotations

from pathlib import Path
import re
import datetime as dt


def parse_mmdd(s: str) -> tuple[int, int]:
    """Parse an 'MM/DD' or 'MM-DD' string into (month, day) with validation."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty date string")
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})$", s)
    if not m:
        raise ValueError(f"Expected MM/DD, got: {s!r}")
    month = int(m.group(1))
    day = int(m.group(2))
    # validate with a non-leap year (EnergyPlus EPW typically is 365-day)
    dt.date(2021, month, day)
    return month, day


def rewrite_first_runperiod(idf_in: Path, idf_out: Path, start_mmdd: str, end_mmdd: str) -> None:
    sm, sd = parse_mmdd(start_mmdd)
    em, ed = parse_mmdd(end_mmdd)

    lines = idf_in.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    # find first RunPeriod object block
    def is_runperiod_start(line: str) -> bool:
        stripped = line.lstrip()
        if stripped.startswith("!"):
            return False
        token = stripped.split(",")[0].strip().lower()
        return token == "runperiod"

    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and is_runperiod_start(line):
            start_idx = i
            continue
        if start_idx is not None and end_idx is None and ";" in line:
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError("Could not locate a RunPeriod object block")

    block = lines[start_idx:end_idx + 1]

    # helper: replace the numeric part before the comma, keep indentation + comments
    def replace_value(line: str, new_val: str) -> str:
        # keep everything after first comma (comments, spacing)
        if "," not in line:
            return line
        prefix = line.split(",")[0]
        suffix = line[len(prefix):]  # includes comma onward
        # preserve original indentation
        indent = prefix[:len(prefix) - len(prefix.lstrip())]
        return f"{indent}{new_val}{suffix}"

    # block layout (typical):
    # 0: RunPeriod,
    # 1: Name
    # 2: Begin Month
    # 3: Begin Day
    # 4: Begin Year (often blank)
    # 5: End Month
    # 6: End Day
    # 7: End Year (often blank)
    #
    # We only touch indices that exist.
    if len(block) >= 7:
        block[1] = replace_value(block[1], "RL_RunPeriod")  # optional
        block[2] = replace_value(block[2], str(sm))
        block[3] = replace_value(block[3], str(sd))
        block[5] = replace_value(block[5], str(em))
        block[6] = replace_value(block[6], str(ed))
    else:
        raise RuntimeError("Unexpected RunPeriod block structure; not enough lines")

    out_lines = lines[:start_idx] + block + lines[end_idx + 1:]
    idf_out.write_text("".join(out_lines), encoding="utf-8")

