# and-hvac-v1

### Connect to vm 
'''
ssh -i ~/.ssh/and_gpu.pem anduser@20.57.140.254
'''

### Build docker image and run locally, dump logs
'''
docker build -t eplus-hello .
docker run --name eplus-test eplus-hello
docker logs eplus-test *>> .\eplus-test-logs.txt
'''

### Mount a host folder as the results directory 
'''
mkdir results | Out-Null

docker run --rm --name eplus-test -v "${PWD}\results:/home/guser/results" eplus-hello

'''

### Smoke test docker file 
'''
docker build -t andruix/eplus:latest -f DockerFile .

docker run --rm \
  -v ~/arduix/and-hvac-v1/shared:/shared \
  andruix/eplus:latest \
  --outdir /shared/results/test_short \
  --rollout-dir /shared/rollouts/inbox/test_short \
  --rollout-id test_short_5zones \
  --zones "PERIMETER_BOT_ZN_3,CORE_BOTTOM,PERIMETER_BOT_ZN_4,PERIMETER_BOT_ZN_2,PERIMETER_BOT_ZN_1" \
  --start-date 01/01 \
  --end-date 01/03 \
  --policy-kind simple \
  --energy-meter "Electricity:HVAC" \
  --reward-scale 3.6e6
'''

### create Shared Folder Once 
'''
mkdir shared -ErrorAction SilentlyContinue | Out-Null
mkdir shared\policy\latest -ErrorAction SilentlyContinue | Out-Null
mkdir shared\rollouts\inbox -ErrorAction SilentlyContinue | Out-Null
mkdir shared\rollouts\done -ErrorAction SilentlyContinue | Out-Null
mkdir shared\logs -ErrorAction SilentlyContinue | Out-Null
'''

### Smoke Test orchestrator file 
'''
source ~/arduix_env/bin/activate

docker build -t andruix/eplus:latest -f DockerFile .

# Add to your shell profile (so it's set every login)
echo 'export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=arduixpoca;AccountKey=<key>;EndpointSuffix=core.windows.net"' >> ~/.bashrc

# Or for current session only
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=arduixpoca;AccountKey=<key>;EndpointSuffix=core.windows.net;EndpointSuffix=core.windows.net"

python3 andruix_orchestrator_td3.py \
  --shared-root /home/anduser/arduix/and-hvac-v1/shared \
  --image andruix/eplus:latest \
  --obs-dim 15 \
  --act-dim 5 \
  --max-workers 3 \
  --rollout-days 7 \
  --batch-size 256 \
  --min-replay-before-train 10000 \
  --train-steps-per-rollout 500 \
  --publish-every-rollouts 1 \
  --gamma 0.98 \
  --actor-lr 1e-4 \
  --critic-lr 1e-4 \
  --policy-noise .2 \
  --tb-run-name denver_july15_mz_1_3_5 \
  --env EPLUS_START_MMDD=07/15 \
  --env EPLUS_END_MMDD=09/22 \
  --env "EPLUS_ZONE=PERIMETER_BOT_ZN_1,PERIMETER_BOT_ZN_2,PERIMETER_BOT_ZN_3,PERIMETER_BOT_ZN_4,CORE_BOTTOM" \
  --env "ANDRUIX_POLICY_KIND=torch" \
  --env "ANDRUIX_POLICY_PATH=/shared/policy/latest/policy.pt" \
  --env "ANDRUIX_REWARD_MODE=raw" \
  --env "ANDRUIX_REWARD_SCALE=1" \
  --env "ANDRUIX_OBS_OCC=1" \
  --env "ANDRUIX_OBS_NO_DOY=0" \
  --env "ANDRUIX_OBS_NO_TREND_15M=0" \
  --env "ANDRUIX_OBS_NO_TREND_60M=0" \
  --env "ANDRUIX_EXPLORE_NOISE=0.15" \
  --env ANDRUIX_TERMINAL_END=0

  sudo rm ./shared/policy/latest/* \
  sudo rm -rf ./shared/rollouts/done/* \
  sudo rm -rf ./shared/rollouts/inbox/* \
  sudo rm -rf ./shared/results/*

'''

### Remove Item for old training
'''
Remove-Item -Recurse -Force .\shared\rollouts\inbox\* -ErrorAction SilentlyContinue `
Remove-Item -Recurse -Force .\shared\rollouts\done\*  -ErrorAction SilentlyContinue `
Remove-Item -Recurse -Force .\shared\policy\latest\*  -ErrorAction SilentlyContinue `
Remove-Item -Recurse -Force .\shared\logs\* -ErrorAction SilentlyContinue
'''

### Start Tensorboard
'''
tensorboard --logdir .\shared\tb --host 0.0.0.0 --port 6006

Then Open

http://localhost:6006

'''
### install docker dependencies
'''
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git python3 python3-venv python3-pip
'''

### Install and enabled docker on linux
'''
# Remove any old Docker versions
sudo apt remove -y docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the Docker group (to run Docker without sudo)
sudo usermod -aG docker $USER

# Log out and log back in (or run `newgrp docker`) for the group change to take effect
exit  # Then SSH back in
'''

### Install azure cli linux
'''
# Install Azure CLI (official Microsoft way)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Log in to Azure (browser will open or you get a code to enter)
az login

# Log in to your specific ACR (this injects a fresh token into Docker)
az acr login --name arduixcr01

# Now try pull (no sudo needed)
docker pull arduixcr01.azurecr.io/arduix/eplus:latest
'''

### Setup  python env 
'''
# Create and activate virtual env
mkdir -p ~/arduix_env
python3 -m venv ~/arduix_env
source ~/arduix_env/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install numpy torch tensorboard tensorboardX  # tensorboardX if needed for older compat
pip install torch torchvision torchaudio  # For full Torch support
pip install azure-storage-blob azure-identity

# If you have a GPU and want CUDA (check with `nvidia-smi` if drivers are installed)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version as needed

# Deactivate when done (but reactivate before running the script)
deactivate

# Verify
source ~/arduix_env/bin/activate
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
deactivate
'''

### setup tensorboard on remote vm 
'''
1) start training script 

2) in another shell ssh in and execute 
# Activate venv again (if needed)
source ~/arduix_env/bin/activate

# Go to the directory where shared/tb lives
cd ~/arduix/and-hvac-v1

# Run TensorBoard pointing at the correct path
tensorboard --logdir shared/tb --host 0.0.0.0 --port 6006

3) Setup ssh tunnel from windows pc to remote vm 
ssh -i C:\Users\succi\.ssh\and_gpu.pem -f -N -L 6006:localhost:6006 anduser@20.57.140.254

4) visit browser 
http://localhost:6006

### Worker timeseries metrics + plotting (debug)

Each rollout worker writes a compact per-timestep timeseries file alongside the normal replay rollout:
- `shared/results/<rollout_id>/timeseries_<rollout_id>.parquet` (preferred)
- falls back to `.csv` if parquet deps aren’t available

#### How to add metrics (WorkerTimeseriesWriter)
The worker builds a single “row” per timestep in the callbacks and appends it via:
`self.ts_writer.append_step(..., extra_scalars={...})`

To add new metrics:
- **Scalar (single value per timestep):** add it to `extra_scalars` in `end_system_timestep_callback`
  - Example: `extra_scalars={"occupied": float(occupied), "comfort_pen": float(comfort_pen)}`
- **Per-zone scalar:** add a dict to one of the per-zone dict args (`zone_temps_c`, `zone_setpoints_heat_c`, `zone_setpoints_cool_c`, `zone_actions_norm`)
  - New per-zone series will show up as columns like `tz_c__CORE_BOTTOM`, `cool_sp_c__CORE_BOTTOM`, etc.

#### Plot a rollout’s timeseries
From your PC (or any machine with Python), run:

```bash
python plot_worker_rollout.py `
  --path .\timeseries\timeseries_fe698c28deb7.parquet `
  --zones "CORE_BOTTOM,PERIMETER_BOT_ZN_3" `
  --resample 15min `
  --start-step 5000 `
  --end-step 6500
```

#### PC dependencies for plotting - Create/activate a venv and install:
```bash
pip install pandas pyarrow matplotlib numpy
```

#### Example (run on your PC Powershell) to copy a single rollout’s parquet down to the current directory:
```bash
scp -i ~/.ssh/and_gpu.pem `
  anduser@20.57.140.254:/home/anduser/arduix/and-hvac-v1/shared/results/<rollout_id>/timeseries_<rollout_id>.parquet `
  .
```

#### Or copy all rollout timeseries files:
```
scp -i ~/.ssh/and_gpu.pem `
  anduser@20.57.140.254:/home/anduser/arduix/and-hvac-v1/shared/results/*/timeseries_*.parquet `
  .
```