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
docker build -t andruix/eplus:latest -f Dockerfile .

docker run --rm -it -v "${pwd}\shared:/shared" andruix/eplus:latest `
  --rollout-id raw-test `
  --rollout-dir /shared/rollouts/inbox/raw-test `
  --outdir /shared/results/raw-test `
  --start-date 06/10 --end-date 06/17 `
  --policy-kind torch `
  --reward-mode raw `
  --reward-scale 3600000
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
docker build -t andruix/eplus:latest -f Dockerfile .

python .\andruix_orchestrator_td3.py `
  --shared-root .\shared `
  --image andruix/eplus:latest `
  --obs-dim 11 `
  --act-dim 1 `
  --max-workers 2 `
  --rollout-days 7 `
  --batch-size 64 `
  --min-replay-before-train 6000 `
  --train-steps-per-rollout 200 `
  --publish-every-rollouts 5 `
  --actor-lr=3e-4 `
  --critic-lr=1e-4 `
  --env EPLUS_START_MMDD=07/15 `
  --env EPLUS_END_MMDD=07/22 `
  --env ANDRUIX_POLICY_KIND=torch `
  --tb-run-name denver_july15_smoke2 `
  --env ANDRUIX_REWARD_MODE=raw `
  --env ANDRUIX_REWARD_SCALE=3600000

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
