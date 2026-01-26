# and-hvac-v1

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

docker run --rm -it -v "$(pwd)/shared:/shared" andruix/eplus:latest --rollout-id smoke-001 --outdir /shared/results/smoke-001 --rollout-dir /shared/rollouts/inbox/smoke-001 --start-date 06/10 --end-date 06/17 
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
  --obs-dim 2 `
  --act-dim 1 `
  --max-workers 1 `
  --rollout-days 7 `
  --batch-size 64 `
  --min-replay-before-train 64 `
  --train-steps-per-rollout 10 `
  --publish-every-rollouts 1 `
  --env ANDRUIX_START_MMDD=06/10 `
  --env ANDRUIX_END_MMDD=06/17 `
  --env ANDRUIX_POLICY_KIND=torch

'''

