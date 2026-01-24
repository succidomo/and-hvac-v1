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
docker run --rm -it -v "$(pwd)/shared:/shared" andruix/eplus:latest \
  --rollout-id smoke-001 \
  --out-dir /shared/rollouts/inbox/smoke-001 \
  --start-mmdd 06/10 --end-mmdd 06/17 \
  --seed 123
'''

