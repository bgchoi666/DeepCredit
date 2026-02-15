from common import *

parser = ArgumentParser(description='Start establishing connections of cluster')
parser.add_argument('--scheduler', help='Scheduler server', default='server1')
parser.add_argument('--worker', help='Worker server', default='server1 server2 server3 server4')
parser.add_argument('--ini', help='Server config ini file', default='server_config.ini')


if __name__ == '__main__':
    ## 1. Parse arguments
    args = parser.parse_args()


    ## 2. Load server config
    configs = ini2dict(args.ini)


    ## 3. Cleanup old cluster
    asyncio.run(run_clients(configs, "pkill dask"))
    sleep(1)


    ## 4. Open scheduler server
    config_scheduler = configs[args.scheduler]
    log_path         = abspath(join(dirname(__file__), 'scheduler.log'))
    cmd_scheduler    = f"source /opt/conda/bin/activate rapids; \
                         rm {log_path}; \
                         nohup dask-scheduler >> {log_path} 2>&1"
    asyncio.run(run_client_background(config_scheduler, cmd_scheduler, timeout=10))


    ## 5. Connect workers to scheduler
    config_workers = []
    cmd_workers    = []
    for config_worker in [config for id, config in configs.items() if id in args.worker.split()]:
        for gpu in range(int(config_worker['gpus'])):
            log_path = abspath(join(dirname(__file__), f'worker--gpu{gpu}.log'))
            cmd_worker = f"source /opt/conda/bin/activate rapids; \
                           export CUDA_VISIBLE_DEVICES={gpu}; \
                           rm {log_path}; \
                           nohup dask-worker {config_scheduler['host']}:{config_scheduler['scheduler_port']} --nthreads {config_worker['nthreads']} --nprocs {config_worker['nprocs']} --memory-limit='20 GiB' --no-reconnect >> {log_path} 2>&1"
            config_workers.append(config_worker)
            cmd_workers.append(cmd_worker)
    asyncio.run(run_clients_background(config_workers, cmd_workers, timeout=10))


    ## 6. Check connections
    client = Client(f"{config_scheduler['host']}:{config_scheduler['scheduler_port']}")
    print(f"* Number of processes for connected workers: {get_worker_info(client)['nprocs']}")