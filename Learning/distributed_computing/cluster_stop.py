from common import *

parser = ArgumentParser(description='Stop established connections of cluster')
parser.add_argument('--scheduler', help='Scheduler server', default='server2')
parser.add_argument('--ini', help='Server config ini file', default='server_config.ini')


if __name__ == '__main__':
    ## 1. Parse arguments
    args = parser.parse_args()


    ## 2. Load server config
    configs = ini2dict(args.ini)


    ## 3. Get scheduler address
    config_scheduler = configs[args.scheduler]
    cmd_scheduler    = "source /opt/conda/bin/activate rapids; \
                        dask-scheduler"


    ## 4. Close cluster
    asyncio.run(run_clients(configs, "pkill dask"))
    # try:
    #     from dask.distributed import Client
    #     client = Client(f"{config_scheduler['host']}:{config_scheduler['scheduler_port']}")
    #     client.shutdown()  # shutdown() may make permission error
    # except:
    #     asyncio.run(run_clients(configs, "pkill dask"))