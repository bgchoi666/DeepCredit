from common import *

parser = ArgumentParser(description='Select generated cluster')
parser.add_argument('--scheduler', help='Scheduler server', default='server1')
parser.add_argument('--ini', help='Server config ini file', default='server_config.ini')


if __name__ == '__main__':
    ## 1. Parse arguments
    args = parser.parse_args()


    ## 2. Load server config
    configs = ini2dict(args.ini)


    ## 3. Get client
    config_scheduler = configs[args.scheduler]
    client = Client(f"{config_scheduler['host']}:{config_scheduler['scheduler_port']}")
    print(client)


    ## 4. Run tasks
    ## 4.1 Define task
    def task(id):
        ## 1. Do something
        ini2dict('server_config.ini')
        sleep(1)
        return dict(id=id)
    params = range(10)

    ## 4.2 Run tasks
    s = time()
    futures = client.map(task, params)
    print("- Futures:", [future.status for future in futures])


    ## 5. Print result
    results = client.gather(futures)
    print(f"* Elapsed time: {time() - s:.2f}s")
    print("- Results:", [future.status for future in futures])
    for result in results:
        print(result)