from common import *

parser = ArgumentParser(description='Select generated cluster')
parser.add_argument('--scheduler', help='Scheduler server', default='server1')
parser.add_argument('--ini', help='Server config ini file', default='server_config.ini')
parser.add_argument('--result_dir', help='Directory path for saved result', default='result')


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
    ### 4.1 Define task
    def task(param):
        id, transfer_info = param

        ## 1. Append entry path to sys.path (current working directory is reset to ~/)
        entry_path = dirname(__file__)
        sys.path.append(entry_path)

        ## 2. Do something
        from common import ini2dict
        ini2dict('server_config.ini')
        sleep(1)
        result = dict(id=id)

        ## 3. Save result with file
        makedirs(transfer_info['result_dir_path'], exist_ok=True)
        src_file_path = abspath(join(transfer_info['result_dir_path'], f'{id}.joblib'))
        dst_file_path = abspath(join(transfer_info['result_dir_path'], f'[{uname()[1]}]{id}.joblib'))
        joblib.dump(result, src_file_path)

        ## 4. Transfer
        config_scheduler = transfer_info['config_scheduler']
        os.system(f"scp -P {config_scheduler['ssh_port']} {src_file_path} {config_scheduler['username']}@{config_scheduler['host']}:{dst_file_path}")
        os.remove(src_file_path)

    ### 4.2 Set parameters
    result_dir_path = abspath(args.result_dir)
    if isdir(result_dir_path):  rmtree(result_dir_path)
    makedirs(result_dir_path)
    ids           = range(10)
    transfer_info = dict(result_dir_path=result_dir_path, config_scheduler=config_scheduler)
    params        = [(id, transfer_info) for id in ids]

    ### 4.3 Run tasks
    s = time()
    futures = client.map(task, params)
    print("- Futures:", [future.status for future in futures])


    ## 5. Print result
    tasks = list(as_completed(futures))  # wait until all tasks are completed
    results = [joblib.load(join(result_dir_path, name)) for name in listdir(result_dir_path)]
    print(f"* Elapsed time: {time() - s:.2f}s")
    print("- Results:", [task.status for task in tasks])
    for name, result in zip(listdir(result_dir_path), results):
        print(f"{name}: {result}")