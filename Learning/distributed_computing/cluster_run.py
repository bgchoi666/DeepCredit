from common import *

parser = ArgumentParser(description='Run a command on a cluster')
parser.add_argument('--cmd', help='Command to run', required=True)
parser.add_argument('--ini', help='Server config ini file', default='server_config.ini')


if __name__ == '__main__':
    ## 1. Parse arguments
    args = parser.parse_args()


    ## 2. Load server config
    configs = ini2dict(args.ini)


    ## 3. Run the command
    asyncio.run(run_clients(configs, args.cmd))