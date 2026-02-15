import sys
import os
from os import listdir, makedirs
from platform import uname
from os.path import join, abspath, isdir, dirname
from shutil import rmtree
from time import time, sleep
import joblib
from configparser import ConfigParser
from argparse import ArgumentParser
import logging

import asyncio, asyncssh
from dask.distributed import Client, as_completed


## ini file parser
def ini2dict(path):
    config = ConfigParser()
    config.read(path)
    return dict(config._sections)


## Run command for remote client
async def run_client(config, cmd):
    async with asyncssh.connect(host=config['host'], port=int(config['ssh_port']), username=config['username']) as conn:
        return await conn.run(cmd)


async def run_clients(configs, cmd):
    ## 1. Create a list of coroutines then, run them in parallel
    tasks   = (run_client(config, cmd) for config in configs.values())
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ## 2. Print results
    print(f"* Command: {cmd} \n")
    for config, result in zip(configs.values(), results):
        task_name = f"[{config['username']}@{config['host']}:{config['ssh_port']}]"
        if isinstance(result, Exception):
            print(f"{task_name} failed: {str(result)}")
        elif result.exit_status != 0:
            print(f"{task_name} exited with status {result.exit_status}")
            print(result.stderr, end='')
        else:
            print(f"{task_name} succeeded")
            print(result.stdout, end='')
        print(75*'-')


async def run_client_background(config, cmd, timeout):
    return await asyncio.wait([run_client(config, cmd)], timeout=timeout)


async def run_clients_background(configs, cmd, timeout):
    if isinstance(cmd, list) and len(cmd) == len(configs):
        params = zip(configs, cmd)
    else:
        params = zip(configs, len(configs)*[cmd])
    return await asyncio.wait([run_client(*param) for param in params], timeout=timeout)


## Utility functions
def get_worker_info(client):
    workers = client._scheduler_identity.get("workers", {})
    return {'nprocs': len(workers), 'nthreads': sum(w["nthreads"] for w in workers.values())}