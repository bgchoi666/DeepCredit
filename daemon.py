# -*- coding: utf-8 -*-
"""
When running the deamon program , server ID and the number of multiprocess are given as parameters.
Ex:
.. code-block::
 python autoDemon.py SERVER_ID n_PROCESS 식별자
"""

import platform
import atexit
import signal
from ENV.DB_Handler import DBHandler
from sqlalchemy.sql import text
import gpustat
import sys, os
import subprocess
from time import sleep
import datetime

dbHandler = DBHandler()

def get_task(server_id, project_name):
    """
    execution process
    :param server_id : server ID
    :param project_name : name of project currently being executed
    """
    log_exe(server_id)
    print(f"Current Server ID : {server_id}")
    print(f"Current Project Name : {project_name}")

    # Check memory.
    if platform.system() == "Windows":
        print(f"Memory check skip.")
    else:
        # Waiting when memory is less than 2G..
        jsonString = gpustat.GPUStatCollection.new_query().jsonify()
        gpu_index = 0
        total_memory = jsonString['gpus'][gpu_index]['memory.total']
        used_memory = jsonString['gpus'][gpu_index]['memory.used']
        for _ in range(100):
            if total_memory - used_memory < 2000: # Mbyte
                print(f"Waiting when memory is less than 2G. The current memory is {total_memory - used_memory} Mb.")
                sys.stdout.flush()
                sleep(60)
            else:
                break

        print(f"The current memory is {used_memory} Mb.")

    # Process execution
    now = datetime.now()
    print(f" >>>>>>>>>> LOG: {now.year}Y {now.month}M {now.day}D {now.hour}h {now.minute}m {now.second}s")
    print(" [ START  ] Run the process.")
    sys.stdout.flush()

    # Change current working directory: (cwd='IndexTracking_Engine_pkg')
    p = subprocess.Popen(['python', 'launcher.py', server_id, project_name])
    stdout, stderr = p.communicate()
    print(f" [ STDOUT ] {stdout}")
    print(f" [ STDERR ] {stderr}")

    print(" ### Process has been terminated. Run it again after a while. ###")
    sys.stdout.flush()
    sleep(2)

def handle_exit():
    """
    Handles errors (termination) for the job being execute.
    """
    print(f"'[ DAEMON ] Exit the {sys.argv[0]}...'")
    if p is not None:
        os.kill(p.pid,signal.SIGTERM)

        #Handles errors (termination) for the job being execute.

        stmt = text("UPDATE om_batch_slide SET state=:state1 WHERE server=:server AND state=:state2")
        param = dict(
            state1 = 'E',
            server = server_id,
            state2 = 'U'
        )
        dbHandler.execute_stmt(stmt, param)

def exits():
    """
    Termination signal handling
    """
    atexit.register(handle_exit) # Register 'handle_exit' execution at the termination
    signal.signal(signal.SIGTERM, handle_exit) # termination signal
    signal.signal(signal.SIGINT, handle_exit) # Keyboard interrupt occurs

def log_exe(server_id):
    """
    Log file is created and the start of this program is recorded.
    :param server_id: Server ID
    """
    if not "logs" in os.listdir(os.getcwd()):
        os.mkdir("logs")
    file_name = "./logs/"+str(server_id)+".log"
    sys.stdout = open(file_name, "a")
    print("------------------------------------------------------------")
    print(f"'[ DAEMON ] Start {server_id}.'")
    print("------------------------------------------------------------")


if __name__ == '__main__':
    thread_list = []
    while 1:
        global p

        server_id = sys.argv[1]
        n_process = int(sys.argv[2])
        project_name = sys.argv[3]

        exits()

        get_task(server_id, project_name)

        ##========== threding ===========##
        """
        print(f" The number of process currently being executed : {threading.active_count() - 1}")
        if (threading.active_count() - 1) < n_process:
            th = threading.Thread(target=get_task, args=(server_id, project_name))
            th.start()

        for thread in thread_list:
            if thread.is_alive():
                continue
            thread.join()
            thread_list.remove(thread)
            print('thread exit: ', thread.name)

        sleep(5)
        """