# -*- coding: utf-8 -*-
import traceback
import sys, os
import shutil
import logging
import datetime
from time import sleep
from ENV.DB_Handler import DBHandler
from sqlalchemy.sql import text
import numpy as np
import json
from main import DeepCredit_main

class Launcher:
    """
    batch autoRun
    :param sleep_time: Wait time before re-run
    """

    def __init__(self, sleep_time):
        self.params = None
        self.dbHandler, self.result = DBHandler(), None

        self.server_id = None

        # batch column parameter
        self.batch_info = None

        self.raw_input = {}

        # 1. Server ID
        try:
            self.server_id = sys.argv[1]
            print(f'## SERVER ID = {self.server_id}')

        except:
            self.exit_handler(message="# Invalid input value. Please check again.", log_flag=False)

        # 2.Declares hardcoding parameters for the class.
        self.sleep_time = sleep_time

        # 3. Start logging.
        # Log settings
        self.logger = logging.getLogger(__name__)
        self.streamHandler = logging.StreamHandler()
        self.logger.addHandler(self.streamHandler)
        self.streamHandler.setLevel(level=logging.WARNING)
        self.now = datetime.now()

        # Start log (create in logs folder)
        if not "logs" in os.listdir(os.getcwd()):
            os.mkdir("logs")

        logfile = f"./logs/{self.server_id}.log"

        # Backup log (if 5M or higher)
        try:
            baklogfile = f"./logs/backup/{self.now.strftime('%Y%m%d_%H%M%S')}.log"
            if os.path.isfile(logfile) and os.stat(logfile).st_size > 5000000 and not os.path.isfile(baklogfile):
                if not "backup" in os.listdir(os.getcwd() + "/logs"):
                    os.mkdir("./logs/backup")
                shutil.move(logfile, baklogfile)
        except:
            print("log file backup error.")

        sys.stdout = open(logfile, "a")
        print(
            "'------------------------------------------------------------------------------------------------------'")
        print(f"'>>> LOG: {self.now.year}Y {self.now.month}M {self.now.day}D {self.now.hour}h {self.now.minute}m'")

    def __del__(self):
        self.dbHandler.engine.dispose()
        print("## Connection is closed.")


    def read_dc_batch(self):
        """
        After reading the first data from 'dc_batch',change the 'state' to Processing('U').
        It is lock so that other process do not read the same data.
        :return: True or False
        """
        try:
            with self.dbHandler.engine.begin() as transaction:
                # 1. Read not state='C'
                #  E : 5일 이내건만 실행 & 20분 후 재시도
                #  U : 1시간 후 재시도
                stmt = text(
                    "SELECT batch_id, state FROM dc_batch WHERE state = 'I' "
                    "ORDER BY FIELD(state, 'I') DESC, batch_id limit 10")
                self.result = self.dbHandler.retrive_stmt(stmt, param=None, transaction=transaction)

                if self.result.empty:
                    print("# There is no more work to be done.")
                    print(f"## Check 'dc_batch' again after {self.sleep_time} seconds.")
                    sleep(self.sleep_time)
                    return False

                else:
                    # 중복 채번으로 성능저하 발생하여 10개 내에서 랜덤으로 변경
                    no = np.random.randint(0, len(self.result))
                    self.result = self.result.iloc[no]

                stmt = text("SELECT * FROM dc_batch WHERE batch_id=:batch_id AND state=:state FOR UPDATE")
                param = dict(batch_id=int(self.result['batch_id']),
                             state=self.result['state'])
                result = self.dbHandler.retrive_stmt(stmt, param, transaction=transaction)

                if result.empty:
                    print("# There is no more work to be done.")
                    print(f"## Check 'dc_batch' again after {self.sleep_time} seconds.")
                    sleep(self.sleep_time)
                    return False

                else:
                    self.result = result.iloc[0]

                # 2. If there is no imported history(if all jobs have been processed), wait and run the task again.
                if self.result is None:
                    print("#   Work already has been done in aother process.")
                    print(f"## Check 'dc_batch' again after {self.sleep_time} seconds.")
                    sleep(self.sleep_time)
                    return False

                # 3. When processing an imported job, change state = 'U' (meaning 'the job is processing').
                # Update server_id name.
                print(f"+ batch_id : {self.result['batch_id']}")

                stmt = text("UPDATE dc_batch SET state='U', serve_server_id=:server_id, work_date = NOW() "
                            "WHERE batch_id=:batch_id")
                param = dict(server_id=self.server_id,
                             batch_id =self.result['batch_id'])
                self.dbHandler.execute_stmt(stmt, param, transaction=transaction)

                return True

        except Exception as ex:
            self.exception_handler(exception_massage=ex, sleep_flag=True)
            return False

    def get_batch_param(self):
        if self.result["mode"] == "predict":
            # Search train_batch param
            stmt = text(f"SELECT batch_param FROM dc_batch WHERE batch_id={self.result['train_batch_id']}")
            batch_param = self.dbHandler.retrive_stmt(stmt).loc[0][0]
            return json.loads(batch_param)
        else: # train
            return self.result['batch_param']

    def set_parameters(self):
        """
        Setting inputs parameters of main program.
        :return: True or False
        """
        try:
            print("## Setting Parameters...")

            self.params = json.loads(self.get_batch_param())

            self.batch_info = dict(batch_id  = self.result['batch_id'],
                                   batch_desc= self.result['batch_desc'],
                                   batch_memo= self.result['batch_memo'],
                                   script_no = self.result['script_no'],
                                   mode      = self.result['model'],
                                   train_batch_id  = self.result['train_batch_id'],
                                   dataset_group   = self.result['dataset_group'],
                                   dataset_version= self.result['dataset_version'],
                                   model_grop= self.result['model_grop'],
                                   gpu_no    = self.result['gpu_no'])

            print("[success] setting parameters")
            return True

        except Exception as ex:
            self.exception_handler(exception_massage=ex, sleep_flag=False)
            return False

    def execute_main(self):
        """
        Main Programs (Batch Launcher System)
        :return: normal or error
        """
        try:
            DeepCredit_main(self.batch_info, self.params)
            return "normal"

        except Exception as ex:
            # If the main program proceeds abnormally, an exception is returned.
            self.exception_handler(exception_massage=ex, sleep_flag=False)
            return ex

    def finish_batch(self, termination):
        """
        Finishing work. Completion processing.
        :param termination: retrun value of execute_main().
        :return: True or False
        """
        print("Start [finish_batch] method")

        try:
            # Update state
            if termination == "normal":
                print("[finish_batch] normal termination")
                sys.stdout.flush()

                stmt = text("UPDATE dc_batch SET state=:state, server_id=:server_id, work_end_date=now() "
                            "WHERE batch_id=:batch_id")
                param = dict(
                    state='C',
                    serve_server_id=self.server_id,
                    batch_id=self.result['batch_id'],
                )
                self.dbHandler.execute_stmt(stmt, param)

            # 2.When abnormally terminate(when exception is received as a parameter), it is modified to state = 'E' in the DB.
            else:
                print("[finish_job] except termination")
                stmt = text("UPDATE dc_batch SET state=:state, work_end_date=now() "
                            "WHERE batch_id=:batch_id")
                param = dict(
                    state='E',
                    batch_id=self.result['batch_id'])
                self.dbHandler.execute_stmt(stmt, param)

            print("Finish [finish_batch] method")
            sys.stdout.flush()
            return True

        except Exception as ex:
            print("Error occur in [finish_batch].")
            sys.stdout.flush()
            self.exception_handler(exception_massage=ex, sleep_flag=False)
            return False


    def exception_handler(self, exception_massage=None, sleep_flag=False):
        """
        The name of the method in which the error occurred is saved in the log.
        Record detailed exception in log file.
        (Reference:https://stackoverflow.com/questions/3702675/how-to-print-the-full-traceback-without-halting-the-program)
        :param exception_massage:
        :param sleep_flag:
        """
        print(f"[ error ] An error has occured in '{sys._getframe(1).f_code.co_name}' ")
        print("#  Error occurred :", exception_massage)
        print(traceback.format_exc())
        if sleep_flag:
            sleep(self.sleep_time)

    def exit_handler(self, message=None, log_flag=False):
        """
        Handle exit.
        :param message: exception massage
        :param log_flag: sleep flag
        """
        print(message)
        print("# Exit the program..")
        if log_flag:
            self.logger.error(message)
            self.logger.error("# Exit the program..")
        exit()


##################################### SETTING ####################################
sleep_time = 10  # When a job fails, the number of seconds to wait to run the job agan.
loop_count = 100  # Maximum number of iterations.
###################################### MAIN #####################################

if __name__ == "__main__":
    print(">>> auto_run execution")
    launcher = Launcher(sleep_time)

    loop = 0
    while loop <= loop_count:
        loop = loop + 1

        print(
            "'------------------------------------------------------------------------------------------------------'")
        sys.stdout.flush()
        print(">> read_dc_batch execution")
        if not launcher.read_dc_batch(): continue  # Read paramerts to 'batch_id' in 'dc_batch'
        print(">> set_parameters execution")
        if not launcher.set_parameters(): continue  # Set paramerts read-> inputs
        print(">> execute_main execution")
        sys.stdout.flush()
        termination = launcher.execute_main()  # Process execution
        print(">> finish_batch execution")
        if not launcher.finish_batch(
            termination): continue  # Handles termination of executed list in 'batch' (state = 'E' of state = 'C')
        sys.exit()