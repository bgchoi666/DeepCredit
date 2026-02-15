from sklearn.model_selection import train_test_split
from Learning.distributed_computing.common import *
from Optimization.optimizer import Optimizer
from Prediction.Save_result import save_result
from Learning.Imbalance import imbalance_data
from Learning.models import *

def data_split(data, params):
    test_size= eval(params["testSize"])
    val_size = eval(params["validationSize"])
    random_state= eval(params["randomState"])

    x_train, x_test = train_test_split(data, test_size=test_size, random_state=random_state)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, test_size=val_size, random_state=random_state)

    # Random selection 70% * 0.5 = 35%
    #idx = np.random.choice(len(y_train), int(len(y_train)*0.35), replace=False)
    #x_train = x_train[idx]
    #y_train = y_train[idx]

    return x_train, x_test

def training(x_train, y_train, batch_param, batch_info):
    ini_path = join(dirname(__file__), 'distributed_computing/server_config.ini')
    scheduler_server = 'server1'

    ## Load server config
    configs = ini2dict(ini_path)

    ## Get client
    config_scheduler = configs[scheduler_server]
    client = Client(f"{config_scheduler['host']}:{config_scheduler['scheduler_port']}")
    print(client)

    param_Args  = dict(x_train =x_train,
                       y_train =y_train,
                       batch_info=batch_info,
                       batch_param =batch_param)
    model_list = get_model_list(batch_info["model_group"]) # model_no list

    ## Set parameters
    parallel_args = [(model_info, param_Args) for _, model_info in model_list.iterrows()]


    def task(parallel_args):
        import sys
        import traceback

        try:
            logfile = f"/opt/data/DeepCredit/test.log"
            sys.stdout = open(logfile, "a")

            model_info, Args = parallel_args
            # model_no: model_list row

            mod_path = join(dirname(dirname(__file__)))
            sys.path.append(mod_path)
            sys.path.append("/opt/data/DeepCredit/")

            from Learning.Imbalance import imbalance_data
            from Learning.models import get_model_id, save_model
            from Prediction.Save_result import save_result

            # Resolving data imbalance through resampling
            x_resampled, y_resampled = imbalance_data(Args['x_train'], Args['y_train'], model_info)
            # Fit
            model_id = get_model_id(model_info["model_id"])['name']
            model = Model_collections(model_id)
            model = model.model_fit(x_resampled, y_resampled, model_info.model_param)
            save_result(None, batch_info, model_info)
            save_model(model, batch_info["batch_id"], model_info, 0)

            # opt = Optimizer("Bayesian")
            # opt.run(parallel_args)

        except Exception as e:
            print(traceback.format_exc())
            raise e

        return

    ## Run tasks
    start_time = time()
    futures = client.map(task, parallel_args)
    tasks = list(as_completed(futures, with_results=False))  # realize lazy function
    print(f"* Elapsed time: {time() - start_time:.2f}s")
    print("- Results:", [task.status for task in tasks])