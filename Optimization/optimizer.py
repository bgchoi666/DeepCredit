import json

import pandas as pd
from GPyOpt.methods import BayesianOptimization
from Learning.models import *
from Learning.Imbalance import imbalance_data
from Prediction.Save_result import save_result

class Bayesian:
    def __init__(self):
        self.args = None

    def domain(self):
        args = deepcopy(self.args)
        model_info = args[0]
        model_param = model_info.model_param

        domains = []
        opt_params = model_param["opt_params"]
        param_type = model_param["param_type"]
        opt_param_list = opt_params["model_param"] + opt_params["resampling_param"]

        for name in opt_param_list:
            # 임시용 : parma_info
            param_info = pd.DataFrame()
            if param_type == 'categorical':
                    n_category = len(param_info['name'])
                    domain = {'name':'name', 'type':param_type, 'domain':np.arange(n_category)}
            elif param_type == 'discrete':
                    domain = {'name': name, 'type': param_type,
                           'domain': np.arange(param_info[name][0], param_info[name][1], dtype=int)}
            elif param_type == 'continuous':
                    domain = {'name': name, 'type':param_type,'domain': list(map(float, param_info[name]))}
            else:
                "Invalid domain parameter type."
                continue

            domains.append(domain)

        return domains

    def objective_function(self, opt_args):
        """

        :param args: optimize parameters -> hyper_param
        :return:
        """
        args = deepcopy(self.args)
        model_info = args[0]
        batch_info = args[1]["batch_info"]

        model_param = json.loads(model_info['model_param'])
        resampling_param = json.loads(model_info['resampling_param'])

        opt_params = model_param.opt_params

        # model_no = args[0]

        # Apply Bayesian parameters
        # copy_input['param'][name] = bool(int(args[0, i])
        opt_param_list = opt_params["model_param"] + opt_params["resampling_param"]

        for i in range(len(opt_param_list)):
            name = model_param.opt_param_list[i]

            if name in opt_params["model_param"]:
                model_param[name] = opt_args[0, i]
            else: # name in opt_param_list["resampling_param"]
                resampling_param[name] = opt_args[0, i]

        model_info["model_param"] = model_param
        model_info["resampling_param"] = resampling_param

        # Resolving data imbalance through resampling.
        x_resampled, y_resampled = imbalance_data(args[1]['x_train'], args[1]['y_train'], model_info)
        ## Fit
        model_id = get_model_id(model_info["model_id"])
        model = Model_collections(model_id)
        model = model.model_fit(x_resampled, y_resampled, model_info.model_param)
        # Performance
        val_acc = model.history.history['val_acc'][-1]

        ## Save
        # Read model_sub_no
        stmt = text("""SELECT MAX(model_sub_no) AS sub_no FROM dc_batch_result
                        WHERE batch_id=:batch_id AND model_no=:model_no""")
        param = dict(batch_id=batch_info['batch_id'],
                     model_no=model_info['model_no'])
        model_sub_no = dbhandler.retrive_stmt(stmt, param).sub_no.iloc[0] + 1

        save_result(None, batch_info, model_info)
        save_model(model, batch_info["batch_id"], model_info.model_no, model_sub_no)

        return 1 - val_acc
        

class Grid:
    def __init__(self):
        self.args = None

    def run(self):
        args = deepcopy(self.args)
        model_no = args[0]
        params = args[1]['param']
        model_name = get_model_id(model_no)
        model = Model_collections(model_name)
        model = model.model_fit(args[1]['x_train'], args[1]['y_train'])

        stmt = text("""SELECT MAX(model_sub_no) AS sub_no FROM dc_batch_result
                                WHERE batch_id=:batch_id AND model_no=:model_no""")
        param = dict(batch_id=args[1]['batch_info']['batch_id'],
                     model_no=model_no)
        model_sub_no = dbhandler.retrive_stmt(stmt, param).sub_no.iloc[0] + 1

        batch_info = params.get("batch_info")
        save_model(model, batch_info, model_no, model_sub_no)

class Optimizer(Bayesian, Grid):
    def __init__(self, method="Grid"):
        super().__init__()
        self.method = method # opt_mode
        self.args = None

    def run(self, *args):
        # model_no, param_Args(x,y,batch_info,param) = parallel_args

        method = self.method
        self.args = args[0]

        if method == "Grid":
            return self._Grid()
        elif method == "Bayesian":
            return self._Bayesian()

    def _Grid(self):
        task = [delayed(self.run())(input) for idx_exp, input in enumerate(self.args)]
        compute(*task, scheduler="threads")

    def _Bayesian(self):
        # self.args : [model_info, Args]
        model_param = json.loads(self.args[0]['model_param'])
        resampling_param = json.loads(self.args[0]['resampling_param'])

        minimize = model_param.get("minimize")

        initial_design_numdata = model_param.get("initial_design_numdata")
        max_iter = model_param.get("max_iter")

       # Create Domain
        domain = self.domain()

       # Bayesian Optimizaion
        Bopt = BayesianOptimization(f=self.objective_function,
                                    domain=domain,
                                    minimize=minimize,
                                    initial_design_numdata=initial_design_numdata)
        Bopt.run_optimization(max_iter=max_iter)