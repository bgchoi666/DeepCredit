from Data_Preprocessing.Read_data import read_data
from Data_Preprocessing.preprocessing import preprocessing
from Learning.models import load_model
from Learning.Training import training
from Prediction.Prediction import predict, get_trained_model_list
from Ensemble.Ensemble import Ensemble
from Prediction.Save_result import save_result
from ENV.env import *

def DeepCredit_main(batch_info, batch_param):
    mode = batch_info["mode"]
    
    if mode != "ensemble":  # 임시
        # Read Data
        data = read_data(batch_info)
    
        X, Y = preprocessing(data, mode, batch_info, batch_param)

    # 학습
    if mode == "train":

        training(X, Y, batch_param, batch_info)

    # 예측
    elif mode == "predict":
        # trained model list
        model_list = get_trained_model_list(batch_info['train_batch_id'])

        def task(batch_info, model_info):
            # Load Model
            model = load_model(batch_info.train_batch_id, model_info.model_no, model_info.model_sub_no)

            # Prdiction
            result = predict(model, X, Y)

            # Save Result
            save_result(result, batch_info, model_info)

        tasks = [delayed(task)(batch_info, model_info) for _, model_info in model_list.iterrows()]
        compute(*tasks, scheduler="threads")

    # 앙상블
    elif mode == "ensemble":
        ensemble = Ensemble()

        print("앙상블 시작")
        start = time.time()

        ensemble.ensemble(batch_info, batch_param)

        end = time.time()
        print("앙상블 종료", end - start, 's')

    # 실예측
    elif mode == "real":
        pass
