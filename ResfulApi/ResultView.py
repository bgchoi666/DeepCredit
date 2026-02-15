# os.chdir("..")
# sys.path.append(".")

import pandas as pd

from ENV.DB_Handler import DBHandler
from sqlalchemy.sql import text

class ResultView:
    def get_batch_info(self, batch_id):
        dbHandler = DBHandler()

        # 베이스 모델 조회
        stmt = text(f"SELECT * FROM dc_batch WHERE batch_id = {batch_id};")

        param = dict(batch_id=batch_id)
        result = dbHandler.retrive_stmt(stmt, param=param)

        return result

    def get_ensemble_dist(self, batch_id, model_no, model_sub_no):
        dbHandler = DBHandler()

        # 베이스 모델 조회
        stmt = text(f"SELECT base_models "
                    f"  FROM dc_batch_result_t4 "
                    f" WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no};")

        param = dict(batch_id=batch_id, model_no=model_no, model_sub_no=model_sub_no)
        result = dbHandler.retrive_stmt(stmt, param=param)

        if len(result) == 0:
            return pd.DataFrame()

        base_models = result['base_models'][0].replace(",", "#").replace("-", ",").split("#")

        model_list = ""
        for row in range(len(base_models)):
            if model_list != "":
                model_list += ","

            model_list += "(" + base_models[row] + ")"

        # 베이스 모델로 예측 결과 조회
        stmt = text(f"SELECT replace(replace(predict_result, '\r\n', ''), ' ', '') predict_result "
            f"FROM dc_batch_result_t4_predict "
            f"WHERE batch_id = {batch_id} AND (model_no, model_sub_no) IN ({model_list})")
        param = dict(batch_id=batch_id, model_list=model_list)
        predict = dbHandler.retrive_stmt(stmt, param=param)

        predict_temp = []

        for row in range(len(predict)):
            predict_temp += predict.iloc[row]['predict_result'].split(',')

        def split_data(row):
            return row.split("-")

        predict_new = pd.DataFrame(list(map(split_data, predict_temp)), columns=['custId', 'y'])
        predict_new = predict_new.astype({'custId': 'str', 'y': 'int64'})

        # 예측 결과 결과 집계
        predict_new = predict_new.groupby('custId').agg(['sum','count'])
        predict_new.columns = ['sum', 'count']  # 연체건수, 전체건수

        # 배치정보 조회
        stmt = text(f"SELECT a.dataset_group, c.data_table, b.customer_column_name, b.label_column_name"
                    f"  FROM dc_batch a, dc_dataset_group b, dc_dataset_version c"
                    f" WHERE b.dataset_group = a.dataset_group"
                    f"   AND c.dataset_group = a.dataset_group"
                    f"   AND c.`version` = a.dataset_version"
                    f"    AND a.batch_id = {batch_id}")
        param = dict(batch_id=batch_id)
        batch_info = dbHandler.retrive_stmt(stmt, param=param)
        data_table = batch_info.iloc[0]['data_table']
        customer_column_name = batch_info.iloc[0]['customer_column_name']
        label_column_name = batch_info.iloc[0]['label_column_name']

        # 실제 결과 조회
        stmt = text(f"SELECT {customer_column_name} custId, {label_column_name} y FROM {data_table}")
        param = dict(data_table=data_table, customer_column_name=customer_column_name,
                     label_column_name=label_column_name, batch_id=batch_id, state=model_no)
        original = dbHandler.retrive_stmt(stmt, param=param)
        original = original.set_index("custId")

        # 예측 결과  결과 + 실제 결과 결합
        resultMerge = pd.merge(predict_new, original, left_index = True, right_index = True, how = 'left')
        resultMerge.columns = ['predNegativeCount', 'predTotalCount', 'positiveCount']

        # 앙상블 결과별 실제 건수 집계
        resultSummary = resultMerge.groupby(['predNegativeCount', 'predTotalCount']).agg(['sum','count']).reset_index()
        resultSummary.columns = ['predNegativeCount', 'predTotalCount', 'realPositiveCount', 'realTotalCount']
        resultSummary['realNegativeCount'] = resultSummary['realTotalCount'] - resultSummary['realPositiveCount']

        result = resultSummary.drop(['realTotalCount'], axis=1)

        # print(result)

        return result

if __name__ == '__main__':
    resultView = ResultView()
    # resultView.get_ensemble_dist(63, 1059, 911)
    resultView.get_ensemble_dist(64, 1059, 793)