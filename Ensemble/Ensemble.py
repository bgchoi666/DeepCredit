[server1]
host=165.246.34.142
import sys

from Learning.distributed_computing.common import *

sys.path.append('./data')
sys.path.append('./module')
sys.path.append('./db')

import random
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy
import scipy

from module.calculate import *
from module.data_process import *
# import db.env as denv

from module.db_process4 import *
from module.learning import *
from module.learning import BinaryClassification
from module.learning_kyh_4 import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

import datetime
import joblib

import sqlalchemy as db
from ENV.DB_Handler import DBHandler

import time
# import math
import concurrent
from functools import partial
import copy
import pymysql
# import json


class Ensemble:
    def __init__(self):
        self.dt_now = datetime.datetime.now()
        self.dbhandler = DBHandler()

        # model_name = "ENSEMBLE"
        # model_range = len(model_predict)
        #
        # batch_id = 131  # db_select_max_batch_id_4("dc_batch_result_t4")+1
        #
        # loop_test = [[0, 10], [10, 20], [20, 30]]  # ,[30,40],[40,50]]
        # loop = [[0, 100000], [100000, 200000], [200000, 300000], [300000, 400000], [400000, 500000], [500000, 600000],
        #         [600000, 700000], [700000, 800000], [800000, 900000], [900000, 1000000]]
        # voting_list = list(range(11, 12))
        #
        # batch_param = {
        #
        # }
        #
        # batch_param["Voting_point"] = str(voting_list[0])
        # batch_param = json.dumps(batch_param)
        # avg_range = 20
        # extract_model = 30
        # ensemble_method = str(extract_model) + "개의 모델 다수결 투표, 연체" + str(voting_list[0]) + "회 이상 연체로 분류," + str(
        #     avg_range) + "번 반복하여 평균, data002"
        #
        # self.db_insert_batch_4(batch_id=batch_id, batch_desc=f"{model_name}", batch_memo=ensemble_method, batch_param=batch_param,
        #                   candidate_id="129", mode="predict", dataset_group="DATA002", dataset_version=2, model_group="M100",
        #                   state="C", user_id="qorxhs@naver.com", display_yn="n", experiment_type=4)
        # self.db_insert_for_test4_pred(model_predict, model_type_subno, batch_id, model_range)
        #
        # for i in range(0, len(loop)):
        #     ensnemble_range = range(loop[i][0], loop[i][1])
        #     self.million_ensemble(rx_test_raw_, batch_id, voting_list, ensnemble_range, model_predict, y_result_20, model_name,
        #                      model_type_subno, model_range)

    def call_the_query_for_ensemble_1(self, num):
        query = f"SELECT b.* FROM dc_candidate_model_list a, dc_batch_result_t3 b " \
                f"WHERE a.batch_id = b.batch_id AND a.model_no = b.model_no AND a.model_sub_no = b.model_sub_no " \
                f"AND a.candidate_id =" + str(num) + " AND data_gubun = 'T'"

        engine = self.dbhandler.get_connection()
        connection = engine.connect()
        metadata = db.MetaData()

        result_proxy = engine.execute(query)
        result_set = result_proxy.fetchall()
        result_set = np.array(result_set)

        return result_set

    def load_the_models_1(self, loaded_db_set):
        model_load_dict = {}
        loaded_model_type_sub_dict = {}
        loaded_model_access = []
        # mode_key = []

        model_total_num = loaded_db_set.shape[0]
        cd = self.dt_now

        for i in range(0, model_total_num):
            # model_type : split structure of model_name EX: DL-DNN-130, by '-'
            model_type = loaded_db_set[i][2].split('-')
            model_key = i
            # the number of model
            model_type_len = len(model_type)
            # the betch_id of seleted model and sub_no
            batch_id_loaded = loaded_db_set[i][1]
            sub_no = loaded_db_set[i][3]

            # saved_model_name = str(batch_id_loaded)+"-"+str(sub_no)
            # classifier model for load to memory by size ex: TABNET-103(2), DL-DNN-103(3)
            if (model_type_len == 2):
                # dictionary key is the loop count + model type[0] : TABNET
                model_num_type = str(i) + "-" + model_type[0]
                # load and save to model_load_dict
                saved_model_name = str(batch_id_loaded) + "-" + str(sub_no)

                # load ver_1
                # model_load_dict[model_key] = torch.load('./model_tabnet_130/'+saved_model_name+'.pt')
                model_load_dict[model_key] = torch.load('./model_tabnet2/' + saved_model_name + '.pt')

            elif (model_type_len == 3):
                # dictionary key is the loop count + model type[0] : EX DNN,XGB,RF
                model_num_type = str(i) + "-" + model_type[1]

                dic_model_type = model_type[0] + "-" + model_type[1] + "/"
                dic_model_resample = model_type[2] + "/"

                if (model_type[1] == "DNN"):
                    model_load_dict[model_key] = (
                        torch.load('./model_save/' + dic_model_type + dic_model_resample + str(sub_no) + ".pt"))

                elif (model_type[1] == "XGB" or model_type[1] == "RF"):
                    model_load_dict[model_key] = (
                        joblib.load('./model_save/' + dic_model_type + dic_model_resample + str(sub_no) + ".pkl"))

                else:
                    print("Error : No Type Model")
                    break

            loaded_model_access.append(model_num_type)
            loaded_model_type_sub_dict[model_key] = str(batch_id_loaded) + "-" + str(sub_no)

            # print(model_load_dict[model_num_type])

        return model_load_dict, loaded_model_access, loaded_model_type_sub_dict

    def return_y_reuslt(self, ry_test, test_x):
        label = ry_test
        test_size = 0.5

        y_result_20 = []
        for i in range(20):
            label = pd.DataFrame(label)
            label_0 = label[label['y1'] == 0]
            label_1 = label[label['y1'] == 1]
            # print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
            data_0 = test_x.loc[label_0.index]
            data_1 = test_x.loc[label_1.index]
            # print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

            x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size=test_size,
                                                                        random_state=i)
            x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size=test_size,
                                                                        random_state=i)

            x_train_c = pd.concat([x_train_0, x_train_1], axis=0)
            y_train_c = pd.concat([y_train_0, y_train_1], axis=0)
            x_test_c = pd.concat([x_test_0, x_test_1], axis=0)
            y_test_c = pd.concat([y_test_0, y_test_1], axis=0)

            y_result_20.append(y_test_c)

        return y_result_20

    def extract_model_learning_(self, test_x, rx_test, ry_test, model_load, model_access_key, model_type_subno):
        model_predict_result = {}
        # loaded_model_access = []

        print("베이스 모델 예측...")
        for i in tqdm((range(len(model_access_key)))):  # tqdm(list(rnum), unit=" 항목", desc="10개 앙상블",leave=True):
            extract_model_type = model_access_key[i].split("-")
            extract_model_type_name = extract_model_type[1]
            split = "-"
            key = i
            model = model_load[key]

            if (extract_model_type_name == "TABNET"):

                pred = model.predict(rx_test.values[:])
                df_pred = pd.DataFrame(pred, index=ry_test.index, columns=['pred'])
                model_predict_result[model_type_subno[key]] = df_pred

            elif (extract_model_type_name == "DNN"):

                pred = DNN_predict(model, rx_test)
                df_pred = pd.DataFrame(pred, index=ry_test.index, columns=['pred'])
                model_predict_result[model_type_subno[key]] = df_pred

            elif (extract_model_type_name == "XGB" or extract_model_type_name == "RF"):

                pred = model.predict(rx_test)
                df_pred = pd.DataFrame(pred, index=ry_test.index, columns=['pred'])
                model_predict_result[model_type_subno[key]] = df_pred

        y_result = self.return_y_reuslt(ry_test, test_x)

        return model_predict_result, y_result

    def ensemble_loop_ver2(self, mdata_list, y_result_20, ce_loop):
        # 각 정해진 sub_no(몇 번의 평균 ex 10, 20, 30.. )에 따라 각 sub_no 넘버별 추출된 모델의 예측 합산 -> t4_detail 저장
        # 추후 sub_no번 다 돌고나면 평균을 내어 t4에 저장
        sub_loop = list(range(0, 20))
        list_m = list(range(0, 30))

        # rseed = random.sample(range(0,200000),1)
        rseed = ce_loop  # rseed_list[ce_loop]
        random.seed(rseed)

        # 랜덤하게 30개의 모델 추출
        # data = random.sample(list(mdata_list.values()), 30)
        data = random.sample(mdata_list, 30)
        pred_array_en = list(range(0, 20))
        for sl in sub_loop:
            # sl번 데이터 의미 : 각 n개의 모델이  m번의 시드번호로 예측한 내역의 합산.
            # ce_loop 데이터 의미 : 몇 번째 앙상블? 랜덤으로 추출한 n개의 모델 번호를 저장.
            pred_array_en[sl] = (sum(list(map(lambda x: data[x].loc[y_result_20[sl].index], list_m))))

        return pred_array_en, rseed

    def ensemble_voting(self, ensemble_range, critical_point, pred_array_en2, y_result_20):
        # pvoting = copy.deepcopy(pred_array_en2)  # pred_array_en2.copy()
        cmc_list_ = [[0 for col in range(20)] for row in ensemble_range]

        print("보팅...")
        for i in tqdm(ensemble_range):
            for k in range(20):
                pred = pred_array_en2[i - ensemble_range[0]][0][k]

                pred.loc[pred['pred'] < critical_point] = 0
                # voting = 5, 합산 5 이상이면 연체 -> 1
                pred.loc[pred['pred'] >= critical_point] = 1

                cmc_list_[i - ensemble_range[0]][k] = list(itertools.chain(*(confusion_matrix(y_result_20[k], pred))))

        return cmc_list_

    def db_insert4_detail(self, Ensemble_bulk_detail_frame):
        ip = '165.246.34.133'
        port = 3307
        user = 'deepcredit'
        passwd = 'value!0328'
        db = 'deep_credit'
        charset = 'utf8'

        db = pymysql.connect(host=ip,
                             port=port,
                             user=user,
                             passwd=passwd,
                             db=db,
                             charset=charset,
                             autocommit=True)

        cursor = db.cursor()

        # query 문
        query = '''
        INSERT INTO dc_batch_result_t4_detail (batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit, interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, true_positive, false_positive, true_negative, false_negative, creation_date) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        # VALUES (%d, %d, %d, %d, %s, %d, %d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %d, %d, %d, %d, %d)"

        changeTuple = [tuple(x) for x in Ensemble_bulk_detail_frame.values]

        # insert
        cursor.executemany(query, changeTuple)

        db.commit()
        db.close()

    def db_insert4_result(self, Ensemble_bulk_result_frame):
        ip = '165.246.34.133'
        port = 3307
        user = 'deepcredit'
        passwd = 'value!0328'
        db = 'deep_credit'
        charset = 'utf8'

        db = pymysql.connect(host=ip,
                             port=port,
                             user=user,
                             passwd=passwd,
                             db=db,
                             charset=charset,
                             autocommit=True)

        cursor = db.cursor()
        # query 문
        query = '''
        INSERT INTO dc_batch_result_t4(
        batch_id, model_no, model_full_name, model_sub_no, avg_true_positive,avg_false_positive, avg_true_negative,avg_false_negative, avg_true_positive_org, avg_false_positive_org,avg_true_negative_org, avg_false_negative_org, avg_real_total_loan, avg_real_gross_profit,avg_real_interest_income, avg_real_write_off, avg_real_total_loan_org, avg_real_gross_profit_org,avg_real_interest_income_org, avg_real_write_off_org,var_total_loan, var_gross_profit, var_interest_income, var_write_off, var_positive_rate,var_negative_rate,var_positive_recall, var_negative_recall, var_positive_precision,var_negative_precision,var_auc, var_total_loan_org, var_gross_profit_org, var_interest_income_org, var_write_off_org,var_positive_rate_org, var_negative_rate_org, var_positive_recall_org, var_negative_recall_org, var_positive_precision_org, var_negative_precision_org, var_auc_org, diversity ,base_models, creation_date)  
        VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)
        '''

        # VALUES (%d, %d, %d, %d, %s, %d, %d, %d, %d, %f, %f, %f, %f, %f, %f, %f, %d, %d, %d, %d, %d)"

        # DataFrame to tuple로 변환
        changeTuple = [tuple(x) for x in Ensemble_bulk_result_frame.values]

        # insert
        cursor.executemany(query, changeTuple)

        db.commit()
        db.close()

    def db_insert_for_test4_pred(self, model_predict, model_type_subno, batch_id, model_range):
        pred_count_size = model_predict[model_type_subno[0]]['pred'].shape[0]
        pred_cus_list = list(range(0, pred_count_size))

        for m in range(0, model_range):
            mtn = model_type_subno[m]
            model_no = mtn.split('-')[0]
            model_sub_no = mtn.split('-')[1]

            for i in range(0, pred_count_size):
                pred_cus_list[i] = (str)(model_predict[mtn].index[i]) + '-' + str(model_predict[mtn]['pred'].iloc[i])

            mbl = (','.join(pred_cus_list))
            db_insert_for_test4_predict(batch_id, model_no, model_sub_no, mbl)

    # def db_bulk_insert_for_test4_detail_ensemble_ver5(self, batch_id, model_name, raw_data, pvoting, en_loop, cmc_list):
    #     batch_id = batch_id
    #     model_no = select_model_no_4(model_name)
    #     model_full_name = "ENSEMBLE"
    #     cd = datetime.datetime.now()
    #     # cm = confusion_matrix(y_result_20[i], pred)
    #     size_en = 20000
    #     ensemble_model_count = 0
    #
    #     model_sub_no_result = list()
    #     sub_no_result = list()  #
    #     detail_list = []
    #     result_list = []
    #
    #     avg_true_positive_ = np.zeros(20)
    #     avg_false_positive_ = np.zeros(20)  # list()#result_set['false_positive'].mean()
    #     avg_true_negative_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
    #     avg_false_negative_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
    #     avg_real_total_loan_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
    #     avg_real_gross_profit_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
    #     avg_real_interest_income_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
    #     avg_real_write_off_ = np.zeros(20)  # list()#result_set['write_off'].mean()
    #     var_total_loan_ = np.zeros(20)  # list()#round(result_set['total_loan'].std(),4)
    #     var_gross_profit_ = np.zeros(20)  # list()#round(result_set['gross_profit'].std(),4)
    #     var_interest_income_ = np.zeros(20)  # list()#round(result_set['interest_income'].std(),4)
    #     var_write_off_ = np.zeros(20)  # list()#round(result_set['write_off'].std(),4)
    #     var_positive_rate_ = np.zeros(20)  # list()#round(result_set['positive_rate'].std(),2)
    #     var_negative_rate_ = np.zeros(20)  # list()#round(result_set['negative_rate'].std(),2)
    #     var_positive_recall_ = np.zeros(20)  # list()#round(result_set['positive_recall'].std(),2)
    #     var_negative_recall_ = np.zeros(20)  # list()#round(result_set['negative_recall'].std(),2)
    #     var_positive_precision_ = np.zeros(20)  # list()#round(result_set['positive_precision'].std(),2)
    #     var_negative_precision_ = np.zeros(20)  # list()#round(result_set['negative_precision'].std(),2)
    #     var_auc_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)
    #
    #     avg_true_positive_org_ = np.zeros(20)  # list()#(result_set['true_positive'].mean())
    #     avg_false_positive_org_ = np.zeros(20)  # list()#(result_set['false_positive'].mean())
    #     avg_true_negative_org_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
    #     avg_false_negative_org_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
    #     avg_real_total_loan_org_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
    #     avg_real_gross_profit_org_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
    #     avg_real_interest_income_org_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
    #     avg_real_write_off_org_ = np.zeros(20)  # list()# result_set['write_off'].mean()
    #     var_total_loan_org_ = np.zeros(20)  # list()#round(result_set['total_loan'].var(),4)
    #     var_gross_profit_org_ = np.zeros(20)  # list()#round(result_set['gross_profit'].var(),4)
    #     var_interest_income_org_ = np.zeros(20)  # list()# round(result_set['interest_income'].var(),4)
    #     var_write_off_org_ = np.zeros(20)  # list()#round(result_set['write_off'].var(),4)
    #     var_positive_rate_org_ = np.zeros(20)  # list()#round(result_set['positive_rate'].var(),2)
    #     var_negative_rate_org_ = np.zeros(20)  # list()#round(result_set['negative_rate'].var(),2)
    #     var_positive_recall_org_ = np.zeros(20)  # list()# round(result_set['positive_recall'].var(),2)
    #     var_negative_recall_org_ = np.zeros(20)  # list()#round(result_set['negative_recall'].var(),2)
    #     var_positive_precision_org_ = np.zeros(20)  # list()#round(result_set['positive_precision'].var(),2)
    #     var_negative_precision_org_ = np.zeros(20)  # list()#round(result_set['negative_precision'].var(),2)
    #     var_auc_org_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)
    #
    #     for i in tqdm(ensnemble_range):
    #         # model_sub_no = i
    #         model_sub_no_result = i
    #         rml = return_model_list(model_type_subno, en_mdodel_list[ensemble_model_count])
    #         # print(len(rml))
    #         # print((rml))
    #         mbl = (','.join(rml))
    #         ensemble_model_count = ensemble_model_count + 1
    #         for k in range(20):
    #             cmc = cmc_list[i][k]
    #             total_count = sum(cmc_list[0][0])
    #             TP, FP, TN, FN = cmc[3], cmc[1], cmc[0], cmc[2]
    #             cal_result = self.calculate_ver2(raw_data, pvoting[i][0][k])
    #             # print(TP, FP, TN, FN )
    #             cloan, cbase_profit, cpredict_profit, cpredict_loan, cbase_loss, cpredict_loss, cy1, cy2 = self.calculate_ver2(
    #                 rx_test_raw_, pvoting[i][0][k])
    #
    #             # print(TP, FP, TN, FN )
    #             # total_loan_ = cal_result['대출실행금액'].sum()
    #             total_loan_ = cloan.sum()
    #             # before_inter = cal_result['[기존]이자수익'].sum()
    #             before_inter = cbase_profit.sum()
    #             # after_inter = cal_result['[예상]이자수익'].sum()
    #             after_inter = cpredict_profit.sum()
    #
    #             # after_total_loan = cal_result['[예상]대출금액'].sum()
    #             after_total_loan = cpredict_loan.sum()
    #             profit_loan = after_inter - before_inter
    #
    #             # before_loss = cal_result['[기존]원금손실'].sum()
    #             before_loss = cbase_loss.sum()
    #             # after_loss = cal_result['[예상]원금손실'].sum()
    #             after_loss = cpredict_loss.sum()
    #
    #             before_profit = before_inter - before_loss
    #             profit_loss = before_loss - after_loss
    #             profit = profit_loan + profit_loss
    #
    #             avg_true_positive_[k] = TP
    #             avg_false_positive_[k] = FP
    #             avg_true_negative_[k] = TN
    #             avg_false_negative_[k] = FN
    #             avg_real_total_loan_[k] = after_total_loan
    #             avg_real_gross_profit_[k] = (after_inter - after_loss)
    #             avg_real_interest_income_[k] = after_inter
    #             avg_real_write_off_[k] = after_loss
    #             var_total_loan_[k] = after_total_loan
    #             var_gross_profit_[k] = (after_inter - after_loss)
    #             var_interest_income_[k] = after_inter
    #             var_write_off_[k] = after_loss
    #             var_positive_rate_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
    #             var_negative_rate_[k] = (round((TN + FN) / total_count * 100, 2))
    #             var_positive_recall_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
    #             var_negative_recall_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
    #             var_positive_precision_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
    #             var_negative_precision_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
    #             var_auc_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #
    #             avg_true_positive_org_[k] = TP
    #             avg_false_positive_org_[k] = FP
    #             avg_true_negative_org_[k] = TN
    #             avg_false_negative_org_[k] = FN
    #             avg_real_total_loan_org_[k] = total_loan_
    #             avg_real_gross_profit_org_[k] = (before_inter - before_loss)
    #             avg_real_interest_income_org_[k] = before_inter
    #             avg_real_write_off_org_[k] = before_loss
    #             var_total_loan_org_[k] = total_loan_
    #             var_gross_profit_org_[k] = (before_inter - before_loss)
    #             var_interest_income_org_[k] = before_inter
    #             var_write_off_org_[k] = before_loss
    #             var_positive_rate_org_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
    #             var_negative_rate_org_[k] = (round((TN + FN) / total_count * 100, 2))
    #             var_positive_recall_org_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
    #             var_negative_recall_org_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
    #             var_positive_precision_org_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
    #             var_negative_precision_org_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
    #             var_auc_org_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #
    #             detail_list.append((batch_id, model_no, i, k, "T", after_total_loan, (after_inter - after_loss),
    #                                 (after_inter), after_loss, (round(FN / (TN + FN + not_nan) * 100, 2)),
    #                                 (round((TN + FN) / total_count * 100, 2)), (round(TP / (TP + FN + not_nan) * 100, 2)),
    #                                 (round(TN / (TN + FP + not_nan) * 100, 2)),
    #                                 (round(TP / (TP + FP + not_nan) * 100, 2)), (round(TN / (TN + FN + not_nan) * 100, 2)),
    #                                 (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)),
    #                                 TP, FP, TN, FN, cd))
    #
    #             # TP, FP, TN, FN = cal_result['y1'].sum(), 0, cal_result['y2'].sum(), 0
    #             TP, FP, TN, FN = cy1.sum(), 0, cy2.sum(), 0
    #
    #             detail_list.append((batch_id, model_no, i, k, "B", total_loan_, (before_inter - before_loss),
    #                                 (before_inter), before_loss,
    #                                 (round(FN / (TN + FN + not_nan) * 100, 2)), (round((TN + FN) / total_count * 100, 2)),
    #                                 (round(TP / (TP + FN + not_nan) * 100, 2)), (round(TN / (TN + FP + not_nan) * 100, 2)),
    #                                 (round(TP / (TP + FP + not_nan) * 100, 2)), (round(TN / (TN + FN + not_nan) * 100, 2)),
    #                                 (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2)),
    #                                 TP, FP, TN, FN, cd))
    #             # base_models.append(mbl)
    #
    #             diversity = 0
    #
    #         result_list.append((batch_id, model_no, model_full_name, model_sub_no_result, avg_true_positive_.mean(),
    #                             avg_false_positive_.mean(), avg_true_negative_.mean(), avg_false_negative_.mean(),
    #                             avg_true_positive_org_.mean(), avg_false_positive_org_.mean(),
    #                             avg_true_negative_org_.mean(), avg_false_negative_org_.mean(), avg_real_total_loan_.mean(),
    #                             avg_real_gross_profit_.mean(), avg_real_interest_income_.mean(), avg_real_write_off_.mean(),
    #                             avg_real_total_loan_org_.mean(), avg_real_gross_profit_org_.mean(),
    #                             avg_real_interest_income_org_.mean(), avg_real_write_off_org_.mean(), var_total_loan_.std(),
    #                             var_gross_profit_.std(), var_interest_income_.std(), var_write_off_.std(),
    #                             var_positive_rate_.std(), var_negative_rate_.std(), var_positive_recall_.std(),
    #                             var_negative_recall_.std(), var_positive_precision_.std(), var_negative_precision_.std(),
    #                             var_auc_.std(), var_total_loan_org_.std(), var_gross_profit_org_.std(),
    #                             var_interest_income_org_.std(), var_write_off_org_.std(), var_positive_rate_org_.std(),
    #                             var_negative_rate_org_.std(), var_positive_recall_org_.std(),
    #                             var_negative_recall_org_.std(), var_positive_precision_org_.std(),
    #                             var_negative_precision_org_.std(), var_auc_org_.std(), diversity, mbl, cd))
    #
    #     # pdlst = pd.DataFrame(lst, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    #     Ensemble_model_detail = pd.DataFrame(detail_list,
    #                                          columns=['batch_id', 'model_no', 'model_sub_no', 'sub_no', 'data_gubun',
    #                                                   'total_loan', 'gross_profit', 'interest_income', 'write_off',
    #                                                   'positive_rate', 'negative_rate', 'positive_recall',
    #                                                   'negative_recall', 'positive_precision', 'negative_precision', 'auc',
    #                                                   'true_positive', 'false_positive', 'true_negative', 'false_negative',
    #                                                   'creation_date'])  # ,ignore_index=True
    #
    #     Ensemble_model_result = pd.DataFrame(result_list,
    #                                          columns=['batch_id', 'model_no', 'model_full_name', 'model_sub_no',
    #                                                   'avg_true_positive', 'avg_false_positive', 'avg_true_negative',
    #                                                   'avg_false_negative', 'avg_true_positive_org',
    #                                                   'avg_false_positive_org', 'avg_true_negative_org',
    #                                                   'avg_false_negative_org', 'avg_real_total_loan',
    #                                                   'avg_real_gross_profit', 'avg_real_interest_income',
    #                                                   'avg_real_write_off', 'avg_real_total_loan_org',
    #                                                   'avg_real_gross_profit_org', 'avg_real_interest_income_org',
    #                                                   'avg_real_write_off_org', 'var_total_loan', 'var_gross_profit',
    #                                                   'var_interest_income', 'var_write_off', 'var_positive_rate',
    #                                                   'var_negative_rate', 'var_positive_recall', 'var_negative_recall',
    #                                                   'var_positive_precision', 'var_negative_precision', 'var_auc',
    #                                                   'var_total_loan_org', 'var_gross_profit_org',
    #                                                   'var_interest_income_org', 'var_write_off_org',
    #                                                   'var_positive_rate_org', 'var_negative_rate_org',
    #                                                   'var_positive_recall_org', 'var_negative_recall_org',
    #                                                   'var_positive_precision_org', 'var_negative_precision_org',
    #                                                   'var_auc_org', 'diversity', 'base_models', 'creation_date'])
    #
    #     return Ensemble_model_detail, Ensemble_model_result

    def return_model_list(self, model_type_subno, mdodel_list):
        return list(map(lambda x: model_type_subno[x], mdodel_list))

    def db_bulk_insert_for_test4_detail_ensemble_ver3(self, batch_id, model_name, raw_data, pvoting, en_loop, cmc_list,
                                                      model_type_subno, en_mdodel_list):
        Ensemble_bulk_detail_frame = pd.DataFrame(
            columns=['batch_id', 'model_no', 'model_sub_no', 'sub_no', 'data_gubun', 'total_loan', 'gross_profit',
                     'interest_income', 'write_off', 'positive_rate', 'negative_rate', 'positive_recall', 'negative_recall',
                     'positive_precision', 'negative_precision', 'auc', 'true_positive', 'false_positive', 'true_negative',
                     'false_negative', 'creation_date'])
        batch_id = batch_id
        model_no = select_model_no_4(model_name)
        model_full_name = "ENSEMBLE"
        cd = datetime.datetime.now()
        # cm = confusion_matrix(y_result_20[i], pred)
        size_en = 20000
        ensemble_model_count = 0
        total_loan = list()  # list(range(size_en))# [0 for col in range(20)] for row in range(10000)]
        gross_profit = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        interest_income = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        write_off = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        positive_rate = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        negative_rate = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        positive_recall = list()  # list(range(size_en))#  [[0 for col in range(20)] for row in range(10000)]
        negative_recall = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        positive_precision = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        negative_precision = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        auc = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        model_sub_no = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
        sub_no = list()  # list(range(size_en))#  [[0 for col in range(20)] for row in range(10000)]
        # negative_precision = [[0 for col in range(20)] for row in range(10000)]
        true_positive = list()
        false_positive = list()
        true_negative = list()
        false_negative = list()
        data_gubun = list()

        avg_true_positive_ = np.zeros(20)
        avg_false_positive_ = np.zeros(20)  # list()#result_set['false_positive'].mean()
        avg_true_negative_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
        avg_false_negative_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
        avg_real_total_loan_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
        avg_real_gross_profit_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
        avg_real_interest_income_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
        avg_real_write_off_ = np.zeros(20)  # list()#result_set['write_off'].mean()
        var_total_loan_ = np.zeros(20)  # list()#round(result_set['total_loan'].std(),4)
        var_gross_profit_ = np.zeros(20)  # list()#round(result_set['gross_profit'].std(),4)
        var_interest_income_ = np.zeros(20)  # list()#round(result_set['interest_income'].std(),4)
        var_write_off_ = np.zeros(20)  # list()#round(result_set['write_off'].std(),4)
        var_positive_rate_ = np.zeros(20)  # list()#round(result_set['positive_rate'].std(),2)
        var_negative_rate_ = np.zeros(20)  # list()#round(result_set['negative_rate'].std(),2)
        var_positive_recall_ = np.zeros(20)  # list()#round(result_set['positive_recall'].std(),2)
        var_negative_recall_ = np.zeros(20)  # list()#round(result_set['negative_recall'].std(),2)
        var_positive_precision_ = np.zeros(20)  # list()#round(result_set['positive_precision'].std(),2)
        var_negative_precision_ = np.zeros(20)  # list()#round(result_set['negative_precision'].std(),2)
        var_auc_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)

        avg_true_positive = list()  # result_set['true_positive'].mean()
        avg_false_positive = list()  # result_set['false_positive'].mean()
        avg_true_negative = list()  # result_set['true_negative'].mean()
        avg_false_negative = list()  # result_set['false_negative'].mean()
        avg_real_total_loan = list()  # result_set['total_loan'].mean()
        avg_real_gross_profit = list()  # result_set['gross_profit'].mean()
        avg_real_interest_income = list()  # result_set['interest_income'].mean()
        avg_real_write_off = list()  # result_set['write_off'].mean()
        var_total_loan = list()  # round(result_set['total_loan'].std(),4)
        var_gross_profit = list()  # round(result_set['gross_profit'].std(),4)
        var_interest_income = list()  # round(result_set['interest_income'].std(),4)
        var_write_off = list()  # round(result_set['write_off'].std(),4)
        var_positive_rate = list()  # round(result_set['positive_rate'].std(),2)
        var_negative_rate = list()  # round(result_set['negative_rate'].std(),2)
        var_positive_recall = list()  # round(result_set['positive_recall'].std(),2)
        var_negative_recall = list()  # round(result_set['negative_recall'].std(),2)
        var_positive_precision = list()  # round(result_set['positive_precision'].std(),2)
        var_negative_precision = list()  # round(result_set['negative_precision'].std(),2)
        var_auc = list()  # round(result_set['auc'].var(),4)

        avg_true_positive_org_ = np.zeros(20)  # list()#(result_set['true_positive'].mean())
        avg_false_positive_org_ = np.zeros(20)  # list()#(result_set['false_positive'].mean())
        avg_true_negative_org_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
        avg_false_negative_org_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
        avg_real_total_loan_org_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
        avg_real_gross_profit_org_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
        avg_real_interest_income_org_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
        avg_real_write_off_org_ = np.zeros(20)  # list()# result_set['write_off'].mean()
        var_total_loan_org_ = np.zeros(20)  # list()#round(result_set['total_loan'].var(),4)
        var_gross_profit_org_ = np.zeros(20)  # list()#round(result_set['gross_profit'].var(),4)
        var_interest_income_org_ = np.zeros(20)  # list()# round(result_set['interest_income'].var(),4)
        var_write_off_org_ = np.zeros(20)  # list()#round(result_set['write_off'].var(),4)
        var_positive_rate_org_ = np.zeros(20)  # list()#round(result_set['positive_rate'].var(),2)
        var_negative_rate_org_ = np.zeros(20)  # list()#round(result_set['negative_rate'].var(),2)
        var_positive_recall_org_ = np.zeros(20)  # list()# round(result_set['positive_recall'].var(),2)
        var_negative_recall_org_ = np.zeros(20)  # list()#round(result_set['negative_recall'].var(),2)
        var_positive_precision_org_ = np.zeros(20)  # list()#round(result_set['positive_precision'].var(),2)
        var_negative_precision_org_ = np.zeros(20)  # list()#round(result_set['negative_precision'].var(),2)
        var_auc_org_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)

        avg_true_positive_org = list()  # (result_set['true_positive'].mean())
        avg_false_positive_org = list()  # (result_set['false_positive'].mean())
        avg_true_negative_org = list()  # result_set['true_negative'].mean()
        avg_false_negative_org = list()  # result_set['false_negative'].mean()
        avg_real_total_loan_org = list()  # result_set['total_loan'].mean()
        avg_real_gross_profit_org = list()  # result_set['gross_profit'].mean()
        avg_real_interest_income_org = list()  # result_set['interest_income'].mean()
        avg_real_write_off_org = list()  # result_set['write_off'].mean()
        var_total_loan_org = list()  # round(result_set['total_loan'].var(),4)
        var_gross_profit_org = list()  # round(result_set['gross_profit'].var(),4)
        var_interest_income_org = list()  # round(result_set['interest_income'].var(),4)
        var_write_off_org = list()  # round(result_set['write_off'].var(),4)
        var_positive_rate_org = list()  # round(result_set['positive_rate'].var(),2)
        var_negative_rate_org = list()  # round(result_set['negative_rate'].var(),2)
        var_positive_recall_org = list()  # round(result_set['positive_recall'].var(),2)
        var_negative_recall_org = list()  # round(result_set['negative_recall'].var(),2)
        var_positive_precision_org = list()  # round(result_set['positive_precision'].var(),2)
        var_negative_precision_org = list()  # round(result_set['negative_precision'].var(),2)
        var_auc_org = list()  # round(result_set['auc'].var(),4)
        base_models = list()

        model_sub_no_result = list()
        sub_no_result = list()  #

        print("결과 생성...")
        for i in tqdm(en_loop):
            # model_sub_no = i
            model_sub_no_result.append(i)
            rml = self.return_model_list(model_type_subno, en_mdodel_list[ensemble_model_count])
            # print(len(rml))
            # print((rml))
            mbl = (','.join(rml))
            ensemble_model_count = ensemble_model_count + 1
            for k in range(20):
                model_sub_no.append(i)
                model_sub_no.append(i)
                sub_no.append(k)
                sub_no.append(k)
                cmc = cmc_list[i - en_loop[0]][k]
                total_count = sum(cmc_list[0][0])
                TP, FP, TN, FN = cmc[3], cmc[1], cmc[0], cmc[2]
                # cal_result = calculate(raw_data, pvoting[i][0][k])
                # cal_result, dataframe type to list type 대체
                cloan, cbase_profit, cpredict_profit, cpredict_loan, cbase_loss, cpredict_loss, cy1, cy2 = self.calculate_ver2(raw_data, pvoting[i - en_loop[0]][0][k])

                # print(TP, FP, TN, FN )
                # total_loan_ = cal_result['대출실행금액'].sum()
                total_loan_ = cloan.sum()
                # before_inter = cal_result['[기존]이자수익'].sum()
                before_inter = cbase_profit.sum()
                # after_inter = cal_result['[예상]이자수익'].sum()
                after_inter = cpredict_profit.sum()

                # after_total_loan = cal_result['[예상]대출금액'].sum()
                after_total_loan = cpredict_loan.sum()
                profit_loan = after_inter - before_inter

                # before_loss = cal_result['[기존]원금손실'].sum()
                before_loss = cbase_loss.sum()
                # after_loss = cal_result['[예상]원금손실'].sum()
                after_loss = cpredict_loss.sum()

                before_profit = before_inter - before_loss
                profit_loss = before_loss - after_loss
                profit = profit_loan + profit_loss

                # print("TP FP TN FN", TP, FP, TN, FN)
                # db_insert_for_test4_detail(

                data_gubun.append('T')
                total_loan.append(after_total_loan)  # 총대출액
                gross_profit.append(after_inter - after_loss)  # 총수익
                interest_income.append(after_inter)  # 총이자수익
                # gross_profit_rate = round((after_inter - after_loss)/after_total_loan*100,2),
                write_off.append(after_loss)
                positive_rate.append(round(FN / (TN + FN + not_nan) * 100, 2))
                negative_rate.append(round((TN + FN) / total_count * 100, 2))
                positive_recall.append(round(TP / (TP + FN + not_nan) * 100, 2))
                negative_recall.append(round(TN / (TN + FP + not_nan) * 100, 2))
                positive_precision.append(round(TP / (TP + FP + not_nan) * 100, 2))
                negative_precision.append(round(TN / (TN + FN + not_nan) * 100, 2))
                auc.append(round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
                # total_count=total_count,
                true_positive.append(TP)
                false_positive.append(FP)
                true_negative.append(TN)
                false_negative.append(FN)
                creation_date = cd  # ) #'1000-01-01')

                avg_true_positive_[k] = TP
                avg_false_positive_[k] = FP
                avg_true_negative_[k] = TN
                avg_false_negative_[k] = FN
                avg_real_total_loan_[k] = after_total_loan
                avg_real_gross_profit_[k] = (after_inter - after_loss)
                avg_real_interest_income_[k] = after_inter
                avg_real_write_off_[k] = after_loss
                var_total_loan_[k] = after_total_loan
                var_gross_profit_[k] = (after_inter - after_loss)
                var_interest_income_[k] = after_inter
                var_write_off_[k] = after_loss
                var_positive_rate_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
                var_negative_rate_[k] = (round((TN + FN) / total_count * 100, 2))
                var_positive_recall_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
                var_negative_recall_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
                var_positive_precision_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
                var_negative_precision_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
                var_auc_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))

                # TP, FP, TN, FN = cal_result['y1'].sum(), 0, cal_result['y2'].sum(), 0
                TP, FP, TN, FN = cy1.sum(), 0, cy2.sum(), 0

                data_gubun.append('B')
                total_loan.append(total_loan_)
                gross_profit.append(before_inter - before_loss)
                # gross_profit_rate = round((before_inter - before_loss)/total_loan*100,2),
                interest_income.append(before_inter)
                write_off.append(before_loss)
                positive_rate.append(round(FN / (TN + FN + not_nan) * 100, 2))
                negative_rate.append(round((TN + FN) / total_count * 100, 2))
                positive_recall.append(round(TP / (TP + FN + not_nan) * 100, 2))
                negative_recall.append(round(TN / (TN + FP + not_nan) * 100, 2))
                positive_precision.append(round(TP / (TP + FP + not_nan) * 100, 2))
                negative_precision.append(round(TN / (TN + FN + not_nan) * 100, 2))
                auc.append(round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
                # total_count=total_count,
                true_positive.append(TP)
                false_positive.append(FP)
                true_negative.append(TN)
                false_negative.append(FN)

                avg_true_positive_org_[k] = TP
                avg_false_positive_org_[k] = FP
                avg_true_negative_org_[k] = TN
                avg_false_negative_org_[k] = FN
                avg_real_total_loan_org_[k] = total_loan_
                avg_real_gross_profit_org_[k] = (before_inter - before_loss)
                avg_real_interest_income_org_[k] = before_inter
                avg_real_write_off_org_[k] = before_loss
                var_total_loan_org_[k] = total_loan_
                var_gross_profit_org_[k] = (before_inter - before_loss)
                var_interest_income_org_[k] = before_inter
                var_write_off_org_[k] = before_loss
                var_positive_rate_org_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
                var_negative_rate_org_[k] = (round((TN + FN) / total_count * 100, 2))
                var_positive_recall_org_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
                var_negative_recall_org_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
                var_positive_precision_org_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
                var_negative_precision_org_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
                var_auc_org_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))

            avg_true_positive.append((avg_true_positive_.mean()))
            avg_false_positive.append((avg_false_positive_.mean()))
            avg_true_negative.append(avg_true_negative_.mean())
            avg_false_negative.append(avg_false_negative_.mean())
            avg_real_total_loan.append(avg_real_total_loan_.mean())
            avg_real_gross_profit.append(avg_real_gross_profit_.mean())
            avg_real_interest_income.append(avg_real_interest_income_.mean())
            avg_real_write_off.append(avg_real_write_off_.mean())
            var_total_loan.append(var_total_loan_.std())
            var_gross_profit.append(var_gross_profit_.std())
            var_interest_income.append(var_interest_income_.std())
            var_write_off.append(var_write_off_.std())
            var_positive_rate.append(var_positive_rate_.std())
            var_negative_rate.append(var_negative_rate_.std())
            var_positive_recall.append(var_positive_recall_.std())
            var_negative_recall.append(var_negative_recall_.std())
            var_positive_precision.append(var_positive_precision_.std())
            var_negative_precision.append(var_negative_precision_.std())
            var_auc.append(var_auc_.std())

            avg_true_positive_org.append(avg_true_positive_org_.mean())
            avg_false_positive_org.append(avg_false_positive_org_.mean())
            avg_true_negative_org.append(avg_true_negative_org_.mean())
            avg_false_negative_org.append(avg_false_negative_org_.mean())
            avg_real_total_loan_org.append(avg_real_total_loan_org_.mean())
            avg_real_gross_profit_org.append(avg_real_gross_profit_org_.mean())
            avg_real_interest_income_org.append(avg_real_interest_income_org_.mean())
            avg_real_write_off_org.append(avg_real_write_off_org_.mean())
            var_total_loan_org.append(var_total_loan_org_.std())
            var_gross_profit_org.append(var_gross_profit_org_.std())
            var_interest_income_org.append(var_interest_income_org_.std())
            var_write_off_org.append(var_write_off_org_.std())
            var_positive_rate_org.append(var_positive_rate_org_.std())
            # var_positive_rate_org.append(var_positive_rate_org_.std())
            var_negative_rate_org.append(var_negative_rate_org_.std())
            var_positive_recall_org.append(var_positive_recall_org_.std())
            var_negative_recall_org.append(var_negative_recall_org_.std())
            var_positive_precision_org.append(var_positive_precision_org_.std())
            var_negative_precision_org.append(var_negative_precision_org_.std())
            var_auc_org.append(var_auc_org_.std())

            base_models.append(mbl)
            creation_date = cd
            diversity = 0

        Ensemble_model_detail = pd.DataFrame(
            {'batch_id': batch_id, 'model_no': model_no, 'model_sub_no': model_sub_no, 'sub_no': sub_no,
             'data_gubun': data_gubun, 'total_loan': total_loan, 'gross_profit': gross_profit,
             'interest_income': interest_income, 'write_off': write_off, 'positive_rate': positive_rate,
             'negative_rate': negative_rate, 'positive_recall': positive_recall, 'negative_recall': negative_recall,
             'positive_precision': positive_precision, 'negative_precision': negative_precision, 'auc': auc,
             'true_positive': true_positive, 'false_positive': false_positive, 'true_negative': true_negative,
             'false_negative': false_negative, 'creation_date': creation_date})  # ,ignore_index=True

        Ensemble_model_result = pd.DataFrame(
            {'batch_id': batch_id, 'model_no': model_no, 'model_full_name': model_full_name,
             'model_sub_no': model_sub_no_result, 'avg_true_positive': avg_true_positive,
             'avg_false_positive': avg_false_positive, 'avg_true_negative': avg_true_negative,
             'avg_false_negative': avg_false_negative, 'avg_true_positive_org': avg_true_positive_org,
             'avg_false_positive_org': avg_false_positive_org, 'avg_true_negative_org': avg_true_negative_org,
             'avg_false_negative_org': avg_false_negative_org, 'avg_real_total_loan': avg_real_total_loan,
             'avg_real_gross_profit': avg_real_gross_profit, 'avg_real_interest_income': avg_real_interest_income,
             'avg_real_write_off': avg_real_write_off, 'avg_real_total_loan_org': avg_real_total_loan_org,
             'avg_real_gross_profit_org': avg_real_gross_profit_org,
             'avg_real_interest_income_org': avg_real_interest_income_org, 'avg_real_write_off_org': avg_real_write_off_org,
             'var_total_loan': var_total_loan, 'var_gross_profit': var_gross_profit,
             'var_interest_income': var_interest_income, 'var_write_off': var_write_off,
             'var_positive_rate': var_positive_rate, 'var_negative_rate': var_negative_rate,
             'var_positive_recall': var_positive_recall, 'var_negative_recall': var_negative_recall,
             'var_positive_precision': var_positive_precision, 'var_negative_precision': var_negative_precision,
             'var_auc': var_auc, 'var_total_loan_org': var_total_loan_org, 'var_gross_profit_org': var_gross_profit_org,
             'var_interest_income_org': var_interest_income_org, 'var_write_off_org': var_write_off_org,
             'var_positive_rate_org': var_positive_rate_org, 'var_negative_rate_org': var_negative_rate_org,
             'var_positive_recall_org': var_positive_recall_org, 'var_negative_recall_org': var_negative_recall_org,
             'var_positive_precision_org': var_positive_precision_org,
             'var_negative_precision_org': var_negative_precision_org, 'var_auc_org': var_auc_org, 'diversity': diversity,
             'base_models': base_models, 'creation_date': creation_date})

        return Ensemble_model_detail, Ensemble_model_result

    # def sub_db_bulk_insert_for_test4_detail_ensemble_ver3(self, en_loop, sn_loop, batch_id, model_name, raw_data, pvoting,
    #                                                       cmc_list, model_type_subno, en_mdodel_list):
    #     Ensemble_bulk_detail_frame = pd.DataFrame(
    #         columns=['batch_id', 'model_no', 'model_sub_no', 'sub_no', 'data_gubun', 'total_loan', 'gross_profit',
    #                  'interest_income', 'write_off', 'positive_rate', 'negative_rate', 'positive_recall', 'negative_recall',
    #                  'positive_precision', 'negative_precision', 'auc', 'true_positive', 'false_positive', 'true_negative',
    #                  'false_negative', 'creation_date'])
    #     batch_id = batch_id
    #     model_no = select_model_no_4(model_name)
    #     model_full_name = "ENSEMBLE"
    #     cd = datetime.datetime.now()
    #     # cm = confusion_matrix(y_result_20[i], pred)
    #     size_en = 20000
    #     ensemble_model_count = 0
    #     total_loan = list()  # list(range(size_en))# [0 for col in range(20)] for row in range(10000)]
    #     gross_profit = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     interest_income = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     write_off = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     positive_rate = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     negative_rate = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     positive_recall = list()  # list(range(size_en))#  [[0 for col in range(20)] for row in range(10000)]
    #     negative_recall = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     positive_precision = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     negative_precision = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     auc = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     model_sub_no = list()  # list(range(size_en))# [[0 for col in range(20)] for row in range(10000)]
    #     sub_no = list()  # list(range(size_en))#  [[0 for col in range(20)] for row in range(10000)]
    #     # negative_precision = [[0 for col in range(20)] for row in range(10000)]
    #     true_positive = list()
    #     false_positive = list()
    #     true_negative = list()
    #     false_negative = list()
    #     data_gubun = list()
    #
    #     avg_true_positive_ = np.zeros(20)
    #     avg_false_positive_ = np.zeros(20)  # list()#result_set['false_positive'].mean()
    #     avg_true_negative_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
    #     avg_false_negative_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
    #     avg_real_total_loan_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
    #     avg_real_gross_profit_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
    #     avg_real_interest_income_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
    #     avg_real_write_off_ = np.zeros(20)  # list()#result_set['write_off'].mean()
    #     var_total_loan_ = np.zeros(20)  # list()#round(result_set['total_loan'].std(),4)
    #     var_gross_profit_ = np.zeros(20)  # list()#round(result_set['gross_profit'].std(),4)
    #     var_interest_income_ = np.zeros(20)  # list()#round(result_set['interest_income'].std(),4)
    #     var_write_off_ = np.zeros(20)  # list()#round(result_set['write_off'].std(),4)
    #     var_positive_rate_ = np.zeros(20)  # list()#round(result_set['positive_rate'].std(),2)
    #     var_negative_rate_ = np.zeros(20)  # list()#round(result_set['negative_rate'].std(),2)
    #     var_positive_recall_ = np.zeros(20)  # list()#round(result_set['positive_recall'].std(),2)
    #     var_negative_recall_ = np.zeros(20)  # list()#round(result_set['negative_recall'].std(),2)
    #     var_positive_precision_ = np.zeros(20)  # list()#round(result_set['positive_precision'].std(),2)
    #     var_negative_precision_ = np.zeros(20)  # list()#round(result_set['negative_precision'].std(),2)
    #     var_auc_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)
    #
    #     avg_true_positive = list()  # result_set['true_positive'].mean()
    #     avg_false_positive = list()  # result_set['false_positive'].mean()
    #     avg_true_negative = list()  # result_set['true_negative'].mean()
    #     avg_false_negative = list()  # result_set['false_negative'].mean()
    #     avg_real_total_loan = list()  # result_set['total_loan'].mean()
    #     avg_real_gross_profit = list()  # result_set['gross_profit'].mean()
    #     avg_real_interest_income = list()  # result_set['interest_income'].mean()
    #     avg_real_write_off = list()  # result_set['write_off'].mean()
    #     var_total_loan = list()  # round(result_set['total_loan'].std(),4)
    #     var_gross_profit = list()  # round(result_set['gross_profit'].std(),4)
    #     var_interest_income = list()  # round(result_set['interest_income'].std(),4)
    #     var_write_off = list()  # round(result_set['write_off'].std(),4)
    #     var_positive_rate = list()  # round(result_set['positive_rate'].std(),2)
    #     var_negative_rate = list()  # round(result_set['negative_rate'].std(),2)
    #     var_positive_recall = list()  # round(result_set['positive_recall'].std(),2)
    #     var_negative_recall = list()  # round(result_set['negative_recall'].std(),2)
    #     var_positive_precision = list()  # round(result_set['positive_precision'].std(),2)
    #     var_negative_precision = list()  # round(result_set['negative_precision'].std(),2)
    #     var_auc = list()  # round(result_set['auc'].var(),4)
    #
    #     avg_true_positive_org_ = np.zeros(20)  # list()#(result_set['true_positive'].mean())
    #     avg_false_positive_org_ = np.zeros(20)  # list()#(result_set['false_positive'].mean())
    #     avg_true_negative_org_ = np.zeros(20)  # list()#result_set['true_negative'].mean()
    #     avg_false_negative_org_ = np.zeros(20)  # list()#result_set['false_negative'].mean()
    #     avg_real_total_loan_org_ = np.zeros(20)  # list()#result_set['total_loan'].mean()
    #     avg_real_gross_profit_org_ = np.zeros(20)  # list()#result_set['gross_profit'].mean()
    #     avg_real_interest_income_org_ = np.zeros(20)  # list()#result_set['interest_income'].mean()
    #     avg_real_write_off_org_ = np.zeros(20)  # list()# result_set['write_off'].mean()
    #     var_total_loan_org_ = np.zeros(20)  # list()#round(result_set['total_loan'].var(),4)
    #     var_gross_profit_org_ = np.zeros(20)  # list()#round(result_set['gross_profit'].var(),4)
    #     var_interest_income_org_ = np.zeros(20)  # list()# round(result_set['interest_income'].var(),4)
    #     var_write_off_org_ = np.zeros(20)  # list()#round(result_set['write_off'].var(),4)
    #     var_positive_rate_org_ = np.zeros(20)  # list()#round(result_set['positive_rate'].var(),2)
    #     var_negative_rate_org_ = np.zeros(20)  # list()#round(result_set['negative_rate'].var(),2)
    #     var_positive_recall_org_ = np.zeros(20)  # list()# round(result_set['positive_recall'].var(),2)
    #     var_negative_recall_org_ = np.zeros(20)  # list()#round(result_set['negative_recall'].var(),2)
    #     var_positive_precision_org_ = np.zeros(20)  # list()#round(result_set['positive_precision'].var(),2)
    #     var_negative_precision_org_ = np.zeros(20)  # list()#round(result_set['negative_precision'].var(),2)
    #     var_auc_org_ = np.zeros(20)  # list()#round(result_set['auc'].var(),4)
    #
    #     avg_true_positive_org = list()  # (result_set['true_positive'].mean())
    #     avg_false_positive_org = list()  # (result_set['false_positive'].mean())
    #     avg_true_negative_org = list()  # result_set['true_negative'].mean()
    #     avg_false_negative_org = list()  # result_set['false_negative'].mean()
    #     avg_real_total_loan_org = list()  # result_set['total_loan'].mean()
    #     avg_real_gross_profit_org = list()  # result_set['gross_profit'].mean()
    #     avg_real_interest_income_org = list()  # result_set['interest_income'].mean()
    #     avg_real_write_off_org = list()  # result_set['write_off'].mean()
    #     var_total_loan_org = list()  # round(result_set['total_loan'].var(),4)
    #     var_gross_profit_org = list()  # round(result_set['gross_profit'].var(),4)
    #     var_interest_income_org = list()  # round(result_set['interest_income'].var(),4)
    #     var_write_off_org = list()  # round(result_set['write_off'].var(),4)
    #     var_positive_rate_org = list()  # round(result_set['positive_rate'].var(),2)
    #     var_negative_rate_org = list()  # round(result_set['negative_rate'].var(),2)
    #     var_positive_recall_org = list()  # round(result_set['positive_recall'].var(),2)
    #     var_negative_recall_org = list()  # round(result_set['negative_recall'].var(),2)
    #     var_positive_precision_org = list()  # round(result_set['positive_precision'].var(),2)
    #     var_negative_precision_org = list()  # round(result_set['negative_precision'].var(),2)
    #     var_auc_org = list()  # round(result_set['auc'].var(),4)
    #     base_models = list()
    #
    #     model_sub_no_result = list()
    #     sub_no_result = list()  #
    #     for i in tqdm(sn_loop):
    #         # model_sub_no = i
    #         model_sub_no_result.append(en_loop[i])
    #         rml = self.return_model_list(model_type_subno, en_mdodel_list[ensemble_model_count])
    #         # print(len(rml))
    #         # print((rml))
    #         mbl = (','.join(rml))
    #         ensemble_model_count = ensemble_model_count + 1
    #         for k in range(20):
    #             model_sub_no.append(en_loop[i])
    #             model_sub_no.append(en_loop[i])
    #             sub_no.append(k)
    #             sub_no.append(k)
    #             cmc = cmc_list[i][k]
    #             total_count = sum(cmc_list[0][0])
    #             TP, FP, TN, FN = cmc[3], cmc[1], cmc[0], cmc[2]
    #             # cal_result = calculate(raw_data, pvoting[i][0][k])
    #             # cal_result, dataframe type to list type 대체
    #             cloan, cbase_profit, cpredict_profit, cpredict_loan, cbase_loss, cpredict_loss, cy1, cy2 = self.calculate_ver2(
    #                 raw_data, pvoting[i][0][k])
    #
    #             # print(TP, FP, TN, FN )
    #             # total_loan_ = cal_result['대출실행금액'].sum()
    #             total_loan_ = cloan.sum()
    #             # before_inter = cal_result['[기존]이자수익'].sum()
    #             before_inter = cbase_profit.sum()
    #             # after_inter = cal_result['[예상]이자수익'].sum()
    #             after_inter = cpredict_profit.sum()
    #
    #             # after_total_loan = cal_result['[예상]대출금액'].sum()
    #             after_total_loan = cpredict_loan.sum()
    #             profit_loan = after_inter - before_inter
    #
    #             # before_loss = cal_result['[기존]원금손실'].sum()
    #             before_loss = cbase_loss.sum()
    #             # after_loss = cal_result['[예상]원금손실'].sum()
    #             after_loss = cpredict_loss.sum()
    #
    #             before_profit = before_inter - before_loss
    #             profit_loss = before_loss - after_loss
    #             profit = profit_loan + profit_loss
    #
    #             # print("TP FP TN FN", TP, FP, TN, FN)
    #             # db_insert_for_test4_detail(
    #
    #             data_gubun.append('T')
    #             total_loan.append(after_total_loan)  # 총대출액
    #             gross_profit.append(after_inter - after_loss)  # 총수익
    #             interest_income.append(after_inter)  # 총이자수익
    #             # gross_profit_rate = round((after_inter - after_loss)/after_total_loan*100,2),
    #             write_off.append(after_loss)
    #             positive_rate.append(round(FN / (TN + FN + not_nan) * 100, 2))
    #             negative_rate.append(round((TN + FN) / total_count * 100, 2))
    #             positive_recall.append(round(TP / (TP + FN + not_nan) * 100, 2))
    #             negative_recall.append(round(TN / (TN + FP + not_nan) * 100, 2))
    #             positive_precision.append(round(TP / (TP + FP + not_nan) * 100, 2))
    #             negative_precision.append(round(TN / (TN + FN + not_nan) * 100, 2))
    #             auc.append(round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #             # total_count=total_count,
    #             true_positive.append(TP)
    #             false_positive.append(FP)
    #             true_negative.append(TN)
    #             false_negative.append(FN)
    #             creation_date = cd  # ) #'1000-01-01')
    #
    #             avg_true_positive_[k] = TP
    #             avg_false_positive_[k] = FP
    #             avg_true_negative_[k] = TN
    #             avg_false_negative_[k] = FN
    #             avg_real_total_loan_[k] = after_total_loan
    #             avg_real_gross_profit_[k] = (after_inter - after_loss)
    #             avg_real_interest_income_[k] = after_inter
    #             avg_real_write_off_[k] = after_loss
    #             var_total_loan_[k] = after_total_loan
    #             var_gross_profit_[k] = (after_inter - after_loss)
    #             var_interest_income_[k] = after_inter
    #             var_write_off_[k] = after_loss
    #             var_positive_rate_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
    #             var_negative_rate_[k] = (round((TN + FN) / total_count * 100, 2))
    #             var_positive_recall_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
    #             var_negative_recall_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
    #             var_positive_precision_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
    #             var_negative_precision_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
    #             var_auc_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #
    #             # TP, FP, TN, FN = cal_result['y1'].sum(), 0, cal_result['y2'].sum(), 0
    #             TP, FP, TN, FN = cy1.sum(), 0, cy2.sum(), 0
    #
    #             data_gubun.append('B')
    #             total_loan.append(total_loan_)
    #             gross_profit.append(before_inter - before_loss)
    #             # gross_profit_rate = round((before_inter - before_loss)/total_loan*100,2),
    #             interest_income.append(before_inter)
    #             write_off.append(before_loss)
    #             positive_rate.append(round(FN / (TN + FN + not_nan) * 100, 2))
    #             negative_rate.append(round((TN + FN) / total_count * 100, 2))
    #             positive_recall.append(round(TP / (TP + FN + not_nan) * 100, 2))
    #             negative_recall.append(round(TN / (TN + FP + not_nan) * 100, 2))
    #             positive_precision.append(round(TP / (TP + FP + not_nan) * 100, 2))
    #             negative_precision.append(round(TN / (TN + FN + not_nan) * 100, 2))
    #             auc.append(round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #             # total_count=total_count,
    #             true_positive.append(TP)
    #             false_positive.append(FP)
    #             true_negative.append(TN)
    #             false_negative.append(FN)
    #
    #             avg_true_positive_org_[k] = TP
    #             avg_false_positive_org_[k] = FP
    #             avg_true_negative_org_[k] = TN
    #             avg_false_negative_org_[k] = FN
    #             avg_real_total_loan_org_[k] = total_loan_
    #             avg_real_gross_profit_org_[k] = (before_inter - before_loss)
    #             avg_real_interest_income_org_[k] = before_inter
    #             avg_real_write_off_org_[k] = before_loss
    #             var_total_loan_org_[k] = total_loan_
    #             var_gross_profit_org_[k] = (before_inter - before_loss)
    #             var_interest_income_org_[k] = before_inter
    #             var_write_off_org_[k] = before_loss
    #             var_positive_rate_org_[k] = (round(FN / (TN + FN + not_nan) * 100, 2))
    #             var_negative_rate_org_[k] = (round((TN + FN) / total_count * 100, 2))
    #             var_positive_recall_org_[k] = (round(TP / (TP + FN + not_nan) * 100, 2))
    #             var_negative_recall_org_[k] = (round(TN / (TN + FP + not_nan) * 100, 2))
    #             var_positive_precision_org_[k] = (round(TP / (TP + FP + not_nan) * 100, 2))
    #             var_negative_precision_org_[k] = (round(TN / (TN + FN + not_nan) * 100, 2))
    #             var_auc_org_[k] = (round((TN / (TN + FP + not_nan) + TP / (TP + FN + not_nan)) * 0.5 * 100, 2))
    #
    #         avg_true_positive.append((avg_true_positive_.mean()))
    #         avg_false_positive.append((avg_false_positive_.mean()))
    #         avg_true_negative.append(avg_true_negative_.mean())
    #         avg_false_negative.append(avg_false_negative_.mean())
    #         avg_real_total_loan.append(avg_real_total_loan_.mean())
    #         avg_real_gross_profit.append(avg_real_gross_profit_.mean())
    #         avg_real_interest_income.append(avg_real_interest_income_.mean())
    #         avg_real_write_off.append(avg_real_write_off_.mean())
    #         var_total_loan.append(var_total_loan_.std())
    #         var_gross_profit.append(var_gross_profit_.std())
    #         var_interest_income.append(var_interest_income_.std())
    #         var_write_off.append(var_write_off_.std())
    #         var_positive_rate.append(var_positive_rate_.std())
    #         var_negative_rate.append(var_negative_rate_.std())
    #         var_positive_recall.append(var_positive_recall_.std())
    #         var_negative_recall.append(var_negative_recall_.std())
    #         var_positive_precision.append(var_positive_precision_.std())
    #         var_negative_precision.append(var_negative_precision_.std())
    #         var_auc.append(var_auc_.std())
    #
    #         avg_true_positive_org.append(avg_true_positive_org_.mean())
    #         avg_false_positive_org.append(avg_false_positive_org_.mean())
    #         avg_true_negative_org.append(avg_true_negative_org_.mean())
    #         avg_false_negative_org.append(avg_false_negative_org_.mean())
    #         avg_real_total_loan_org.append(avg_real_total_loan_org_.mean())
    #         avg_real_gross_profit_org.append(avg_real_gross_profit_org_.mean())
    #         avg_real_interest_income_org.append(avg_real_interest_income_org_.mean())
    #         avg_real_write_off_org.append(avg_real_write_off_org_.mean())
    #         var_total_loan_org.append(var_total_loan_org_.std())
    #         var_gross_profit_org.append(var_gross_profit_org_.std())
    #         var_interest_income_org.append(var_interest_income_org_.std())
    #         var_write_off_org.append(var_write_off_org_.std())
    #         var_positive_rate_org.append(var_positive_rate_org_.std())
    #         # var_positive_rate_org.append(var_positive_rate_org_.std())
    #         var_negative_rate_org.append(var_negative_rate_org_.std())
    #         var_positive_recall_org.append(var_positive_recall_org_.std())
    #         var_negative_recall_org.append(var_negative_recall_org_.std())
    #         var_positive_precision_org.append(var_positive_precision_org_.std())
    #         var_negative_precision_org.append(var_negative_precision_org_.std())
    #         var_auc_org.append(var_auc_org_.std())
    #
    #         base_models.append(mbl)
    #         creation_date = cd
    #         diversity = 0
    #
    #     Ensemble_model_detail = pd.DataFrame(
    #         {'batch_id': batch_id, 'model_no': model_no, 'model_sub_no': model_sub_no, 'sub_no': sub_no,
    #          'data_gubun': data_gubun, 'total_loan': total_loan, 'gross_profit': gross_profit,
    #          'interest_income': interest_income, 'write_off': write_off, 'positive_rate': positive_rate,
    #          'negative_rate': negative_rate, 'positive_recall': positive_recall, 'negative_recall': negative_recall,
    #          'positive_precision': positive_precision, 'negative_precision': negative_precision, 'auc': auc,
    #          'true_positive': true_positive, 'false_positive': false_positive, 'true_negative': true_negative,
    #          'false_negative': false_negative, 'creation_date': creation_date})  # ,ignore_index=True
    #
    #     Ensemble_model_result = pd.DataFrame(
    #         {'batch_id': batch_id, 'model_no': model_no, 'model_full_name': model_full_name,
    #          'model_sub_no': model_sub_no_result, 'avg_true_positive': avg_true_positive,
    #          'avg_false_positive': avg_false_positive, 'avg_true_negative': avg_true_negative,
    #          'avg_false_negative': avg_false_negative, 'avg_true_positive_org': avg_true_positive_org,
    #          'avg_false_positive_org': avg_false_positive_org, 'avg_true_negative_org': avg_true_negative_org,
    #          'avg_false_negative_org': avg_false_negative_org, 'avg_real_total_loan': avg_real_total_loan,
    #          'avg_real_gross_profit': avg_real_gross_profit, 'avg_real_interest_income': avg_real_interest_income,
    #          'avg_real_write_off': avg_real_write_off, 'avg_real_total_loan_org': avg_real_total_loan_org,
    #          'avg_real_gross_profit_org': avg_real_gross_profit_org,
    #          'avg_real_interest_income_org': avg_real_interest_income_org, 'avg_real_write_off_org': avg_real_write_off_org,
    #          'var_total_loan': var_total_loan, 'var_gross_profit': var_gross_profit,
    #          'var_interest_income': var_interest_income, 'var_write_off': var_write_off,
    #          'var_positive_rate': var_positive_rate, 'var_negative_rate': var_negative_rate,
    #          'var_positive_recall': var_positive_recall, 'var_negative_recall': var_negative_recall,
    #          'var_positive_precision': var_positive_precision, 'var_negative_precision': var_negative_precision,
    #          'var_auc': var_auc, 'var_total_loan_org': var_total_loan_org, 'var_gross_profit_org': var_gross_profit_org,
    #          'var_interest_income_org': var_interest_income_org, 'var_write_off_org': var_write_off_org,
    #          'var_positive_rate_org': var_positive_rate_org, 'var_negative_rate_org': var_negative_rate_org,
    #          'var_positive_recall_org': var_positive_recall_org, 'var_negative_recall_org': var_negative_recall_org,
    #          'var_positive_precision_org': var_positive_precision_org,
    #          'var_negative_precision_org': var_negative_precision_org, 'var_auc_org': var_auc_org, 'diversity': diversity,
    #          'base_models': base_models, 'creation_date': creation_date})
    #
    #     return Ensemble_model_detail, Ensemble_model_result

    def calculate_ver2(self, raw_data, predict):
        temp = raw_data.loc[predict.index]
        pred = predict['pred']
        y1 = temp['y1']
        y2 = (~(y1.astype(bool))).astype(int)
        pred2 = (~(pred.astype(bool))).astype(int)  # rx_test_raw_ 18

        loan = temp['대출실행금액']
        bond = temp['금리']
        base_profit = loan * bond * 0.01 * y2  # 25
        base_loss = loan * y1  # 26
        predict_loan = loan * pred2  # 27
        predict_profit = loan * bond * 0.01 * pred2 * y2
        preditc_loss = loan * pred2 * y1  # 33

        return loan, base_profit, predict_profit, predict_loan, base_loss, preditc_loss, y1, y2

    # def sub_ensemble(self, ensnemble_range, model_predict, y_result_20):
    #     print("병렬처리 시작")
    #     start = time.time()
    #
    #     func = partial(self.ensemble_loop_ver2, model_predict, y_result_20)  # list_pred_sum)
    #     pred_array_en2 = list(range(0, 1))
    #
    #     with concurrent.futures.ProcessPoolExecutor(10) as executor:
    #         pred_array_en2 = list(executor.map(func, ensnemble_range))
    #
    #     end = time.time()
    #     print("병렬처리 수행 시각", end - start, 's')
    #
    #     return pred_array_en2


    def sub_voting(self, sub_range, voting_n, pred_array_en_m, y_result_20):
        avg_range = 20
        extract_model = 30

        # en_mdodel_list = list(ensnemble_range)

        for vn in voting_n:
            cmc_list, pvoting = self.ensemble_voting(sub_range, vn, pred_array_en_m, y_result_20)
        print(len(cmc_list))
        return cmc_list, pvoting


    # def sub_build_bulk_dataframe(self, sub_range, ensnemble_range, batch_id, model_name, rx_test_raw_, pvoting, cmc_list,
    #                              model_type_subno, model_range):
    #     for i in sub_range:
    #         random.seed(ensnemble_range[i])
    #         en_mdodel_list[i] = (random.sample(range(model_range), 30))
    #
    #     print("build dataframe 시작")
    #     start = time.time()
    #     Ensemble_bulk_detail_frame, Ensemble_bulk_result_frame = self.sub_db_bulk_insert_for_test4_detail_ensemble_ver3(
    #         ensnemble_range, sub_range, batch_id, model_name, rx_test_raw_, pvoting, cmc_list, model_type_subno,
    #         en_mdodel_list)
    #     end = time.time()
    #     print("build dataframe 시각", end - start, 's')
    #
    #     print("벌크 인서트 시작")
    #     start = time.time()
    #     self.db_insert4_result(Ensemble_bulk_result_frame)
    #     self.db_insert4_detail(Ensemble_bulk_detail_frame)
    #     end = time.time()
    #     print("벌크 수행 시각", end - start, 's')


    # def million_ensemble(self, rx_test_raw_, batch_id, voting_point, ensnemble_range, model_predict, y_result_20, model_name,
    #                      model_type_subno, model_range):
    #     # batch_id = db_select_max_batch_id_4("dc_batch_result_t4")+1
    #
    #     pred_array_en_m = self.sub_ensemble(ensnemble_range, model_predict, y_result_20)
    #     sub_range = range(0, len(ensnemble_range))
    #
    #     cmc_list, pvoting = self.sub_voting(sub_range, voting_point, pred_array_en_m, y_result_20)
    #     print('cmc_list')
    #     # print(cmc_list[0])
    #     # print(cmc_list[1])
    #     self.sub_build_bulk_dataframe(sub_range, ensnemble_range, batch_id, model_name, rx_test_raw_, pvoting, cmc_list,
    #                              model_type_subno, model_range)

    def split_range(self, source_range, num):
        each = int((source_range[1] - source_range[0]) / num)

        retlist = []

        for i in range(num):
            retlist.append((source_range[0] + each * i, source_range[0] + each * (i + 1)))

        if source_range[1] > retlist[-1][1]:
            retlist[-1] = (retlist[-1][0], source_range[1])

        return retlist

    def ensemble(self, batch_info, batch_param):
        data_name = 'NICE_IBK_002'

        # 전처리 데이터, 레이블, 원본 데이터
        data, label, raw_data = load_and_preprocessing("./data/" + data_name + ".csv", '계좌번호', "./data/ibk_datatype.csv", '구분')

        # 학습:테스트 = 6:4
        train_x, test_x, train_y, test_y = split_same_ratio(data, label, test_size=0.4, random_state=0)

        # 전체 테스트 데이터셋 선택.
        rx_test, trash0, ry_test, trash1 = split_same_ratio(test_x, test_y, test_size=1, random_state=0)

        # 원본에서 계좌번호, 대출금액, 금리, 레이블 추출
        rx_test_raw = raw_data.loc[rx_test.index]
        rx_test_raw_ = rx_test_raw[['대출실행금액', '금리', 'y1']]

        # 추천 베이스 모델 목록 조회
        result_set = self.call_the_query_for_ensemble_1(batch_info['candidate_id'])
        
        # 베이스 모델 로딩(374개 서버 20초)
        model_load, model_access_key, model_type_subno = self.load_the_models_1(result_set)
        
        # 베이스 모델로 테스트 데이터 예측(서버 35초, 로컬 53초)
        model_predict, y_result_20 = self.extract_model_learning_(test_x, rx_test, ry_test, model_load, model_access_key, model_type_subno)
        range_list = self.split_range([0, 999999], 1000)

        # 병렬 처리 파라메터 생성
        param_Args = dict(model_predict=model_predict,
                          y_result_20=y_result_20,
                          rx_test_raw_=rx_test_raw_,
                          model_type_subno=model_type_subno,
                          batch_info=batch_info,
                          batch_param=batch_param)

        parallel_args = [(range, param_Args) for range in range_list]

        # range_param = range_list[2]

        def task(parallel_args):
            import sys
            import traceback

            logfile = f"/opt/data/DeepCredit/ensemble.log"
            sys.stdout = open(logfile, "a")

            mod_path = join(dirname(dirname(__file__)))
            sys.path.append(mod_path)
            sys.path.append("/opt/data/DeepCredit/")

            try:
                from Ensemble.Ensemble import Ensemble
                # from Learning.Imbalance import imbalance_data
                # from Learning.models import get_model_id, save_model
                # from Prediction.Save_result import save_result

                ensemble = Ensemble()

                range_param, Args = parallel_args

                model_predict = Args['model_predict']
                y_result_20 = Args['y_result_20']
                rx_test_raw_ = Args['rx_test_raw_']
                model_type_subno = Args['model_type_subno']
                batch_info = Args['batch_info']
                batch_param = Args['batch_param']

                # 앙상블 반복 횟수
                ensemble_range = range(range_param[0], range_param[1])

                pred_array_en2 = list(ensemble_range)
                en_model_list = list(ensemble_range)
                model_range = len(model_predict)

                # 앙상블(30개 * 20번)
                # TODO: 앙상블 갯수 및 반복 횟수 파라메터 참조
                print("앙상블...")
                model_predict_list = list(model_predict.values())
                for i in tqdm(range(0, len(pred_array_en2))):
                    seed = pred_array_en2[i]
                    pred_array_en2[i] = ensemble.ensemble_loop_ver2(model_predict_list, y_result_20, seed)

                    random.seed(seed)
                    en_model_list[i] = (random.sample(range(model_range), 30))

                # 보팅
                # TODO: 연체 기준 파라메터 처리
                cmc_list = ensemble.ensemble_voting(ensemble_range, 11, pred_array_en2, y_result_20)

                # 상세 결과 생성 및 저장
                # TODO : 소스 정리
                detail_frame, result_frame = ensemble.db_bulk_insert_for_test4_detail_ensemble_ver3(
                    batch_info["batch_id"],
                    "ENSEMBLE",
                    rx_test_raw_,
                    pred_array_en2,
                    ensemble_range,
                    cmc_list,
                    model_type_subno,
                    en_model_list)

                # 모델별(모델번호, 모델보조번호) 고객 예측 결과 저장
                # TODO: 한번만 실행
                if ensemble_range[0] == 0:
                    ensemble.db_insert_for_test4_pred(model_predict, model_type_subno, batch_info["batch_id"], model_range)

                # 배치 결과, 상세결과 저장
                ensemble.db_insert4_result(result_frame)
                ensemble.db_insert4_detail(detail_frame)

            except Exception as e:
                print(traceback.format_exc())
                raise e

            return

        # 병렬처리 환경 파일 로딩
        ini_path = join(dirname(dirname(__file__)), 'Learning/distributed_computing/server_config.ini')
        scheduler_server = 'server1'

        configs = ini2dict(ini_path)

        ## 스케줄러 연결
        config_scheduler = configs[scheduler_server]
        client = Client(f"{config_scheduler['host']}:{config_scheduler['scheduler_port']}")
        print(client)

        ## 병렬 처리 수행
        start_time = time.time()
        futures = client.map(task, parallel_args)
        tasks = list(as_completed(futures, with_results=False))  # realize lazy function
        print(f"* Elapsed time: {time.time() - start_time:.2f}s")
        print("- Results:", [task.status for task in tasks])