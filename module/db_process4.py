import sys

sys.path.append('../')
# import import_ipynb
# import numpy as np
# import db.env as denv

# import db.DB_Handler
from ENV.DB_Handler import DBHandler
import sqlalchemy as sqc
import pandas as pd
from datetime import datetime
from pytz import timezone, utc

kor = timezone("Asia/Seoul")

def init_db():
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    
    return engine

def db_select_max_batch_id_4(table_name) : 
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT MAX(batch_id) FROM {table_name}"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0]

def db_select_max_batch_id_4_from_t(table_name) : 
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT MAX(batch_id) FROM {table_name}"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0]

def select_model_no_4(model_full_name) : 
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT model_no FROM dc_model_list WHERE model_full_name = '{model_full_name}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0]

def db_insert_batch_4(batch_id, batch_desc, batch_memo,batch_param, candidate_id, mode, dataset_group, dataset_version, model_group, state, user_id, display_yn, experiment_type):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    table = sqc.Table('dc_batch', metadata, autoload=True, autoload_with = engine)

    query = sqc.insert(table).values(batch_id = batch_id, batch_desc = batch_desc, batch_memo = batch_memo,candidate_id=candidate_id, batch_param=batch_param , mode = mode, dataset_group = dataset_group, dataset_version = dataset_version, model_group = model_group, state = state, user_id = user_id, display_yn = display_yn, experiment_type = experiment_type, creation_date = datetime.now().astimezone(kor))
    result_proxy = engine.execute(query)
    result_proxy.close()

#xxxxxxx
# def db_insert_for_test4(batch_id, model_no, model_full_name, model_sub_no) :
#     dbhandler = DBHandler()
#     engine = dbhandler.get_connection()
#     connection = engine.connect()
#     metadata = sqc.MetaData()
#
#     query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no, avg_true_positive, avg_false_positive, avg_true_negative, avg_false_negative, avg_real_total_loan, avg_real_gross_profit, avg_real_interest_income, avg_real_write_off, avg_true_positive_org, avg_false_positive_org, avg_true_negative_org, avg_false_negative_org, real_total_loan_org, avg_real_gross_profit_org, avg_real_gross_profit_org ,avg_real_interest_income_org, avg_real_write_off_org, diversity, base_model, creation_date) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no}, {avg_true_positive}, {avg_false_positive}, {avg_true_negative}, {avg_false_negative}, {avg_real_total_loan}, {avg_real_gross_profit}, {avg_real_interest_income}, {avg_real_write_off}, {avg_true_positive_org}, {avg_false_positive_org}, {avg_true_negative_org}, {avg_false_negative_org}, {real_total_loan_org}, {avg_real_gross_profit_org} , {avg_real_interest_income_org}, {avg_real_write_off_org}, {diversity}, {base_model}, {creation_date})"
#
#     result_proxy = engine.execute(query)
#     result_proxy.close()

#ooooooo
def db_insert_for_test4_1(batch_id, model_no, model_full_name, model_sub_no) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()

    query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no})"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_result_t4_2(batch_id, model_no, model_full_name, model_sub_no,diversity,base_models,creation_date) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    #diversity = {round(diversity_avg,6)},\
    #base_models = '{base_models}',\
    #creation_date = '{creation_date}'\

    query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no, diversity,base_models,creation_date) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no} ,{diversity},'{base_models}', '{creation_date}')"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

#oooooooo
def db_insert_for_test4_detail(batch_id, model_no, model_sub_no, sub_no, 
                               data_gubun, total_loan, gross_profit, interest_income, 
                               write_off, positive_rate, negative_rate, positive_recall, 
                               negative_recall, positive_precision, negative_precision, 
                               auc, true_positive, false_positive, true_negative, 
                               false_negative, creation_date) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    query = f"INSERT INTO dc_batch_result_t4_detail (batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit, interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, true_positive, false_positive, true_negative, false_negative, creation_date) VALUES ({batch_id}, {model_no}, {model_sub_no}, {sub_no}, '{data_gubun}', {total_loan}, {gross_profit}, {interest_income}, {write_off}, {positive_rate}, {negative_rate}, {positive_recall}, {negative_recall}, {positive_precision}, {negative_precision}, {auc}, {true_positive}, {false_positive}, {true_negative}, {false_negative}, '{creation_date}')"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

#oooooooo
def db_insert_for_test4_predict(batch_id, model_no, model_sub_no, predict_result) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    query = f"INSERT INTO dc_batch_result_t4_predict (batch_id, model_no, model_sub_no, predict_result) VALUES ({batch_id}, {model_no}, {model_sub_no}, '{predict_result}')"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def update_for_t4_detail(batch_id, model_no, model_sub_no, base_models, diversity_avg, creation_date):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    #평균
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'T'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    query = f"UPDATE dc_batch_result_t4 SET     avg_true_positive = {result_set['true_positive'].mean()},    avg_false_positive = {result_set['false_positive'].mean()},    avg_true_negative = {result_set['true_negative'].mean()},    avg_false_negative = {result_set['false_negative'].mean()},    avg_real_total_loan = {result_set['total_loan'].mean()},    avg_real_gross_profit = {result_set['gross_profit'].mean()},    avg_real_interest_income = {result_set['interest_income'].mean()},    avg_real_write_off = {result_set['write_off'].mean()}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'T'}"
    result_proxy = engine.execute(query)
    result_proxy.close()
                           
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'B'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    query = f"UPDATE dc_batch_result_t4 SET     avg_true_positive_org = {(result_set['true_positive'].mean())},    avg_false_positive_org = {(result_set['false_positive'].mean())},    avg_true_negative_org = {result_set['true_negative'].mean()},    avg_false_negative_org = {result_set['false_negative'].mean()},    avg_real_total_loan_org = {result_set['total_loan'].mean()},    avg_real_gross_profit_org = {result_set['gross_profit'].mean()},    avg_real_interest_income_org = {result_set['interest_income'].mean()},    avg_real_write_off_org = {result_set['write_off'].mean()}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'B'}"
    result_proxy = engine.execute(query)
    result_proxy.close()
    
    #분산 
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'T'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    ###INSERT 시점 'positive_rate' 분산 값 확인용
    print('positive_rate  var',round(result_set['positive_rate'].var(),4))
    print(type(round(result_set['positive_rate'].var(),4)))
    print('negative_rate  var',round(result_set['negative_rate'].var(),4))
    print(type(round(result_set['negative_rate'].var(),4)))
    query = f"UPDATE dc_batch_result_t4 SET     var_total_loan = {round(result_set['total_loan'].var(),4)},    var_gross_profit = {round(result_set['gross_profit'].var(),4)},    var_interest_income = {round(result_set['interest_income'].var(),4)},    var_write_off = {round(result_set['write_off'].var(),4)},    var_positive_rate = {round(result_set['positive_rate'].var(),2)},    var_negative_rate = {round(result_set['negative_rate'].var(),2)},    var_positive_recall = {round(result_set['positive_recall'].var(),2)},    var_negative_recall = {round(result_set['negative_recall'].var(),2)},    var_positive_precision = {round(result_set['positive_precision'].var(),2)},    var_negative_precision = {round(result_set['negative_precision'].var(),2)},    var_auc = {round(result_set['auc'].var(),4)}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'T'}"
    result_proxy = engine.execute(query)
    result_proxy.close()
                          
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'B'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    query = f"UPDATE dc_batch_result_t4 SET     var_total_loan_org = {round(result_set['total_loan'].var(),4)},    var_gross_profit_org = {round(result_set['gross_profit'].var(),4)},    var_interest_income_org = {round(result_set['interest_income'].var(),4)},    var_write_off_org = {round(result_set['write_off'].var(),4)},    var_positive_rate_org = {round(result_set['positive_rate'].var(),2)},    var_negative_rate_org = {round(result_set['negative_rate'].var(),2)},    var_positive_recall_org = {round(result_set['positive_recall'].var(),2)},    var_negative_recall_org = {round(result_set['negative_recall'].var(),2)},    var_positive_precision_org = {round(result_set['positive_precision'].var(),2)},    var_negative_precision_org = {round(result_set['negative_precision'].var(),2)},    var_auc_org = {round(result_set['auc'].var(),4)}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'B'}"
    result_proxy = engine.execute(query)
    result_proxy.close()

    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'T'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    print('diver',{round(diversity_avg,6)})
    print('d otype',type(diversity_avg))
    print('d type',({type(round(diversity_avg,6))}))
    
    query = f"UPDATE dc_batch_result_t4 SET     diversity = {round(diversity_avg,6)},    base_models = '{base_models}',    creation_date = '{creation_date}'    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'T'}"
    result_proxy = engine.execute(query)
    result_proxy.close()

def update_for_result_t4_2(batch_id, model_no, model_sub_no):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    #평균
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'T'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    #query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no, diversity,base_models,creation_date) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no} ,{diversity},'{base_models}', '{creation_date}')"

    query = f"UPDATE dc_batch_result_t4 SET     avg_true_positive = {result_set['true_positive'].mean()},    avg_false_positive = {result_set['false_positive'].mean()},    avg_true_negative = {result_set['true_negative'].mean()},    avg_false_negative = {result_set['false_negative'].mean()},    avg_real_total_loan = {result_set['total_loan'].mean()},    avg_real_gross_profit = {result_set['gross_profit'].mean()},    avg_real_interest_income = {result_set['interest_income'].mean()},    avg_real_write_off = {result_set['write_off'].mean()},    var_total_loan = {round(result_set['total_loan'].var(),4)},    var_gross_profit = {round(result_set['gross_profit'].var(),4)},    var_interest_income = {round(result_set['interest_income'].var(),4)},    var_write_off = {round(result_set['write_off'].var(),4)},    var_positive_rate = {round(result_set['positive_rate'].var(),2)},    var_negative_rate = {round(result_set['negative_rate'].var(),2)},    var_positive_recall = {round(result_set['positive_recall'].var(),2)},    var_negative_recall = {round(result_set['negative_recall'].var(),2)},    var_positive_precision = {round(result_set['positive_precision'].var(),2)},    var_negative_precision = {round(result_set['negative_precision'].var(),2)},    var_auc = {round(result_set['auc'].var(),4)}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'T'}"
    result_proxy = engine.execute(query)
    result_proxy.close()
                           
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'B'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    query = f"UPDATE dc_batch_result_t4 SET     avg_true_positive_org = {(result_set['true_positive'].mean())},    avg_false_positive_org = {(result_set['false_positive'].mean())},    avg_true_negative_org = {result_set['true_negative'].mean()},    avg_false_negative_org = {result_set['false_negative'].mean()},    avg_real_total_loan_org = {result_set['total_loan'].mean()},    avg_real_gross_profit_org = {result_set['gross_profit'].mean()},    avg_real_interest_income_org = {result_set['interest_income'].mean()},    avg_real_write_off_org = {result_set['write_off'].mean()},    var_total_loan_org = {round(result_set['total_loan'].var(),4)},    var_gross_profit_org = {round(result_set['gross_profit'].var(),4)},    var_interest_income_org = {round(result_set['interest_income'].var(),4)},    var_write_off_org = {round(result_set['write_off'].var(),4)},    var_positive_rate_org = {round(result_set['positive_rate'].var(),2)},    var_negative_rate_org = {round(result_set['negative_rate'].var(),2)},    var_positive_recall_org = {round(result_set['positive_recall'].var(),2)},    var_negative_recall_org = {round(result_set['negative_recall'].var(),2)},    var_positive_precision_org = {round(result_set['positive_precision'].var(),2)},    var_negative_precision_org = {round(result_set['negative_precision'].var(),2)},    var_auc_org = {round(result_set['auc'].var(),4)}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'B'}"
    result_proxy = engine.execute(query)
    result_proxy.close()

def insert_for_result_t4_2_combine(batch_id, model_no,  model_full_name, model_sub_no,diversity,base_models,creation_date):
    
    #diversity = {round(diversity_avg,6)},\
    #base_models = '{base_models}',\
    #creation_date = '{creation_date}'\    

    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()

    #평균
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'T'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    #query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no, diversity,base_models,creation_date) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no} ,{diversity},'{base_models}', '{creation_date}')"

    avg_true_positive = result_set['true_positive'].mean()
    avg_false_positive = result_set['false_positive'].mean()
    avg_true_negative = result_set['true_negative'].mean()
    avg_false_negative = result_set['false_negative'].mean()
    avg_real_total_loan = result_set['total_loan'].mean()
    avg_real_gross_profit = result_set['gross_profit'].mean()
    avg_real_interest_income = result_set['interest_income'].mean()
    avg_real_write_off = result_set['write_off'].mean()
    var_total_loan = round(result_set['total_loan'].std(),4)
    var_gross_profit = round(result_set['gross_profit'].std(),4)
    var_interest_income = round(result_set['interest_income'].std(),4)
    var_write_off = round(result_set['write_off'].std(),4)
    var_positive_rate = round(result_set['positive_rate'].std(),2)
    var_negative_rate = round(result_set['negative_rate'].std(),2)
    var_positive_recall = round(result_set['positive_recall'].std(),2)
    var_negative_recall = round(result_set['negative_recall'].std(),2)
    var_positive_precision = round(result_set['positive_precision'].std(),2)
    var_negative_precision = round(result_set['negative_precision'].std(),2)
    var_auc = round(result_set['auc'].var(),4)
    #WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'T'}"
    result_proxy = engine.execute(query)
    result_proxy.close()  
        
    query = f"SELECT * FROM dc_batch_result_t4_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{'B'}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    avg_true_positive_org = (result_set['true_positive'].mean())
    avg_false_positive_org = (result_set['false_positive'].mean())
    avg_true_negative_org = result_set['true_negative'].mean()
    avg_false_negative_org = result_set['false_negative'].mean()
    avg_real_total_loan_org = result_set['total_loan'].mean()
    avg_real_gross_profit_org = result_set['gross_profit'].mean()
    avg_real_interest_income_org = result_set['interest_income'].mean()
    avg_real_write_off_org = result_set['write_off'].mean()
    var_total_loan_org = round(result_set['total_loan'].var(),4)
    var_gross_profit_org = round(result_set['gross_profit'].var(),4)
    var_interest_income_org = round(result_set['interest_income'].var(),4)
    var_write_off_org = round(result_set['write_off'].var(),4)
    var_positive_rate_org = round(result_set['positive_rate'].var(),2)
    var_negative_rate_org = round(result_set['negative_rate'].var(),2)
    var_positive_recall_org = round(result_set['positive_recall'].var(),2)
    var_negative_recall_org = round(result_set['negative_recall'].var(),2)
    var_positive_precision_org = round(result_set['positive_precision'].var(),2)
    var_negative_precision_org = round(result_set['negative_precision'].var(),2)
    var_auc_org = round(result_set['auc'].var(),4)
    #WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no}" # AND data_gubun = {'B'}"
    result_proxy = engine.execute(query)
    result_proxy.close()
        
    query = f"INSERT INTO dc_batch_result_t4 (batch_id, model_no, model_full_name, model_sub_no,avg_true_positive, avg_false_positive, avg_true_negative, avg_false_negative, avg_real_total_loan,avg_real_gross_profit, avg_real_interest_income, avg_real_write_off, var_positive_rate, var_negative_rate,var_positive_recall, var_negative_recall, var_positive_precision, var_negative_precision, var_auc, avg_true_positive_org, avg_false_positive_org, avg_true_negative_org, avg_false_negative_org, avg_real_total_loan_org,avg_real_gross_profit_org, avg_real_interest_income_org, avg_real_write_off_org, var_positive_rate_org, var_negative_rate_org,var_positive_recall_org, var_negative_recall_org, var_positive_precision_org, var_negative_precision_org, var_auc_org,diversity ,base_models, creation_date) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no},{avg_true_positive}, {avg_false_positive}, {avg_true_negative}, {avg_false_negative}, {avg_real_total_loan},{avg_real_gross_profit}, {avg_real_interest_income}, {avg_real_write_off}, {var_positive_rate}, {var_negative_rate},{var_positive_recall}, {var_negative_recall}, {var_positive_precision}, {var_negative_precision}, {var_auc},{avg_true_positive_org}, {avg_false_positive_org}, {avg_true_negative_org}, {avg_false_negative_org}, {avg_real_total_loan_org},{avg_real_gross_profit_org}, {avg_real_interest_income_org}, {avg_real_write_off_org}, {var_positive_rate_org}, {var_negative_rate_org},{var_positive_recall_org}, {var_negative_recall_org}, {var_positive_precision_org}, {var_negative_precision_org}, {var_auc_org},{diversity},'{base_models}', '{creation_date}')"
    result_proxy = engine.execute(query)
    result_proxy.close()