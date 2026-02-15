import sys

import numpy as np
from ENV.DB_Handler import DBHandler
import sqlalchemy as sqc
import pandas as pd
from datetime import datetime
from pytz import timezone, utc
from sklearn.model_selection import train_test_split

kor = timezone("Asia/Seoul")

def init_db():
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    
    return engine

def select_candidate_id(candidate_id):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT model_no, model_sub_no FROM dc_candidate_model_list WHERE candidate_id = '{candidate_id}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    return result_set

def db_select_max_batch_id(table_name) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT MAX(batch_id) FROM {table_name}"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0]

def select_model_no(model_full_name) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT model_no FROM dc_model_list WHERE model_full_name = '{model_full_name}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0]

def select_model_name(model_no) : 
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    query = f"SELECT model_id, resampling_method FROM dc_model_list WHERE model_no = '{model_no}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchone()
    return result_set[0], result_set[1]

def db_insert_batch(batch_id, batch_desc, batch_memo, mode, dataset_group, dataset_version, model_group, state, user_id, display_yn, experiment_type):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    table = sqc.Table('dc_batch', metadata, autoload=True, autoload_with = engine)

    query = sqc.insert(table).values(batch_id = batch_id, batch_desc = batch_desc, batch_memo = batch_memo, mode = mode, dataset_group = dataset_group, dataset_version = dataset_version, model_group = model_group, state = state, user_id = user_id, display_yn = display_yn, experiment_type = experiment_type, creation_date = datetime.now().astimezone(kor))
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_test2(batch_id, model_no, model_full_name, model_sub_no, total_count, true_positive, false_positive, true_negative, false_negative, true_positive_test2, false_positive_test2, true_negative_test2, false_negative_test2,true_positive_org, false_positive_org, true_negative_org, false_negative_org,true_positive_test2_org, false_positive_test2_org, true_negative_test2_org, false_negative_test2_org, real_total_loan, real_gross_profit, real_interest_income, real_write_off, real_total_loan_test2, real_gross_profit_test2, real_interest_income_test2, real_write_off_test2, real_total_loan_org, real_gross_profit_org, real_interest_income_org, real_write_off_org, real_total_loan_test2_org, real_gross_profit_test2_org, real_interest_income_test2_org, real_write_off_test2_org) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    table = sqc.Table('dc_batch', metadata, autoload=True, autoload_with = engine)
    result_table = sqc.Table('dc_batch_result', metadata, autoload=True, autoload_with = engine)
    
    # query = db.select([table])
    # result_proxy = engine.execute(query)
    # result_set = result_proxy.fetchall()
    # print(result_set)
    
    query = sqc.insert(result_table).values(
        batch_id = batch_id,
        model_no = model_no,
        model_full_name = model_full_name,
        model_sub_no = model_sub_no,
        total_count = total_count,
        true_positive = true_positive,
        false_positive = false_positive,
        true_negative = true_negative,
        false_negative = false_negative,
        true_positive_test2 = true_positive_test2,
        false_positive_test2 = false_positive_test2,
        true_negative_test2 = true_negative_test2,
        false_negative_test2 = false_negative_test2,
        true_positive_org = true_positive_org,
        false_positive_org = false_positive_org,
        true_negative_org = true_negative_org,
        false_negative_org = false_negative_org,
        true_positive_test2_org = true_positive_test2_org,
        false_positive_test2_org = false_positive_test2_org,
        true_negative_test2_org = true_negative_test2_org,
        false_negative_test2_org = false_negative_test2_org,
        real_total_loan = real_total_loan,
        real_gross_profit = real_gross_profit,
        real_interest_income = real_interest_income,
        real_write_off = real_write_off,
        real_total_loan_test2 = real_total_loan_test2,
        real_gross_profit_test2 = real_gross_profit_test2,
        real_interest_income_test2 = real_interest_income_test2,
        real_write_off_test2 = real_write_off_test2,
        real_total_loan_org = real_total_loan_org,
        real_gross_profit_org = real_gross_profit_org,
        real_interest_income_org = real_interest_income_org,
        real_write_off_org = real_write_off_org,
        real_total_loan_test2_org = real_total_loan_test2_org,
        real_gross_profit_test2_org = real_gross_profit_test2_org,
        real_interest_income_test2_org = real_interest_income_test2_org,
        real_write_off_test2_org = real_write_off_test2_org)
        
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_test3(batch_id, model_no, model_full_name, model_sub_no, data_gubun) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()

    query = f"INSERT INTO dc_batch_result_t3 (batch_id, model_no, model_full_name, model_sub_no, data_gubun) VALUES ({batch_id}, {model_no}, '{model_full_name}', {model_sub_no}, '{data_gubun}')"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_test3_detail(batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit, gross_profit_rate, interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, total_count, true_positive, false_positive, true_negative, false_negative) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    query = f"INSERT INTO dc_batch_result_t3_detail (batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit, gross_profit_rate, interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, total_count, true_positive, false_positive, true_negative, false_negative) VALUES ({batch_id}, {model_no}, {model_sub_no}, {sub_no}, '{data_gubun}', {total_loan}, {gross_profit}, {gross_profit_rate}, {interest_income}, {write_off}, {positive_rate}, {negative_rate}, {positive_recall}, {negative_recall}, {positive_precision}, {negative_precision}, {auc}, {total_count}, {true_positive}, {false_positive}, {true_negative}, {false_negative})"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_test4(batch_id, model_no, model_full_name, model_sub_no, avg_true_positive, avg_false_positive, avg_true_negative, avg_false_negative, avg_true_positive_org, avg_false_positive_org,avg_true_negative_org, avg_false_negative_org, avg_real_total_loan, avg_real_gross_profit, avg_real_interest_income, avg_real_write_off, avg_real_total_loan_org, avg_real_gross_profit_org, avg_real_interest_income_org, avg_real_write_off_org, var_total_loan, var_gross_profit, var_gross_profit_rate, var_interest_income, var_write_off, var_positive_rate, var_negative_rate, var_positive_recall, var_negative_recall, var_positive_precision, var_negative_precision, var_auc, var_total_loan_org, var_gross_profit_org, var_gross_profit_rate_org, var_interest_income_org, var_write_off_org, var_positive_rate_org, var_negative_rate_org, var_positive_recall_org, var_negative_recall_org, var_positive_precision_org, var_negative_precision_org, var_auc_org, diversity, base_models, creation_date) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()

    query = f"INSERT INTO dc_batch_result_t3 (batch_id, model_no, model_full_name, model_sub_no, avg_true_positive, avg_false_positive, avg_true_negative, avg_false_negative, avg_true_positive_org, avg_false_positive_org,avg_true_negative_org, avg_false_negative_org, avg_real_total_loan, avg_real_gross_profit, avg_real_interest_income, avg_real_write_off, avg_real_total_loan_org, avg_real_gross_profit_org, avg_real_interest_income_org, avg_real_write_off_org, var_total_loan, var_gross_profit, var_gross_profit_rate, var_interest_income, var_write_off, var_positive_rate, var_negative_rate, var_positive_recall, var_negative_recall, var_positive_precision, var_negative_precision, var_auc, var_total_loan_org, var_gross_profit_org, var_gross_profit_rate_org, var_interest_income_org, var_write_off_org, var_positive_rate_org, var_negative_rate_org, var_positive_recall_org, var_negative_recall_org, var_positive_precision_org, var_negative_precision_org, var_auc_org, diversity, base_models, creation_date) VALUES ({batch_id}, {model_no}, {model_full_name}, {model_sub_no}, {avg_true_positive}, {avg_false_positive}, {avg_true_negative}, {avg_false_negative}, {avg_true_positive_org}, {avg_false_positive_org}, {avg_true_negative_org}, {avg_false_negative_org}, {avg_real_total_loan}, {avg_real_gross_profit}, {avg_real_interest_income}, {avg_real_write_off}, {avg_real_total_loan_org}, {avg_real_gross_profit_org}, {avg_real_interest_income_org}, {avg_real_write_off_org}, {var_total_loan}, {var_gross_profit}, {var_gross_profit_rate}, {var_interest_income}, {var_write_off}, {var_positive_rate}, {var_negative_rate}, {var_positive_recall}, {var_negative_recall}, {var_positive_precision}, {var_negative_precision}, {var_auc}, {var_total_loan_org}, {var_gross_profit_org}, {var_gross_profit_rate_org}, {var_interest_income_org}, {var_write_off_org}, {var_positive_rate_org}, {var_negative_rate_org}, {var_positive_recall_org}, {var_negative_recall_org}, {var_positive_precision_org}, {var_negative_precision_org}, {var_auc_org}, {diversity}, {base_models, creation_date})"
    result_proxy = engine.execute(query)
    result_proxy.close()

def db_insert_for_test4_detail(batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit, interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, total_count, true_positive, false_positive, true_negative, false_negative) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    
    query = f"INSERT INTO dc_batch_result_t4_detail (batch_id, model_no, model_sub_no, sub_no, data_gubun, total_loan, gross_profit,  interest_income, write_off, positive_rate, negative_rate, positive_recall, negative_recall, positive_precision, negative_precision, auc, total_count, true_positive, false_positive, true_negative, false_negative) VALUES ({batch_id}, {model_no}, {model_sub_no}, {sub_no}, '{data_gubun}', {total_loan}, {gross_profit}, {interest_income}, {write_off}, {positive_rate}, {negative_rate}, {positive_recall}, {negative_recall}, {positive_precision}, {negative_precision}, {auc}, {total_count}, {true_positive}, {false_positive}, {true_negative}, {false_negative})"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def update_for_t3_detail(batch_id, model_no, model_sub_no, data_gubun):
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
    query = f"SELECT * FROM dc_batch_result_t3_detail WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{data_gubun}'"
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = pd.DataFrame(result_set)
    query = f"UPDATE dc_batch_result_t3 SET     avg_total_loan = {result_set['total_loan'].mean()},     avg_gross_profit = {result_set['gross_profit'].mean()},    avg_gross_profit_rate = {round(result_set['gross_profit_rate'].mean(),2)},    avg_interest_income = {result_set['interest_income'].mean()},    avg_write_off = {result_set['write_off'].mean()},    avg_positive_rate = {round(result_set['positive_rate'].mean(),2)},    avg_negative_rate = {round(result_set['negative_rate'].mean(),2)},    avg_positive_recall = {round(result_set['positive_recall'].mean(),2)},    avg_negative_recall = {round(result_set['negative_recall'].mean(),2)},    avg_positive_precision = {round(result_set['positive_precision'].mean(),2)},    avg_negative_precision = {round(result_set['negative_precision'].mean(),2)},    avg_auc = {round(result_set['auc'].mean(),2)},    var_total_loan = {round(result_set['total_loan'].std(),2)},    var_gross_profit = {round(result_set['gross_profit'].std(),2)},    var_gross_profit_rate = {round(result_set['gross_profit_rate'].std(),2)},    var_interest_income = {round(result_set['interest_income'].std(),2)},    var_write_off = {round(result_set['write_off'].std(),2)},    var_positive_rate = {round(result_set['positive_rate'].std(),2)},    var_negative_rate = {round(result_set['negative_rate'].std(),2)},    var_positive_recall = {round(result_set['positive_recall'].std(),2)},    var_negative_recall = {round(result_set['negative_recall'].std(),2)},    var_positive_precision = {round(result_set['positive_precision'].std(),2)},    var_negative_precision = {round(result_set['negative_precision'].std(),2)},    var_auc = {round(result_set['auc'].std(),2)},    true_positive = {result_set['true_positive'].mean()},    false_positive = {result_set['false_positive'].mean()},    true_negative = {result_set['true_negative'].mean()},    false_negative = {result_set['false_negative'].mean()}    WHERE batch_id = {batch_id} AND model_no = {model_no} AND model_sub_no = {model_sub_no} AND data_gubun = '{data_gubun}'"
    
    result_proxy = engine.execute(query)
    result_proxy.close()

def select_top100_db_by_model(batch_id, model_no, limit):
    query = f"SELECT model_sub_no FROM dc_batch_result WHERE batch_id={batch_id} AND model_no={model_no} ORDER BY real_gross_profit DESC LIMIT {limit}"
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    # metadata = db.MetaData()
   
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = np.array(result_set)
    result_set.resize(limit)
    return result_set

def select_model_no_distinct(batch_id) :
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    # metadata = db.MetaData()
    
    query = f"SELECT DISTINCT model_no, model_full_name FROM dc_batch_result WHERE batch_id={batch_id}"
    
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = np.array(result_set)
    return result_set

def merge_data_by_randomstate(data, label, random_state_list, how_to_merge):
    
    if how_to_merge == 'union':
        index_name = data.index.name
        column_name = data.reset_index().columns
        result = pd.DataFrame(columns = column_name)
        result.set_index(index_name, inplace = True)
    elif how_to_merge == 'inter':
        result = data.copy()
    
    for no in random_state_list:
        rx_train, rx_test, ry_train, ry_test = train_test_split(data, label, test_size = 0.5, random_state = no)
        result = pd.concat((result, rx_train))
        if how_to_merge == 'union':
            result = result.drop_duplicates()
        elif how_to_merge == 'inter':
            result = result[result.duplicated()==True]
    return result

def select_for(query):
    query = query
    dbhandler = DBHandler()
    engine = dbhandler.get_connection()
    connection = engine.connect()
    metadata = sqc.MetaData()
   
    result_proxy = engine.execute(query)
    result_set = result_proxy.fetchall()
    result_set = np.array(result_set)
    return result_set

# if __name__ == "__main__":
#     print(select_for_ensemble())