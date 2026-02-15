from ENV.env import *


def save_result(cm, batch_info, model_info):
    if batch_info["mode"] == "train":
        total = None
        TN, FP, FN, TP = None, None, None, None
    else: # preidct
        total = cm.sum()
        TN = cm[0]
        FP = cm[1]
        FN = cm[2]
        TP = cm[3]

    result = dict(batch_id=int(batch_info['batch_id']),
                  model_no=model_info.model_no,
                  model_full_name=model_info.model_full_name,
                  model_sub_no=0,
                  model_param=model_info.model_param,
                  resampling_param=model_info.resampling_param,
                  total_count=total,
                  true_positive =TP,
                  false_positive=FP,
                  true_negative =TN,
                  false_negative=FN,
                  real_total_loan=None,
                  real_gross_profit=None,
                  real_interest_income=None,
                  real_write_off=None)

    stmt = text("INSERT INTO dc_batch_result "
                "(batch_id, model_no, model_full_name, model_sub_no, model_param, resampling_param, "
                "total_count, true_positive, false_positive, true_negative, false_negative, "
                "real_total_loan, real_gross_profit, real_interest_income, real_write_off) "
                "VALUES (:batch_id, :model_no, :model_full_name, :model_sub_no, "
                ":model_param, :resampling_param, "
                ":total_count, :true_positive, :false_positive, "
                ":true_negative, :false_negative, :real_total_loan, :real_gross_profit, "
                ":real_interest_income, :real_write_off)")
    param = result
    dbhandler.execute_stmt(stmt, param)