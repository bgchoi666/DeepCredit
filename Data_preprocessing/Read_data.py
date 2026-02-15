from ENV.env import *


def read_data(batch_info):
    stmt = text("SELECT a.data_table, b.customer_column_name, b.label_column_name FROM dc_dataset_version a, dc_dataset_group b "
                "WHERE a.dataset_group = b.dataset_group AND a.dataset_group = :dataset_group AND a.`version` = :version")
    param = dict(dataset_group=batch_info['dataset_group'],
                 version=batch_info['dataset_version'])
    result = dbhandler.retrive_stmt(stmt, param)

    if result.empty:
        raise ValueError("Error: Unknown data_table.")

    try:
        stmt = text(f"SELECT * FROM {result['data_table'][0]}")
        # param = dict(data_table=result['data_table'][0])
        data = dbhandler.retrive_stmt(stmt, param).set_index(result["customer_column_name"][0])

    except Exception as e:
        raise e

    if data.empty:
        raise ValueError("Error: Empty data_table.")

    return data