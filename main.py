from interface import DeepCredit_main
from ENV.env import *

dbhandler = DBHandler()
stmt = text("SELECT * FROM dc_batch WHERE batch_id=2")
batch_info = dbhandler.retrive_stmt(stmt).loc[0]

if batch_info["mode"] in ("predict", "ensemble"):
    # Search train_batch param
    stmt = text(f"SELECT batch_param FROM dc_batch WHERE batch_id={batch_info['train_batch_id']}")
    batch_param = dbhandler.retrive_stmt(stmt).loc[0][0]
    batch_param = json.loads(batch_param)
else:
    batch_param = json.loads(batch_info["batch_param"])


DeepCredit_main(batch_info, batch_param)