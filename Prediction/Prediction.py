from sklearn.metrics import accuracy_score, confusion_matrix
from ENV.env import *

def predict(model, x_test, y_test):
    y_pred = model.predict(x_test)

    # mdoel_id: DNN
    if y_pred[0] != 0 and y_pred[0] != 1:
        y_pred = np.where(y_pred.reshape(-1)>0.5, 1, 0)

    cm = confusion_matrix(y_test, y_pred.reshape(-1)).ravel()

    # accuracy_score(y_test, y_pred)
    return cm

def get_trained_model_list(train_batch_id):
    stmt = text(f"SELECT * FROM dc_batch_result WHERE batch_id={train_batch_id}")
    return dbhandler.retrive_stmt(stmt)