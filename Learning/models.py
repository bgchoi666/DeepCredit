from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import paramiko
from ENV.env import *


# 모델 저장 서버 정보(임시로 고정)
SERVER_IP = "165.246.34.142"
PORT_NO = 22
USER_ID = "deep"
USER_PASS = "credit!0721"
FILE_PATH = "Learning/models"
PATH = "/opt/data/DeepCredit/models/" # 모델 저장 위치

class Model_collections:
   def __init__(self, model_name):
      self.model_name = model_name

   def model_fit(self, x_train, y_train, model_param):
      model_param = json.loads(model_param)
      if self.model_name == 'XGB':
         return  self.XGBoost(x_train, y_train, model_param)
      elif self.model_name == 'DNN':
         return self.DNN(x_train, y_train, model_param)
      elif self.model_name == 'RF':
         return self.RandomForest(x_train, y_train, model_param)

   @staticmethod
   def XGBoost(x_train, y_train, model_param):
      XGBoost = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
      model = XGBoost.fit(x_train, y_train)
      return model

   @staticmethod
   def DNN(x_train, y_train, model_param):
      model = tf.keras.models.Sequential([
         tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(1, activation='sigmoid')
      ])

      model.compile(
         optimizer='adam',
         loss='binary_crossentropy',
         metrics=['acc']
      )

      model.fit(x_train, y_train, epochs=int(model_param["epochs"]), validation_split=0.2, batch_size=32)
      return model

   @staticmethod
   def RandomForest(x_train, y_train, model_param):
      RF = RandomForestClassifier(n_estimators=100, random_state=0)
      model = RF.fit(x_train, y_train)
      return model

def get_model_id(model_id):
   patten = re.compile(r'(?P<model_type>[A-Z]+)-(?P<model_name>[A-Z]+)')
   matchobj = patten.search(model_id)
   model_name = matchobj.group('model_name')
   model_type = matchobj.group('model_type')
   return {"type":model_type, "name":model_name}

def save_model(model, batch_id, model_info, model_sub_no):
   FILE_NAME = f"{model_info.model_no}/{model_sub_no}"

   remote_home_path = PATH
   remote_file_path = 'T' + str(batch_id) + '/'

   model_type = get_model_id(model_info.model_id)["type"]
   if model_type == "ML":
      joblib.dump(model, PATH + remote_file_path + FILE_NAME + '.pkl')
   else:
      model.save(PATH + remote_file_path + FILE_NAME+ '.h5')

   transport = paramiko.Transport(SERVER_IP, PORT_NO)
   transport.connect(username=USER_ID, password=USER_PASS)

   sftp = paramiko.SFTPClient.from_transport(transport)
   forder_list = sftp.listdir(remote_home_path)

   if forder_list.count(remote_file_path) == 0:  # 리모트에 폴더 없으면 생성
      sftp.mkdir(remote_home_path + '/' + remote_file_path)

   print("SFTP PUT : ", remote_home_path + '/' + remote_file_path + '/' + FILE_NAME)

   try:
      sftp.put(FILE_PATH + '/' + FILE_NAME, remote_home_path + '/' + remote_file_path + '/' + FILE_NAME)
      sftp.close()
      transport.close()
      print("Save model complete!")

   except Exception as e:
      raise e

def load_model(train_batch_id, model_info, model_sub_no):
   FILE_NAME = f"{model_info.model_no}/{model_sub_no}"

   remote_home_path = PATH
   remote_file_path = 'T' + str(train_batch_id) + '/'

   transport = paramiko.Transport(SERVER_IP, PORT_NO)
   transport.connect(username=USER_ID, password=USER_PASS)

   print("SFTP GET : ", remote_home_path + '/' + remote_file_path + '/' + FILE_NAME)

   try:
      sftp = paramiko.SFTPClient.from_transport(transport)
      sftp.get(remote_home_path + '/' + remote_file_path + '/' + FILE_NAME, FILE_PATH + '/' + FILE_NAME)
      sftp.close()
      transport.close()

      model_type = get_model_id(model_info.model_id)["type"]
      if model_type == "ML":
         model = joblib.load(PATH + remote_file_path + FILE_NAME + '.pkl')
      else: # "DL"
         model = tf.keras.models.load_model(PATH + remote_file_path + FILE_NAME + '.h5')
      print("Load model complete!")
      return model

   except Exception as e:
      raise e

def get_model_list(model_group):
   # retrun: DataFrame
   stmt = text("SELECT * FROM dc_model_list WHERE model_group_id=:model_group")
   params = dict(model_group=model_group)
   model_list = dbhandler.retrive_stmt(stmt, params)
   return model_list.query("use_yn=='Y'")