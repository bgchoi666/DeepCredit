#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
import import_ipynb
from module.data_process import *
from module.calculate import *
from module.Resampling_Imbalanced_data import *
#from db.db_process import *

import itertools

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from keras import datasets
from pytorch_tabnet.tab_model import TabNetClassifier


# In[2]:


def tabnet(x_train, y_train):
    
    model = TabNetClassifier(verbose =0)
    
    model.fit(
    X_train=x_train, y_train=y_train,
    #eval_set=[(x_train, y_train), (x_test, y_test)]
    )
    
    return model


# In[ ]:





# In[3]:


def learning_test_4(data, label, no, sampling, model_name, batch_id, raw_data, iteration): #xgb_params, rf_params
    
    x_train, x_test, y_train, y_test = split_same_ratio(data, label, test_size = 0.5, random_state=no)
    print(1)
       
    if sampling=='ADASYN':   
        sample_x, sample_y = Adasyn(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMOTE':
        sample_x, sample_y = Smote(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMTOMEK':
        sample_x, sample_y = SmoteTmoek(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'RUS':
        sample_x, sample_y = RUS(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ENN':
        sample_x, sample_y = ENN_(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'NCR':
        sample_x, sample_y = NCR(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+NCR':
        sample_x, sample_y = ADASYN_NCR(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+RUS':
        sample_x, sample_y = ADASYN_RUS(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    else :
        print("잘못된 sampling parameter 입력")
    print(2)
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        print(4)
    else:
        print("잘못된 모델입력")
    #db_insert_for_test4(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'T')
    #db_insert_for_test4(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'B')
    
    print(3)
    
    for i in range(0,iteration):
        rx_test, trash0, ry_test, trash1 = split_same_ratio(x_test, y_test, test_size = 0.4, random_state=i)
        
        if model_name == 'DL-DNN':
            pred = model.predict(rx_test)
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['result'])
            df_pred['pred'] = 0
            df_pred.loc[df_pred['result'] > 0.5, 'pred'] = 1
            df_pred.drop(['result'], axis=1, inplace=True)
            pred = pd.DataFrame(df_pred, columns = ['pred'])
        elif model_name == 'TABNET':
            pred = model.predict(rx_test.values[:])
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['pred'])
        else :
            pred = model.predict(rx_test)
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['pred'])
        
        result = calculate(raw_data, df_pred)
        
        total_loan = result['대출실행금액'].sum()
        before_inter = result['[기존]이자수익'].sum()
        after_inter = result['[예상]이자수익'].sum()

        after_total_loan = result['[예상]대출금액'].sum()
        profit_loan = after_inter - before_inter
        before_loss = result['[기존]원금손실'].sum()
        after_loss = result['[예상]원금손실'].sum()

        before_profit = before_inter - before_loss
        profit_loss = before_loss - after_loss
        profit = profit_loan + profit_loss
    
        cm = confusion_matrix(ry_test,pred)

        cmc = list(itertools.chain(*cm))
        
        total_count = cm.sum()
        TP, FP, TN, FN = cmc[3], cmc[1], cmc[0], cmc[2]      

    return model
        


# In[4]:


#model_no 50count -> 1 up loop
#sub_no 50count -> 0 


# In[5]:


def learning_test_4_2(train_x, test_x, train_y, test_y, sampling, model_name,no , raw_data):
#data, label, no, sampling, model_name, batch_id, raw_data,
#iteration): #xgb_params, rf_params
#def learning_test_3(train_x, test_x, train_y, test_y, 
#no, sampling, model_name, batch_id, raw_data, xgb_params, rf_params, iteration)    
    #x_train, x_test, y_train, y_test = split_same_ratio(data, label, test_size = 0.5, random_state=no)
    x_train, trash0, y_train, trash1 = split_same_ratio(train_x, train_y, test_size = 0.4, random_state=no)
    print(1)
       
    if sampling=='ADASYN':   
        sample_x, sample_y = Adasyn(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMOTE':
        sample_x, sample_y = Smote(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMTOMEK':
        sample_x, sample_y = SmoteTmoek(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'RUS':
        sample_x, sample_y = RUS(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ENN':
        sample_x, sample_y = ENN_(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'NCR':
        sample_x, sample_y = NCR(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+NCR':
        sample_x, sample_y = ADASYN_NCR(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+RUS':
        sample_x, sample_y = ADASYN_RUS(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    else :
        print("잘못된 sampling parameter 입력")
    print(2)
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        print(4)
    else:
        print("잘못된 모델입력")
    #db_insert_for_test4(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'T')
    #db_insert_for_test4(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'B')
    
    print(3)
            
    return model
        


# In[ ]:



