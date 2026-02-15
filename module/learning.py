import sys
sys.path.append('..')
# import import_ipynb
from module.data_process import *
from module.calculate import *
from module.Resampling_Imbalanced_data import *
# from db.db_test import *

import itertools

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import TensorDataset, Dataset, DataLoader

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from keras import datasets
from pytorch_tabnet.tab_model import TabNetClassifier

import joblib
import pickle

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()

        self.layer_1 = nn.Linear(x_train.shape[1], 64)
        self.layer_2 = nn.Linear(64,64)
        self.layer_out = nn.Linear(64,1)

        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


# In[ ]:


def DNN(x_train, y_train, epochs=5, batch_size=64, lr = 0.001):
    
    x_train, y_train= x_train.values[:], y_train.values[:]
    x_train, y_train= torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float()
    
    device = torch.device("cuda:1")
   
    class traindata(Dataset):
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data
            
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return len(self.x_data)
    
    #x_train = torch.FloatTensor(x_train)
    train_data = traindata(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    
#     class testdata(Dataset):
#         def __init__(self, x_data):
#             self.x_data = x_data
            
#         def __getitem__(self, index):
#             return self.x_data[index]

#         def __len__(self):
#             return len(self.x_data)

    #test_data = testdata(torch.FloatTensor(x_test))
    
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last=True)
    #test_loader = DataLoader(dataset = test_data, batch_size=1)
    global BinaryClassification
    class BinaryClassification(nn.Module):
        def __init__(self):
            super(BinaryClassification, self).__init__()

            self.layer_1 = nn.Linear(x_train.shape[1], 64)
            self.layer_2 = nn.Linear(64,64)
            self.layer_out = nn.Linear(64,1)

            self.relu = nn.ReLU()
            self.dropout= nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.batchnorm2 = nn.BatchNorm1d(64)

        def forward(self, inputs):
            x = self.layer_1(inputs)
            x = self.relu(x)
            x = self.batchnorm1(x)
            x = self.layer_2(x)
            x = self.relu(x)
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)

            return x
   
    model = BinaryClassification()
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    def binary_acc (y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        #print(y_pred_tag)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc*100)
        
        return acc
    
    model.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        #print(f'Epoch {e+0:03}: | Loss : {epoch_loss/len(train_loader):.5f} | Acc : {epoch_acc/len(train_loader):.3f}')
    return model


# In[ ]:


def DNN_predict(model, x_test):
    
    x_test= x_test.values[:]
    x_test= torch.from_numpy(x_test).float()
    
    device = torch.device("cuda:1")
    
    class testdata(Dataset):
        def __init__(self, x_data):
            self.x_data = x_data

        def __getitem__(self, index):
            return self.x_data[index]

        def __len__(self):
            return len(self.x_data)
    
    test_data = testdata(torch.FloatTensor(x_test))
    
    test_loader = DataLoader(dataset = test_data, batch_size=1)
    y_pred_list = []

    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            if y_test_pred >0.5:
                y_test_pred =1
            else:
                y_test_pred = 0
                
            #print(y_test_pred)
            #y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_test_pred)

    return y_pred_list


# In[ ]:


def RFC(x_train, y_train, params):

    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    #params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['min_samples_split'] = int(params['min_samples_split'])

    model = RandomForestClassifier(**params)
        # bootstrap=True, 
        # class_weight=None, 
        # criterion='gini',
        # max_depth=None, 
        # max_features='auto', 
        # max_leaf_nodes=None,
        # min_impurity_decrease=0.0, 
        # min_impurity_split=None,
        # min_samples_leaf=1, 
        # min_samples_split=2,
        # min_weight_fraction_leaf=0.0, 
        # n_estimators=1000, #결정 트리의 갯수 지정 default=10
        # n_jobs=None,
        # oob_score=False, 
        # random_state=None, 
        # verbose=0,
        # warm_start=False)
    
    model.fit(x_train, y_train)
    #print('Model accuracy score with 1000 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, pred)))
    
    return model


# In[ ]:


def XGB_params(x_train, y_train, params):
    
    params['colsample_bylevel'] = params['colsample_bylevel']
    params['colsample_bytree'] = params['colsample_bytree']
    params['gamma'] = params['gamma']
    params['learning_rate'] = params['learning_rate']
    params['max_delta_step'] = params['max_delta_step']
    params['learning_rate'] = params['learning_rate']
    params['subsample'] = params['subsample']
    params['colsample_bynode'] = params['colsample_bynode']
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_child_weight'] = params['min_child_weight']
    params['use_label_encoder'] = False
    params['eval_metric'] = 'logloss'
   
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train)
    
    return model


# In[ ]:


def tabnet(x_train, y_train):
    
    model = TabNetClassifier(verbose =0)
    
    model.fit(
    X_train=x_train, y_train=y_train,
    #eval_set=[(x_train, y_train), (x_test, y_test)]
    )
    
    return model


# In[ ]:


# def learning_test_1(x_train, x_test, y_train, y_test, no, mode, model_name, batch_id):
    
#     rx_train, rx_test, ry_train, ry_test = split_same_ratio(x_train, y_train, test_size = 0.5, random_state=no)
    
    
#     if mode=='adasyn':   
#         sample_x, sample_y = Adasyn(rx_train, ry_train)
#         model_no =1
#     elif mode == 'smote':
#         sample_x, sample_y = Smote(rx_train, ry_train)
#         model_no =2
#     elif mode == 'SMTOMEK':
#         sample_x, sample_y = SmoteTmoek(rx_train, ry_train)
#         model_no =3
#     elif mode == 'rus':
#         sample_x, sample_y = RUS(rx_train, ry_train)
#         model_no =4
#     elif mode == 'enn':
#         sample_x, sample_y = ENN_(rx_train, ry_train)
#         model_no =5
#     elif mode == 'ncr':
#         sample_x, sample_y = NCR(rx_train, ry_train)
#         model_no =6
#     elif mode == 'ADASYN+NCR':
#         sample_x, sample_y = ADASYN_NCR(rx_train, ry_train)
#         model_no =7
#     elif mode == 'ADASYN+RUS':
#         sample_x, sample_y = ADASYN_RUS(rx_train, ry_train)
#         model_no =8
#     else :
#         print("잘못된 sampling parameter 입력")
    
#     if model_name == 'ML-XGB':
#         model = XGB_params(sample_x, sample_y, x_test, y_test, xgb_params)
#     elif model_name == 'ML-RF':
#         model = RFC(sample_x, sample_y, x_test, y_test, rfc_params)
#     elif model_name == 'DL-DNN':
#         model = DNN(sample_x, sample_y)
#     else:
#         print("잘못된 모델입력")
    
    
#     pred = model.predict(x_test)
#     if model_name == 'DL-DNN':
#         df_pred = pd.DataFrame(pred, index = y_test.index, columns = ['result'])
#         df_pred['pred'] = 0
#         df_pred.loc[df_pred['result'] > 0.5, 'pred'] = 1
#         df_pred.drop(['result'], axis=1, inplace=True)
#         pred = pd.DataFrame(df_pred, columns = ['pred'])
#     else :
#         df_pred = pd.DataFrame(pred, index = y_test.index, columns = ['pred'])
#     y = pd.concat([y_test,df_pred],axis=1)
#     money = cal_money(raw_data, df_pred)
#     total_loan = money['대출실행금액'].sum()
#     before_inter = money['[기존]이자수익'].sum()
#     after_inter = money['[예상]이자수익'].sum()
    
#     after_total_loan = money['[예상]대출금액'].sum()
#     profit_loan = after_inter - before_inter
#     before_loss = money['[기존]원금손실'].sum()
#     after_loss = money['[예상]원금손실'].sum()
    
#     before_profit = before_inter - before_loss
#     profit_loss = before_loss - after_loss
#     profit = profit_loan + profit_loss

#     pr = list([profit_loan, profit_loss, profit])
    
#     cm = confusion_matrix(y_test,pred)
#     cmc = list(itertools.chain(*cm)) + pr
#     cmcc = pd.DataFrame([cmc], index=[no], columns = ['TN','FN','FP','TP', '이자수익 감소', '원금손실 감소', '총이익'])
#     db_insert(batch_id,model_no,f'{model_name}-{mode}',no,cm.sum(),cmc[3],cmc[1],cmc[0],cmc[2],after_total_loan,after_inter - after_loss,after_inter,after_loss)
    


# In[ ]:


def learning_test_2(x_train, x_test1, x_test2, y_train, y_test1, y_test2, no, sampling, model_name, batch_id, raw_data, xgb_params, rf_params):
    
    rx_train, rx_test, ry_train, ry_test = split_same_ratio(x_train, y_train, test_size = 0.5, random_state=no)
    
    
    if sampling=='ADASYN':   
        sample_x, sample_y = Adasyn(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMOTE':
        sample_x, sample_y = Smote(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMTOMEK':
        sample_x, sample_y = SmoteTmoek(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'RUS':
        sample_x, sample_y = RUS(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ENN':
        sample_x, sample_y = ENN_(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'NCR':
        sample_x, sample_y = NCR(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+NCR':
        sample_x, sample_y = ADASYN_NCR(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+RUS':
        sample_x, sample_y = ADASYN_RUS(rx_train, ry_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    else :
        print("잘못된 sampling parameter 입력")
    
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
    else:
        print("잘못된 모델입력")
    
    pred1 = model.predict(x_test1)
    pred2 = model.predict(x_test2)
    
    if model_name == 'DL-DNN':
        df_pred1 = pd.DataFrame(pred1, index = y_test1.index, columns = ['result'])
        df_pred1['pred'] = 0
        df_pred1.loc[df_pred1['result'] > 0.5, 'pred'] = 1
        df_pred1.drop(['result'], axis=1, inplace=True)
        pred1 = pd.DataFrame(df_pred1, columns = ['pred'])

        
        df_pred2 = pd.DataFrame(pred2, index = y_test2.index, columns = ['result'])
        df_pred2['pred'] = 0
        df_pred2.loc[df_pred2['result'] > 0.5, 'pred'] = 1
        df_pred2.drop(['result'], axis=1, inplace=True)
        pred2 = pd.DataFrame(df_pred2, columns = ['pred'])

        
    else :
        df_pred1 = pd.DataFrame(pred1, index = y_test1.index, columns = ['pred'])
        df_pred2 = pd.DataFrame(pred2, index = y_test2.index, columns = ['pred'])
        
    test_1_result = calculate(raw_data, df_pred1)
    test_2_result = calculate(raw_data, df_pred2)
    
    test1_total_loan = test_1_result['대출실행금액'].sum()
    test1_before_inter = test_1_result['[기존]이자수익'].sum()
    test1_after_inter = test_1_result['[예상]이자수익'].sum()
    
    test1_after_total_loan = test_1_result['[예상]대출금액'].sum()
    test1_profit_loan = test1_after_inter - test1_before_inter
    test1_before_loss = test_1_result['[기존]원금손실'].sum()
    test1_after_loss = test_1_result['[예상]원금손실'].sum()
    
    test1_before_profit = test1_before_inter - test1_before_loss
    test1_profit_loss = test1_before_loss - test1_after_loss
    test1_profit = test1_profit_loan + test1_profit_loss
    
    test2_total_loan = test_2_result['대출실행금액'].sum()
    test2_before_inter = test_2_result['[기존]이자수익'].sum()
    test2_after_inter = test_2_result['[예상]이자수익'].sum()
    
    test2_after_total_loan = test_2_result['[예상]대출금액'].sum()
    test2_profit_loan = test2_after_inter - test2_before_inter
    test2_before_loss = test_2_result['[기존]원금손실'].sum()
    test2_after_loss = test_2_result['[예상]원금손실'].sum()
    
    test2_before_profit = test2_before_inter - test2_before_loss
    test2_profit_loss = test2_before_loss - test2_after_loss
    test2_profit = test2_profit_loan + test2_profit_loss
    
    test1_cm = confusion_matrix(y_test1,pred1)
    test2_cm = confusion_matrix(y_test2, pred2)
    
    test1_cmc = list(itertools.chain(*test1_cm))
    test2_cmc = list(itertools.chain(*test2_cm))
    
    db_insert_for_test2(
        batch_id=batch_id,
        model_no=model_no,
        model_full_name=model_full_name,
        model_sub_no=no,
        total_count=test1_cm.sum(),
        true_positive=test1_cmc[3],
        false_positive=test1_cmc[1],
        true_negative=test1_cmc[0],
        false_negative=test1_cmc[2],
        true_positive_test2=test2_cmc[3],
        false_positive_test2=test2_cmc[1],
        true_negative_test2=test2_cmc[0],
        false_negative_test2=test2_cmc[2],
        true_positive_org=test_1_result['y1'].sum(),
        false_positive_org=0,
        true_negative_org=test_1_result['y2'].sum(),
        false_negative_org=0,
        true_positive_test2_org=test_2_result['y1'].sum(),
        false_positive_test2_org=0,
        true_negative_test2_org=test_2_result['y2'].sum(),
        false_negative_test2_org=0,
        real_total_loan=test1_after_total_loan,
        real_gross_profit=test1_after_inter - test1_after_loss,
        real_interest_income=test1_after_inter,
        real_write_off=test1_after_loss,
        real_total_loan_test2=test2_after_total_loan,
        real_gross_profit_test2=test2_after_inter - test2_after_loss,
        real_interest_income_test2=test2_after_inter,
        real_write_off_test2=test2_after_loss,
        real_total_loan_org=test1_total_loan,
        real_gross_profit_org=test1_before_inter - test1_before_loss,
        real_interest_income_org=test1_before_inter,
        real_write_off_org=test1_before_loss,
        real_total_loan_test2_org=test2_total_loan,
        real_gross_profit_test2_org=test2_before_inter - test2_before_loss,
        real_interest_income_test2_org=test2_before_inter,
        real_write_off_test2_org=test2_before_loss)    


# In[ ]:


not_nan = 0.00000000000000000001


# In[ ]:


def learning_test_3(train_x, test_x, train_y, test_y, no, sampling, model_name, batch_id, raw_data, xgb_params, rf_params, iteration):
    
    x_train, trash0, y_train, trash1 = split_same_ratio(train_x, train_y, test_size = 0.4, random_state=no)
      
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
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
        torch.save(model, f'./model_save/{model_name}/{sampling}/{no}.pt')
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        model.save_model(f'./model_save/{model_name}/{sampling}/{no}')
        #모델 로드 방법
        #model = TabNetClassifier()
        #model.load_model(f'./model_save/{model_name}/{sampling}/{no}.zip')

    else:
        print("잘못된 모델입력")
    db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'T')
    db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'B')

    
    for i in range(0,iteration):
        if i==0:
            trash0, rx_test, trash1, ry_test = split_same_ratio(test_x, test_y, test_size = 0.5, random_state=1)
            
        rx_test, trash0, ry_test, trash1 = split_same_ratio(test_x, test_y, test_size = 0.5, random_state=i)
        
        if model_name == 'DL-DNN':
            pred = DNN_predict(model, rx_test)
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['pred'])
#             df_pred['pred'] = 0
#             df_pred.loc[df_pred['result'] > 0.5, 'pred'] = 1
#             df_pred.drop(['result'], axis=1, inplace=True)
#             pred = pd.DataFrame(df_pred, columns = ['pred'])
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

        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'T',
            total_loan = after_total_loan, 
            gross_profit = after_inter - after_loss, 
            gross_profit_rate = round((after_inter - after_loss)/(after_total_loan+not_nan)*100,2), 
            interest_income = after_inter, 
            write_off = after_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)
       
        TP, FP, TN, FN = result['y1'].sum(), 0, result['y2'].sum(), 0
        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'B',
            total_loan = total_loan, 
            gross_profit = before_inter - before_loss, 
            gross_profit_rate = round((before_inter - before_loss)/total_loan*100,2), 
            interest_income = before_inter, 
            write_off = before_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)

    update_for_t3_detail(batch_id, model_no, no, 'T')
    update_for_t3_detail(batch_id, model_no, no, 'B')


# In[ ]:


def learning_test_rebuilding(train_x, test_x, train_y, test_y, no, sampling, model_name, batch_id, raw_data, xgb_params, rf_params, iteration):
    
    x_train, trash0, y_train, trash1 = split_same_ratio(train_x, train_y, test_size = 0.4, random_state=0)
      
    if sampling=='ADASYN':   
        sample_x, sample_y = Adasyn_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMOTE':
        sample_x, sample_y = Smote_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'SMTOMEK':
        sample_x, sample_y = SmoteTmoek_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'RUS':
        sample_x, sample_y = RUS_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ENN':
        sample_x, sample_y = ENN_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'NCR':
        sample_x, sample_y = NCR_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+NCR':
        sample_x, sample_y = ADASYN_NCR_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    elif sampling == 'ADASYN+RUS':
        sample_x, sample_y = ADASYN_RUS_rebuild(x_train, y_train)
        model_full_name = f"{model_name}-{sampling}"
        model_no = select_model_no(model_full_name)
    else :
        print("잘못된 sampling parameter 입력")
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
        #joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
        #joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
        #torch.save(model, f'./model_save/{model_name}/{sampling}/{no}.pt')
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        #model.save_model(f'./model_save/{model_name}/{sampling}/{no}')
        #모델 로드 방법
        #model = TabNetClassifier()
        #model.load_model(f'./model_save/{model_name}/{sampling}/{no}.zip')

    else:
        print("잘못된 모델입력")
    db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'T')
    #db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'B')

    
    for i in range(0,iteration):
          
        rx_test, trash0, ry_test, trash1 = split_same_ratio(test_x, test_y, test_size = 0.5, random_state=0)
        
        if model_name == 'DL-DNN':
            pred = DNN_predict(model, rx_test)
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['pred'])
#             df_pred['pred'] = 0
#             df_pred.loc[df_pred['result'] > 0.5, 'pred'] = 1
#             df_pred.drop(['result'], axis=1, inplace=True)
#             pred = pd.DataFrame(df_pred, columns = ['pred'])
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

        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'T',
            total_loan = after_total_loan, 
            gross_profit = after_inter - after_loss, 
            gross_profit_rate = round((after_inter - after_loss)/(after_total_loan+not_nan)*100,2), 
            interest_income = after_inter, 
            write_off = after_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)
       
        TP, FP, TN, FN = result['y1'].sum(), 0, result['y2'].sum(), 0
        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'B',
            total_loan = total_loan, 
            gross_profit = before_inter - before_loss, 
            gross_profit_rate = round((before_inter - before_loss)/total_loan*100,2), 
            interest_income = before_inter, 
            write_off = before_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)

    update_for_t3_detail(batch_id, model_no, no, 'T')
    #update_for_t3_detail(batch_id, model_no, no, 'B')


# In[ ]:


def learning_from_model(train_x, test_x, train_y, test_y, no, sampling, model_name, batch_id, raw_data, iteration):
    
    x_train, trash0, y_train, trash1 = split_same_ratio(train_x, train_y, test_size = 0.4, random_state=no)
      
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
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'DL-DNN':
        model = torch.load(f'./model_save/{model_name}/{sampling}/{no}.pt', map_location= torch.device('cuda:1'))
        
        #model = DNN(sample_x, sample_y)
        #torch.save(model, f'./model_save/{model_name}/{sampling}/{no}.pt')
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        model.save_model(f'./model_save/{model_name}/{sampling}/{no}')
        #모델 로드 방법
        #model = TabNetClassifier()
        #model.load_model(f'./model_save/{model_name}/{sampling}/{no}.zip')

    else:
        print("잘못된 모델입력")
    db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'T')
    db_insert_for_test3(batch_id=batch_id, model_no=model_no, model_full_name = model_full_name, model_sub_no=no, data_gubun = 'B')

    
    for i in range(0,iteration):
        if i==0:
            trash0, rx_test, trash1, ry_test = split_same_ratio(test_x, test_y, test_size = 0.5, random_state=1)
            
        rx_test, trash0, ry_test, trash1 = split_same_ratio(test_x, test_y, test_size = 0.5, random_state=i)
        
        if model_name == 'DL-DNN':
            pred = DNN_predict(model, rx_test)
            df_pred = pd.DataFrame(pred, index = ry_test.index, columns = ['pred'])
#             df_pred['pred'] = 0
#             df_pred.loc[df_pred['result'] > 0.5, 'pred'] = 1
#             df_pred.drop(['result'], axis=1, inplace=True)
#             pred = pd.DataFrame(df_pred, columns = ['pred'])
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

        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'T',
            total_loan = after_total_loan, 
            gross_profit = after_inter - after_loss, 
            gross_profit_rate = round((after_inter - after_loss)/(after_total_loan+not_nan)*100,2), 
            interest_income = after_inter, 
            write_off = after_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)
       
        TP, FP, TN, FN = result['y1'].sum(), 0, result['y2'].sum(), 0
        db_insert_for_test3_detail(
            batch_id = batch_id, 
            model_no = model_no, 
            model_sub_no = no, 
            sub_no = i, 
            data_gubun = 'B',
            total_loan = total_loan, 
            gross_profit = before_inter - before_loss, 
            gross_profit_rate = round((before_inter - before_loss)/total_loan*100,2), 
            interest_income = before_inter, 
            write_off = before_loss, 
            positive_rate= round(FN/(TN+FN+not_nan)*100,2), 
            negative_rate= round((TN+FN)/total_count*100,2), 
            positive_recall= round(TP/(TP+FN+not_nan)*100,2), 
            negative_recall= round(TN/(TN+FP+not_nan)*100,2), 
            positive_precision=round(TP/(TP+FP+not_nan)*100,2), 
            negative_precision=round(TN/(TN+FN+not_nan)*100,2), 
            auc=round((TN/(TN+FP+not_nan) + TP/(TP+FN+not_nan))*0.5*100,2), 
            total_count=total_count,
            true_positive=TP,
            false_positive=FP,
            true_negative=TN,
            false_negative=FN)

    update_for_t3_detail(batch_id, model_no, no, 'T')
    update_for_t3_detail(batch_id, model_no, no, 'B')


# In[ ]:


def learning_test(train_x, test_x, train_y, test_y, no, sampling, model_name):
    
    x_train, trash0, y_train, trash1 = split_same_ratio(train_x, train_y, test_size = 0.4, random_state=no)
    xgb_params = {'colsample_bylevel': 0.7291996510410845, 'colsample_bynode': 0.9217797235560365, 'colsample_bytree': 0.28228138569574335, 'gamma': 2.4088338550775488, 'learning_rate': 0.7924886027821245, 'max_delta_step': 0.24518006861811276, 'max_depth': 12.02746581851463, 'min_child_weight': 2.1503470455682905, 'n_estimators': 219.75992411764736, 'subsample': 0.7352863422651585}
    rf_params = {'max_depth': 25.0, 'min_samples_leaf': 0.3, 'min_samples_split': 13.021112920432707, 'n_estimators': 71.97795349233064}
      
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
    if model_name == 'ML-XGB':
        model = XGB_params(sample_x, sample_y, xgb_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'ML-RF':
        model = RFC(sample_x, sample_y, rf_params)
        joblib.dump(model, f'./model_save/{model_name}/{sampling}/{no}.pkl')
    elif model_name == 'DL-DNN':
        model = DNN(sample_x, sample_y)
        torch.save(model, f'./model_save/{model_name}/{sampling}/{no}.pt')
    elif model_name == 'TABNET':
        if(sample_x.shape[0]==1025):
            sample_x.drop(sample_x.index[1024], inplace=True)
            sample_y.drop(sample_y.index[1024], inplace=True)
        model = tabnet(sample_x.values[:], sample_y.values[:].reshape(len(sample_y),))
        model.save_model(f'./model_save/{model_name}/{sampling}/{no}')
        #모델 로드 방법
        #model = TabNetClassifier()
        #model.load_model(f'./model_save/{model_name}/{sampling}/{no}.zip')

    else:
        print("잘못된 모델입력")
    
    return model


# In[ ]:



