#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
#sys.path.append('/data')
#sys.path.append('root/CSS/final/module')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import category_encoders as ce


# In[2]:


def load_and_preprocessing(datafile,datafile_index,datatypefile,datatypefile_index):
    data = pd.read_csv(datafile, index_col=datafile_index, encoding='utf-8')
    raw_data = data.copy()
    datatype = pd.read_csv(datatypefile, index_col = datatypefile_index, encoding = 'utf-8')
    
    #데이터 타입 파일로 부터 (전처리 완료된 파일) 원본 데이터를 categorical, numerical을 구분한다.
    cat_datatype = datatype[datatype['t']=='categorical'].T
    num_datatype = datatype[datatype['t']=='numerical'].T
    cat_datatype.columns.tolist()

    data.columns.tolist()
    cat_col = list(set(data.columns.tolist()).intersection(cat_datatype.columns.tolist()))
    num_col = list(set(data.columns.tolist()).intersection(num_datatype.columns.tolist()))
    cat_data = data[cat_col]
    num_data = data[num_col]
    #print('카테고리 데이터 : ', cat_data.shape,'뉴메릭 데이터 : ', num_data.shape)
    
    cat_nom_datatype = datatype[datatype['tt']=='norminal'].T
    cat_ord_datatype = datatype[datatype['tt']=='ordinal'].T
    cat_nom_col = list(set(data.columns.tolist()).intersection(cat_nom_datatype.columns.tolist()))
    cat_ord_col = list(set(data.columns.tolist()).intersection(cat_ord_datatype.columns.tolist()))
    cat_nom_data = data[cat_nom_col]
    cat_ord_data = data[cat_ord_col]
    #print('카테고리 노미널 데이터 : ', cat_nom_data.shape,'카테고리 오디널 데이터 : ',cat_ord_data.shape)
    
    cat_nom_str = [var for var in cat_nom_data.columns if cat_nom_data[var].dtype=='object']
    cat_nom_int = [col for col in cat_nom_data.columns if cat_nom_data[col].dtype!='object']
    #print('카테고리 노미널 스트링 데이터 : ', len(cat_nom_str),'카테고리 노미널 인트 데이터 : ',len(cat_nom_int))
    
    cat_ord_str = [var for var in cat_ord_data.columns if cat_ord_data[var].dtype=='object']
    cat_ord_int = [col for col in cat_ord_data.columns if cat_ord_data[col].dtype!='object']
    #print('카테고리 오디널 스트링 데이터 : ', len(cat_ord_str),'카테고리 오디널 인트 데이터 : ',len(cat_ord_int))
    
    scaler = RobustScaler()
    data[num_col] = scaler.fit_transform(data[num_col])

    encoder = ce.OneHotEncoder(cols = cat_nom_col)
    data = encoder.fit_transform(data)
        
    label = data['y1']
    data = data.drop(['y1'], axis=1)
    
    return data, label, raw_data


# In[44]:


def split_same_ratio(data, label, test_size,random_state):
    
    label = pd.DataFrame(label)
    label_0 = label[label['y1']==0]
    label_1 = label[label['y1']==1]
    #print("0의 갯수 : ", label_0.shape, "1의 갯수 : ", label_1.shape)
    data_0 = data.loc[label_0.index]
    data_1 = data.loc[label_1.index]
    #print("0의 갯수 : ", data_0.shape, "1의 갯수 : ", data_1.shape)

    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(data_0, label_0, test_size = test_size, random_state = random_state)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_1, label_1, test_size = test_size, random_state = random_state)
    
    x_train = pd.concat([x_train_0, x_train_1], axis =0)
    y_train = pd.concat([y_train_0, y_train_1], axis =0)
    x_test = pd.concat([x_test_0, x_test_1], axis =0)
    y_test = pd.concat([y_test_0, y_test_1], axis =0)
    
    return x_train, x_test, y_train, y_test


# In[43]:


if __name__ == "__main__":
    data, label, raw_data = load_and_preprocessing("../data/NICE_IBK_002.csv", '계좌번호', "../data/ibk_datatype.csv", '구분')
    x_train, x_test, y_train, y_test = split_same_ratio(data, label, test_size = 0.2, random_state =3)
    x_train1, x_val, y_train1, y_val = split_same_ratio(x_train, y_train, test_size=0.25, random_state=3)
    print(data)


# In[ ]:



