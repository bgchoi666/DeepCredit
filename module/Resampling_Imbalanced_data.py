#!/usr/bin/env python
# coding: utf-8

# # Resmapling Library

# In[8]:


import import_ipynb
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
#from Library_Box import * 


# # Over-Sampling : ADASYN

# In[ ]:


def Adasyn(x_data, y_data, n_neighbors=5):
    adasyn = ADASYN(n_neighbors = n_neighbors)
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled 


# # Over-Sampling : SMOTE

# In[ ]:


def Smote(x_data,y_data, k_neighbors=5):
    sm = SMOTE(k_neighbors=k_neighbors)
    x_resampled, y_resampled = sm.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled


# # Over-Sampling : SMOTETomek

# In[ ]:


def SmoteTmoek(x_data, y_data, random_state=0):
    smt = SMOTETomek()
    x_resampled, y_resampled = smt.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled


# # Under-Sampling : RandomUnderSampling 

# In[5]:


def RUS(x_data, y_data):
    Rus = RandomUnderSampler(sampling_strategy='majority')
    x_resampled, y_resampled = Rus.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled


# # Under-Sampling : EditedNearestNeighbours

# In[3]:


def ENN_(x_data, y_data, n_neighbors=3): 
    Enn = EditedNearestNeighbours(kind_sel="all", n_neighbors=n_neighbors)
    x_resampled, y_resampled = Enn.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled  


# # Under-Sampling : Edited Nearest Neighbours

# In[12]:


def NCR(x_data,y_data,n_neighbors=3):
    Ncr = NeighbourhoodCleaningRule(kind_sel="all", n_neighbors=n_neighbors)
    x_resampled, y_resampled = Ncr.fit_resample(x_data, y_data)
    
    return x_resampled, y_resampled


# # Hybrid-Sampling : ADSYN + NCR

# In[13]:


def ADASYN_NCR(x_data,y_data, n_neighbors=5):
    adasyn = ADASYN(n_neighbors = n_neighbors)
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

    Ncr = NeighbourhoodCleaningRule(kind_sel="all", n_neighbors=n_neighbors)
    x_resampled_H, y_resampled_H = Ncr.fit_resample(x_resampled, y_resampled)

    return x_resampled_H, y_resampled_H


# # # Hybrid-Sampling : ADSYN + Rus

# In[2]:


def ADASYN_RUS(x_data,y_data, n_neighbors=3):
    adasyn = ADASYN(n_neighbors = n_neighbors)
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

    Rus = RandomUnderSampler(sampling_strategy='majority')
    x_resampled_H, y_resampled_H = Rus.fit_resample(x_resampled, y_resampled)
    
    return x_resampled_H, y_resampled_H
