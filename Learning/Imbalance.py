# Resmapling Library
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import *
from imblearn.over_sampling import *
import json

def imbalance_data(x, y, model_info):
    x_resampled, y_resampled = resampling_method(x, y, model_info)
    return x_resampled, y_resampled

def resampling_method(x, y, model_info):
    method= model_info.resampling_method
    param = json.loads(model_info.resampling_param)

    if method == "ADASYN":
        return Adasyn(x, y, param)
    elif method == "SMOTE":
        return Smote(x, y)
    elif method == "SMTOMEK": # ???
        return SmoteTmoek(x, y)
    elif method == "RUS":
        return RUS(x, y, param)
    elif method == "ENN":
        return ENN(x, y, param)
    elif method == "CNN":
        return CNN(x, y, param)
    elif method == "NCR":
        return NCR(x, y, param)
    elif method == "ADASYN_NCR":
        return ADASYN_NCR(x, y, param)
    elif method == "ADASYN_CNN":
        return ADASYN_CNN(x, y, param)
    else:
        return "Invalid resampling method."


# Over-Sampling : ADASYN
def Adasyn(x_data, y_data, param):
    adasyn = ADASYN(random_state=eval(param["random_state"])) # random_state = 2
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Over-Sampling : SMOTE
def Smote(x_data, y_data):
    sm = SMOTE()
    x_resampled, y_resampled = sm.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Over-Sampling : SMOTETomek
def SmoteTmoek(x_data, y_data):
    smt = SMOTETomek()
    x_resampled, y_resampled = smt.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Under-Sampling : RandomUnderSampling
def RUS(x_data, y_data, param):
    # sampling_strategy='majority'
    Rus = RandomUnderSampler(sampling_strategy=param["sampling_strategy"])
    x_resampled, y_resampled = Rus.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Under-Sampling : EditedNearestNeighbours
def ENN(x_data, y_data, param):
    # kind_sel="all", n_neighbors=5
    Enn = EditedNearestNeighbours(kind_sel=param["kind_sel"], n_neighbors=eval(param["n_neighbors"]))
    x_resampled, y_resampled = Enn.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Under-Sampling : Condensed Nearest Neighbour
def CNN(x_data, y_data, param):
    # n_neighbors=5
    Cnn = CondensedNearestNeighbour(n_neighbors=eval(param["n_neighbors"]))
    x_resampled, y_resampled = Cnn.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Under-Sampling : Edited Nearest Neighbours
def NCR(x_data, y_data, param):
    # kind_sel="all", n_neighbors=5
    Ncr = NeighbourhoodCleaningRule(kind_sel=param["kind_sel"],
                                    n_neighbors=eval(param["n_neighbors"]))
    x_resampled, y_resampled = Ncr.fit_resample(x_data, y_data)

    return x_resampled, y_resampled

# Hybrid-Sampling : ADSYN + NCR
def ADASYN_NCR(x_data,y_data, param):
    # kind_sel="all", n_neighbors=5
    adasyn = ADASYN()
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

    Ncr = NeighbourhoodCleaningRule(kind_sel=param["kind_sel"],
                                    n_neighbors=eval(param["n_neighbors"]))
    x_resampled_H, y_resampled_H = Ncr.fit_resample(x_resampled, y_resampled)

    return x_resampled_H, y_resampled_H

# Hybrid-Sampling : ADSYN + CNN
def ADASYN_CNN(x_data, y_data, param):
    # n_neighbors=5
    adasyn = ADASYN()
    x_resampled, y_resampled = adasyn.fit_resample(x_data, y_data)

    Cnn = CondensedNearestNeighbour(n_neighbors=eval(param["n_neighbors"]))
    x_resampled_H, y_resampled_H = Cnn.fit_resample(x_resampled, y_resampled)

    return x_resampled_H, y_resampled_H