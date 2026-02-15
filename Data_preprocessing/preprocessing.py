import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from Data_Preprocessing.transformer import Transformer


def drop_column(df):
    # 모든 데이터가 동일한 컬럼
    drop_cols = df.columns[df.nunique() == 1]
    df = df.drop(columns=drop_cols)
    return df

def preprocessing(data, mode, batch_info, batch_param):
    tf = Transformer(batch_info)

    X, Y = data.iloc[:, :-1], data.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=eval(batch_param["testSize"]))

    if mode == "train":
        tf.fit(X_train)
        X = tf.transform(X_train)
        Y = Y_train
    else:
        X = tf.transform(X_test)
        Y = Y_test
    return X, Y

def preprocessing_beta(df, datatype):
    # 카테고리, 뉴메릭 데이터 구분
    cat_datatype = datatype.T[datatype.T['type'] == 'ca'].T
    num_datatype = datatype.T[datatype.T['type'] == 'num'].T
    cat_datatype.columns.tolist()
    df.columns.tolist()
    cat_col = list(set(df.columns.tolist()).intersection(cat_datatype.columns.tolist()))
    num_col = list(set(df.columns.tolist()).intersection(num_datatype.columns.tolist()))
    cat_data = df[cat_col]
    num_data = df[num_col]
    print('카테고리 데이터 : ', cat_data.shape, '뉴메릭 데이터 : ', num_data.shape)

    # 카테고리 데이터중 num, ord 구분
    cat_nom_datatype = datatype.T[datatype.T['typeoftype'] == 'nom'].T
    cat_ord_datatype = datatype.T[datatype.T['typeoftype'] == 'ord'].T
    cat_nom_col = list(set(df.columns.tolist()).intersection(cat_nom_datatype.columns.tolist()))
    cat_ord_col = list(set(df.columns.tolist()).intersection(cat_ord_datatype.columns.tolist()))
    cat_nom_data = df[cat_nom_col]
    cat_ord_data = df[cat_ord_col]
    print('카테고리 노미널 데이터 : ', cat_nom_data.shape, '카테고리 오디널 데이터 : ', cat_ord_data.shape)

    cat_nom_str = [var for var in cat_nom_data.columns if cat_nom_data[var].dtype == 'object']
    cat_nom_int = [col for col in cat_nom_data.columns if cat_nom_data[col].dtype != 'object']
    print('카테고리 노미널 스트링 데이터 : ', len(cat_nom_str), '카테고리 노미널 인트 데이터 : ', len(cat_nom_int))

    cat_ord_str = [var for var in cat_ord_data.columns if cat_ord_data[var].dtype == 'object']
    cat_ord_int = [col for col in cat_ord_data.columns if cat_ord_data[col].dtype != 'object']
    print('카테고리 오디널 스트링 데이터 : ', len(cat_ord_str), '카테고리 오디널 인트 데이터 : ', len(cat_ord_int))

    encoder = ce.OneHotEncoder(cols=cat_nom_col)
    df = encoder.fit_transform(df)
    print(f"df.shape: {df.shape}")

    data = df.drop(['y'], axis=1)
    label = df['y']

    return data, label

def data_scaler(data):
    columns = data.columns
    scaler = RobustScaler()
    data = scaler.fit_transform(data)

    data = pd.DataFrame(data, columns=columns)
    return data