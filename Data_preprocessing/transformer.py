import pymysql.err
import sqlalchemy.exc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from ENV.env import *
import pickle


class Transformer:
    def __init__(self, batch_info):
        self.batch_id = batch_info['batch_id']
        self.version = batch_info['dataset_version']
        self.dataset_group = batch_info['dataset_group']
        # self.db_handler = DB_Handler('deep_credit')
        # self.con = DB_Handler('deep_credit').connection
        self.db_handler = dbhandler
        self.con = dbhandler.connection
        self.scaler = None
        self.imputer = None
        self.df_mgmt = None
        self.dt_cols = None
        self.data_col_idx_dict = None
        self.col_after_transform = []
        self.imputer_file = '_'.join([self.dataset_group, str(self.version)]) + '_imputer.bin'
        self.scaler_file = '_'.join([self.dataset_group, str(self.version)]) + '_scaler.bin'

        sql = f"select data_table from dc_dataset_version where version = {self.version} and dataset_group = '{self.dataset_group}'"
        self.data_table_name = pd.read_sql(sql, self.con).data_table.values[0]

        sql = f"select * from dc_dataset_management where version = {self.version} and dataset_group = '{self.dataset_group}' and use_yn = 'Y'"
        self.df_mgmt = pd.read_sql(sql, self.con)
        self.item_codes = self.df_mgmt.item_code.to_list()

    def _transformer(self, method):

        #imputation and scaling
        if method in ['mean', 'median', 'mode', 'default']:
            field_name = 'missing_method'
        elif method in ['min-max', 'z-score']:
            field_name = 'normalization_method'
        elif method in ['one-hot', 'ordinal']:
            field_name = 'encoding_method'

        #replacing values
        if method == 'replace':
            field_name = 'outlier_process'
            method_df = self.df_mgmt[self.df_mgmt[field_name] != '']
        else:
            method_df = self.df_mgmt[self.df_mgmt[field_name] == method]

        # get the rows with method
        columns = method_df.item_code.to_list()

        # select columns not only exist in mgmt table
        columns = [c for c in columns if c in self.dt_cols]

        # return {column name : default value} dict
        if method == 'default':
            return {c: v for c, v in zip(columns, method_df.missing_default_value.to_list())}
        # return {column name : replace value dict} dict
        elif method == 'replace':
            return {c: v for c, v in zip(columns, method_df.outlier_process.apply(lambda x :eval(x)).values)}
        else:
            self.col_after_transform = self.col_after_transform + columns

        # find the index with column name from management table
        col_idx = list(map(self.data_col_idx_dict.get, columns))
        # x = [x for _, x in sorted(zip(a, columns))]

        if not col_idx:
            return

        method2transformer = {'min-max': MinMaxScaler(),
                              'z-score': StandardScaler(),
                              'one-hot': OneHotEncoder(handle_unknown='ignore'),
                              'ordinal': OrdinalEncoder(), #OrdinalEncoder(handle_unknown='ignore'),
                              'mean': SimpleImputer(strategy='mean'),
                              'median': SimpleImputer(strategy='median'),
                              'mode': SimpleImputer(strategy='most_frequent')
                              }

        transformer = method2transformer.get(method)

        if not transformer:
            print("No normalization nor encoding method exists")
            return

        return (method, transformer, col_idx)

    def _column_transformers(self, transformer='imputer' or 'scaler'):
        if transformer == 'imputer':
            return ColumnTransformer(
                transformers=list(filter(None, [
                    self._transformer('mean'),
                    self._transformer('median'),
                    self._transformer('mode'),
                ]))
            )
        elif transformer == 'scaler':
            return ColumnTransformer(
                transformers=list(filter(None, [
                    self._transformer('min-max'),
                    self._transformer('z-score'),
                    self._transformer('one-hot'),
                    self._transformer('ordinal'),
                ]))
            )
        else:
            return

    def get_data(self):
        sql = f"select * from {self.data_table_name} "
        return pd.read_sql(sql, self.con)

    def _filter_data(self, X):

        X = X[[c for c in X.columns if c in self.item_codes]]

        # make column, index dictionary for pipe
        self.dt_cols = X.columns
        self.data_col_idx_dict = {c: i for i, c in enumerate(X.columns)}

        # fill up missing values with default values
        fill_dict = self._transformer('default')
        if fill_dict:
            X.fillna(fill_dict, inplace = True)

        # replace values with outlier dictionary
        fill_dict = self._transformer('replace')
        if fill_dict:
            X.replace(fill_dict, inplace = True)
        # for item_code, replace_dict in enumerate(self._transformer('replace')):
        #     X[item_code].map(replace_dict).fillna(X[item_code], inplace= True)

        return X

    def _save_model(self):
        # saving imputer and scaler
        imputer_blob = pickle.dumps(self.imputer)
        scaler_blob = pickle.dumps(self.scaler)
        sql = "REPLACE INTO dc_preprocessing (batch_id, item_code, imputing_model, scaling_model) VALUES (%s,%s,%s,%s)"
        try:
            self.con.execute(sql, (self.batch_id, self.dataset_group, imputer_blob, scaler_blob,))
        except sqlalchemy.exc.IntegrityError as err:
            print('\n',err,'\n')

    def _load_model(self):
        # saving imputer and scaler
        sql = f"SELECT * FROM dc_preprocessing where batch_id = {self.batch_id} and item_code = '{self.dataset_group}'"
        df = pd.read_sql(sql, self.con)
        imputer_blob = df.imputing_model.values[0]
        scaler_blob = df.scaling_model.values[0]
        self.imputer = pickle.loads(imputer_blob)
        self.scaler = pickle.loads(scaler_blob)

    def fit(self, X):
        X = self._filter_data(X)
        self.imputer = self._column_transformers('imputer')

        if len(X.columns) != len(self.col_after_transform):
            print("some columns missing in imputation")
            return

        self.imputer.fit(X)

        # because it's only fit, replicate the column order like after imputation. so scaler can match.
        X = X[self.col_after_transform].values

        # reset the column name, index dictionary since column order changed after imputation
        self.data_col_idx_dict = {c: i for i, c in enumerate(self.col_after_transform)}
        self.col_after_transform = []

        self.scaler = self._column_transformers('scaler')
        self.scaler.fit(X)

        self._save_model()

    def fit_transform(self, X):
        X = self._filter_data(X)

        # create imputer
        self.imputer = self._column_transformers('imputer')

        if len(X.columns) != len(self.col_after_transform):
            print("some columns missing in imputation")
            return

        X = self.imputer.fit_transform(X)

        # reset the column name, index dictionary since column order changed after imputation
        self.data_col_idx_dict = {c: i for i, c in enumerate(self.col_after_transform)}
        self.col_after_transform = []

        self.scaler = self._column_transformers('scaler')
        X = self.scaler.fit_transform(X)
        self._save_model()
        return X

    def transform(self, X):
        X = self._filter_data(X)
        self._load_model()
        return self.scaler.transform(self.imputer.transform(X))


if __name__ == '__main__':
    batch_info = dict(batch_id=1000,
                      dataset_group='KCB-YC-001',
                      dataset_version=1,
                      )
    tf = Transformer(batch_info)

    data = tf.get_data()
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train = tf.fit_transform(X_train)
    # X_test = tf.transform(X_test)

    tf.fit(X_train)
    X_train = tf.transform(X_train)
    X_test = tf.transform(X_test)
    print("column order :\n", tf.col_after_transform)
    print()
    print("train: ", X_train.shape)
    print("test: ", X_test.shape)