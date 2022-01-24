import pandas as pd
import datetime as dt
import pickle
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, \
    log_loss, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

import datetime as dt


# Значимые признаки, которые сохраняем в features
features_columns = ['id', 'buy_time', '158', '0', '2', '144', '27', '250', '13', '229', '223', '4',
                   '14', '224', '191', '222', '248', '201', '134', '192']


def preprocessing(df1, df2, col1, col2):
    # Удаление признака, признанного за ненужный:
    df1.drop(columns='Unnamed: 0', inplace=True)

    # Преобразование признака времени
    df1[col2] = df1[col2].apply(lambda x: dt.datetime.fromtimestamp(x))
    df2[col2] = df2[col2].apply(lambda x: dt.datetime.fromtimestamp(x))

    # присоединение датасета features - df2
    df1.sort_values(col1, ascending = True, inplace=True)
    df2.sort_values(col1, ascending = True, inplace=True)

    df = pd.merge_asof(df1,
                       df2,
                       on= col1,
                       by= col2,
                       direction='nearest')

    # Создание новых признаков
    df['offer_month'] = df[col2].dt.month
    df['offer_day'] = df[col2].dt.day

    # Удаление признака buy_tine
    df.drop(columns= col2, inplace=True)

    return df


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)




if __name__ == '__main__':
    PROJECT_NAME = input('Name of your project: ')
    USERS_DATA_PATH = input('Path to users data file: ') # data_test.csv
    FEATURES_PATH = input('Path to features file: ') # features.csv
    try:
        print('Opening datasets...')
        initial_test_df = pd.read_csv(USERS_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {USERS_DATA_PATH} is not found. Please, сheck the path and try again!')


    try:
        print('Collecting data...')
        features_df = pd.read_csv(FEATURES_PATH, sep='\t', usecols=features_columns)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'File {FEATURES_PATH} is not found. Please, сheck the path and try again!')

    print('Preprocessing ...')
    df_for_predictions = preprocessing(initial_test_df, features_df, 'id', 'buy_time')

    print('Making predictions ...')
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    df_for_predictions['target'] = target_pred_proba = model.predict_proba(df_for_predictions)[:, 1]

    print('Saving results ... ')
    df_for_predictions['target'].to_csv(PROJECT_NAME + '_predictions_' + str(dt.date.today()))

    print('Completed !')

