"""XGBoost"""

from __future__ import print_function
import gc
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_data():
    """Load dataset"""
    train_p = '../input/train_p'
    prop_p = '../input/prop_p'
    sample_p = '../input/sample_p'

    if os.path.exists(train_p):
        train = pd.read_pickle(train_p)
    else:
        train = pd.read_csv('../input/train_2016_v2.csv')
        train.to_pickle(train_p)

    if os.path.exists(prop_p):
        prop = pd.read_pickle(prop_p)
    else:
        prop = pd.read_csv('../input/properties_2016.csv')
        print('Binding to float32')
        for col, dtype in zip(prop.columns, prop.dtypes):
            if dtype == np.float64:
                prop[col] = prop[col].astype(np.float32)
        prop.to_pickle(prop_p)

    if os.path.exists(sample_p):
        sample = pd.read_pickle(sample_p)
    else:
        sample = pd.read_csv('../input/sample_submission.csv')
        sample.to_pickle(sample_p)
    return prop, train, sample


def binning(field, bincnt, qnt=False):
    """Sort into bins"""
    if not qnt:
        out = pd.qcut(field, bincnt, labels=False)
    else:
        out = pd.cut(field, bincnt, labels=False)
    return out.astype(float)


def feature_engineering(prop):
    """Create custom features"""
    # ratio of bed to bath
    prop['bedbathratio'] = prop['bedroomcnt'] / prop['bathroomcnt']

    print('Encoding categorical features ...')

    cat_features = [
        'regionidcity',
        'yearbuilt'
    ]

    for cat in cat_features:
        prop[cat] = prop[cat].fillna(-1)
        lbl = LabelEncoder()
        lbl.fit(list(prop[cat].values))
        prop[cat] = lbl.transform(list(prop[cat].values))
    return prop


def create_trainset(train, prop):
    """Create training dataset"""
    df_train = train.merge(prop, how='left', on='parcelid')
    df_train = df_train[df_train.logerror > -0.21]
    df_train = df_train[df_train.logerror < 0.27]

    x_train = df_train.drop([
        'parcelid', 'logerror', 'transactiondate',
        'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values
    y_mean = np.mean(y_train)
    print(x_train.shape, y_train.shape)

    train_columns = x_train.columns

    for col in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[col] = (x_train[col] is True)

    del df_train
    gc.collect()

    split = 80000
    x_train, y_train, x_valid, y_valid = \
        x_train[:split], y_train[:split], x_train[split:], y_train[split:]

    print('Building DMatrix...')

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    del x_train, x_valid
    gc.collect()

    print('Training ...')

    params = {}
    params['eta'] = 0.037
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'
    params['max_depth'] = 5
    params['subsample'] = 0.80
    params['lambda'] = 0.8
    params['alpha'] = 0.4
    params['base_score'] = y_mean
    params['silent'] = 1

    clf = xgb.train(params, d_train, 10000, [(d_train, 'train'),
                    (d_valid, 'valid')], early_stopping_rounds=50,
                    verbose_eval=10)

    del d_train, d_valid

    return train_columns, clf


def main():
    """Run the main function"""
    print('Loading data ...')
    prop, train, sample = load_data()

    print('Feature engineering ...')
    prop = feature_engineering(prop)

    print('Creating training set ...')
    train_columns, clf = create_trainset(train, prop)

    print('Building test set ...')

    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')

    del prop
    gc.collect()

    x_test = df_test[train_columns]
    for col in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[col] = (x_test[col] is True)

    del df_test, sample
    gc.collect()

    d_test = xgb.DMatrix(x_test)

    del x_test
    gc.collect()

    print('Predicting on test ...')

    p_test = clf.predict(d_test)

    del d_test
    gc.collect()

    sub = pd.read_csv('../input/sample_submission.csv')
    for col in sub.columns[sub.columns != 'ParcelId']:
        sub[col] = p_test

    print('Writing csv ...')
    sub.to_csv('results/xgb_starter.csv', index=False, float_format='%.4f')

if __name__ == '__main__':
    main()
