"""XGBoost"""

from __future__ import print_function
import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder


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
    # prop['city'] = prop['rawcensustractandblock'][0:4]

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
    params = {}
    params['learning_rate'] = 0.037
    params['boosting_type'] = 'dart'
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = 0.8
    params['num_leaves'] = 60
    params['verbose'] = 0
    params['min_hessian'] = 1

    df_train = train.merge(prop, how='left', on='parcelid')
    df_train = df_train[df_train.logerror > -0.21]
    df_train = df_train[df_train.logerror < 0.27]

    x_train = df_train.drop([
        'parcelid', 'logerror', 'transactiondate',
        'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values
    print(x_train.shape, y_train.shape)

    train_columns = x_train.columns

    for col in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[col] = (x_train[col] is True)

    del df_train
    gc.collect()

    print('Training ...')

    x_train = x_train.values.astype(np.float32, copy=False)

    d_train = lgb.Dataset(x_train, label=y_train)

    cv_scores = []
    rounds = []
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
    for dev_index, val_index in kf.split(range(x_train.shape[0])):
        dev_X, val_X = x_train[dev_index, :], x_train[val_index, :]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        e_train = lgb.Dataset(dev_X, label=dev_y)
        e_valid = lgb.Dataset(val_X, label=val_y)
        clf = lgb.train(
            params, e_train, 1000, [e_valid], verbose_eval=50,
            early_stopping_rounds=50)
        score = clf.best_score['valid_0']['l1']
        cv_scores.append(score)
        rounds.append(clf.best_iteration)
    print(cv_scores)
    print(round(np.mean(cv_scores), 8))
    print(rounds)
    print(np.mean(rounds))

    del x_train
    gc.collect()

    clf = lgb.train(params, d_train, np.max(rounds))

    del d_train
    gc.collect()

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

    print('Predicting on test ...')

    p_test = clf.predict(x_test)

    del x_test
    gc.collect()

    sub = pd.read_csv('../input/sample_submission.csv')
    for col in sub.columns[sub.columns != 'ParcelId']:
        sub[col] = p_test

    print('Writing csv ...')
    sub.to_csv('results/lgbm_starter.csv', index=False, float_format='%.4f')

if __name__ == '__main__':
    main()
