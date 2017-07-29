"""Stack model"""
from __future__ import print_function

from datetime import datetime
import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
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
        train = pd.read_csv('../input/train_2016_v2.csv',
                            parse_dates=['transactiondate'])
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


def binner(field, bincnt, qnt=False):
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
    prop['city'] = prop['rawcensustractandblock'][0:4]

    # prop['taxamount_bin'] = binner(prop['taxamount'], 12, True)

    abc_list = []
    classes = 20
    for i in xrange(97, 123):
        abc_list.append(str(chr(i)))
    prop_lon, lon_bins = pd.qcut(prop['longitude'],
                                 classes, retbins=True,
                                 labels=abc_list[0:classes])
    prop_lat, lat_bins = pd.qcut(prop['latitude'],
                                 classes, retbins=True,
                                 labels=abc_list[0:classes])
    prop_lon = prop_lon.astype(object)
    prop_lat = prop_lat.astype(object)
    prop['grid'] = prop_lon + prop_lat

    print('Encoding categorical features ...')

    cat_features = [
        'regionidcity',
        'yearbuilt',
        'grid',
        'city'
    ]

    for cat in cat_features:
        prop[cat] = prop[cat].fillna(-1)
        lbl = LabelEncoder()
        lbl.fit(list(prop[cat].values))
        prop[cat] = lbl.transform(list(prop[cat].values))
    return prop


def create_lgb_trainset(train, prop):
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
        'fireplacecnt',
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
            params, e_train, 1000, [e_valid], verbose_eval=200,
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


def create_xgb_trainset(train, prop):
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


def run_gb(train, prop, sample, clf, train_columns, model):
    """Gradient Boosting"""
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

    if model == 'XGB':
        x_test = xgb.DMatrix(x_test)

    print('Predicting on test ...')

    p_test = clf.predict(x_test)

    del x_test
    gc.collect()
    return p_test


def get_time_features(df):
    """Time features"""
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df


def MAE(y, ypred):
    """Calculate MAE"""
    return np.sum([abs(y[i] - ypred[i]) for i in range(len(y))]) / len(y)


def run_ols():
    """Run OLS"""
    np.random.seed(17)
    random.seed(17)

    prop, train, sample = load_data()

    train = pd.merge(train, prop, how='left', on='parcelid')
    y = train['logerror'].values

    del prop
    gc.collect()

    exc = [train.columns[c] for c in range(len(train.columns))
           if train.dtypes[c] == 'O'] + ['logerror', 'parcelid']
    col = [c for c in train.columns if c not in exc]

    train = get_time_features(train[col])
    print(np.any(np.isnan(train)))
    print(np.all(np.isfinite(train)))

    reg = LinearRegression(n_jobs=-1)
    reg.fit(train, y)
    print('fit...')
    print(MAE(y, reg.predict(train)))

    del train, y
    gc.collect()
    return reg, col


def ensemble(lgb_pred, xgb_pred, reg, col, prop, sample):
    """Combine models"""
    XGB_WEIGHT = 0.6300
    BASELINE_WEIGHT = 0.0056
    OLS_WEIGHT = 0.0550
    BASELINE_PRED = 0.0115

    test_dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01',
                  '2017-11-01', '2017-12-01']
    test_columns = ['201610', '201611', '201612', '201710', '201711', '201712']

    print("\nCombining XGBoost, LightGBM, and baseline predictions ...")
    lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
    xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
    baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
    pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED +\
        lgb_weight * lgb_pred

    print("\nCombined XGB/LGB/baseline predictions:")
    print(pd.DataFrame(pred0).head())

    print("\nPredicting with OLS and combining with XGB/LGB/baseline ...")

    test = pd.merge(sample, prop, how='left', left_on='ParcelId',
                    right_on='parcelid')

    test['transactiondate'] = '2016-01-01'
    test = get_time_features(test[col])

    for i in range(len(test_dates)):
        test['transactiondate'] = test_dates[i]
        pred = OLS_WEIGHT * reg.predict(get_time_features(test)) + \
            (1 - OLS_WEIGHT) * pred0
        sample[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
        print('predict...', i)

    print("\nCombined XGB/LGB/baseline/OLS predictions:")
    print(sample.head())
    del sample['parcelid']

    print("\nWriting results to disk ...")
    sample.to_csv('results/sub{}.csv'
                  .format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False)

    print("\nFinished ...")


def main():
    """Run the main function"""
    print('Loading data ...')
    prop, train, sample = load_data()

    print('Feature engineering ...')
    prop = feature_engineering(prop)

    print('Creating LGB model ...')
    ltrain_columns, lclf = create_lgb_trainset(train, prop)

    print('Creating XGB model ...')
    xtrain_columns, xclf = create_xgb_trainset(train, prop)

    print('Predicting with LGB ...')
    lgb_results = run_gb(train, prop, sample, lclf, ltrain_columns, 'LGB')

    print('Predicting with XGB ...')
    xgb_results = run_gb(train, prop, sample, xclf, xtrain_columns, 'XGB')

    print('Fit OLS model ...')
    reg, col = run_ols()

    print('Ensembling')
    ensemble(lgb_results, xgb_results, reg, col, prop, sample)


if __name__ == '__main__':
    main()
