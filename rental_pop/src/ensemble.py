"""
Multi-model ensemble
"""
import glob
import time
import pandas as pd

PATH = '../output/ensemble/'
MODELS = glob.glob(PATH + '*.csv')

interest_levels = ['low', 'medium', 'high']

tau_test = {
    'low': 0.69195995, 
    'medium': 0.23108864,
    'high': 0.07695141, 
}

def correct(df, train=True, verbose=False):
    if train:
        tau = tau_train
    else:
        tau = tau_test
        
    index = df['listing_id']
    df_sum = df[interest_levels].sum(axis=1)
    df_correct = df[interest_levels].copy()
    
    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)
    
    for c in interest_levels:
        df_correct[c] /= df_sum

    for i in range(20):
        for c in interest_levels:
            df_correct[c] *= tau[c] / df_correct[c].mean()

        df_sum = df_correct.sum(axis=1)

        for c in interest_levels:
            df_correct[c] /= df_sum
    
    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)

    df_correct = pd.concat([index, df_correct], axis=1)
    return df_correct

def create_ensemble():
    """ simple averaging ensemble """
    model_list = {}
    for i in xrange(len(MODELS)):
        model_list[MODELS[i]] = pd.read_csv(MODELS[i])

    allm = pd.concat(model_list.values())
    ensem = allm.groupby(level=0).mean()
    timestamp = str(int(time.time()))
    ensem['listing_id'] = [int(i) for i in ensem['listing_id']]
    ensem = correct(ensem, train=False, verbose=True)
    ensem.to_csv(PATH + 'out/' + 'ensemble_' + timestamp + '.csv', index=False)
    return

if __name__ == '__main__':
    create_ensemble()
    print('ensemble results created')
