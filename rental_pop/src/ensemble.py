"""
Multi-model ensemble
"""
import glob
import time
import pandas as pd

PATH = '../output/ensemble/'
MODELS = glob.glob(PATH + '*.csv')

def create_ensemble():
    """ simple averaging ensemble """
    model_list = {}
    for i in xrange(len(MODELS)):
        model_list[MODELS[i]] = pd.read_csv(MODELS[i])

    allm = pd.concat(model_list.values())
    ensem = allm.groupby(level=0).mean()
    timestamp = str(int(time.time()))
    ensem.to_csv(PATH + 'out/' + 'ensemble_' + timestamp + '.csv', index=False)
    return

if __name__ == '__main__':
    create_ensemble()
    print('ensemble results created')
