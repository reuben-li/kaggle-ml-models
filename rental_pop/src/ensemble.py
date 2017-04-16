"""
Multi-model ensemble
"""

import os
import pandas as pd

PATH = '../output/ensemble/'
MODELS = os.listdir(PATH)

def create_ensemble():
    """ simple averaging ensemble """
    model_list = {}
    for i in xrange(len(MODELS)):
        model_list[MODELS[i]] = pd.read_csv(PATH + MODELS[i])

    allm = pd.concat(model_list.values())
    ensem = allm.groupby(level=0).mean()
    ensem.to_csv(PATH + 'ensemble.csv', index=False)
    return

if __name__ == '__main__':
    create_ensemble()
    print('ensemble results created')
