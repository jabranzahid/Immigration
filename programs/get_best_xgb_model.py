'''
This generates the best model. The indices are derived as described
in xgboot_optimize.ipynb
'''

import xgboost as xgb
import numpy as np


def get_best_indices():

    index = [  4,   5,  12,  31,  35,  36,  37,  39,  41,  42,  43,  44,  45,
              46,  51,  52,  53,  56,  58,  59,  62,  63,  64,  65,  66,  67,
              68,  69,  70,  73,  74,  75,  76,  77,  78,  81,  82,  83,  89,
              95,  96,  98,  99, 101, 102, 103, 104, 108, 113, 114, 115, 116,
             119, 126, 129, 130, 131, 133, 134, 135, 141, 143, 148, 153, 154,
             156, 157, 158, 159, 160, 161, 162, 163, 164, 167]

    return index

def get_best_xgb_model(para = None):

    if para is None:
        para = {'colsample_bytree': 0.4,
                 'learning_rate': 0.1,
                 'max_depth': 6,
                 'min_child_weight': 8,
                 'n_estimators': 65,
                 'subsample': 0.9174674408083486}

    reg = xgb.XGBRegressor(**para,objective='reg:squarederror')
    index = get_best_indices()

    return reg, index