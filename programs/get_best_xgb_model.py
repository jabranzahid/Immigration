'''
This generates the best model. The indices are derived as described
in xgboot_optimize.ipynb
'''

import xgboost as xgb
import numpy as np


def get_best_xgb_model(para = None):

    if para is None:
        para = {'alpha': 5, 
                'colsample_bytree': 0.4, 
                'gamma': 10, 
                'learning_rate': 0.12999999999999998, 
                'max_depth': 7, 
                'min_child_weight': 5, 
                'n_estimators': 46, 
                'subsample': 0.8779937359366555}
    reg = xgb.XGBClassifier(**para,objective='binary:logistic')

    return reg