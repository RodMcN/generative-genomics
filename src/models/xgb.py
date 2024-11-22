import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import joblib
from io import BytesIO


class XGBWrapper:
    def __init__(self, params, random_state=42):
        self.params = params
        self.model = None
        self.random_state = random_state
        self.class_weights = None


    def fit(self, train_x, train_y, val_x, val_y):
        
        self.class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        self.class_weights = dict(enumerate(self.class_weights))
        sample_weights = np.array([self.class_weights[i] for i in train_y])
        sample_weight_eval_set = [np.array([self.class_weights[i] for i in val_y])]
        
        
        self.model = XGBClassifier(**self.params,
                          random_state=42,
                        #   n_jobs=min(os.cpu_count() // 2, 32),
                          objective='multi:softmax',
                          booster="gbtree",
                          eval_metric="mlogloss", 
                          num_class=len(np.unique(train_y)),
                          early_stopping_rounds=10,
                          # verbosity=0,
                          tree_method="hist", device="cuda"
                          )

        self.model.fit(
            train_x, 
            train_y, 
            eval_set=[(val_x, val_y)],
            verbose=True,
            sample_weight=sample_weights,
            sample_weight_eval_set=sample_weight_eval_set
        )
    
    
    def evaluate(self, data):
        ...
        
    
    def save(self, path):
        model_buffer = BytesIO()
        self.model.save_model(model_buffer)
        attributes = {
            'params': self.params,
            'random_state': self.random_state,
            'class_weights': self.class_weights,
            'model': model_buffer.getvalue()
        }
        joblib.dump(attributes, path)


    @classmethod
    def load(cls, path):
        attributes = joblib.load(path)
        instance = cls(params=attributes['params'], random_state=attributes['random_state'])
        instance.class_weights = attributes['class_weights']
        model_buffer = BytesIO(attributes['model'])
        instance.model = XGBClassifier()
        instance.model.load_model(model_buffer)
        return instance
