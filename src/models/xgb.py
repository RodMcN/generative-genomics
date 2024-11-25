import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import tempfile
from sklearn.metrics import f1_score


class XGBWrapper:
    def __init__(self, params, name, random_state=42):
        self.params = params
        self.name = name
        self.model = None
        self.random_state = random_state
        self.class_weights = None
        self.history = None


    def fit(self, train_x, train_y, val_x, val_y):
        
        self.class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        self.class_weights = dict(enumerate(self.class_weights))
        sample_weights = np.array([self.class_weights[i] for i in train_y])
        sample_weight_eval_set = [np.array([self.class_weights[i] for i in val_y])]
        
        
        self.model = XGBClassifier(**self.params,
                          random_state=42,
                          objective='multi:softmax',
                          booster="gbtree",
                          eval_metric="mlogloss", 
                          num_class=len(np.unique(train_y)),
                          tree_method="hist", device="cuda"
                          )

        self.model.fit(
            train_x, 
            train_y, 
            eval_set=[(val_x, val_y)],
            verbose=True,
            sample_weight=sample_weights,
            sample_weight_eval_set=sample_weight_eval_set,
        )
    
    
    def evaluate(self, test_x, test_y, categories, category_mapping=None):
        test_y_pred = self.model.predict(test_x)
                
        # Map integer labels to class names
        test_y_series = pd.Series(test_y).map(lambda x: categories[x])
        test_y_pred_series = pd.Series(test_y_pred).map(lambda x: categories[x])
        
        if category_mapping is not None:
            output_cols = ["mapped_cell_type", "f1_score"]
            
            mapped_test_y = test_y_series.map(category_mapping).fillna(test_y_series)
            mapped_test_y_pred = test_y_pred_series.map(category_mapping).fillna(test_y_pred_series)
            
            mapped_categories = sorted(mapped_test_y.unique())
        else:
            output_cols = ["cell_type", "f1_score"]
            mapped_test_y = test_y_series
            mapped_test_y_pred = test_y_pred_series
            mapped_categories = sorted(mapped_test_y.unique())
            
        output_rows = []

        # F1 per class
        f1_per_class = f1_score(mapped_test_y, mapped_test_y_pred, labels=mapped_categories, average=None)
        for cat, score in zip(mapped_categories, f1_per_class):
            output_rows.append((cat, score))

        # Macro-Averaged F1
        macro_f1 = f1_score(mapped_test_y, mapped_test_y_pred, average='macro')
        output_rows.append(("macro averaged", macro_f1))

        # Micro-Averaged F1
        micro_f1 = f1_score(mapped_test_y, mapped_test_y_pred, average='micro')
        output_rows.append(("micro averaged", micro_f1))

        return pd.DataFrame(output_rows, columns=output_cols)
        
    
    def save(self, path):
        if self.model is not None:
            with tempfile.NamedTemporaryFile() as tmp_file:
                # dump xgboost model to a temp file to save as bytes
                # janky but keeps everything together without pickling model
                self.model.save_model(tmp_file.name)
                tmp_file.seek(0)
                model = tmp_file.read()
        else:
            model = None
        attributes = {
            'params': self.params,
            'name': self.name,
            'random_state': self.random_state,
            'class_weights': self.class_weights,
            'model': model
        }
        joblib.dump(attributes, path)


    @classmethod
    def load(cls, path):
        attributes = joblib.load(path)
        instance = cls(params=attributes['params'], 
                    name=attributes['name'],
                    random_state=attributes['random_state'])
        instance.class_weights = attributes['class_weights']
        model_data = attributes['model']
        if model_data is not None:
            with tempfile.NamedTemporaryFile() as tmp_file:
                # load the model from the stored bytes
                # write to a temp file for load_model
                tmp_file.write(model_data)
                tmp_file.flush()
                instance.model = XGBClassifier()
                instance.model.load_model(tmp_file.name)
        else:
            instance.model = None
        return instance
