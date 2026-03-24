'''
Model.py file.

This will be used to:
- Load the feature matrix 
- Time-based split by end_date - sort, slice into 60/20/20, save 3 seperate parquets
- MLflow experiment setup
- XGBoost pipeline with scale_pos_weight = 2.47
- Metric logging structure - Brier Score, AUC-ROC, log-loss
'''

import polars as pl 
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import mlflow 
from xgboost import XGBClassifier
import os 
import logging 
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_SPLIT = 0.6
CALIBRATION_SPLIT = 0.2
TEST_SPLIT = 0.2
SCALE_POS_WEIGHT = 2.47

mlflow.set_experiment('polymarket-conformal')

class Model:
    def __init__(self):
        self.feature_matrix = None 
        # Load data into model class 
        self._load_data(path=self.feature_matrix)

        logger.info("Successfully initalised model")
    
    def _load_data(self, path):
        try:
            self.feature_matrix = pl.read_parquet('data/processed/feature_matrix.parquet')
        except Exception as e:
            logger.error(f'Error reading data from {path}')
            return None

    def data_split(self):
        try:
            # Ensure data is there and not None 
            if self.feature_matrix is None:
                logger.warning(f"Feature matrix is None")
                return None

            # Copy data from local feature_matrix  
            data_copy = self.feature_matrix.clone().sort('end_date')
        
            # Determine total markets to effectively split by instead of a single value
            total_markets = len(data_copy)
            train_cutoff = int(total_markets * 0.6)
            calib_cutoff = int(total_markets * 0.8) 

            # Slice data using determined splits 
            train = data_copy[:train_cutoff]
            calibration = data_copy[train_cutoff: calib_cutoff]
            test = data_copy[calib_cutoff:]

            # Verify 
            for name, split in [("Train", train), ('Calibration', calibration), ('Test', test)]:
                print(f"\n{name}")
                print(f" Markets: {len(split)}")
                print(f" Date range: {split['end_date'].min()} -> {split['end_date'].max()}")
                print(f" Yes rate: {split['resolved_yes'].mean():.2%}")
        
            return train, calibration, test
        
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None
        
    def get_X_y(self, df: pl.DataFrame):
        EXCLUDE_COLS = ['market_id', 'resolved_yes', 'end_date', 'price_end']
        FEATURE_COLS = [c for c in self.feature_matrix.columns if c not in EXCLUDE_COLS]

        X = df.select(FEATURE_COLS)
        y = df.select('resolved_yes')

        return X, y
    
    def train_model(self, X_train, y_train, X_test, y_test):
        params = {
            'scale_pos_weight': SCALE_POS_WEIGHT,
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'min_child_weight': 5,
            'eval_metric': 'logloss',
            'random_state': 42,
        }

        with mlflow.start_run(run_name='baseline'):
            # Log paramerts
            mlflow.log_params(params)

            # Train model
            model = XGBClassifier(**params)
            model.fit(
                X_train.to_numpy(),
                y_train.to_numpy().ravel()
            )

            # Get predictions
            y_pred_proba = model.predict_proba(X_test.to_numpy())[:, 1]
            y_pred = model.predict(X_test.to_numpy())

            # Calculate metrics
            brier = brier_score_loss(y_test.to_numpy(), y_pred_proba)
            auc = roc_auc_score(y_test.to_numpy(), y_pred_proba)
            ll = log_loss(y_test.to_numpy(), y_pred_proba)

            # Log metrics
            mlflow.log_metric('brier_score', brier)
            mlflow.log_metric('auc_roc', auc)
            mlflow.log_metric('log_loss', ll)
            
            # Log model
            mlflow.xgboost.log_model(model, 'model')

            logger.info(f"Brier: {brier:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")

        return model
        
    def hyperparameter_search(self, X_train, y_train, X_test, y_test, n_trails=100):

        def objective(trial):
            # Optuna suggests values for each parameter
            # suggest_int means pick a while number between the two values
            # suggest_float means pick a decimal between the two values
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': SCALE_POS_WEIGHT,
                'random_state': 42,
                'eval_metric': 'logloss',
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0), # L1 regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0), # L2 regularization
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0) # Feature subsampling
            }        

            # Each trail is logged as a seperate child run inside mlflow
            with mlflow.start_run(run_name=f"trail_{trial.number}", nested=True):
                mlflow.log_params(params)

                model = XGBClassifier(**params)
                model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())

                y_pred_proba = model.predict_proba(X_test.to_numpy())[:, 1]

                auc = roc_auc_score(y_test.to_numpy(), y_pred_proba)
                brier = brier_score_loss(y_test.to_numpy(), y_pred_proba)

                mlflow.log_metric('auc_roc', auc)
                mlflow.log_metric('brier_score', brier)

            # Return the alue Optuna should optimise
            # We return brier score and tell the study to minise it
            return brier
        
        # Wrap the whole search in a parent MLflow run 
        # All traials will nest inside this one run in the UI
        with mlflow.start_run(run_name='optuna_search'):
            # direction='minimize' because lower brier score is better
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trails)

        logger.info(f"Best trail: {study.best_trial.number}")
        logger.info(f"Best Brier Score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params
    
    def train_final_model(self, X_train, y_train, X_test, y_test, best_params):
        final_params = {
            **best_params,
            'scale_pos_weight': SCALE_POS_WEIGHT,
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        with mlflow.start_run(run_name='final_model'):
            mlflow.log_params(final_params)

            model = XGBClassifier(**final_params)
            model.fit(X_train.to_numpy(), y_train.to_numpy().ravel())

            y_pred_proba = model.predict_proba(X_test.to_numpy())[:, 1]
            y_pred = model.predict(X_test.to_numpy())

            brier = brier_score_loss(y_test.to_numpy(), y_pred_proba)
            auc = roc_auc_score(y_test.to_numpy(), y_pred_proba)
            ll = log_loss(y_test.to_numpy(), y_pred_proba)

            mlflow.log_metric('brier_score', brier)
            mlflow.log_metric('auc_roc', auc)
            mlflow.log_metric('log_loss', ll)

            # Register the model in the MLFlow model registery
            mlflow.xgboost.log_model(
                model,
                'final_model',
                registered_model_name='polymarket-xgboost'
            )

            logger.info(f"Final mode - Brier: {brier:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")
            
        return model
    
    def run(self):
        # 1. Split
        train, calibration, test = self.data_split()

        # 2. Save each file 
        train.write_parquet('data/model/train.parquet')
        calibration.write_parquet('data/model/calibration.parquet')
        test.write_parquet('data/model/test.parquet')

        # 3. Seperate features and targets
        X_train, y_train = self.get_X_y(train)
        X_test, y_test = self.get_X_y(test)

        # 4. Train baseline model
        model = self.train_model(X_train, y_train, X_test, y_test)

        # 5. Hyperparameter search
        best_params = self.hyperparameter_search(X_train, y_train, X_test, y_test)

        # 6. Fit final model
        final_model = self.train_final_model(X_train, y_train, X_test, y_test, best_params)

        logger.info(f"Search complete. Best params: {best_params}")

        
if __name__ == '__main__':
    model = Model()
    model.run()