from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score

import numpy as np
import polars as pl
import logging
import mlflow
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score 

EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]

class CalibratedModel:
    '''
    Manual Platt scaling swapper to replaced CalibratedClassifierCV which removed the prefit support
    '''
    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt = platt_scaler
        self.classes_ = base_model.classes_
    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        calibrated_yes = self.platt.predict_proba(raw)[:, 1]
        calibrated_no = 1 - calibrated_yes
        return np.column_stack([calibrated_no, calibrated_yes])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_experiment('polymarket-conformal')

class Conformal:
    def __init__(self):
        # Initalise Models 
        self.xgb_model = None
        self.calib_model = None
        self.mapie_model = None 
        
        # Initalise Datasets
        self.train = None
        self.calibration = None
        self.test = None 
        self.feature_cols = None

        # Slits
        self.X_train = None 
        self.y_train = None 

        self.X_calib = None
        self.y_calib = None
        
        self.X_test = None
        self.y_test = None

        # Load all data 
        self._load()

        logger.info("Successfully initalised Conformal Class")

    def _load(self):
        try:
            # Step 1 - Load XGBoost model from MLflow registery
            self.xgb_model = mlflow.xgboost.load_model('models:/polymarket-xgboost/5')

            logger.info("Successfully loaded model")

            # Step 2 - Load Train, Calibration, Test tables 
            train = pl.read_parquet('data/model/train.parquet')
            calibration = pl.read_parquet('data/model/calibration.parquet')
            test = pl.read_parquet('data/model/test.parquet')

            self.feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]

            # Create splits and return X, y matrix's
            self.X_train, self.y_train = self._get_X_y(train)
            self.X_calib, self.y_calib = self._get_X_y(calibration) 
            self.X_test, self.y_test = self._get_X_y(test)

            logger.info(f"Successfully loaded model datasets and split into train, calibration, test splits")
        except Exception as e:
            logger.error(e)
            return None

    def _get_X_y(self, df: pl.DataFrame):
        X = df.select(self.feature_cols)
        y = df.select('resolved_yes')
        return X, y

    def calibrate(self):
        try:
            # before calibration
            y_prob_before = self.xgb_model.predict_proba(
                self.X_test.to_numpy()
            )[:, 1]

            frac_pos_before, mean_pred_before = calibration_curve(
                self.y_test.to_numpy().ravel(),
                y_prob_before,
                n_bins=10
            )

            brier_score = brier_score_loss(
                self.y_test.to_numpy().ravel(),
                y_prob_before
            )

            # Get raw XGBoost probabilities on calibration set
            raw_calib_probs = self.xgb_model.predict_proba(
                self.X_calib.to_numpy()
            )[:, 1].reshape(-1, 1)

            # Fit sigmoid correction on top - manual Platt scaling
            platt = LogisticRegression()
            platt.fit(raw_calib_probs, self.y_calib.to_numpy().ravel())

            # Wrap into sklearn-compatible model
            self.calib_model = CalibratedModel(self.xgb_model, platt)

            # After calibration, predict probability
            y_prob_after = self.calib_model.predict_proba(
                self.X_test.to_numpy()
            )[:, 1]

            frac_pos_after, mean_pred_after = calibration_curve(
                self.y_test.to_numpy().ravel(),
                y_prob_after,
                n_bins=10
            )

            brier_after = brier_score_loss(
                self.y_test.to_numpy().ravel(),
                y_prob_after
            )

            # Plot both graphs to see comparison
            fix, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, frac, mean, title, brier in [
                (axes[0], frac_pos_before, mean_pred_before,
                 f'Before Calibration\nBrier: {brier_score:.4f}', brier_score),
                (axes[1], frac_pos_after, mean_pred_after,
                 f'After Calibration\nBrier: {brier_after:.4f}', brier_after) 
            ]:
                ax.plot(mean, frac, marker='o', label='Model')
                ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
                ax.set_xlabel('Mean predicted probability')
                ax.set_ylabel('Fraction of positives')
                ax.set_title(title)
                ax.legend()
            
            plt.tight_layout()
            plt.savefig('calibration_comparison.png')
            plt.close()

            # Log to MLflow
            with mlflow.start_run(run_name='calibration', nested=True):
                mlflow.log_metric('brier_before', brier_score)
                mlflow.log_metric('brier_after', brier_after)
                mlflow.log_metric('brier_improvement', brier_score - brier_after)
                mlflow.log_artifact('calibration_comparison.png')

            logger.info(
                f"Calibration complete - "
                f"Brier before: {brier_score:.4f} |"
                f"Brier after: {brier_after:.4f}"
            )

        except Exception as e:
            logger.error(F"Error in calibrate: {e}")

    def conformalize(self):
        try:
            self.mapie_model = SplitConformalClassifier(
                estimator=self.calib_model,
                confidence_level=0.9,
                conformity_score='lac',
                prefit=True
            )

            self.mapie_model.conformalize(
                self.X_calib.to_numpy(),
                self.y_calib.to_numpy().ravel()
            )

            logger.info("Successfully conformalized model")

        except Exception as e:
            logger.error(f"Error in conformalize: {e}")

    def verify_coverage(self):
        try:
            # Get prediction sets on test set
            y_pred, y_pred_sets = self.mapie_model.predict_set(
                self.X_test.to_numpy()
            )

            y_true = self.y_test.to_numpy().ravel().astype(int)

            # y_pred_sets shape: (n_samples, n_classes)
            # For each market, chekc if true lable is in prediction set
            n_samples = len(y_true)

            covered = 0
            n_yes_only = 0
            n_no_only = 0
            n_uncertain = 0

            for i in range(n_samples):
                true_label = y_true[i]
                pred_set = y_pred_sets[i]

                # Classify signal type
                yes_in_set = bool(pred_set[1][0])
                no_in_set = bool(pred_set[0][0])


                # Check coverage
                if bool(y_pred_sets[i][true_label][0]):
                    covered += 1

                if yes_in_set and not no_in_set:
                    n_yes_only += 1
                elif no_in_set and not yes_in_set:
                    n_no_only += 1
                else:
                    n_uncertain += 1
            
            coverage = covered / n_samples
            ambiguity_rate = n_uncertain / n_samples
            signal_rate = (n_yes_only + n_no_only) / n_samples

            # Log in mlflow
            with mlflow.start_run(run_name='coverage_verification', nested=True):
                mlflow.log_metric('emperical_coverage', coverage)
                mlflow.log_metric('ambiguity_rate', ambiguity_rate)
                mlflow.log_metric('signal_rate', signal_rate)
                mlflow.log_metric('n_yes_signals', n_yes_only)
                mlflow.log_metric('n_no_signals', n_no_only)
                mlflow.log_metric('n_uncertain', n_uncertain)

            logger.info(f"Coverage:       {coverage:.4f} (must be >= 0.90)")
            logger.info(f"Ambiguity rate: {ambiguity_rate:.4f}")
            logger.info(f"Signal rate:    {signal_rate:.4f}")
            logger.info(f"YES signals:    {n_yes_only}")
            logger.info(f"NO signals:     {n_no_only}")
            logger.info(f"Uncertain:      {n_uncertain}")

            if coverage < 0.90:
                logger.warning(
                    f"Coverage {coverage:.4f} is below 0.90 -"
                    f"check calibration set integrity"
                )
            return y_pred, y_pred_sets
        
        except Exception as e:
            logger.error(f"Error in verify_coverage: {e}")

    def explore_alpha(self):
        try:
            # Define alpha test set
            alphas = [0.05, 0.10, 0.20]
            results = []

            for alpha in alphas:
                # Determine confidence interval
                confidence_level = 1 - alpha

                # Fit mapie model with new alpha
                mapie_alpha = SplitConformalClassifier(
                    estimator=self.calib_model,
                    confidence_level=confidence_level,
                    conformity_score='lac',
                    prefit=True
                )

                mapie_alpha.conformalize(
                    self.X_calib.to_numpy(),
                    self.y_calib.to_numpy().ravel()
                )

                y_pred, y_pred_sets = mapie_alpha.predict_set(
                    self.X_test.to_numpy()
                )

                y_true = self.y_test.to_numpy().ravel().astype(int)
                n_samples = len(y_true)

                covered = int(sum(
                    bool(y_pred_sets[i][y_true[i]][0])
                    for i in range(n_samples)
                ))
                uncertain = int(sum(
                    bool(y_pred_sets[i][0][0]) and bool(y_pred_sets[i][1][0])
                    for i in range(n_samples)
                ))

                coverage = covered / n_samples
                ambiguity_rate = uncertain / n_samples
                signal_rate = 1 - ambiguity_rate

                results.append({
                    'alpha': alpha,
                    'confidence': confidence_level,
                    'coverage': coverage,
                    'ambiguity_rate': ambiguity_rate,
                    'signal_rate': signal_rate
                })

                logger.info(
                    f"Alpha {alpha} | Coverage: {float(coverage):.4f} | "
                    f"Signal rate: {float(signal_rate):.4f} | "
                    f"Ambiguity: {float(ambiguity_rate):.4f}"
                )

            # Plot coverage and signal rate vs alpha
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].plot(
                alphas,
                [r['coverage'] for r in results],
                marker='o', label='Empirical coverage'
            )

            axes[0].plot(
                alphas,
                [1 - a for a in alphas],
                linestyle='--', label='Theoretical coverage'
            )

            axes[0].set_xlabel('Alpha')
            axes[0].set_ylabel('Coverage')
            axes[0].set_title('Empirical vs Theoretical Coverage')
            axes[0].legend()

            axes[1].plot(
                alphas,
                [r['signal_rate'] for r in results],
                marker='o', label='Signal rate'
            )

            axes[1].plot(
                alphas,
                [r['ambiguity_rate'] for r in results],
                marker='o', label='Ambiguity rate'
            )

            axes[1].set_xlabel('Alpha')
            axes[1].set_ylabel('Rate')
            axes[1].set_title('Signal Rate vs Alpha')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig('alpha_exploration.png')
            plt.close()

            # Log to MLflow
            with mlflow.start_run(run_name='alpha_exploration', nested=True):
                for r in results:
                    mlflow.log_metric(
                        f"coverage_alpha_{r['alpha']}",
                        r['coverage']
                    )
                    mlflow.log_metric(
                        f"signal_rate_alpha_{r['alpha']}",
                        r['signal_rate']
                    )
                    mlflow.log_metric(
                        f"ambiguity_alpha_{r['alpha']}",
                        r['ambiguity_rate']
                    )
                mlflow.log_artifact('alpha_exploration.png')

        except Exception as e:
            logger.error(f"Error in explore_alpha: {e}")

    def predict_signal(self, prediction_set, current_price: float) -> dict:
        '''
        Takes a predition set and current market price. 
        Returns a signal dictionary with the signal type and context

        prediction_set: shape (2, 1) - [NO, YES] as True/False
        current_price: float between 0 and 1 (current YES probability)
        '''
        yes_in_set = bool(prediction_set[1][0])
        no_in_set = bool(prediction_set[0][0])

        if yes_in_set and not no_in_set:
            signal = 'GREEN'
            description = 'Model confident YES'
        elif no_in_set and not yes_in_set:
            signal = 'RED'
            description = 'Model confident NO'
        else:
            signal = 'YELLOW'
            description = 'Model uncertain'

        return {
            'signal': signal,
            'description': description,
            'current_price': current_price,
            'yes_in_set': yes_in_set,
            'no_in_set': no_in_set
        }


    def run(self):
        with mlflow.start_run(run_name='conformal_pipeline'):
            self.calibrate()
            self.conformalize()
            self.verify_coverage()
            self.explore_alpha()

            # Register MAPIE model in MLflow
            mlflow.sklearn.log_model(
                self.mapie_model,
                'mapie_model',
                registered_model_name='polymarket-mapie'
            )

            mlflow.log_param('feature_cols', self.feature_cols)
            mlflow.log_param('exclude_cols', EXCLUDE_COLS)
            mlflow.log_param('confidence_level', 0.9)
            mlflow.log_param('conformity_score', 'lac')

            logger.info("Confromal pipeline complete")
if __name__ == '__main__':
    cp = Conformal()
    cp.calibrate()
    cp.conformalize()


    # Sample markets spread across the test set
    indices = [0, 200, 500, 1000, 1500, 2000, 2500, 2800]

    y_pred, y_pred_sets = cp.mapie_model.predict_set(cp.X_test.to_numpy())
    y_true = cp.y_test.to_numpy().ravel().astype(int)

    for i in indices:
        signal = cp.predict_signal(y_pred_sets[i], current_price=0.5)
        print(f"Market {i} | True: {'YES' if y_true[i] else 'NO'} | Signal: {signal['signal']} | {signal['description']}")
