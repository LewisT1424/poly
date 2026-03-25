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
POLYMARKET_FEE = 0.02 

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

        self.test_prices = None

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

            self.test_prices = test.select('price_end')
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


    def backtest(self):
        try:
            # Get prediction sets for all test markets
            y_pred, y_pred_sets = self.mapie_model.predict_set(
                self.X_test.to_numpy()
            )

            # Get market prices for disagreement analysis
            market_prices = self.test_prices.to_numpy().ravel()

            y_true = self.y_test.to_numpy().ravel().astype(int)
            n_samples = len(y_true)

            results = []

            for i in range(n_samples):
                market_price = float(market_prices[i])
                # Get signal for thism market using current_price=0.5 as placeholder
                # In production this would be the live market price
                signal = self.predict_signal(y_pred_sets[i], current_price=market_price)
                true_label = y_true[i]

                if signal['signal'] == 'GREEN':
                    disagreement = market_price < 0.5
                    # Green is correct if market actually resolved YES (1)
                    correct = bool(true_label == 1)
                    if correct:
                        gross = 1.0 - market_price
                        profit = gross - (gross * POLYMARKET_FEE)
                    else:
                        profit = -market_price # no fee on loses

                elif signal['signal'] == 'RED':
                    disagreement = market_price > 0.5
                    # RED is correct if market actually resolved NO (0)
                    correct = bool(true_label == 0)
                    if correct:
                        gross = market_price
                        profit = gross - (gross * POLYMARKET_FEE)
                    else:
                        profit = -(1.0 - market_price) # No fee on loses

                else:
                    disagreement = False
                    correct = None
                    profit = 0.0
                
                results.append({
                    'signal': signal['signal'],
                    'correct': correct,
                    'true_label': true_label,
                    'market_price': market_price,
                    'disagreement': disagreement,
                    'profit': profit
                })

            # Add this temporarily before pl.DataFrame(results)
            results_df = pl.DataFrame(results).with_columns(
                pl.col('correct').cast(pl.Boolean)
            )
            

            # Seperate by signal type for accuracy calculations
            green = results_df.filter(pl.col('signal') == 'GREEN')
            red = results_df.filter(pl.col('signal') == 'RED')
            actionable = results_df.filter(pl.col('signal') != 'YELLOW')

            # Calculate accuracy per signal type
            # If no signals of type exists, default to 0.0
            green_accuracy = green['correct'].mean() if len(green) > 0 else 0.0
            red_accuracy = red['correct'].mean() if len(red) > 0 else 0.0
            overall_accuracy = actionable['correct'].mean() if len(actionable) > 0 else 0.0

            # Disagreement analysis
            disagree = actionable.filter(pl.col('disagreement') == True)
            agree = actionable.filter(pl.col('disagreement') == False)

            disagree_accuracy = disagree['correct'].mean() if len(disagree) > 0 else 0.0
            agree_accuracy = agree['correct'].mean() if len(agree) > 0 else 0.0
            avg_profit = disagree['profit'].mean() if len(disagree) >0 else 0.0
            edge = float(disagree_accuracy) - float(agree_accuracy)

            # Signal coverage - what proportion of markets got an actionable signal
            signal_coverage = len(actionable) / n_samples
            

            # Log all backtest metrics to MLflow
            with mlflow.start_run(run_name='backtest', nested=True):
                mlflow.log_metric('green_accuracy', float(green_accuracy))
                mlflow.log_metric('red_accuracy', float(red_accuracy))
                mlflow.log_metric('overall_accuracy', float(overall_accuracy))
                mlflow.log_metric('signal_coverage', float(signal_coverage))
                mlflow.log_metric('n_green', len(green))
                mlflow.log_metric('n_red', len(red))
                mlflow.log_metric('n_yellow', len(results_df.filter(pl.col('signal') == 'YELLOW')))
                mlflow.log_metric('disagree_accuracy', float(disagree_accuracy))
                mlflow.log_metric('agree_accuracy', float(agree_accuracy))
                mlflow.log_metric('avg_profit', float(avg_profit))
                mlflow.log_metric('edge', float(edge))
                mlflow.log_metric('n_disagreements', len(disagree))

            # Plot signal accuracy bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            signals = ['GREEN', "RED", 'Overall']
            accuracies = [float(green_accuracy), float(red_accuracy), float(overall_accuracy)]
            colours = ['green', 'red', 'steelblue']

            bars = ax.bar(signals, accuracies, color=colours, alpha=0.7, edgecolor='black')
            ax.axhline(y=0.5, linestyle='--', color='grey', label='Random baseline (50%)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Backtest Signal Accuracy')
            ax.set_ylim(0, 1)
            ax.legend()

            # Add accuracy value lables on top of each bar
            for bar, acc in zip(bars, accuracies):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{acc:.1%}',
                    ha='center', va='bottom', fontweight='bold'
                )

            plt.tight_layout()
            plt.savefig('backtest_accuracy.png')
            plt.close()

            # Log chat to mlflow
            with mlflow.start_run(run_name='backtest', nested=True):
                mlflow.log_artifact('backtest_accuracy.png')

            # Print results
            logger.info(f"Total test markets:    {n_samples}")
            logger.info(f"GREEN signals:         {len(green)} → {float(green_accuracy):.1%} accurate")
            logger.info(f"RED signals:           {len(red)} → {float(red_accuracy):.1%} accurate")
            logger.info(f"YELLOW (uncertain):    {len(results_df.filter(pl.col('signal') == 'YELLOW'))}")
            logger.info(f"Overall accuracy:      {float(overall_accuracy):.1%}")
            logger.info(f"Signal coverage:       {float(signal_coverage):.1%}")


            # Honest interpretation
            if float(overall_accuracy) > 0.65:
                logger.info("Signal quality: STRONG — meaningfully above random baseline")
            elif float(overall_accuracy) > 0.55:
                logger.info("Signal quality: MODERATE — above random but marginal edge")
            else:
                logger.info("Signal quality: WEAK — close to random baseline, review model")

            logger.info(f"--- Disagreement Analysis ---")
            logger.info(f"Disagreement signals:  {len(disagree)}")
            logger.info(f"Disagreement accuracy: {float(disagree_accuracy):.1%}")
            logger.info(f"Agreement accuracy:    {float(agree_accuracy):.1%}")
            logger.info(f"Edge:                  {float(edge):+.1%}")
            logger.info(f"Avg profit per trade:  {float(avg_profit):.3f}")

            if float(edge) > 0.05:
                logger.info("Edge assessment: POSITIVE — model adds value beyond market price")
            elif float(edge) > 0:
                logger.info("Edge assessment: MARGINAL — small positive edge, needs more data")
            else:
                logger.info("Edge assessment: NONE — model does not beat market price alone")


            logger.info('----------------------------------------------')
            # At the end of the backtest loop, save results with price
            results_df = results_df.with_columns(
                pl.Series('market_price', market_prices)
            )

            # Filter to disagreements only
            disagreements = results_df.filter(pl.col('disagreement') == True)

            # Bin the disagreement prices
            disagreements_binned = disagreements.with_columns(
                pl.when(pl.col('market_price') < 0.1).then(pl.lit('0.0-0.1'))
                .when(pl.col('market_price') < 0.3).then(pl.lit('0.1-0.3'))
                .when(pl.col('market_price') < 0.5).then(pl.lit('0.3-0.5'))
                .when(pl.col('market_price') < 0.7).then(pl.lit('0.5-0.7'))
                .when(pl.col('market_price') < 0.9).then(pl.lit('0.7-0.9'))
                .otherwise(pl.lit('0.9-1.0'))
                .alias('price_bin')
            )

            print(disagreements_binned.group_by('price_bin').agg([
                pl.col('market_price').count().alias('count'),
                pl.col('correct').mean().alias('accuracy')
            ]).sort('price_bin'))

            return results_df

        except Exception as e:
            logger.error(f"Error happened during backtest: {e}")

    def run(self):
        with mlflow.start_run(run_name='conformal_pipeline'):
            self.calibrate()
            self.conformalize()
            self.verify_coverage()
            self.explore_alpha()
            self.backtest()

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
    cp.run()