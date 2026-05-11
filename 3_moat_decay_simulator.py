import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MoatDecaySimulator")

class MoatDecaySimulator:
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.panel = pd.DataFrame()
        self.decay_metrics = []

    def load_data(self):
        logger.info(f"Loading Excess Returns panel from {self.data_path}")
        self.panel = pd.read_csv(self.data_path)
        self.panel['Period_End'] = pd.to_datetime(self.panel['Period_End'])
        self.panel.sort_values(by=['Ticker', 'Period_End'], inplace=True)

    def _run_adf_test(self, series: pd.Series, ticker: str) -> dict:
        clean_series = series.dropna()
        
        if len(clean_series) < 12:  
            logger.warning(f"Insufficient data for ADF test on {ticker}")
            return {"ADF_Statistic": np.nan, "p_value": np.nan, "Is_Mean_Reverting": False}

        adf_result = adfuller(clean_series, autolag='AIC')

        p_value = adf_result[1]
        is_mean_reverting = p_value < 0.05
        
        return {
            "ADF_Statistic": adf_result[0],
            "p_value": p_value,
            "Is_Mean_Reverting": is_mean_reverting
        }

    def _calibrate_ornstein_uhlenbeck(self, series: pd.Series, ticker: str) -> dict:
        df = pd.DataFrame({'X': series.dropna()})
        if len(df) < 12:
            return {"Lambda": np.nan, "Half_Life_Years": np.nan}

        df['X_lag'] = df['X'].shift(1)
        df['dX'] = df['X'] - df['X_lag']
        df = df.dropna()

        X_indep = df['X_lag']
        X_indep = sm.add_constant(X_indep) # Add alpha intercept
        y_dep = df['dX']

        model = sm.OLS(y_dep, X_indep).fit()

        beta = model.params['X_lag']

        dt = 0.25

        if beta >= 0:
            moat_decay_factor = 0.0001 
            half_life_years = 99.9     
        else:
            moat_decay_factor = -beta / dt
            half_life_years = np.log(2) / moat_decay_factor

        return {
            "Beta": beta,
            "Lambda_MDF": moat_decay_factor,
            "Half_Life_Years": half_life_years
        }

    def run_simulation(self, output_path: str):
        self.load_data()
        logger.info("Initiating 10-Year Moat Decay Simulation across all cohorts...")

        for ticker, group in self.panel.groupby('Ticker'):
            excess_returns = group['Excess_Return']

            adf_stats = self._run_adf_test(excess_returns, ticker)

            ou_stats = self._calibrate_ornstein_uhlenbeck(excess_returns, ticker)

            result = {
                "Ticker": ticker,
                "Mean_Excess_Return": excess_returns.mean(),
                "ADF_p_value": adf_stats['p_value'],
                "Is_Mean_Reverting": adf_stats['Is_Mean_Reverting'],
                "Moat_Decay_Factor_Lambda": ou_stats['Lambda_MDF'],
                "Half_Life_Years": ou_stats['Half_Life_Years']
            }
            self.decay_metrics.append(result)
            
            logger.info(f"{ticker} | Mean Reverting: {result['Is_Mean_Reverting']} | "
                        f"MDF (Lambda): {result['Moat_Decay_Factor_Lambda']:.4f} | "
                        f"Half-Life: {result['Half_Life_Years']:.2f} yrs")

        results_df = pd.DataFrame(self.decay_metrics)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Simulation Complete. Decay metrics exported to {output_path}")
        
        return results_df

if __name__ == "__main__":
    simulator = MoatDecaySimulator(data_path="excess_returns_panel.csv")

    calibrated_decay_factors = simulator.run_simulation(output_path="moat_decay_calibration.csv")
    