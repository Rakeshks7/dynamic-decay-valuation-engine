import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinancialMetricCalculator")

class EconomicProfitEngine:
    
    def __init__(self, data_path: str, tax_rate: float = 0.21):
        self.data_path = data_path
        self.tax_rate = tax_rate
        self.panel = pd.DataFrame()

        self.wacc_assumptions = {
            "NVDA": 0.160,   
            "MSFT": 0.115,   
            "META": 0.120,   
            "AMZN": 0.115,   
            "GOOGL": 0.115,  
            "NOW": 0.075     
        }

    def load_data(self):
        logger.info(f"Loading engineered metrics from {self.data_path}")
        self.panel = pd.read_csv(self.data_path)
        self.panel['Period_End'] = pd.to_datetime(self.panel['Period_End'])
        self.panel.sort_values(by=['Ticker', 'Period_End'], inplace=True)

    def _capitalize_rd(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        amortization_quarters = 20 if ticker == "NOW" else 12 

        df['RD_Amortization'] = df['R&D_Expense'].rolling(
            window=amortization_quarters, min_periods=1).mean()

        df['Capitalized_RD_Asset'] = df['R&D_Expense'].rolling(
            window=amortization_quarters, min_periods=1).sum() - \
            (df['RD_Amortization'] * amortization_quarters / 2) 

        df['Capitalized_RD_Asset'] = df['Capitalized_RD_Asset'].clip(lower=0)
        return df

    def calculate_roic(self) -> pd.DataFrame:
        logger.info("Calculating Adjusted NOPAT and Invested Capital...")
        processed_groups = []
        
        for ticker, group in self.panel.groupby('Ticker'):
            group = group.copy()

            group['Operating_Income'] = group['Operating_Margin'] * group['Revenues']

            group = self._capitalize_rd(group, ticker)

            group['Adjusted_EBIT'] = group['Operating_Income'] + group['R&D_Expense'] - group['RD_Amortization']

            group['NOPAT'] = group['Adjusted_EBIT'] * (1 - self.tax_rate)

            group['Net_PPE_Proxy'] = (group['CapEx'] - group['Depreciation']).cumsum()
            group['Invested_Capital'] = group['Net_PPE_Proxy'].clip(lower=group['CapEx']) + group['Capitalized_RD_Asset']

            group['ROIC'] = (group['NOPAT'] * 4) / group['Invested_Capital']

            group['ROIC'] = group['ROIC'].replace([np.inf, -np.inf], np.nan)
            
            processed_groups.append(group)
            
        self.panel = pd.concat(processed_groups, ignore_index=True)
        return self.panel

    def calculate_excess_return(self) -> pd.DataFrame:
        logger.info("Computing Excess Returns (ROIC - WACC) using risk-adjusted WACC...")

        self.panel['WACC'] = self.panel['Ticker'].map(self.wacc_assumptions)

        self.panel['Excess_Return'] = self.panel['ROIC'] - self.panel['WACC']

        final_df = self.panel.dropna(subset=['Excess_Return']).copy()

        final_df['ROIC'] = final_df['ROIC'].round(4)
        final_df['Excess_Return'] = final_df['Excess_Return'].round(4)
        
        return final_df

    def run_calculator(self, output_path: str):
        self.load_data()
        self.calculate_roic()
        final_metrics = self.calculate_excess_return()

        export_df = final_metrics[[
            'Ticker', 'Period_End', 'NOPAT', 'Invested_Capital', 
            'ROIC', 'WACC', 'Excess_Return'
        ]]
        
        export_df.to_csv(output_path, index=False)
        logger.info(f"Financial Metric Calculation complete. Excess Returns saved to {output_path}")
        return export_df

if __name__ == "__main__":
    calculator = EconomicProfitEngine(data_path="engineered_metrics_panel.csv")

    economic_profit_panel = calculator.run_calculator(output_path="excess_returns_panel.csv")
    print("Metrics calculated. Dataset is primed for the Ornstein-Uhlenbeck Mean Reversion Simulation.")