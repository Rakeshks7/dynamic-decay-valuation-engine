import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataAggregationEngine")

class FinancialAggregationEngine:
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.panel = pd.DataFrame()

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)

        df['Period_End'] = pd.to_datetime(df['Period_End'])

        financial_cols = [
            'PaymentsToAcquirePropertyPlantAndEquipment', 
            'DepreciationAndAmortization', 
            'ResearchAndDevelopmentExpense', 
            'OperatingIncomeLoss', 
            'Revenues'
        ]
        
        for col in financial_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        self.panel = df
        return self.panel

    def stitch_financial_timeline(self) -> pd.DataFrame:
        logger.info("Stitching financial timelines across thousands of filings...")
        df = self.panel.copy()

        df = df.sort_values(by=['Ticker', 'Period_End'])

        stitched_data = []
        for ticker, group in df.groupby('Ticker'):
            group = group.set_index('Period_End')

            group = group[~group.index.duplicated(keep='last')]

            stitched_group = group.resample('QE').apply(lambda x: x.iloc[-1] if not x.empty else np.nan)

            if 'Ticker' in stitched_group.columns:
                stitched_group = stitched_group.drop(columns=['Ticker'])
            stitched_group['Ticker'] = ticker
            
            stitched_data.append(stitched_group.reset_index())
            
        self.panel = pd.concat(stitched_data, ignore_index=True)
        return self.panel

    def extract_key_metrics(self) -> pd.DataFrame:
        logger.info("Extracting Key Metrics: CapEx, Depreciation, and Operating Margins...")
        df = self.panel.copy()

        df['CapEx'] = df['PaymentsToAcquirePropertyPlantAndEquipment'].fillna(0).abs()

        df['Depreciation'] = df['DepreciationAndAmortization'].fillna(0).abs()

        df['Revenues'] = df['Revenues'].replace(0, np.nan)
        df['Operating_Margin'] = df['OperatingIncomeLoss'] / df['Revenues']

        df['R&D_Expense'] = df['ResearchAndDevelopmentExpense'].fillna(0)

        df['Operating_Margin'] = df['Operating_Margin'].replace([np.inf, -np.inf], np.nan).round(4)

        engineered_panel = df[[
            'Ticker', 
            'Period_End', 
            'Revenues', 
            'Operating_Margin', 
            'CapEx', 
            'Depreciation', 
            'R&D_Expense'
        ]]
        
        self.panel = engineered_panel
        return self.panel

    def run_pipeline(self, output_path: str):
        self.load_and_prepare_data()
        self.stitch_financial_timeline()
        final_panel = self.extract_key_metrics()

        final_panel = final_panel.dropna(subset=['Revenues', 'CapEx', 'Depreciation'], how='all')
        
        final_panel.to_csv(output_path, index=False)
        logger.info(f"Aggregation complete. Engineered panel saved to {output_path}")
        return final_panel

if __name__ == "__main__":
    engine = FinancialAggregationEngine(raw_data_path="xbrl_sector_panel_2016_2026.csv")

    processed_financials = engine.run_pipeline(output_path="engineered_metrics_panel.csv")
    print("Data Processing Complete. Panel is ready for ROIC/WACC spread calculation.")