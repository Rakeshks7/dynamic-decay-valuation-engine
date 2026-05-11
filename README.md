# The Dynamic Decay Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SSRN Working Paper](https://img.shields.io/badge/SSRN-Read_The_Paper-darkred.svg)](INSERT_YOUR_SSRN_LINK_HERE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production_Ready-success.svg)]()

This repository contains the core Python architecture for the **Dynamic Decay DCF Model**, an institutional quantitative framework designed to re-engineer Terminal Value calculations for hyper-capital-intensive technology firms. 

This engine was developed as the empirical backbone for the manuscript: *The Accelerated Decay: Re-engineering Terminal Value in Hyper-Capital-Intensive Technology*.

## Theoretical Framework
Standard Discounted Cash Flow (DCF) models rely on industrial-era assumptions of persistent physical capital and stable perpetuity (e.g., the Gordon Growth Model). When applied to modern AI infrastructure and semiconductor hyperscalers, these models fail to account for the thermodynamics of silicon decay and rapid frontier obsolescence. 

This engine corrects the "Terminal Value Trap" by:
1. **Capitalizing R&D** on truncated, hyper-accelerated amortization schedules.
2. **Reclassifying Disguised Maintenance CapEx** to unmask true Free Cash Flow.
3. **Calibrating a Moat Decay Factor (λ)** using the Ornstein-Uhlenbeck stochastic process to force exponential decay on terminal growth rates.

## Core Architecture
The pipeline is divided into three distinct modules:

* `1_sec_ingestion_pipeline.py`: Programmatically scrapes and parses historical eXtensible Business Reporting Language (XBRL) 10-K and 10-Q filings from the SEC EDGAR database.
* `2_financial_aggregation_engine.py`: Stitches multi-year XBRL data, isolates CapEx/D&A mismatches, and calculates true Economic Profit (ROIC - WACC).
* `3_moat_decay_simulator.py`: Ingests historical excess returns, applies Augmented Dickey-Fuller (ADF) testing for stationarity, and utilizes an Ornstein-Uhlenbeck simulation to calculate the entropic half-life of a firm's competitive advantage.

## Citation
If you utilize this quantitative framework or data pipeline in your own research or portfolio modeling, please cite the foundational paper:

Rakesh K. (2026). The Accelerated Decay: Re-engineering Terminal Value in Hyper-Capital-Intensive Technology. Social Science Research Network (SSRN). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6731365

## Disclaimer

For Educational and Research Purposes Only.
The code and financial models provided in this repository do not constitute financial advice, investment recommendations, or an offer to buy or sell any securities. Quantitative simulations are based on historical data and theoretical thermodynamic frameworks; past performance and historical mean-reversion metrics are not indicative of future results. The author assumes no liability for any financial losses or damages incurred from the application of this code to live trading environments or institutional capital allocation.