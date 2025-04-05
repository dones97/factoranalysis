
# Fama-French Factor Analysis for Indian Stocks

This Streamlit app calculates Fama-French three-factor model betas for Indian stocks, estimates expected returns,
and visualizes factor contributions over time.

## Features

- Calculate Fama-French three factors for the Indian market (using NIFTY 500 as market proxy)
- Estimate rolling factor betas for any Indian stock
- Calculate expected returns based on factor exposures
- Visualize historical factor contributions
- Download all data for further analysis
- Display model assumptions and limitations

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run fama_french_streamlit_app_updated.py
   ```

## Usage

1. Enter a valid stock ticker (add .NS for NSE stocks, e.g., RELIANCE.NS)
2. Select the date range for analysis
3. Enter the risk-free rate (default is 6%)
4. View the results in the different tabs

## Data Sources

- Market Return: NIFTY 500 Index (^CRSLDX)
- Size Factor: IIFL vs NIFTY 500
- Value Factor: JUNIORBEES vs KOTAKBKETF
- Risk-free Rate: User input (default 6% annual)

## Notes

- The app uses a 36-month rolling window for beta estimation
- All expected returns are annualized
- The app requires an internet connection to download data from Yahoo Finance
- The app will automatically adjust to the available data if a stock has less than 10 years of history
