import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Fama-French 5-Factor Model Analysis",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Fama-French 5-Factor Model Analysis")
st.markdown('''
This app performs stock analysis using the Fama-French 5-Factor Model to estimate expected returns.
The model includes:
- Market factor (Mkt-RF)
- Size factor (SMB - Small Minus Big)
- Value factor (HML - High Minus Low)
- Profitability factor (RMW - Robust Minus Weak)
- Momentum factor (WML - Winners Minus Losers)
''')

# Sidebar for user inputs
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS for NSE, AAPL for NASDAQ)", "RELIANCE.NS")

st.sidebar.subheader("Analysis Period")
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)  # Default to 10 years
start_date = st.sidebar.date_input("Start Date", start_date)
end_date = st.sidebar.date_input("End Date", end_date)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date")

# Function to calculate Fama-French factors (weekly)
@st.cache_data(ttl=24*3600)
def calculate_fama_french_factors(start_date, end_date):
    try:
        nifty500 = yf.download('^CRSLDX', start=start_date, end=end_date, progress=False)
        # Resample to weekly (Friday) and compute returns
        nifty500_weekly = nifty500['Close'].resample('W-FRI').last().pct_change().dropna()
        if isinstance(nifty500_weekly, pd.DataFrame):
            nifty500_weekly = nifty500_weekly.iloc[:, 0]
        dates = nifty500_weekly.index
        
        # Mock Fama-French factors (weekly)
        np.random.seed(42)
        smb = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        hml = pd.Series(np.random.normal(0.0015, 0.025, len(dates)), index=dates)
        rmw = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        wml = pd.Series(np.random.normal(0.002, 0.03, len(dates)), index=dates)
        
        ff_factors = pd.DataFrame({
            'Mkt-RF': nifty500_weekly,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'WML': wml
        })
        return ff_factors
    except Exception as e:
        st.error(f"Error calculating Fama-French factors: {e}")
        return None

# Function to get stock data (weekly)
@st.cache_data(ttl=24*3600)
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        stock_weekly = stock_data['Close'].resample('W-FRI').last().pct_change().dropna()
        if isinstance(stock_weekly, pd.DataFrame):
            stock_weekly = stock_weekly.iloc[:, 0]
        stock_weekly.name = ticker
        return stock_weekly, stock_data
    except Exception as e:
        st.error(f"Error getting stock data: {e}")
        return None, None

# Function to calculate factor betas
def calculate_factor_betas(stock_returns, ff_factors):
    try:
        aligned_data = pd.concat([stock_returns, ff_factors], axis=1).dropna()
        y = aligned_data.iloc[:, 0]
        X = sm.add_constant(aligned_data.iloc[:, 1:])
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        st.error(f"Error calculating factor betas: {e}")
        return None

# Function to calculate factor contributions
def calculate_factor_contributions(betas, avg_factors):
    contributions = {}
    for factor in avg_factors.index:
        if factor in betas:
            contributions[factor] = betas[factor] * avg_factors[factor]
    if 'const' in betas:
        contributions['Alpha'] = betas['const']
    return contributions

# Function to calculate historical annual beta factors
def calculate_annual_betas(stock_returns, ff_factors):
    aligned_data = pd.concat([stock_returns, ff_factors], axis=1).dropna()
    annual_betas = {}
    for year in sorted(aligned_data.index.year.unique()):
        data_year = aligned_data[aligned_data.index.year == year]
        if len(data_year) >= 20:  # require at least 20 weeks of data
            y = data_year.iloc[:, 0]
            X = sm.add_constant(data_year.iloc[:, 1:])
            model_year = sm.OLS(y, X).fit()
            # Store only factor betas (excluding alpha)
            annual_betas[year] = model_year.params.drop('const')
    return pd.DataFrame(annual_betas).T

# Main analysis function
def run_analysis():
    with st.spinner("Fetching data and performing analysis..."):
        ff_factors = calculate_fama_french_factors(start_date, end_date)
        if ff_factors is not None:
            stock_returns, stock_data = get_stock_data(ticker, start_date, end_date)
            if stock_returns is not None and stock_data is not None:
                # Stock Price Chart
                st.subheader(f"{ticker} Stock Price")
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data = stock_data.copy()
                    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
                if f"Close_{ticker}" in stock_data.columns:
                    close_col = f"Close_{ticker}"
                elif "Close" in stock_data.columns:
                    close_col = "Close"
                else:
                    st.error("Close column not found in stock data")
                    return
                fig_price = px.line(stock_data, y=close_col, title=f"{ticker} Stock Price")
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Factor Betas Calculation
                model = calculate_factor_betas(stock_returns, ff_factors)
                if model is not None:
                    st.subheader("Fama-French 5-Factor Model Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Factor Betas")
                        betas = model.params
                        betas_df = pd.DataFrame({
                            'Factor': betas.index,
                            'Beta': betas.values,
                            'P-Value': model.pvalues,
                            'Significant': model.pvalues < 0.05
                        })
                        betas_df['Beta'] = betas_df['Beta'].round(4)
                        betas_df['P-Value'] = betas_df['P-Value'].round(4)
                        st.dataframe(betas_df)
                        st.markdown(f"**R-squared:** {model.rsquared:.4f}")
                        st.markdown(f"**Adjusted R-squared:** {model.rsquared_adj:.4f}")
                    
                    # Expected Return, Volatility & Sharpe Ratio Calculations
                    avg_factors = ff_factors.mean()
                    expected_weekly_return = (
                        betas['const'] +
                        betas['Mkt-RF'] * avg_factors['Mkt-RF'] +
                        betas['SMB'] * avg_factors['SMB'] +
                        betas['HML'] * avg_factors['HML'] +
                        betas['RMW'] * avg_factors['RMW'] +
                        betas['WML'] * avg_factors['WML']
                    )
                    expected_annual_return = (1 + expected_weekly_return)**52 - 1
                    
                    # Expected volatility (standard deviation)
                    factor_betas = model.params.drop('const')
                    cov_factors = ff_factors.cov()
                    predicted_variance_weekly = np.dot(np.dot(factor_betas.values, cov_factors.values), factor_betas.values) + model.mse_resid
                    expected_weekly_std = np.sqrt(predicted_variance_weekly)
                    expected_annual_std = expected_weekly_std * np.sqrt(52)
                    
                    # Annualized Sharpe Ratio (assuming risk-free rate = 0)
                    sharpe_ratio = expected_annual_return / expected_annual_std if expected_annual_std != 0 else np.nan
                    
                    # Factor Contributions
                    contributions = calculate_factor_contributions(betas, avg_factors)
                    
                    with col2:
                        st.markdown("### Expected Returns, Volatility & Sharpe Ratio")
                        st.markdown(f"**Expected Weekly Return:** {expected_weekly_return:.6f} ({expected_weekly_return*100:.4f}%)")
                        st.markdown(f"**Expected Annual Return:** {expected_annual_return:.4f} ({expected_annual_return*100:.2f}%)")
                        st.markdown(f"**Expected Weekly Volatility:** {expected_weekly_std:.6f} ({expected_weekly_std*100:.4f}%)")
                        st.markdown(f"**Expected Annual Volatility:** {expected_annual_std:.4f} ({expected_annual_std*100:.2f}%)")
                        st.markdown(f"**Annualized Sharpe Ratio:** {sharpe_ratio:.4f}")
                    
                    # Factor Contribution Table & Stacked Bar Chart
                    st.subheader("Factor Contributions to Expected Return")
                    contributions_df = pd.DataFrame({
                        'Factor': list(contributions.keys()),
                        'Contribution': list(contributions.values())
                    })
                    total_contribution = contributions_df['Contribution'].sum()
                    contributions_df['Percentage'] = contributions_df['Contribution'] / total_contribution * 100
                    contributions_df['Contribution'] = contributions_df['Contribution'].round(4)
                    contributions_df['Percentage'] = contributions_df['Percentage'].round(2)
                    st.dataframe(contributions_df)
                    
                    # Stacked Bar Chart: Create a dummy column for stacking
                    contributions_df["Overall"] = "Expected Return"
                    fig_stack = px.bar(contributions_df, 
                                       x="Contribution", 
                                       y="Overall", 
                                       color="Factor", 
                                       orientation="h",
                                       title="Stacked Contribution to Expected Return")
                    st.plotly_chart(fig_stack, use_container_width=True)
                    
                    # Historical Annual Beta Factors
                    annual_betas_df = calculate_annual_betas(stock_returns, ff_factors)
                    if not annual_betas_df.empty:
                        st.subheader("Historical Annual Beta Factors")
                        st.dataframe(annual_betas_df.round(4))
                        fig_annual = px.line(annual_betas_df, 
                                             x=annual_betas_df.index, 
                                             y=annual_betas_df.columns,
                                             labels={'value': 'Beta', 'variable': 'Factor', 'index': 'Year'},
                                             title="Historical Annual Beta Factors")
                        st.plotly_chart(fig_annual, use_container_width=True)
                    
                    # Other visualizations (Rolling Betas, Correlation Matrix, Factor Returns, Cumulative Returns)
                    st.subheader("Rolling Factor Betas (52-Week Window)")
                    rolling_window = min(52, len(stock_returns))
                    if rolling_window >= 20:
                        rolling_betas = {}
                        aligned_data = pd.concat([stock_returns, ff_factors], axis=1).dropna()
                        for i in range(rolling_window, len(aligned_data) + 1):
                            window_data = aligned_data.iloc[i - rolling_window:i]
                            y = window_data.iloc[:, 0]
                            X = sm.add_constant(window_data.iloc[:, 1:])
                            try:
                                window_model = sm.OLS(y, X).fit()
                                for factor, beta in window_model.params.items():
                                    if factor not in rolling_betas:
                                        rolling_betas[factor] = []
                                    rolling_betas[factor].append(beta)
                            except Exception:
                                pass
                        rolling_dates = aligned_data.index[rolling_window - 1:]
                        rolling_betas_df = pd.DataFrame(rolling_betas, index=rolling_dates[:len(list(rolling_betas.values())[0])])
                        fig_rolling = px.line(
                            rolling_betas_df,
                            y=rolling_betas_df.columns[1:],  # Exclude constant
                            labels={'value': 'Beta', 'variable': 'Factor'},
                            title='Rolling Factor Betas (52-Week Window)'
                        )
                        st.plotly_chart(fig_rolling, use_container_width=True)
                    
                    st.subheader("Factor Correlation Matrix")
                    corr_matrix = ff_factors.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title='Factor Correlation Matrix'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.subheader("Weekly Factor Returns")
                    factor_stats = ff_factors.describe().T
                    factor_stats['Ann_Return'] = (1 + factor_stats['mean'])**52 - 1
                    factor_stats['Ann_Volatility'] = factor_stats['std'] * np.sqrt(52)
                    factor_stats = factor_stats.round(4)
                    st.dataframe(factor_stats)
                    fig_ff = px.line(
                        ff_factors,
                        y=ff_factors.columns,
                        labels={'value': 'Return', 'variable': 'Factor'},
                        title='Weekly Factor Returns'
                    )
                    st.plotly_chart(fig_ff, use_container_width=True)
                    
                    st.subheader("Cumulative Factor Returns")
                    cumulative_returns = (1 + ff_factors).cumprod()
                    fig_cum = px.line(
                        cumulative_returns,
                        y=cumulative_returns.columns,
                        labels={'value': 'Cumulative Return', 'variable': 'Factor'},
                        title='Cumulative Factor Returns'
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                    
                    with st.expander("View Detailed Model Results"):
                        st.text(model.summary())
                        st.markdown("### Model Formula")
                        formula_parts = []
                        for factor, beta in betas.items():
                            if factor == 'const':
                                formula_parts.append(f"{beta:.4f}")
                            else:
                                formula_parts.append(f"{beta:.4f} × {factor}")
                        formula = " + ".join(formula_parts)
                        st.markdown(f"**Expected Return = {formula}**")
                        st.markdown("### Annualization Formula")
                        st.markdown("**Annual Return = (1 + Weekly Return)^52 - 1**")
                        
                # Display Assumptions at the End
                st.markdown("---")
                st.markdown("### Assumptions")
                st.markdown("""
                - **Risk-Free Rate:** Assumed to be 0%.
                - **Market Proxy:** NIFTY 500 (ticker `^CRSLDX`) is used as the market proxy.
                - **Data Frequency:** Weekly returns computed from the closing prices on Fridays.
                - **Annualization:** 52 weeks per year is assumed for annualizing returns and volatility.
                - **Fama-French Factors:** Factors (SMB, HML, RMW, WML) are simulated using mock data.
                - **Rolling and Annual Betas:** 
                  - Rolling betas are computed using a 52‑week moving window.
                  - Historical annual betas are derived from regressions on each calendar year (with at least 20 weeks of data).
                """)
                st.markdown("Fama-French 5-Factor Model Analysis App | Created with Streamlit")

# Run the analysis when the app loads
run_analysis()
