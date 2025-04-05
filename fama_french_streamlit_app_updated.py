
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Indian Stock Factor Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Fama-French Factor Analysis for Indian Stocks")
st.markdown("""
This app calculates Fama-French three-factor model betas for Indian stocks, estimates expected returns,
and visualizes factor contributions over time.
""")

# Sidebar for inputs
st.sidebar.header("Parameters")

# Stock selection
stock_ticker = st.sidebar.text_input("Stock Ticker (add .NS for NSE stocks)", "RELIANCE.NS")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)  # Default to 10 years

start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

# Risk-free rate input
risk_free_rate = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=6.0) / 100

@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def calculate_fama_french_factors(start_date, end_date):
    """
    Calculate Fama-French three factors for Indian market
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the three Fama-French factors
    """
    
    # Download market data (NIFTY 500 as market proxy - CHANGE #1)
    nifty500 = yf.download('^CRSLDX', start=start_date, end=end_date, progress=False)  # NIFTY 500 index
    nifty500_monthly = nifty500['Close'].resample('M').last().pct_change().squeeze()
    
    # Download size factor proxies (IIFL for larger stocks)
    iifl_data = yf.download('IIFL.NS', start=start_date, end=end_date, progress=False)
    iifl_monthly = iifl_data['Close'].resample('M').last().pct_change().squeeze()
    
    # Download value and growth proxies
    value_data = yf.download('JUNIORBEES.NS', start=start_date, end=end_date, progress=False)
    growth_data = yf.download('KOTAKBKETF.NS', start=start_date, end=end_date, progress=False)
    
    value_monthly = value_data['Close'].resample('M').last().pct_change().squeeze()
    growth_monthly = growth_data['Close'].resample('M').last().pct_change().squeeze()
    
    # Create risk-free rate series (using input annual rate as proxy)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    rf_monthly = pd.Series(data=risk_free_rate/12, index=dates)  # Converting annual rate to monthly
    
    # Create the factor dataframe
    ff_factors = pd.DataFrame({
        'Rm-Rf': nifty500_monthly - rf_monthly,
        'SMB': iifl_monthly - nifty500_monthly,
        'HML': value_monthly - growth_monthly
    })
    
    # Drop any rows with NaN values
    ff_factors = ff_factors.dropna()
    
    return ff_factors

@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def get_stock_data(ticker, start_date, end_date):
    """
    Get stock price data and calculate monthly returns
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    tuple
        Monthly stock returns and data availability information
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Check data availability (CHANGE #3)
    if stock_data.empty:
        return None, {"available": False, "message": f"No data available for {ticker}"}
    
    # Calculate the actual data range
    actual_start_date = stock_data.index[0]
    actual_end_date = stock_data.index[-1]
    
    # Calculate years of data available
    years_available = (actual_end_date - actual_start_date).days / 365.25
    
    # Create data availability info
    data_info = {
        "available": True,
        "actual_start_date": actual_start_date,
        "actual_end_date": actual_end_date,
        "years_available": years_available,
        "message": f"Data available for {years_available:.2f} years ({actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')})"
    }
    
    # Calculate monthly returns
    stock_monthly = stock_data['Close'].resample('M').last().pct_change().squeeze()
    
    return stock_monthly, data_info

def calculate_factor_betas(stock_returns, ff_factors, window=36):
    """
    Calculate rolling factor betas using regression
    
    Parameters:
    -----------
    stock_returns : pandas.Series
        Monthly stock returns
    ff_factors : pandas.DataFrame
        Fama-French factors
    window : int
        Rolling window in months
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing rolling betas and alpha
    """
    # Align data
    aligned_data = pd.concat([stock_returns, ff_factors], axis=1).dropna()
    aligned_data.columns = ['Stock'] + list(ff_factors.columns)
    
    # Initialize DataFrames for results
    rolling_betas = pd.DataFrame(index=aligned_data.index, 
                                columns=['Alpha', 'Beta_Mkt', 'Beta_SMB', 'Beta_HML', 'R2'])
    
    # Calculate rolling betas
    for i in range(window, len(aligned_data)):
        window_data = aligned_data.iloc[i-window:i]
        
        X = window_data[ff_factors.columns]
        X = sm.add_constant(X)
        y = window_data['Stock']
        
        model = sm.OLS(y, X).fit()
        
        current_date = aligned_data.index[i-1]
        rolling_betas.loc[current_date, 'Alpha'] = model.params['const']
        rolling_betas.loc[current_date, 'Beta_Mkt'] = model.params['Rm-Rf']
        rolling_betas.loc[current_date, 'Beta_SMB'] = model.params['SMB']
        rolling_betas.loc[current_date, 'Beta_HML'] = model.params['HML']
        rolling_betas.loc[current_date, 'R2'] = model.rsquared
    
    return rolling_betas.dropna()

def calculate_expected_return(betas, factor_means, rf_rate):
    """
    Calculate expected return using the Fama-French three-factor model
    
    Parameters:
    -----------
    betas : pandas.Series or dict
        Factor betas (Beta_Mkt, Beta_SMB, Beta_HML)
    factor_means : pandas.Series
        Mean factor returns
    rf_rate : float
        Risk-free rate (monthly)
        
    Returns:
    --------
    dict
        Expected return and factor contributions
    """
    # Calculate expected return components
    market_contribution = betas['Beta_Mkt'] * factor_means['Rm-Rf']
    smb_contribution = betas['Beta_SMB'] * factor_means['SMB']
    hml_contribution = betas['Beta_HML'] * factor_means['HML']
    
    # Calculate total expected return
    expected_return = rf_rate + market_contribution + smb_contribution + hml_contribution
    
    # Annualize
    annual_rf = rf_rate * 12
    annual_market = market_contribution * 12
    annual_smb = smb_contribution * 12
    annual_hml = hml_contribution * 12
    annual_expected_return = expected_return * 12
    
    return {
        'Risk-Free Rate': annual_rf,
        'Market Premium Contribution': annual_market,
        'SMB Contribution': annual_smb,
        'HML Contribution': annual_hml,
        'Total Expected Return': annual_expected_return
    }

def calculate_historical_contributions(betas, ff_factors):
    """
    Calculate historical factor contributions
    
    Parameters:
    -----------
    betas : pandas.DataFrame
        Rolling factor betas
    ff_factors : pandas.DataFrame
        Fama-French factors
        
    Returns:
    --------
    pandas.DataFrame
        Historical factor contributions
    """
    # Align data
    aligned_factors = ff_factors[ff_factors.index.isin(betas.index)]
    
    # Calculate contributions
    contributions = pd.DataFrame(index=betas.index)
    contributions['Market'] = betas['Beta_Mkt'] * aligned_factors['Rm-Rf']
    contributions['SMB'] = betas['Beta_SMB'] * aligned_factors['SMB']
    contributions['HML'] = betas['Beta_HML'] * aligned_factors['HML']
    contributions['Alpha'] = betas['Alpha']
    contributions['Total'] = contributions['Market'] + contributions['SMB'] + contributions['HML'] + contributions['Alpha']
    
    return contributions

def plot_rolling_betas(betas):
    """
    Plot rolling factor betas
    
    Parameters:
    -----------
    betas : pandas.DataFrame
        Rolling factor betas
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(betas.index, betas['Beta_Mkt'], label='Market Beta')
    ax.plot(betas.index, betas['Beta_SMB'], label='SMB Beta')
    ax.plot(betas.index, betas['Beta_HML'], label='HML Beta')
    
    ax.set_title('Rolling Factor Betas (36-Month Window)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_factor_contributions(contributions):
    """
    Plot historical factor contributions
    
    Parameters:
    -----------
    contributions : pandas.DataFrame
        Historical factor contributions
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(contributions.index, contributions['Market'], label='Market')
    ax.plot(contributions.index, contributions['SMB'], label='SMB')
    ax.plot(contributions.index, contributions['HML'], label='HML')
    ax.plot(contributions.index, contributions['Alpha'], label='Alpha')
    ax.plot(contributions.index, contributions['Total'], label='Total Return', color='black', linewidth=2)
    
    ax.set_title('Historical Factor Contributions to Monthly Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return Contribution')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_cumulative_contributions(contributions):
    """
    Plot cumulative factor contributions
    
    Parameters:
    -----------
    contributions : pandas.DataFrame
        Historical factor contributions
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Calculate cumulative returns
    cumulative = (1 + contributions).cumprod()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cumulative.index, cumulative['Market'], label='Market')
    ax.plot(cumulative.index, cumulative['SMB'], label='SMB')
    ax.plot(cumulative.index, cumulative['HML'], label='HML')
    ax.plot(cumulative.index, cumulative['Alpha'], label='Alpha')
    ax.plot(cumulative.index, cumulative['Total'], label='Total Return', color='black', linewidth=2)
    
    ax.set_title('Cumulative Factor Contributions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_contribution_breakdown(expected_return_components):
    """
    Plot expected return breakdown
    
    Parameters:
    -----------
    expected_return_components : dict
        Components of expected return
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Extract components
    components = {k: v for k, v in expected_return_components.items() if k != 'Total Expected Return'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.bar(components.keys(), components.values())
    
    # Add total expected return line
    ax.axhline(y=expected_return_components['Total Expected Return'], color='r', linestyle='-', 
               label=f"Total Expected Return: {expected_return_components['Total Expected Return']:.2%}")
    
    # Customize plot
    ax.set_title('Expected Return Breakdown (Annualized)')
    ax.set_ylabel('Return Contribution')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    return fig

# Main app logic
try:
    # Calculate Fama-French factors
    with st.spinner('Calculating Fama-French factors...'):
        ff_factors = calculate_fama_french_factors(start_date_input, end_date_input)
    
    # Get stock data
    with st.spinner(f'Downloading data for {stock_ticker}...'):
        stock_returns, data_info = get_stock_data(stock_ticker, start_date_input, end_date_input)
    
    # Check if data is available (CHANGE #3)
    if not data_info["available"]:
        st.error(data_info["message"])
        st.stop()
    else:
        # Display data availability information
        st.info(data_info["message"])
    
    # Calculate factor betas
    with st.spinner('Calculating factor betas...'):
        rolling_betas = calculate_factor_betas(stock_returns, ff_factors)
    
    # Calculate factor contributions
    with st.spinner('Calculating factor contributions...'):
        contributions = calculate_historical_contributions(rolling_betas, ff_factors)
    
    # Calculate expected returns using the most recent betas
    latest_betas = rolling_betas.iloc[-1]
    factor_means = ff_factors.mean()
    expected_return_components = calculate_expected_return(latest_betas, factor_means, risk_free_rate/12)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Factor Betas", "Expected Returns", "Historical Contributions", "Data Tables", "Assumptions"])
    
    with tab1:
        st.header("Factor Betas Analysis")
        
        # Display latest betas
        st.subheader("Latest Factor Betas")
        latest_betas_df = pd.DataFrame({
            'Beta': latest_betas[['Beta_Mkt', 'Beta_SMB', 'Beta_HML']].values
        }, index=['Market', 'SMB', 'HML'])
        st.dataframe(latest_betas_df)
        
        # Plot rolling betas
        st.subheader("Rolling Factor Betas")
        beta_fig = plot_rolling_betas(rolling_betas)
        st.pyplot(beta_fig)
        
        # Display R-squared
        st.subheader("Model Fit (R²)")
        r2_fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rolling_betas.index, rolling_betas['R2'])
        ax.set_title('Rolling R² (36-Month Window)')
        ax.set_xlabel('Date')
        ax.set_ylabel('R²')
        ax.grid(True)
        st.pyplot(r2_fig)
    
    with tab2:
        st.header("Expected Returns Analysis")
        
        # Display expected return components
        st.subheader("Expected Return Breakdown (Annualized)")
        expected_return_df = pd.DataFrame({
            'Component': list(expected_return_components.keys()),
            'Value': list(expected_return_components.values())
        })
        expected_return_df['Value'] = expected_return_df['Value'].map('{:.2%}'.format)
        st.dataframe(expected_return_df)
        
        # Plot expected return breakdown
        st.subheader("Expected Return Components")
        er_fig = plot_contribution_breakdown(expected_return_components)
        st.pyplot(er_fig)
    
    with tab3:
        st.header("Historical Factor Contributions")
        
        # Plot factor contributions
        st.subheader("Monthly Factor Contributions")
        contrib_fig = plot_factor_contributions(contributions)
        st.pyplot(contrib_fig)
        
        # Plot cumulative contributions
        st.subheader("Cumulative Factor Contributions")
        cum_contrib_fig = plot_cumulative_contributions(contributions)
        st.pyplot(cum_contrib_fig)
        
        # Display annual contribution statistics
        st.subheader("Annual Contribution Statistics")
        annual_contrib = contributions.resample('Y').sum()
        st.dataframe(annual_contrib)
    
    with tab4:
        st.header("Data Tables")
        
        # Display Fama-French factors
        st.subheader("Fama-French Factors")
        st.dataframe(ff_factors)
        
        # Display rolling betas
        st.subheader("Rolling Factor Betas")
        st.dataframe(rolling_betas)
        
        # Display factor contributions
        st.subheader("Monthly Factor Contributions")
        st.dataframe(contributions)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download Fama-French factors
            csv = ff_factors.to_csv()
            st.download_button(
                label="Download Fama-French Factors",
                data=csv,
                file_name="fama_french_factors.csv",
                mime="text/csv",
            )
        
        with col2:
            # Download rolling betas
            csv = rolling_betas.to_csv()
            st.download_button(
                label="Download Rolling Betas",
                data=csv,
                file_name="rolling_betas.csv",
                mime="text/csv",
            )
        
        with col3:
            # Download factor contributions
            csv = contributions.to_csv()
            st.download_button(
                label="Download Factor Contributions",
                data=csv,
                file_name="factor_contributions.csv",
                mime="text/csv",
            )
    
    # CHANGE #2: Add Assumptions tab
    with tab5:
        st.header("Model Assumptions")
        
        st.subheader("Fama-French Factor Proxies")
        assumptions_data = [
            ["Market Factor (Rm-Rf)", "NIFTY 500 Index (^CRSLDX) minus Risk-Free Rate", "Represents the excess return of the market over the risk-free rate"],
            ["Size Factor (SMB)", "IIFL.NS minus NIFTY 500", "Represents the return difference between small and big companies"],
            ["Value Factor (HML)", "JUNIORBEES.NS minus KOTAKBKETF.NS", "Represents the return difference between high and low book-to-market companies"],
            ["Risk-Free Rate", f"{risk_free_rate:.2%} annual rate (user input)", "Represents the return on a risk-free investment"]
        ]
        
        assumptions_df = pd.DataFrame(assumptions_data, columns=["Factor", "Proxy Used", "Description"])
        st.table(assumptions_df)
        
        st.subheader("Methodology Assumptions")
        methodology_data = [
            ["Beta Estimation", "36-month rolling window", "Assumes that factor exposures are relatively stable over 3-year periods"],
            ["Return Calculation", "Monthly returns", "Assumes that monthly returns provide sufficient granularity for the analysis"],
            ["Expected Return", "Based on historical factor means", "Assumes that historical factor returns are indicative of future returns"],
            ["Data Availability", f"{data_info['years_available']:.2f} years of data for {stock_ticker}", f"Analysis based on data from {data_info['actual_start_date'].strftime('%Y-%m-%d')} to {data_info['actual_end_date'].strftime('%Y-%m-%d')}"]
        ]
        
        methodology_df = pd.DataFrame(methodology_data, columns=["Assumption", "Implementation", "Implication"])
        st.table(methodology_df)
        
        st.subheader("Model Limitations")
        st.markdown("""
        - The Fama-French three-factor model may not capture all relevant risk factors for Indian stocks
        - The proxies used for the factors may not perfectly represent the theoretical factors
        - Historical returns may not be indicative of future returns
        - The model assumes linear relationships between factors and returns
        - The 36-month rolling window may be too short or too long for some stocks
        - The model does not account for transaction costs, taxes, or other practical considerations
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Tips for troubleshooting:")
    st.info("1. Make sure the stock ticker is valid (add .NS for NSE stocks)")
    st.info("2. Try a different date range (some data might not be available)")
    st.info("3. Check if all required data sources are accessible")
