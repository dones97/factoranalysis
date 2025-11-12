"""
Factor Calculation Module
Calculates Fama-French style factors (SMB, HML, RMW, WML) from constituent stock data.
This follows the academic methodology for creating factor portfolios.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class FactorCalculator:
    """
    Calculates Fama-French style factors from stock data.
    """

    def __init__(self, constituents: List[str], start_date: str, end_date: str):
        """
        Initialize the factor calculator.

        Args:
            constituents: List of ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.constituents = constituents
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.prices = None
        self.fundamentals = {}

    def download_price_data(self, batch_size: int = 50, delay: float = 1.0) -> pd.DataFrame:
        """
        Download price data for all constituents with rate limiting.

        Args:
            batch_size: Number of stocks to download in each batch
            delay: Delay in seconds between batches

        Returns:
            DataFrame with weekly closing prices for all stocks
        """
        print(f"Downloading price data for {len(self.constituents)} stocks...")

        all_prices = {}
        failed_tickers = []

        # Process in batches to avoid rate limiting
        for i in range(0, len(self.constituents), batch_size):
            batch = self.constituents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.constituents)-1)//batch_size + 1}")

            for ticker in batch:
                try:
                    data = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    if not data.empty:
                        # Handle MultiIndex columns from yfinance
                        if isinstance(data.columns, pd.MultiIndex):
                            # Extract Close price column
                            close_data = data['Close']
                            if isinstance(close_data, pd.DataFrame):
                                close_data = close_data.iloc[:, 0]
                        else:
                            close_data = data['Close']

                        # Resample to weekly (Friday)
                        weekly = close_data.resample('W-FRI').last()
                        all_prices[ticker] = weekly
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    print(f"Failed to download {ticker}: {str(e)[:50]}")
                    failed_tickers.append(ticker)

            # Rate limiting delay between batches
            if i + batch_size < len(self.constituents):
                time.sleep(delay)

        if failed_tickers:
            print(f"Failed to download {len(failed_tickers)} stocks: {failed_tickers[:10]}...")

        # Create DataFrame with all prices
        if all_prices:
            self.prices = pd.DataFrame(all_prices)
            print(f"Successfully downloaded data for {len(all_prices)} stocks")
        else:
            print("ERROR: No price data could be downloaded for any stocks")
            self.prices = pd.DataFrame()

        return self.prices

    def get_market_caps(self) -> Dict[str, float]:
        """
        Get current market capitalization for all stocks.

        Returns:
            Dictionary mapping ticker to market cap
        """
        print("Fetching market capitalization data...")
        market_caps = {}

        for ticker in self.constituents:
            try:
                info = yf.Ticker(ticker).info
                mc = info.get('marketCap', None)
                if mc and mc > 0:
                    market_caps[ticker] = mc
            except:
                pass

        print(f"Retrieved market caps for {len(market_caps)} stocks")
        return market_caps

    def get_book_to_market(self) -> Dict[str, float]:
        """
        Calculate book-to-market ratios for all stocks.

        Returns:
            Dictionary mapping ticker to B/M ratio
        """
        print("Calculating book-to-market ratios...")
        bm_ratios = {}

        for ticker in self.constituents:
            try:
                info = yf.Ticker(ticker).info
                book_value = info.get('bookValue', 0)
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

                if book_value > 0 and current_price > 0:
                    bm_ratios[ticker] = book_value / current_price
            except:
                pass

        print(f"Calculated B/M ratios for {len(bm_ratios)} stocks")
        return bm_ratios

    def get_profitability(self) -> Dict[str, float]:
        """
        Calculate operating profitability (Operating Income / Book Equity).

        Returns:
            Dictionary mapping ticker to profitability metric
        """
        print("Calculating profitability metrics...")
        profitability = {}

        for ticker in self.constituents:
            try:
                ticker_obj = yf.Ticker(ticker)

                # Get financial statements
                financials = ticker_obj.financials
                balance_sheet = ticker_obj.balance_sheet

                if not financials.empty and not balance_sheet.empty:
                    # Get most recent data
                    op_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 0

                    # Try multiple names for equity
                    equity = 0
                    for equity_name in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                        if equity_name in balance_sheet.index:
                            equity = balance_sheet.loc[equity_name].iloc[0]
                            break

                    if equity > 0 and op_income is not None:
                        profitability[ticker] = op_income / equity
            except:
                pass

        print(f"Calculated profitability for {len(profitability)} stocks")
        return profitability

    def get_momentum(self) -> Dict[str, float]:
        """
        Calculate momentum (12-month return, skipping most recent month).

        Returns:
            Dictionary mapping ticker to momentum score
        """
        print("Calculating momentum scores...")
        momentum = {}

        if self.prices is None:
            print("Price data not available. Download prices first.")
            return momentum

        for ticker in self.prices.columns:
            try:
                prices = self.prices[ticker].dropna()
                if len(prices) < 52:  # Need at least 1 year of weekly data
                    continue

                # Return from t-52 weeks to t-4 weeks (skip most recent month)
                momentum[ticker] = (prices.iloc[-5] / prices.iloc[-52]) - 1
            except:
                pass

        print(f"Calculated momentum for {len(momentum)} stocks")
        return momentum

    def calculate_portfolio_returns(self, stock_list: List[str],
                                   weights: str = 'equal') -> pd.Series:
        """
        Calculate returns for a portfolio of stocks.

        Args:
            stock_list: List of ticker symbols in the portfolio
            weights: 'equal' for equal-weighted or 'value' for value-weighted

        Returns:
            Series of portfolio returns
        """
        if self.prices is None:
            raise ValueError("Price data not available")

        # Get prices for stocks in the portfolio
        portfolio_prices = self.prices[[s for s in stock_list if s in self.prices.columns]]

        if portfolio_prices.empty:
            return pd.Series(dtype=float)

        # Calculate returns for each stock
        returns = portfolio_prices.pct_change()

        if weights == 'equal':
            # Equal-weighted average
            portfolio_returns = returns.mean(axis=1)
        else:
            # For now, use equal-weighted (value-weighting requires market cap time series)
            portfolio_returns = returns.mean(axis=1)

        return portfolio_returns.dropna()

    def calculate_smb(self) -> pd.Series:
        """
        Calculate SMB (Small Minus Big) factor.
        Sorts stocks by market cap and creates long-short portfolio.

        Returns:
            Series of SMB factor returns
        """
        print("\nCalculating SMB factor...")

        market_caps = self.get_market_caps()

        if len(market_caps) < 10:
            print("Insufficient market cap data for SMB")
            return pd.Series(dtype=float)

        # Sort by market cap
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1])

        # Split into Small (bottom 50%) and Big (top 50%)
        mid_point = len(sorted_stocks) // 2
        small_stocks = [s[0] for s in sorted_stocks[:mid_point]]
        big_stocks = [s[0] for s in sorted_stocks[mid_point:]]

        print(f"Small portfolio: {len(small_stocks)} stocks")
        print(f"Big portfolio: {len(big_stocks)} stocks")

        # Calculate returns
        small_returns = self.calculate_portfolio_returns(small_stocks)
        big_returns = self.calculate_portfolio_returns(big_stocks)

        # Align indices
        smb = small_returns.subtract(big_returns, fill_value=0)

        print(f"SMB factor calculated: {len(smb)} periods, mean={smb.mean():.4f}, std={smb.std():.4f}")
        return smb

    def calculate_hml(self) -> pd.Series:
        """
        Calculate HML (High Minus Low) factor.
        Sorts stocks by book-to-market and creates long-short portfolio.

        Returns:
            Series of HML factor returns
        """
        print("\nCalculating HML factor...")

        bm_ratios = self.get_book_to_market()

        if len(bm_ratios) < 10:
            print("Insufficient B/M data for HML")
            return pd.Series(dtype=float)

        # Sort by B/M ratio
        sorted_stocks = sorted(bm_ratios.items(), key=lambda x: x[1])

        # High B/M (Value) = top 30%, Low B/M (Growth) = bottom 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        low_bm_stocks = [s[0] for s in sorted_stocks[:cutoff]]  # Growth
        high_bm_stocks = [s[0] for s in sorted_stocks[-cutoff:]]  # Value

        print(f"Value (High B/M) portfolio: {len(high_bm_stocks)} stocks")
        print(f"Growth (Low B/M) portfolio: {len(low_bm_stocks)} stocks")

        # Calculate returns
        high_returns = self.calculate_portfolio_returns(high_bm_stocks)
        low_returns = self.calculate_portfolio_returns(low_bm_stocks)

        # Align indices
        hml = high_returns.subtract(low_returns, fill_value=0)

        print(f"HML factor calculated: {len(hml)} periods, mean={hml.mean():.4f}, std={hml.std():.4f}")
        return hml

    def calculate_rmw(self) -> pd.Series:
        """
        Calculate RMW (Robust Minus Weak) factor.
        Sorts stocks by profitability and creates long-short portfolio.

        Returns:
            Series of RMW factor returns
        """
        print("\nCalculating RMW factor...")

        profitability = self.get_profitability()

        if len(profitability) < 10:
            print("Insufficient profitability data for RMW")
            return pd.Series(dtype=float)

        # Sort by profitability
        sorted_stocks = sorted(profitability.items(), key=lambda x: x[1])

        # Robust (high profit) = top 30%, Weak (low profit) = bottom 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        weak_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        robust_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Robust (high profit) portfolio: {len(robust_stocks)} stocks")
        print(f"Weak (low profit) portfolio: {len(weak_stocks)} stocks")

        # Calculate returns
        robust_returns = self.calculate_portfolio_returns(robust_stocks)
        weak_returns = self.calculate_portfolio_returns(weak_stocks)

        # Align indices
        rmw = robust_returns.subtract(weak_returns, fill_value=0)

        print(f"RMW factor calculated: {len(rmw)} periods, mean={rmw.mean():.4f}, std={rmw.std():.4f}")
        return rmw

    def calculate_wml(self) -> pd.Series:
        """
        Calculate WML (Winners Minus Losers) factor.
        Sorts stocks by past returns and creates long-short portfolio.

        Returns:
            Series of WML factor returns
        """
        print("\nCalculating WML factor...")

        momentum = self.get_momentum()

        if len(momentum) < 10:
            print("Insufficient momentum data for WML")
            return pd.Series(dtype=float)

        # Sort by momentum
        sorted_stocks = sorted(momentum.items(), key=lambda x: x[1])

        # Winners (high momentum) = top 30%, Losers (low momentum) = bottom 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        loser_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        winner_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Winners portfolio: {len(winner_stocks)} stocks")
        print(f"Losers portfolio: {len(loser_stocks)} stocks")

        # Calculate returns
        winner_returns = self.calculate_portfolio_returns(winner_stocks)
        loser_returns = self.calculate_portfolio_returns(loser_stocks)

        # Align indices
        wml = winner_returns.subtract(loser_returns, fill_value=0)

        print(f"WML factor calculated: {len(wml)} periods, mean={wml.mean():.4f}, std={wml.std():.4f}")
        return wml

    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all Fama-French factors and market return.

        Returns:
            DataFrame with columns: Mkt-RF, SMB, HML, RMW, WML
        """
        print("\n" + "="*60)
        print("CALCULATING FAMA-FRENCH FACTORS")
        print("="*60)

        # Download price data first
        self.download_price_data()

        # Get market return
        print("\nCalculating market return...")
        market_data = yf.download('^NSEI', start=self.start_date, end=self.end_date,
                                 progress=False, auto_adjust=False)

        # Handle MultiIndex columns
        if isinstance(market_data.columns, pd.MultiIndex):
            market_close = market_data['Close'].iloc[:, 0] if isinstance(market_data['Close'], pd.DataFrame) else market_data['Close']
        else:
            market_close = market_data['Close']

        market_return = market_close.resample('W-FRI').last().pct_change().dropna()
        print(f"Market return calculated: {len(market_return)} periods")

        # Calculate each factor
        smb = self.calculate_smb()
        hml = self.calculate_hml()
        rmw = self.calculate_rmw()
        wml = self.calculate_wml()

        # Combine into single DataFrame
        factors = pd.DataFrame({
            'Mkt-RF': market_return,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'WML': wml
        })

        # Fill missing values with 0 (for periods where factor couldn't be calculated)
        factors = factors.fillna(0)

        print("\n" + "="*60)
        print("FACTOR CALCULATION COMPLETE")
        print("="*60)
        print(f"\nTotal periods: {len(factors)}")
        print(f"Date range: {factors.index[0]} to {factors.index[-1]}")
        print("\nFactor Statistics (annualized):")
        print(factors.mean() * 52)
        print("\nFactor Volatility (annualized):")
        print(factors.std() * np.sqrt(52))

        return factors


def load_nifty_500_constituents(file_path: str = None) -> List[str]:
    """
    Load NIFTY 500 constituent tickers from file.

    Args:
        file_path: Path to CSV file with constituents

    Returns:
        List of ticker symbols
    """
    if file_path and pd.io.common.file_exists(file_path):
        df = pd.read_csv(file_path)
        # Assume column is named 'Ticker' or 'Symbol'
        col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        tickers = df[col].tolist()
        print(f"Loaded {len(tickers)} constituents from {file_path}")
        return tickers
    else:
        # Fallback: Use a subset of major stocks if constituent file not available
        print("Constituent file not found. Using default stock list...")
        return get_default_stock_list()


def get_default_stock_list() -> List[str]:
    """
    Returns a default list of major Indian stocks as fallback.
    """
    # Top 50 NIFTY stocks as a reasonable proxy
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS',
        'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
        'WIPRO.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'TECHM.NS', 'POWERGRID.NS',
        'NTPC.NS', 'TATASTEEL.NS', 'ONGC.NS', 'ADANIPORTS.NS', 'INDUSINDBK.NS',
        'COALINDIA.NS', 'TATAMOTORS.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'CIPLA.NS',
        'DRREDDY.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'HINDALCO.NS',
        'SHREECEM.NS', 'UPL.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS',
        'APOLLOHOSP.NS', 'ADANIENT.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'TATACONSUM.NS'
    ]


if __name__ == "__main__":
    # Example usage
    constituents = get_default_stock_list()

    # Calculate factors for last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    calculator = FactorCalculator(
        constituents=constituents,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    factors = calculator.calculate_all_factors()

    # Save to file
    output_file = '../data/ff_factors.parquet'
    factors.to_parquet(output_file)
    print(f"\nFactors saved to {output_file}")
