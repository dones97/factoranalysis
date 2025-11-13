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
        Calculate operating profitability (Operating Income / Revenue).
        This is the operating margin, a key profitability metric.

        Returns:
            Dictionary mapping ticker to profitability metric
        """
        print("Calculating profitability metrics (Operating Margin)...")
        profitability = {}

        for ticker in self.constituents:
            try:
                ticker_obj = yf.Ticker(ticker)

                # Get financial statements
                financials = ticker_obj.financials

                if not financials.empty:
                    # Get most recent data
                    op_income = None
                    revenue = None

                    # Try to get Operating Income
                    if 'Operating Income' in financials.index:
                        op_income = financials.loc['Operating Income'].iloc[0]
                    elif 'EBIT' in financials.index:
                        op_income = financials.loc['EBIT'].iloc[0]

                    # Try to get Revenue (multiple possible names)
                    for revenue_name in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                        if revenue_name in financials.index:
                            revenue = financials.loc[revenue_name].iloc[0]
                            break

                    # Calculate operating margin
                    if revenue is not None and revenue > 0 and op_income is not None:
                        profitability[ticker] = op_income / revenue
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

    def get_asset_growth(self) -> Dict[str, float]:
        """
        Calculate asset growth rate (change in total assets year-over-year).
        Used for CMA (Conservative Minus Aggressive) factor.

        Returns:
            Dictionary mapping ticker to asset growth rate
        """
        print("Calculating asset growth rates...")
        asset_growth = {}

        for ticker in self.constituents:
            try:
                ticker_obj = yf.Ticker(ticker)

                # Get balance sheet data
                balance_sheet = ticker_obj.balance_sheet

                if not balance_sheet.empty and len(balance_sheet.columns) >= 2:
                    # Get total assets for most recent two periods
                    total_assets_current = None
                    total_assets_prev = None

                    # Try different possible names for Total Assets
                    for asset_name in ['Total Assets', 'TotalAssets']:
                        if asset_name in balance_sheet.index:
                            total_assets_current = balance_sheet.loc[asset_name].iloc[0]
                            total_assets_prev = balance_sheet.loc[asset_name].iloc[1]
                            break

                    # Calculate asset growth rate
                    if (total_assets_current is not None and
                        total_assets_prev is not None and
                        total_assets_prev > 0):
                        asset_growth[ticker] = (total_assets_current / total_assets_prev) - 1
            except:
                pass

        print(f"Calculated asset growth for {len(asset_growth)} stocks")
        return asset_growth

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

    def calculate_cma(self) -> pd.Series:
        """
        Calculate CMA (Conservative Minus Aggressive) factor.
        Sorts stocks by asset growth and creates long-short portfolio.
        Conservative = low asset growth, Aggressive = high asset growth.

        Returns:
            Series of CMA factor returns
        """
        print("\nCalculating CMA factor...")

        asset_growth = self.get_asset_growth()

        if len(asset_growth) < 10:
            print("Insufficient asset growth data for CMA")
            return pd.Series(dtype=float)

        # Sort by asset growth
        sorted_stocks = sorted(asset_growth.items(), key=lambda x: x[1])

        # Conservative (low growth) = bottom 30%, Aggressive (high growth) = top 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        conservative_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        aggressive_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Conservative (low growth) portfolio: {len(conservative_stocks)} stocks")
        print(f"Aggressive (high growth) portfolio: {len(aggressive_stocks)} stocks")

        # Calculate returns
        conservative_returns = self.calculate_portfolio_returns(conservative_stocks)
        aggressive_returns = self.calculate_portfolio_returns(aggressive_stocks)

        # Align indices
        cma = conservative_returns.subtract(aggressive_returns, fill_value=0)

        print(f"CMA factor calculated: {len(cma)} periods, mean={cma.mean():.4f}, std={cma.std():.4f}")
        return cma

    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all Fama-French factors and market return.

        Returns:
            DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, WML
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
        hml = self.cal
