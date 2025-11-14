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

    def __init__(self, constituents: List[str], start_date: str, end_date: str, filter_by_data_availability: bool = True):
        """
        Initialize the factor calculator.

        Args:
            constituents: List of ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            filter_by_data_availability: If True, pre-filter stocks to only those with fundamental data
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.prices = None
        self.fundamentals = {}

        # Filter constituents by data availability if requested
        if filter_by_data_availability and len(constituents) > 200:
            print(f"\n{'='*60}")
            print(f"PRE-FILTERING STOCKS FOR DATA AVAILABILITY")
            print(f"{'='*60}")
            print(f"Initial stock count: {len(constituents)}")
            self.constituents = self.filter_stocks_with_fundamentals(constituents)
            print(f"Filtered stock count: {len(self.constituents)}")
            print(f"{'='*60}\n")
        else:
            self.constituents = constituents

    def filter_stocks_with_fundamentals(self, tickers: List[str], sample_size: int = 100, batch_size: int = 20) -> List[str]:
        """
        Pre-filter stocks to identify which ones have fundamental data available.
        Tests a sample first, then applies filtering based on success rate.

        Args:
            tickers: List of ticker symbols to filter
            sample_size: Number of stocks to test initially
            batch_size: Batch size for testing

        Returns:
            List of tickers that have fundamental data available
        """
        import time

        print(f"Testing data availability for up to {sample_size} stocks...")
        valid_stocks = []
        test_count = min(sample_size, len(tickers))

        for i in range(0, test_count, batch_size):
            batch = tickers[i:i+batch_size]
            print(f"  Testing batch {i//batch_size + 1}/{(test_count-1)//batch_size + 1} ({len(batch)} stocks)...")

            for ticker in batch:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info

                    # Check if stock has the fundamental data we need
                    has_book_value = info.get('bookValue', 0) > 0
                    has_price = info.get('currentPrice', info.get('regularMarketPrice', 0)) > 0

                    # Try to get financial data
                    has_financials = False
                    try:
                        financials = ticker_obj.financials
                        has_financials = not financials.empty
                    except:
                        pass

                    # If stock has at least book value and price, it's usable
                    if has_book_value and has_price:
                        valid_stocks.append(ticker)
                except Exception as e:
                    # Skip stocks that error out
                    pass

            # Small delay between batches
            if i + batch_size < test_count:
                time.sleep(1)

        success_rate = len(valid_stocks) / test_count if test_count > 0 else 0
        print(f"\nData availability test results:")
        print(f"  Tested: {test_count} stocks")
        print(f"  Valid: {len(valid_stocks)} stocks")
        print(f"  Success rate: {success_rate*100:.1f}%")

        # If success rate is very low (<5%), return the valid stocks we found
        # If success rate is decent (>5%), we can expand to all stocks
        if success_rate < 0.05:
            print(f"\n  WARNING: Low success rate ({success_rate*100:.1f}%)")
            print(f"  Using only the {len(valid_stocks)} stocks with confirmed data")
            return valid_stocks
        elif success_rate > 0.20:
            print(f"\n  Good success rate! Testing all {len(tickers)} stocks...")
            # Test all stocks since we have good coverage
            return self._test_all_stocks(tickers, batch_size)
        else:
            # Moderate success - use the tested ones
            print(f"\n  Moderate success rate. Using {len(valid_stocks)} validated stocks.")
            return valid_stocks

    def _test_all_stocks(self, tickers: List[str], batch_size: int = 20) -> List[str]:
        """Helper method to test all stocks when success rate is good."""
        import time

        valid_stocks = []
        total = len(tickers)

        for i in range(0, total, batch_size):
            batch = tickers[i:i+batch_size]
            if i % (batch_size * 10) == 0:  # Progress every 10 batches
                print(f"  Tested {i}/{total} stocks... ({len(valid_stocks)} valid so far)")

            for ticker in batch:
                try:
                    info = yf.Ticker(ticker).info
                    if (info.get('bookValue', 0) > 0 and
                        info.get('currentPrice', info.get('regularMarketPrice', 0)) > 0):
                        valid_stocks.append(ticker)
                except:
                    pass

            if i + batch_size < total:
                time.sleep(1)

        print(f"  Final: {len(valid_stocks)}/{total} stocks have fundamental data")
        return valid_stocks

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
        import time
        print("Calculating book-to-market ratios...")
        bm_ratios = {}
        batch_size = 50
        total = len(self.constituents)

        for i in range(0, total, batch_size):
            batch = self.constituents[i:i+batch_size]
            print(f"  Processing B/M batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} stocks)...")

            for ticker in batch:
                try:
                    info = yf.Ticker(ticker).info
                    book_value = info.get('bookValue', 0)
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

                    if book_value > 0 and current_price > 0:
                        bm_ratios[ticker] = book_value / current_price
                except Exception as e:
                    # Silently skip errors but could log them
                    pass

            # Rate limiting: pause between batches
            if i + batch_size < total:
                time.sleep(2)  # 2 second delay between batches

        print(f"Calculated B/M ratios for {len(bm_ratios)} stocks")
        return bm_ratios

    def get_profitability(self) -> Dict[str, float]:
        """
        Calculate operating profitability (Operating Income / Revenue).
        This is the operating margin, a key profitability metric.

        Returns:
            Dictionary mapping ticker to profitability metric
        """
        import time
        print("Calculating profitability metrics (Operating Margin)...")
        profitability = {}
        batch_size = 50
        total = len(self.constituents)

        for i in range(0, total, batch_size):
            batch = self.constituents[i:i+batch_size]
            print(f"  Processing profitability batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} stocks)...")

            for ticker in batch:
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
                except Exception as e:
                    # Silently skip errors
                    pass

            # Rate limiting: pause between batches
            if i + batch_size < total:
                time.sleep(2)  # 2 second delay between batches

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
        import time
        print("Calculating asset growth rates...")
        asset_growth = {}
        batch_size = 50
        total = len(self.constituents)

        for i in range(0, total, batch_size):
            batch = self.constituents[i:i+batch_size]
            print(f"  Processing asset growth batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} stocks)...")

            for ticker in batch:
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
                except Exception as e:
                    # Silently skip errors
                    pass

            # Rate limiting: pause between batches
            if i + batch_size < total:
                time.sleep(2)  # 2 second delay between batches

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

    def get_revenue_growth(self) -> Dict[str, float]:
        """
        Get revenue growth rates from yfinance .info property.
        Used for RGR (Revenue Growth Rate) factor.

        Returns:
            Dictionary mapping ticker to revenue growth rate
        """
        print("Fetching revenue growth data...")
        revenue_growth = {}

        for ticker in self.constituents:
            try:
                info = yf.Ticker(ticker).info
                growth = info.get('revenueGrowth')

                if growth is not None and not np.isnan(growth):
                    revenue_growth[ticker] = growth
            except Exception as e:
                # Silently skip errors
                pass

        print(f"Retrieved revenue growth for {len(revenue_growth)} stocks")
        return revenue_growth

    def calculate_rgr(self) -> pd.Series:
        """
        Calculate RGR (Revenue Growth Rate) factor.
        High revenue growth - Low revenue growth.

        Returns:
            Series of RGR factor returns
        """
        print("\nCalculating RGR factor...")
        revenue_growth = self.get_revenue_growth()

        if len(revenue_growth) < 10:
            print("Insufficient revenue growth data for RGR")
            return pd.Series(dtype=float)

        # Sort by revenue growth
        sorted_stocks = sorted(revenue_growth.items(), key=lambda x: x[1])

        # High growth = top 30%, Low growth = bottom 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        low_growth_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        high_growth_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Low growth portfolio: {len(low_growth_stocks)} stocks")
        print(f"High growth portfolio: {len(high_growth_stocks)} stocks")

        # Calculate returns
        low_growth_returns = self.calculate_portfolio_returns(low_growth_stocks)
        high_growth_returns = self.calculate_portfolio_returns(high_growth_stocks)

        # RGR = High growth - Low growth
        rgr = high_growth_returns.subtract(low_growth_returns, fill_value=0)

        print(f"RGR factor calculated: {len(rgr)} periods, mean={rgr.mean():.4f}, std={rgr.std():.4f}")
        return rgr

    def get_quality_scores(self) -> Dict[str, float]:
        """
        Calculate quality scores combining profitability and safety metrics.
        Modified QMJ that excludes growth (to avoid overlap with RGR).

        Returns:
            Dictionary mapping ticker to quality score
        """
        print("Calculating quality scores (profitability + safety)...")
        quality_data = []

        for ticker in self.constituents:
            try:
                info = yf.Ticker(ticker).info

                # Profitability metrics (60% weight)
                roe = info.get('returnOnEquity')
                roa = info.get('returnOnAssets')
                profit_margin = info.get('profitMargins')

                # Safety metric (40% weight)
                debt_to_equity = info.get('debtToEquity')

                # Only include if we have at least profitability metrics
                if roe is not None and roa is not None and profit_margin is not None:
                    quality_data.append({
                        'ticker': ticker,
                        'roe': roe if not np.isnan(roe) else 0,
                        'roa': roa if not np.isnan(roa) else 0,
                        'profit_margin': profit_margin if not np.isnan(profit_margin) else 0,
                        'safety': 1 / (1 + debt_to_equity) if (debt_to_equity is not None and not np.isnan(debt_to_equity)) else 0.5  # Neutral if missing
                    })
            except Exception as e:
                # Silently skip errors
                pass

        if not quality_data:
            print("No quality data available")
            return {}

        # Convert to DataFrame for easier percentile ranking
        df = pd.DataFrame(quality_data)

        # Calculate percentile ranks for each component (0 to 1 scale)
        df['roe_rank'] = df['roe'].rank(pct=True)
        df['roa_rank'] = df['roa'].rank(pct=True)
        df['profit_margin_rank'] = df['profit_margin'].rank(pct=True)
        df['safety_rank'] = df['safety'].rank(pct=True)

        # Composite quality score
        # Profitability: 60% (20% each for ROE, ROA, Profit Margin)
        # Safety: 40%
        df['quality_score'] = (
            0.20 * df['roe_rank'] +
            0.20 * df['roa_rank'] +
            0.20 * df['profit_margin_rank'] +
            0.40 * df['safety_rank']
        )

        quality_scores = dict(zip(df['ticker'], df['quality_score']))
        print(f"Calculated quality scores for {len(quality_scores)} stocks")
        return quality_scores

    def calculate_qmj(self) -> pd.Series:
        """
        Calculate QMJ (Quality Minus Junk) factor.
        Modified version focusing on profitability + safety (excludes growth).

        Returns:
            Series of QMJ factor returns
        """
        print("\nCalculating QMJ factor...")
        quality_scores = self.get_quality_scores()

        if len(quality_scores) < 10:
            print("Insufficient quality data for QMJ")
            return pd.Series(dtype=float)

        # Sort by quality score
        sorted_stocks = sorted(quality_scores.items(), key=lambda x: x[1])

        # Quality (high score) = top 30%, Junk (low score) = bottom 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        junk_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        quality_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Junk (low quality) portfolio: {len(junk_stocks)} stocks")
        print(f"Quality (high quality) portfolio: {len(quality_stocks)} stocks")

        # Calculate returns
        junk_returns = self.calculate_portfolio_returns(junk_stocks)
        quality_returns = self.calculate_portfolio_returns(quality_stocks)

        # QMJ = Quality - Junk
        qmj = quality_returns.subtract(junk_returns, fill_value=0)

        print(f"QMJ factor calculated: {len(qmj)} periods, mean={qmj.mean():.4f}, std={qmj.std():.4f}")
        return qmj

    def get_liquidity_scores(self) -> Dict[str, float]:
        """
        Calculate Amihud illiquidity measure for each stock.
        Illiquidity = Average(|Return| / Dollar Volume) over rolling window.

        Returns:
            Dictionary mapping ticker to illiquidity score (higher = less liquid)
        """
        print("Calculating liquidity scores (Amihud measure)...")
        liquidity_scores = {}

        if self.prices is None:
            print("Price data not available for liquidity calculation")
            return {}

        for ticker in self.constituents:
            if ticker not in self.prices.columns:
                continue

            try:
                # Get price and volume data
                ticker_prices = self.prices[ticker]

                # Calculate returns
                returns = ticker_prices.pct_change().dropna()

                # Get volume (assuming it's in the DataFrame - we'll need to handle this)
                # For now, we'll use a simplified turnover approach with market cap
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap', 0)
                avg_volume = info.get('averageVolume', 0)

                if market_cap > 0 and avg_volume > 0:
                    # Calculate average price for dollar volume
                    avg_price = ticker_prices.tail(60).mean()

                    # Simplified Amihud measure
                    # (In practice, we'd use daily |return| / dollar_volume)
                    # Here we use inverse turnover as a proxy
                    avg_dollar_volume = avg_volume * avg_price
                    turnover = avg_dollar_volume / market_cap if market_cap > 0 else 0

                    # Illiquidity = 1 / turnover (higher = less liquid)
                    if turnover > 0:
                        liquidity_scores[ticker] = 1 / turnover
                    else:
                        liquidity_scores[ticker] = 999999  # Very illiquid

            except Exception as e:
                # Silently skip errors
                pass

        print(f"Calculated liquidity scores for {len(liquidity_scores)} stocks")
        return liquidity_scores

    def calculate_liq(self) -> pd.Series:
        """
        Calculate LIQ (Liquidity) factor.
        Illiquid (low liquidity) - Liquid (high liquidity).

        Returns:
            Series of LIQ factor returns
        """
        print("\nCalculating LIQ factor...")
        liquidity_scores = self.get_liquidity_scores()

        if len(liquidity_scores) < 10:
            print("Insufficient liquidity data for LIQ")
            return pd.Series(dtype=float)

        # Sort by illiquidity score (higher = less liquid)
        sorted_stocks = sorted(liquidity_scores.items(), key=lambda x: x[1])

        # Liquid (low illiquidity) = bottom 30%, Illiquid (high illiquidity) = top 30%
        cutoff = int(len(sorted_stocks) * 0.3)
        liquid_stocks = [s[0] for s in sorted_stocks[:cutoff]]
        illiquid_stocks = [s[0] for s in sorted_stocks[-cutoff:]]

        print(f"Liquid (low illiquidity) portfolio: {len(liquid_stocks)} stocks")
        print(f"Illiquid (high illiquidity) portfolio: {len(illiquid_stocks)} stocks")

        # Calculate returns
        liquid_returns = self.calculate_portfolio_returns(liquid_stocks)
        illiquid_returns = self.calculate_portfolio_returns(illiquid_stocks)

        # LIQ = Illiquid - Liquid (expect positive if illiquidity earns premium)
        liq = illiquid_returns.subtract(liquid_returns, fill_value=0)

        print(f"LIQ factor calculated: {len(liq)} periods, mean={liq.mean():.4f}, std={liq.std():.4f}")
        return liq

    def calculate_all_factors(self) -> pd.DataFrame:
        """
        Calculate all Fama-French factors and market return.

        Returns:
            DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, WML, RGR, QMJ, LIQ
        """
        print("\n" + "="*60)
        print("CALCULATING FAMA-FRENCH FACTORS (9-FACTOR MODEL)")
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

        # Calculate original 6 factors
        smb = self.calculate_smb()
        hml = self.calculate_hml()
        rmw = self.calculate_rmw()
        cma = self.calculate_cma()
        wml = self.calculate_wml()

        # Calculate new 3 factors
        rgr = self.calculate_rgr()
        qmj = self.calculate_qmj()
        liq = self.calculate_liq()

        # Combine into single DataFrame
        factors = pd.DataFrame({
            'Mkt-RF': market_return,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma,
            'WML': wml,
            'RGR': rgr,
            'QMJ': qmj,
            'LIQ': liq
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
