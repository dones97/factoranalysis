# Fama-French Factor Calculation System

This system pre-calculates Fama-French style factors (SMB, HML, RMW, WML) from constituent stock data and stores them for use in the portfolio analyzer app.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (Monthly)                 │
│                                                             │
│  Trigger: 1st of every month at 2 AM UTC                   │
│  Also: Manual trigger via workflow_dispatch                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              scripts/update_factors.py                      │
│                                                             │
│  1. Load NIFTY 500 constituents                            │
│  2. Download 10 years of price data                        │
│  3. Calculate factor portfolios                            │
│  4. Save to data/ff_factors.parquet                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              data/ff_factors.parquet                        │
│                                                             │
│  Columns: Mkt-RF, SMB, HML, RMW, CMA, WML                  │
│  Index: Weekly Friday dates                                │
│  Format: Parquet (compressed time series)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│       portfolio_analyzer_0428_fixed.py (Streamlit App)     │
│                                                             │
│  fetch_ff_factors():                                        │
│  1. Try load from parquet (fast)                           │
│  2. Fallback to CSV if needed                              │
│  3. Fallback to simple calculation if files missing        │
└─────────────────────────────────────────────────────────────┘
```

## Factor Calculation Methodology

### 1. SMB (Small Minus Big)
- **Method**: Sort all stocks by market capitalization
- **Portfolios**:
  - Small: Bottom 50% by market cap
  - Big: Top 50% by market cap
- **Factor**: Equal-weighted return of Small portfolio - Equal-weighted return of Big portfolio

### 2. HML (High Minus Low / Value)
- **Method**: Sort all stocks by Book-to-Market ratio
- **Portfolios**:
  - High B/M (Value): Top 30% by B/M ratio
  - Low B/M (Growth): Bottom 30% by B/M ratio
- **Factor**: Return of Value portfolio - Return of Growth portfolio

### 3. RMW (Robust Minus Weak / Profitability)
- **Method**: Sort all stocks by Operating Margin (Operating Income / Revenue)
- **Portfolios**:
  - Robust (High Margin): Top 30% by operating margin
  - Weak (Low Margin): Bottom 30% by operating margin
- **Factor**: Return of Robust portfolio - Return of Weak portfolio
- **Note**: Operating margin is a cleaner measure of profitability efficiency

### 4. WML (Winners Minus Losers / Momentum)
- **Method**: Sort all stocks by past 12-month return (skipping most recent month)
- **Portfolios**:
  - Winners: Top 30% by momentum
  - Losers: Bottom 30% by momentum
- **Factor**: Return of Winners portfolio - Return of Losers portfolio

### 5. CMA (Conservative Minus Aggressive / Investment)
- **Method**: Sort all stocks by asset growth rate (year-over-year change in total assets)
- **Portfolios**:
  - Conservative (Low Growth): Bottom 30% by asset growth
  - Aggressive (High Growth): Top 30% by asset growth
- **Factor**: Return of Conservative portfolio - Return of Aggressive portfolio
- **Note**: Asset growth measures investment aggressiveness; low growth firms tend to have higher returns

### 6. Mkt-RF (Market Risk Premium)
- **Method**: Weekly returns of NIFTY 50 index (^NSEI)

## Files

### Core Scripts
- `scripts/calculate_factors.py`: Main factor calculation logic (FactorCalculator class)
- `scripts/update_factors.py`: Orchestration script for GitHub Actions

### Data Files
- `data/nifty_500_constituents.csv`: List of stocks to use for factor calculation
- `data/ff_factors.parquet`: Pre-calculated factor returns (efficient storage)
- `data/ff_factors.csv`: Same data in CSV format (for inspection)

### Configuration
- `.github/workflows/update_factors.yml`: GitHub Actions workflow
- `requirements.txt`: Python dependencies

## Manual Factor Update

To manually update factors locally:

```bash
# Navigate to project directory
cd mgmt638

# Install dependencies
pip install -r requirements.txt

# Run update script
python scripts/update_factors.py
```

This will:
1. Download constituent stock data
2. Calculate factors for the last 10 years
3. Save to `data/ff_factors.parquet` and `data/ff_factors.csv`

## GitHub Actions Setup

The workflow is configured to run automatically, but you can also trigger it manually:

1. Go to your GitHub repository
2. Click "Actions" tab
3. Select "Update Fama-French Factors" workflow
4. Click "Run workflow" button

## Updating Constituents

To update the list of stocks used for factor calculation:

1. Edit `data/nifty_500_constituents.csv`
2. Add/remove tickers (must be valid yfinance tickers with .NS or .BO suffix)
3. Commit and push changes
4. Next monthly run will use updated list

Example format:
```csv
Ticker,Company_Name
RELIANCE.NS,Reliance Industries Ltd
TCS.NS,Tata Consultancy Services Ltd
```

## Fallback Behavior

The Streamlit app has three levels of fallback:

1. **Primary**: Load pre-calculated factors from `data/ff_factors.parquet`
2. **Secondary**: Load from `data/ff_factors.csv` if parquet unavailable
3. **Tertiary**: Calculate simple approximations using market indices:
   - Mkt-RF: NIFTY 50 (^NSEI)
   - SMB: NIFTY Midcap 50 (^NSEMDCP50) - NIFTY 50
   - HML: NIFTY Value 20 (NV20.NS) - NIFTY 50
   - RMW/CMA/WML: Alpha 50 (ALPHA.NS) with scaling

## Performance Considerations

- **Factor calculation time**: ~5-10 minutes for 50 stocks, ~30-60 minutes for 500 stocks
- **API rate limits**: yfinance may throttle; script includes delays between batches
- **File size**: Parquet file is ~50-100 KB for 10 years of weekly data
- **App load time**: <1 second when using pre-calculated factors

## Monitoring

Check GitHub Actions tab for:
- Workflow run status
- Calculation logs
- Error messages

The workflow will:
- ✅ Commit changes if factor data updated successfully
- ⚠️ Skip commit if no changes detected
- ❌ Fail if calculation encounters errors

## Troubleshooting

### Issue: Factors not updating
- Check GitHub Actions logs for errors
- Verify `nifty_500_constituents.csv` has valid tickers
- Check yfinance API availability

### Issue: App shows "fallback calculation" warning
- Verify `data/ff_factors.parquet` exists in repository
- Check file is being committed by GitHub Actions
- Ensure Streamlit can access the data folder

### Issue: Calculation takes too long
- Reduce number of constituents in CSV file
- Increase delay between batches in `calculate_factors.py`
- Use a smaller stock universe (e.g., NIFTY 50 instead of NIFTY 500)

## Future Enhancements

Potential improvements:
1. Add historical constituent data (avoid survivorship bias)
2. Implement value-weighted portfolio returns
3. Add more factors (BAB, Quality, etc.)
4. Store factor characteristics (correlations, volatilities)
5. Add data quality checks and validation
6. Implement incremental updates (only calculate new periods)

## References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
