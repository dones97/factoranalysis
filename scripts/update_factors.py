"""
Update Factors Script
Runs monthly to update pre-calculated Fama-French factors.
Designed to be executed by GitHub Actions.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add scripts directory to path to import calculate_factors
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from calculate_factors import FactorCalculator, load_nifty_500_constituents, get_default_stock_list
import pandas as pd


def update_factors():
    """
    Main function to update factors.
    This will be called by GitHub Actions monthly.
    """
    print("="*60)
    print("FACTOR UPDATE SCRIPT")
    print(f"Run date: {datetime.now()}")
    print("="*60)

    # Determine paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    constituents_file = data_dir / 'nifty_500_constituents.csv'
    output_file = data_dir / 'ff_factors.parquet'

    # Load constituents
    if constituents_file.exists():
        print(f"\nLoading constituents from {constituents_file}")
        constituents = load_nifty_500_constituents(str(constituents_file))
    else:
        print(f"\nConstituents file not found: {constituents_file}")
        print("Using default NIFTY 50 stock list")
        constituents = get_default_stock_list()

    # Calculate factors for last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    print(f"\nCalculating factors from {start_date.date()} to {end_date.date()}")

    try:
        calculator = FactorCalculator(
            constituents=constituents,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        factors = calculator.calculate_all_factors()

        # Save to parquet format (efficient for time series)
        print(f"\nSaving factors to {output_file}")
        factors.to_parquet(output_file, compression='snappy')

        # Also save as CSV for easy inspection
        csv_file = data_dir / 'ff_factors.csv'
        print(f"Saving factors to {csv_file}")
        factors.to_csv(csv_file)

        print("\n" + "="*60)
        print("UPDATE COMPLETE")
        print("="*60)
        print(f"\nFactor data saved successfully!")
        print(f"Records: {len(factors)}")
        print(f"Date range: {factors.index[0]} to {factors.index[-1]}")

        # Print summary statistics
        print("\nFactor Summary (Annualized):")
        print("\nMean Returns:")
        print((factors.mean() * 52 * 100).round(2))
        print("\nVolatility:")
        print((factors.std() * (52**0.5) * 100).round(2))

        return True

    except Exception as e:
        print(f"\nERROR: Factor calculation failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = update_factors()
    sys.exit(0 if success else 1)
