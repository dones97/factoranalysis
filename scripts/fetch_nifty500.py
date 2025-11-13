"""
Fetch NIFTY 500 constituent list from NSE
This script downloads the latest NIFTY 500 constituents and saves them to CSV
"""

import pandas as pd
import requests
from io import StringIO
import time

def fetch_nifty500_from_nse():
    """
    Fetch NIFTY 500 constituents from NSE India website
    """
    # Try multiple URLs
    urls = [
        'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv',
        'https://archives.nseindia.com/content/indices/ind_nifty500list.csv',
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for url in urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                print(f"Success! Fetched {len(df)} stocks")
                return df
        except Exception as e:
            print(f"Failed: {e}")
            time.sleep(2)

    return None

def create_comprehensive_nifty500():
    """
    Create a comprehensive NIFTY 500 list with major Indian stocks
    This is a fallback if NSE fetch fails
    """
    # Combine NIFTY 50, NIFTY Next 50, NIFTY Midcap 150, and NIFTY Smallcap 250
    stocks = []

    # NIFTY 50 (Top 50 large caps)
    nifty_50 = [
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

    # NIFTY Next 50 (51-100)
    nifty_next50 = [
        'ICICIGI.NS', 'PIDILITIND.NS', 'DLF.NS', 'INDIGO.NS', 'PNB.NS',
        'HAVELLS.NS', 'VEDL.NS', 'GODREJCP.NS', 'BOSCHLTD.NS', 'ABB.NS',
        'BANKBARODA.NS', 'AMBUJACEM.NS', 'ADANIGREEN.NS', 'SRF.NS', 'SIEMENS.NS',
        'DABUR.NS', 'BERGEPAINT.NS', 'GAIL.NS', 'HINDPETRO.NS', 'TORNTPHARM.NS',
        'MARICO.NS', 'COLPAL.NS', 'ACC.NS', 'IOC.NS', 'LUPIN.NS',
        'BEL.NS', 'MOTHERSON.NS', 'ADANIPOWER.NS', 'SAIL.NS', 'TATAPOWER.NS',
        'CANBK.NS', 'ZYDUSLIFE.NS', 'BIOCON.NS', 'TRENT.NS', 'HAL.NS',
        'MCDOWELL-N.NS', 'NAUKRI.NS', 'BANDHANBNK.NS', 'ATUL.NS', 'MRF.NS',
        'BAJAJHLDNG.NS', 'CONCOR.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'LTIM.NS',
        'DMART.NS', 'CHOLAFIN.NS', 'LICHSGFIN.NS', 'MUTHOOTFIN.NS', 'PAGEIND.NS'
    ]

    # Major Midcap stocks (101-250)
    midcap_stocks = [
        'ASTRAL.NS', 'AUROPHARMA.NS', 'BALKRISIND.NS', 'BATAINDIA.NS', 'BHARATFORG.NS',
        'BHEL.NS', 'CANFINHOME.NS', 'CHAMBLFERT.NS', 'CROMPTON.NS', 'CUMMINSIND.NS',
        'DEEPAKNTR.NS', 'DIXON.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS',
        'GLENMARK.NS', 'GMRINFRA.NS', 'GNFC.NS', 'GODREJPROP.NS', 'GRANULES.NS',
        'GUJGASLTD.NS', 'HIND UNILVR.NS', 'HINDCOPPER.NS', 'HONAUT.NS', 'IDFCFIRSTB.NS',
        'IEX.NS', 'IGL.NS', 'INDHOTEL.NS', 'INDUSTOWER.NS', 'INTELLECT.NS',
        'IRCTC.NS', 'JINDALSTEL.NS', 'JKCEMENT.NS', 'JUBLFOOD.NS', 'KAJARIACER.NS',
        'KEI.NS', 'L&TFH.NS', 'LALPATHLAB.NS', 'LAURUSLABS.NS', 'MANAPPURAM.NS',
        'MFSL.NS', 'METROPOLIS.NS', 'MPHASIS.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS',
        'NMDC.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'OIL.NS', 'PETRONET.NS',
        'PFC.NS', 'PIIND.NS', 'PNB.NS', 'POLYCAB.NS', 'PVRINOX.NS',
        'RAMCOCEM.NS', 'RBLBANK.NS', 'RECLTD.NS', 'SAIL.NS', 'SBICARD.NS',
        'SCHAEFFLER.NS', 'SRF.NS', 'STAR.NS', 'SUNTV.NS', 'SUPREMEIND.NS',
        'TATACHEM.NS', 'TATACOMM.NS', 'TATAELXSI.NS', 'TIINDIA.NS', 'TORNTPOWER.NS',
        'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'UNITDSPR.NS', 'VOLTAS.NS',
        'WHIRLPOOL.NS', 'ZEEL.NS', 'ZENSARTECH.NS', 'ABCAPITAL.NS', 'ABFRL.NS',
        'AEGISLOG.NS', 'AFFLE.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'AMARAJABAT.NS',
        'AMBER ПО.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ATUL.NS',
        'AUBANK.NS', 'AUROPHARMA.NS', 'AVENDUS.NS', 'BAJAJCON.NS', 'BAJAJHLDNG.NS',
        'BANDHANBNK.NS', 'BASF.NS', 'BSOFT.NS', 'CANFINHOME.NS', 'CAPLIPOINT.NS',
        'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS',
        'CENTURYTEX.NS', 'CESC.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHOLAHLDNG.NS',
        'CIEINDIA.NS', 'CLEAN.NS', 'COFORGE.NS', 'COLPAL.NS', 'CONCORDBIO.NS',
        'COROMANDEL.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS',
        'DBCORP.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELTACORP.NS', 'DEVYANI.NS',
        'DHANI.NS', 'DHUNSERI.NS', 'DISHTV.NS', 'DIXON.NS', 'DOLLAR IND.NS',
        'EIHOTEL.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS',
        'ENGINERSIN.NS', 'EQUITAS.NS', 'ERIS.NS', 'ESABINDIA.NS', 'ESCORTS.NS'
    ]

    # Additional stocks to reach 500
    smallcap_stocks = [
        'AARTIDRUGS.NS', 'AAVAS.NS', 'ACE.NS', 'ADANIGAS.NS', 'ADANITRANS.NS',
        'AETHER.NS', 'AFFLE.NS', 'AGARIND.NS', 'AHLEAST.NS', 'AJANTPHARM.NS',
        'AKZOINDIA.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'AMARAJABAT.NS', 'AMBER.NS',
        'ANGELONE.NS', 'ANURAS.NS', 'APLAPOLLO.NS', 'APLLTD.NS', 'APOLLOPIPE.NS',
        'ARMANFIN.NS', 'ARTEMISMED.NS', 'ARVIND.NS', 'ASAHIINDIA.NS', 'ASHIANA.NS',
        'ASIANHOTNR.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAMICRO.NS', 'ATGL.NS',
        'ATUL.NS', 'AVALON.NS', 'AVANTIFEED.NS', 'AXISCADES.NS', 'BAJAJCON.NS',
        'BALAJITELB.NS', 'BALAMINES.NS', 'BALMLAWRIE.NS', 'BALRAMCHIN.NS', 'BANCOINDIA.NS',
        'BATAINDIA.NS', 'BAYERCROP.NS', 'BBL.NS', 'BEML.NS', 'BHARATGEAR.NS',
        'BHARATRAS.NS', 'BHELINFRA.NS', 'BHUSANSTL.NS', 'BIKAJI.NS', 'BIRLACORPN.NS',
        'BLAL.NS', 'BLUEDART.NS', 'BLUESTARCO.NS', 'BLS.NS', 'BBTC.NS',
        'BOROLTD.NS', 'BOROSIL.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'BSOFT.NS',
        'BSE.NS', 'CAMPUS.NS', 'CANBANK.NS', 'CAPF.NS', 'CAPLIPOINT.NS',
        'CARBORUNIV.NS', 'CARERATING.NS', 'CARTRADE.NS', 'CASTROLIND.NS', 'CCL.NS',
        'CDSL.NS', 'CEATLTD.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS',
        'CERA.NS', 'CHALET.NS', 'CHEMFAB.NS', 'CHENNPETRO.NS', 'CHOICEIN.NS',
        'CHOLAHLDNG.NS', 'CIEINDIA.NS', 'CLEAN.NS', 'CMS.NS', 'COCHINSHIP.NS',
        'COFFEEDAY.NS', 'COMPUAGE.NS', 'CONCORDBIO.NS', 'CONFIPET.NS', 'CORDS.NS',
        'COROMANDEL.NS', 'COX&KINGS.NS', 'CRANESSOFT.NS', 'CREDITACC.NS', 'CRISIL.NS',
        'CROMPTON.NS', 'CSBBANK.NS', 'CUB.NS', 'CUPID.NS', 'CYIENT.NS',
        'DABUR.NS', 'DALMIASUG.NS', 'DBCORP.NS', 'DBL.NS', 'DCAL.NS',
        'DCBBANK.NS', 'DCMSHRIRAM.NS', 'DCWLTD.NS', 'DDL.NS', 'DEEPAKFERT.NS',
        'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DELTACORP.NS', 'DEN.NS', 'DHANUKA.NS',
        'DHFL.NS', 'DHANI.NS', 'DHUNSERI.NS', 'DIAMINES.NS', 'DICIND.NS',
        'DIGISPICE.NS', 'DLINKINDIA.NS', 'DOLLAR IND.NS', 'DREDGECORP.NS', 'DRREDDY.NS',
        'DSPBLACKROCK.NS', 'DTIL.NS', 'DUCOL.NS', 'DYNAMATECH.NS', 'EASTSILK.NS',
        'EDELWEISS.NS', 'EFCL.NS', 'EIDPARRY.NS', 'EIH.NS', 'EIHOTEL.NS',
        'ELECTCAST.NS', 'ELECTHERM.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMA MILTD.NS',
        'EMCO.NS', 'EMUDHRA.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITAS.NS',
        'ERIS.NS', 'ESABINDIA.NS', 'ESCORTS.NS', 'ETHOSLTD.NS', 'EVEREADY.NS'
    ]

    # Combine all lists
    stocks.extend(nifty_50)
    stocks.extend(nifty_next50)
    stocks.extend(midcap_stocks)
    stocks.extend(smallcap_stocks)

    # Remove duplicates while preserving order
    stocks = list(dict.fromkeys(stocks))

    return stocks

def main():
    """
    Main function to fetch and save NIFTY 500 constituents
    """
    print("Fetching NIFTY 500 constituents...")

    # Try to fetch from NSE
    df = fetch_nifty500_from_nse()

    if df is not None:
        # Add .NS suffix for yfinance
        if 'Symbol' in df.columns:
            df['Ticker'] = df['Symbol'] + '.NS'
        elif 'Company Name' in df.columns and 'Symbol' not in df.columns:
            # Extract symbol from other columns
            pass

        # Keep only Ticker and Company_Name columns
        output_df = df[['Ticker', 'Company Name']].rename(columns={'Company Name': 'Company_Name'})
    else:
        print("\nNSE fetch failed. Using comprehensive stock list...")
        stocks = create_comprehensive_nifty500()

        # Create DataFrame
        company_names = [ticker.replace('.NS', '') for ticker in stocks]
        output_df = pd.DataFrame({
            'Ticker': stocks,
            'Company_Name': company_names
        })

    # Save to CSV
    output_file = '../data/nifty_500_constituents.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(output_df)} stocks to {output_file}")
    print(f"\nFirst 10 stocks:")
    print(output_df.head(10))
    print(f"\nLast 10 stocks:")
    print(output_df.tail(10))

if __name__ == "__main__":
    main()
