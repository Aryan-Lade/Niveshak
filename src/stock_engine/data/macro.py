import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RAW_MACRO_DIR = Path("data_store/raw/macro")

def download_and_cache_macro():
    RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    tickers = {
        "^NSEI": "NIFTY_50",
        "^INDIAVIX": "INDIA_VIX",
        "USDINR=X": "USD_INR",
        "^GSPC": "S&P_500",
        "CL=F": "CRUDE_OIL"
    }
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, period="max", progress=False)
            if data.empty:
                logger.warning(f"No data downloaded for {ticker}")
                continue
            data = data[["Close"]].rename(columns={"Close": name})
            data.to_parquet(RAW_MACRO_DIR / f"{name}.parquet")
            logger.info(f"Downloaded and cached {ticker} as {name}.parquet")
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_and_cache_macro()