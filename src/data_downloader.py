"""
Project MIDAS v2 — Data Downloader
Downloads 5 years of XAUUSD data (M1, M3, M5, M15) from MT5.
Run this on your laptop with MT5 open and logged in.

Usage:
    python data_downloader.py           # Download all timeframes
    python data_downloader.py --tf M5   # Download specific timeframe
    python data_downloader.py --verify  # Verify existing data
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.")
    print("Run: pip install MetaTrader5")
    sys.exit(1)

# Import config — adjust path if running from different directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MT5_SYMBOL, DATA_RAW, DATA_START_YEAR, DATA_END_YEAR,
    MT5_TIMEFRAMES, get_mt5_timeframe
)


def initialize_mt5():
    """Initialize MT5 connection."""
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.terminal_info()
    print(f"MT5 connected: {info.name}")
    print(f"Build: {info.build}")

    # Check symbol is available
    symbol_info = mt5.symbol_info(MT5_SYMBOL)
    if symbol_info is None:
        print(f"Symbol {MT5_SYMBOL} not found. Make sure it's visible in Market Watch.")
        mt5.shutdown()
        sys.exit(1)

    if not symbol_info.visible:
        mt5.symbol_select(MT5_SYMBOL, True)

    print(f"Symbol: {MT5_SYMBOL} | Spread: {symbol_info.spread} | Point: {symbol_info.point}")
    return True


def download_timeframe(tf_name: str, chunk_months: int = 6):
    """
    Download data for a single timeframe in chunks.
    MT5 has limits on how much data you can request at once,
    so we chunk by months and concatenate.

    Args:
        tf_name: Timeframe name (M1, M3, M5, M15)
        chunk_months: Months per chunk (smaller = more reliable)
    """
    tf_mt5 = get_mt5_timeframe(tf_name)
    if tf_mt5 is None:
        print(f"Invalid timeframe: {tf_name}")
        return None

    start_date = datetime(DATA_START_YEAR, 1, 1)
    end_date = datetime.now()

    print(f"\n{'='*60}")
    print(f"Downloading {MT5_SYMBOL} {tf_name}")
    print(f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    all_data = []
    current_start = start_date
    chunk_num = 0

    while current_start < end_date:
        chunk_end = current_start + timedelta(days=chunk_months * 30)
        if chunk_end > end_date:
            chunk_end = end_date

        # Fetch rates
        rates = mt5.copy_rates_range(MT5_SYMBOL, tf_mt5, current_start, chunk_end)

        if rates is not None and len(rates) > 0:
            df_chunk = pd.DataFrame(rates)
            all_data.append(df_chunk)
            chunk_num += 1
            print(f"  Chunk {chunk_num}: {current_start.strftime('%Y-%m')} to "
                  f"{chunk_end.strftime('%Y-%m')} — {len(rates):,} candles")
        else:
            print(f"  Chunk {chunk_num + 1}: {current_start.strftime('%Y-%m')} to "
                  f"{chunk_end.strftime('%Y-%m')} — NO DATA (may be weekend/holiday gap)")

        current_start = chunk_end

    if not all_data:
        print(f"ERROR: No data downloaded for {tf_name}")
        return None

    # Concatenate and clean
    df = pd.concat(all_data, ignore_index=True)

    # Convert time column (MT5 returns Unix timestamp)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Drop duplicates (chunk overlaps)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Rename columns to standard format
    df = df.rename(columns={
        "time": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume",  # MT5 uses tick_volume for forex/metals
        "spread": "spread",
        "real_volume": "real_volume",
    })

    # Keep only essential columns
    cols_to_keep = ["datetime", "open", "high", "low", "close", "volume", "spread"]
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available]

    # Save to CSV
    output_file = DATA_RAW / f"XAUUSD_{tf_name}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n  SAVED: {output_file}")
    print(f"  Total candles: {len(df):,}")
    print(f"  Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    return df


def verify_data():
    """Verify all downloaded data files exist and print summary."""
    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    all_good = True

    for tf_name in MT5_TIMEFRAMES:
        filepath = DATA_RAW / f"XAUUSD_{tf_name}.csv"

        if not filepath.exists():
            print(f"  {tf_name}: MISSING ❌")
            all_good = False
            continue

        df = pd.read_csv(filepath, parse_dates=["datetime"])
        duration = df["datetime"].iloc[-1] - df["datetime"].iloc[0]
        years = duration.days / 365.25

        # Check for gaps (more than expected interval between candles)
        expected_minutes = MT5_TIMEFRAMES[tf_name]
        time_diffs = df["datetime"].diff().dt.total_seconds() / 60
        # Filter out weekend gaps (>48 hours) and just count trading gaps
        trading_gaps = time_diffs[(time_diffs > expected_minutes * 3) & (time_diffs < 48 * 60)]

        print(f"  {tf_name}: {len(df):>10,} candles | "
              f"{df['datetime'].iloc[0].strftime('%Y-%m-%d')} to "
              f"{df['datetime'].iloc[-1].strftime('%Y-%m-%d')} | "
              f"{years:.1f} years | "
              f"Gaps (>3x interval): {len(trading_gaps)} ✅")

        # Sanity checks
        if len(df) < 1000:
            print(f"    ⚠️  WARNING: Very few candles for {tf_name}")
            all_good = False

        # Check for price anomalies
        price_range = df["close"].max() - df["close"].min()
        if df["close"].min() < 500 or df["close"].max() > 5000:
            print(f"    ⚠️  WARNING: Unusual price range: "
                  f"${df['close'].min():.2f} - ${df['close'].max():.2f}")

        # Check for zero/null values
        null_count = df[["open", "high", "low", "close"]].isnull().sum().sum()
        zero_count = (df[["open", "high", "low", "close"]] == 0).sum().sum()
        if null_count > 0 or zero_count > 0:
            print(f"    ⚠️  WARNING: {null_count} nulls, {zero_count} zeros found")
            all_good = False

    print()
    if all_good:
        print("✅ All data files look good!")
    else:
        print("⚠️  Some issues found. Review warnings above.")

    return all_good


def main():
    parser = argparse.ArgumentParser(description="MIDAS v2 Data Downloader")
    parser.add_argument("--tf", type=str, help="Download specific timeframe (M1/M3/M5/M15)")
    parser.add_argument("--verify", action="store_true", help="Verify existing data only")
    args = parser.parse_args()

    if args.verify:
        verify_data()
        return

    # Initialize MT5
    initialize_mt5()

    try:
        if args.tf:
            # Download single timeframe
            if args.tf not in MT5_TIMEFRAMES:
                print(f"Invalid timeframe: {args.tf}. Choose from: {list(MT5_TIMEFRAMES.keys())}")
                return
            download_timeframe(args.tf)
        else:
            # Download all timeframes
            print(f"\nDownloading all timeframes for {MT5_SYMBOL}...")
            print(f"Timeframes: {list(MT5_TIMEFRAMES.keys())}")
            print(f"This may take a few minutes for M1 data...\n")

            for tf_name in MT5_TIMEFRAMES:
                download_timeframe(tf_name)

        # Verify after download
        verify_data()

    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()