"""Daily preprocessor for raw BTC/USDT data."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)


class DailyPreprocessor:
    """Processes daily raw data into structured format for feature engineering."""

    def __init__(
        self,
        bucket_name: str = "btc-orderbook-data",
        project_id: str = "my-project-779482",
        credentials_path: str | None = None,
        raw_prefix: str = "raw",
        processed_prefix: str = "processed",
        local_work_dir: str = "./data/preprocessed",
    ):
        """Initialize daily preprocessor.

        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account JSON file
            raw_prefix: Prefix for raw data in GCS
            processed_prefix: Prefix for processed data in GCS
            local_work_dir: Local directory for temporary processing
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.raw_prefix = raw_prefix.rstrip("/")
        self.processed_prefix = processed_prefix.rstrip("/")
        self.local_work_dir = Path(local_work_dir)

        # Create local work directory
        self.local_work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client
        if credentials_path:
            import os

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Initialized GCS client for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise

    def _download_blob(self, blob_name: str, local_path: Path) -> None:
        """Download a blob from GCS to local file."""
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        logger.debug(f"Downloaded {blob_name} to {local_path}")

    def _upload_blob(self, local_path: Path, blob_name: str) -> None:
        """Upload a local file to GCS."""
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")

    def _list_blobs_for_date(self, date: datetime) -> list[str]:
        """List all raw data blobs for a specific date."""
        date_str = date.strftime("%Y/%m/%d")
        prefix = f"{self.raw_prefix}/{date_str}/"

        blobs = list(self.bucket.list_blobs(prefix=prefix))
        blob_names = [blob.name for blob in blobs]

        logger.info(f"Found {len(blob_names)} raw files for {date_str}")
        return blob_names

    def _parse_orderbook_data(self, file_path: Path) -> pd.DataFrame:
        """Parse orderbook JSONL file into DataFrame."""
        records = []

        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Calculate orderbook features
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])

                    if bids and asks:
                        best_bid = float(bids[0][0]) if bids else 0
                        best_ask = float(asks[0][0]) if asks else 0
                        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                        spread = best_ask - best_bid if best_bid and best_ask else 0
                        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0

                        # Calculate depth metrics
                        bid_volume_5 = sum(float(b[1]) for b in bids[:5])
                        ask_volume_5 = sum(float(a[1]) for a in asks[:5])
                        bid_volume_10 = sum(float(b[1]) for b in bids[:10])
                        ask_volume_10 = sum(float(a[1]) for a in asks[:10])

                        # Order imbalance
                        total_bid_vol = bid_volume_10
                        total_ask_vol = ask_volume_10
                        order_imbalance = (
                            (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                            if (total_bid_vol + total_ask_vol) > 0
                            else 0
                        )

                        record = {
                            "timestamp": data["timestamp"],
                            "event_time": data["event_time"],
                            "mid_price": mid_price,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread": spread,
                            "spread_pct": spread_pct,
                            "bid_volume_5": bid_volume_5,
                            "ask_volume_5": ask_volume_5,
                            "bid_volume_10": bid_volume_10,
                            "ask_volume_10": ask_volume_10,
                            "order_imbalance": order_imbalance,
                        }
                        records.append(record)

                except Exception as e:
                    logger.warning(f"Error parsing orderbook line: {e}")
                    continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()

        return df

    def _parse_trade_data(self, file_path: Path) -> pd.DataFrame:
        """Parse trade JSONL file into DataFrame."""
        records = []

        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    record = {
                        "timestamp": data["timestamp"],
                        "event_time": data["event_time"],
                        "trade_id": data["trade_id"],
                        "price": float(data["price"]),
                        "quantity": float(data["quantity"]),
                        "is_buyer_maker": data["is_buyer_maker"],
                    }
                    records.append(record)

                except Exception as e:
                    logger.warning(f"Error parsing trade line: {e}")
                    continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()

        return df

    def _aggregate_trade_features(
        self, trades_df: pd.DataFrame, freq: str = "1min"
    ) -> pd.DataFrame:
        """Aggregate trade data into features."""
        if trades_df.empty:
            return pd.DataFrame()

        # Separate buy and sell trades
        buy_trades = trades_df[~trades_df["is_buyer_maker"]]
        sell_trades = trades_df[trades_df["is_buyer_maker"]]

        # Aggregate features
        agg_features = pd.DataFrame()

        # OHLCV
        ohlc = trades_df["price"].resample(freq).ohlc()
        volume = trades_df["quantity"].resample(freq).sum()

        agg_features["open"] = ohlc["open"]
        agg_features["high"] = ohlc["high"]
        agg_features["low"] = ohlc["low"]
        agg_features["close"] = ohlc["close"]
        agg_features["volume"] = volume

        # Trade count
        agg_features["trade_count"] = trades_df["trade_id"].resample(freq).count()

        # Buy/Sell volumes
        agg_features["buy_volume"] = buy_trades["quantity"].resample(freq).sum()
        agg_features["sell_volume"] = sell_trades["quantity"].resample(freq).sum()

        # Buy/Sell trade counts
        agg_features["buy_count"] = buy_trades["trade_id"].resample(freq).count()
        agg_features["sell_count"] = sell_trades["trade_id"].resample(freq).count()

        # VWAP
        trades_df["value"] = trades_df["price"] * trades_df["quantity"]
        vwap = trades_df["value"].resample(freq).sum() / trades_df["quantity"].resample(freq).sum()
        agg_features["vwap"] = vwap

        # Fill NaN values
        agg_features = agg_features.fillna(method="ffill").fillna(0)

        return agg_features

    def _merge_data(
        self, orderbook_df: pd.DataFrame, trade_features_df: pd.DataFrame, freq: str = "1min"
    ) -> pd.DataFrame:
        """Merge orderbook and trade data."""
        # Resample orderbook data
        if not orderbook_df.empty:
            orderbook_resampled = orderbook_df.resample(freq).agg(
                {
                    "mid_price": "last",
                    "best_bid": "last",
                    "best_ask": "last",
                    "spread": "mean",
                    "spread_pct": "mean",
                    "bid_volume_5": "mean",
                    "ask_volume_5": "mean",
                    "bid_volume_10": "mean",
                    "ask_volume_10": "mean",
                    "order_imbalance": "mean",
                }
            )
        else:
            orderbook_resampled = pd.DataFrame()

        # Merge data
        if not orderbook_resampled.empty and not trade_features_df.empty:
            merged = pd.merge(
                trade_features_df,
                orderbook_resampled,
                left_index=True,
                right_index=True,
                how="outer",
            )
        elif not trade_features_df.empty:
            merged = trade_features_df
        elif not orderbook_resampled.empty:
            merged = orderbook_resampled
        else:
            merged = pd.DataFrame()

        # Forward fill missing values
        if not merged.empty:
            merged = merged.fillna(method="ffill").fillna(0)

        return merged

    async def process_date(self, date: datetime) -> str | None:  # noqa: C901
        """Process all data for a specific date.

        Args:
            date: Date to process

        Returns:
            Path to processed file in GCS if successful, None otherwise
        """
        logger.info(f"Processing data for {date.strftime('%Y-%m-%d')}")

        try:
            # List raw files for date
            blob_names = self._list_blobs_for_date(date)
            if not blob_names:
                logger.warning(f"No raw data found for {date.strftime('%Y-%m-%d')}")
                return None

            # Download and parse files
            orderbook_dfs = []
            trade_dfs = []

            for blob_name in blob_names:
                local_path = self.local_work_dir / Path(blob_name).name
                self._download_blob(blob_name, local_path)

                try:
                    if "orderbook" in blob_name:
                        df = self._parse_orderbook_data(local_path)
                        if not df.empty:
                            orderbook_dfs.append(df)
                    elif "trades" in blob_name or "trade" in blob_name:
                        df = self._parse_trade_data(local_path)
                        if not df.empty:
                            trade_dfs.append(df)

                finally:
                    # Clean up local file
                    local_path.unlink()

            # Combine dataframes
            orderbook_df = pd.concat(orderbook_dfs) if orderbook_dfs else pd.DataFrame()
            trades_df = pd.concat(trade_dfs) if trade_dfs else pd.DataFrame()

            # Sort by timestamp
            if not orderbook_df.empty:
                orderbook_df = orderbook_df.sort_index()
            if not trades_df.empty:
                trades_df = trades_df.sort_index()

            # Aggregate trade features
            trade_features = self._aggregate_trade_features(trades_df)

            # Merge all data
            merged_df = self._merge_data(orderbook_df, trade_features)

            if merged_df.empty:
                logger.warning(f"No data to process for {date.strftime('%Y-%m-%d')}")
                return None

            # Save processed data
            output_filename = f"btcusdt_{date.strftime('%Y%m%d')}_1min.parquet"
            local_output = self.local_work_dir / output_filename
            merged_df.to_parquet(local_output, compression="snappy")

            # Upload to GCS
            date_str = date.strftime("%Y/%m/%d")
            gcs_path = f"{self.processed_prefix}/{date_str}/{output_filename}"
            self._upload_blob(local_output, gcs_path)

            # Clean up local file
            local_output.unlink()

            logger.info(
                f"Successfully processed {len(merged_df)} records for {date.strftime('%Y-%m-%d')}"
            )
            return gcs_path

        except Exception as e:
            logger.error(f"Error processing date {date.strftime('%Y-%m-%d')}: {e}")
            return None

    async def process_date_range(self, start_date: datetime, end_date: datetime) -> list[str]:
        """Process data for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of processed file paths in GCS
        """
        processed_files = []
        current_date = start_date

        while current_date <= end_date:
            result = await self.process_date(current_date)
            if result:
                processed_files.append(result)
            current_date += timedelta(days=1)

        logger.info(f"Processed {len(processed_files)} days of data")
        return processed_files


async def main() -> None:
    """Example usage of DailyPreprocessor."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Use configuration settings
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config import (
        GCP_PROJECT_ID,
        GCS_BUCKET,
        GCP_CREDENTIALS_PATH,
        PROCESSED_DATA_DIR,
    )  # noqa: I001

    # Create preprocessor
    preprocessor = DailyPreprocessor(
        bucket_name=GCS_BUCKET,
        project_id=GCP_PROJECT_ID,
        credentials_path=GCP_CREDENTIALS_PATH,
        local_work_dir=str(PROCESSED_DATA_DIR),
    )

    # Process yesterday's data
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    result = await preprocessor.process_date(yesterday)

    if result:
        logger.info(f"Processed data saved to: {result}")
    else:
        logger.warning("No data processed")


if __name__ == "__main__":
    asyncio.run(main())
