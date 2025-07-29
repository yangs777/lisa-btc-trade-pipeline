"""
Daily Preprocessor for Orderbook Data
Runs at 04:05 KST (19:05 UTC) to process daily data
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta, timezone
from google.cloud import storage
import logging
import schedule
import time
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_preprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyPreprocessor:
    """Processes raw orderbook data into daily feature datasets"""
    
    def __init__(self,
                 bucket_name: str = "btc-orderbook-data",
                 project_id: str = "my-project-779482",
                 local_temp_dir: str = "./data/temp"):
        
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.local_temp_dir = Path(local_temp_dir)
        self.local_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Metrics
        self.metrics = {
            'days_processed': 0,
            'files_processed': 0,
            'errors': 0,
            'last_run': None
        }
        
    def run_daily_job(self, date: Optional[datetime] = None):
        """Run preprocessing for a specific date (default: yesterday)"""
        try:
            # Default to yesterday (KST)
            if date is None:
                kst = timezone(timedelta(hours=9))
                now = datetime.now(kst)
                date = (now - timedelta(days=1)).date()
            else:
                date = date.date()
                
            logger.info(f"Starting daily preprocessing for {date}")
            
            # Download raw data for the date
            raw_files = self._download_raw_data(date)
            if not raw_files:
                logger.warning(f"No raw data found for {date}")
                return
                
            # Process orderbook data
            orderbook_df = self._process_orderbook_data(raw_files['orderbook'])
            
            # Process trades data
            trades_df = self._process_trades_data(raw_files['trades'])
            
            # Merge and create features
            features_df = self._create_daily_features(orderbook_df, trades_df)
            
            # Quality checks
            if not self._quality_checks(features_df):
                logger.error(f"Quality checks failed for {date}")
                self.metrics['errors'] += 1
                return
                
            # Upload processed data
            self._upload_processed_data(features_df, date)
            
            # Cleanup temp files
            self._cleanup_temp_files()
            
            # Update metrics
            self.metrics['days_processed'] += 1
            self.metrics['last_run'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully processed data for {date}")
            
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            self.metrics['errors'] += 1
            raise
            
    def _download_raw_data(self, date: datetime.date) -> dict:
        """Download raw data files for a specific date"""
        try:
            date_str = date.strftime("%Y/%m/%d")
            raw_files = {'orderbook': [], 'trades': []}
            
            # List blobs for the date
            prefix = f"raw/{date_str}/"
            blobs = self.storage_client.list_blobs(self.bucket_name, prefix=prefix)
            
            for blob in blobs:
                local_path = self.local_temp_dir / Path(blob.name).name
                
                # Download file
                blob.download_to_filename(str(local_path))
                logger.info(f"Downloaded {blob.name}")
                
                # Categorize file
                if 'orderbook' in blob.name:
                    raw_files['orderbook'].append(local_path)
                elif 'trades' in blob.name:
                    raw_files['trades'].append(local_path)
                    
                self.metrics['files_processed'] += 1
                
            return raw_files
            
        except Exception as e:
            logger.error(f"Error downloading raw data: {e}")
            raise
            
    def _process_orderbook_data(self, files: List[Path]) -> pd.DataFrame:
        """Process orderbook snapshot files"""
        try:
            dfs = []
            
            for file_path in sorted(files):
                df = pd.read_parquet(file_path)
                dfs.append(df)
                
            # Combine all dataframes
            if not dfs:
                return pd.DataFrame()
                
            df = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Set datetime as index
            df = df.set_index('datetime')
            
            logger.info(f"Processed {len(df)} orderbook snapshots")
            return df
            
        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")
            raise
            
    def _process_trades_data(self, files: List[Path]) -> pd.DataFrame:
        """Process trades data files"""
        try:
            dfs = []
            
            for file_path in sorted(files):
                df = pd.read_parquet(file_path)
                dfs.append(df)
                
            # Combine all dataframes
            if not dfs:
                return pd.DataFrame()
                
            df = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Aggregate trades by second
            df_agg = df.set_index('datetime').resample('1S').agg({
                'price': ['first', 'last', 'mean', 'std'],
                'quantity': ['sum', 'count'],
                'is_buyer_maker': ['sum', 'count']
            })
            
            # Flatten column names
            df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
            
            # Calculate additional features
            df_agg['buy_volume'] = df_agg['quantity_sum'] - df_agg['is_buyer_maker_sum']
            df_agg['sell_volume'] = df_agg['is_buyer_maker_sum']
            df_agg['trade_imbalance'] = (df_agg['buy_volume'] - df_agg['sell_volume']) / df_agg['quantity_sum']
            
            logger.info(f"Processed {len(df)} trades into {len(df_agg)} aggregated records")
            return df_agg
            
        except Exception as e:
            logger.error(f"Error processing trades data: {e}")
            raise
            
    def _create_daily_features(self, orderbook_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily feature dataset"""
        try:
            # Merge orderbook and trades data
            df = orderbook_df.join(trades_df, how='outer')
            
            # Forward fill orderbook data (orderbook state persists)
            orderbook_cols = [col for col in orderbook_df.columns if col in df.columns]
            df[orderbook_cols] = df[orderbook_cols].fillna(method='ffill')
            
            # Fill remaining NaNs
            df = df.fillna(0)
            
            # Add time features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['second'] = df.index.second
            df['day_of_week'] = df.index.dayofweek
            
            # Calculate additional orderbook features
            df['bid_ask_ratio'] = df['best_bid'] / df['best_ask']
            df['log_mid_price'] = np.log(df['mid_price'])
            df['mid_price_change'] = df['mid_price'].diff()
            df['mid_price_pct_change'] = df['mid_price'].pct_change()
            
            # Calculate depth features
            for i in range(5):
                df[f'bid_depth_{i}'] = df[f'bid_{i}_price'] * df[f'bid_{i}_qty']
                df[f'ask_depth_{i}'] = df[f'ask_{i}_price'] * df[f'ask_{i}_qty']
                
            # Total depth
            df['total_bid_depth'] = sum(df[f'bid_depth_{i}'] for i in range(5))
            df['total_ask_depth'] = sum(df[f'ask_depth_{i}'] for i in range(5))
            df['depth_imbalance'] = (df['total_bid_depth'] - df['total_ask_depth']) / (df['total_bid_depth'] + df['total_ask_depth'])
            
            # Rolling features (various windows)
            for window in [10, 30, 60, 300]:  # seconds
                df[f'mid_price_mean_{window}s'] = df['mid_price'].rolling(window).mean()
                df[f'mid_price_std_{window}s'] = df['mid_price'].rolling(window).std()
                df[f'spread_mean_{window}s'] = df['spread'].rolling(window).mean()
                df[f'volume_sum_{window}s'] = df['quantity_sum'].rolling(window).sum()
                
            # Remove NaN rows from rolling calculations
            df = df.dropna()
            
            # Select final features (exclude raw bid/ask prices)
            feature_cols = [col for col in df.columns if not any(x in col for x in ['bid_', 'ask_']) or any(x in col for x in ['depth', 'ratio'])]
            df_features = df[feature_cols]
            
            logger.info(f"Created feature dataset with {len(df_features)} rows and {len(df_features.columns)} features")
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise
            
    def _quality_checks(self, df: pd.DataFrame) -> bool:
        """Perform quality checks on processed data"""
        try:
            # Check minimum rows (should have most seconds of the day)
            min_rows = 86400 * 0.8  # 80% of seconds in a day
            if len(df) < min_rows:
                logger.warning(f"Data has only {len(df)} rows, expected at least {min_rows}")
                return False
                
            # Check for extreme values
            if df['mid_price'].min() <= 0:
                logger.error("Found non-positive prices")
                return False
                
            if df['spread_pct'].max() > 10:  # 10% spread is too high
                logger.warning("Found extremely high spreads")
                
            # Check for long gaps
            time_diff = df.index.to_series().diff()
            max_gap = time_diff.max()
            if max_gap > timedelta(minutes=5):
                logger.warning(f"Found gap of {max_gap} in data")
                
            # Check for data consistency
            null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if null_pct > 0.1:  # More than 10% nulls
                logger.error(f"Data has {null_pct:.2%} null values")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in quality checks: {e}")
            return False
            
    def _upload_processed_data(self, df: pd.DataFrame, date: datetime.date):
        """Upload processed features to GCS"""
        try:
            # Save to parquet
            filename = f"features_{date.strftime('%Y%m%d')}.parquet"
            local_path = self.local_temp_dir / filename
            df.to_parquet(local_path, compression='snappy')
            
            # Upload to GCS
            gcs_path = f"daily_feats/{date.year}/{date.strftime('%m')}/{filename}"
            blob = self.bucket.blob(gcs_path)
            
            blob.metadata = {
                'processing_time': datetime.now(timezone.utc).isoformat(),
                'rows': str(len(df)),
                'columns': str(len(df.columns)),
                'date': date.isoformat()
            }
            
            blob.upload_from_filename(str(local_path))
            
            logger.info(f"Uploaded processed data to gs://{self.bucket_name}/{gcs_path}")
            
        except Exception as e:
            logger.error(f"Error uploading processed data: {e}")
            raise
            
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file_path in self.local_temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            
    def schedule_daily_run(self):
        """Schedule daily preprocessing at 04:05 KST (19:05 UTC)"""
        # Schedule job
        schedule.every().day.at("19:05").do(self.run_daily_job)
        
        logger.info("Scheduled daily preprocessing at 19:05 UTC (04:05 KST)")
        
        # Run scheduler
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes on error

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily orderbook data preprocessor')
    parser.add_argument('--date', type=str, help='Process specific date (YYYY-MM-DD)')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    args = parser.parse_args()
    
    # Check for service account key
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        sys.exit(1)
        
    preprocessor = DailyPreprocessor()
    
    if args.date:
        # Process specific date
        date = datetime.strptime(args.date, '%Y-%m-%d')
        preprocessor.run_daily_job(date)
    elif args.schedule:
        # Run on schedule
        preprocessor.schedule_daily_run()
    else:
        # Process yesterday
        preprocessor.run_daily_job()

if __name__ == "__main__":
    main()