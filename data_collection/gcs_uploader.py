"""
GCS Uploader for Orderbook Data
Automatically uploads collected parquet files to Google Cloud Storage
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from google.cloud import storage
import threading
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gcs_uploader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GCSUploader:
    """Uploads collected orderbook data to Google Cloud Storage"""
    
    def __init__(self, 
                 bucket_name: str = "btc-orderbook-data",
                 local_dir: str = "./data/raw",
                 upload_interval_minutes: int = 5,
                 delete_after_upload: bool = False,
                 project_id: str = "my-project-779482"):
        
        self.bucket_name = bucket_name
        self.local_dir = Path(local_dir)
        self.upload_interval_minutes = upload_interval_minutes
        self.delete_after_upload = delete_after_upload
        self.project_id = project_id
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Track uploaded files
        self.uploaded_files = set()
        self._load_uploaded_files()
        
        # Metrics
        self.metrics = {
            'files_uploaded': 0,
            'bytes_uploaded': 0,
            'failed_uploads': 0,
            'last_upload_time': None
        }
        
        self.running = False
        
    def _load_uploaded_files(self):
        """Load list of already uploaded files from local cache"""
        cache_file = self.local_dir / ".uploaded_files"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.uploaded_files = set(line.strip() for line in f)
                logger.info(f"Loaded {len(self.uploaded_files)} uploaded files from cache")
            except Exception as e:
                logger.error(f"Failed to load uploaded files cache: {e}")
                
    def _save_uploaded_files(self):
        """Save list of uploaded files to local cache"""
        cache_file = self.local_dir / ".uploaded_files"
        try:
            with open(cache_file, 'w') as f:
                for filename in sorted(self.uploaded_files):
                    f.write(f"{filename}\n")
        except Exception as e:
            logger.error(f"Failed to save uploaded files cache: {e}")
            
    def start(self):
        """Start the uploader service"""
        self.running = True
        logger.info(f"Starting GCS uploader for bucket: {self.bucket_name}")
        
        # Schedule upload job
        schedule.every(self.upload_interval_minutes).minutes.do(self.upload_pending_files)
        
        # Run initial upload
        self.upload_pending_files()
        
        # Start scheduler loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(10)
                
    def stop(self):
        """Stop the uploader service"""
        self.running = False
        # Final upload
        self.upload_pending_files()
        logger.info("GCS uploader stopped")
        
    def upload_pending_files(self):
        """Upload all pending parquet files to GCS"""
        try:
            # Find parquet files
            parquet_files = list(self.local_dir.glob("*.parquet"))
            pending_files = [f for f in parquet_files if f.name not in self.uploaded_files]
            
            if not pending_files:
                return
                
            logger.info(f"Found {len(pending_files)} files to upload")
            
            for file_path in pending_files:
                try:
                    self._upload_file(file_path)
                except Exception as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
                    self.metrics['failed_uploads'] += 1
                    
            # Save updated cache
            self._save_uploaded_files()
            self.metrics['last_upload_time'] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Error in upload process: {e}")
            
    def _upload_file(self, file_path: Path):
        """Upload a single file to GCS"""
        try:
            # Determine GCS path based on file timestamp
            filename = file_path.name
            
            # Extract date from filename (format: BTCUSDT_orderbook_YYYYMMDD_HHMMSS.parquet)
            parts = filename.split('_')
            if len(parts) >= 4:
                date_str = parts[2]  # YYYYMMDD
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                
                # Determine file type
                file_type = parts[1]  # 'orderbook' or 'trades'
                
                # Create GCS path
                gcs_path = f"raw/{year}/{month}/{day}/{file_type}/{filename}"
            else:
                # Fallback path
                gcs_path = f"raw/misc/{filename}"
                
            # Upload to GCS
            blob = self.bucket.blob(gcs_path)
            
            # Set metadata
            blob.metadata = {
                'upload_time': datetime.now(timezone.utc).isoformat(),
                'local_path': str(file_path),
                'file_size': file_path.stat().st_size
            }
            
            # Upload with retry
            retry_count = 3
            for attempt in range(retry_count):
                try:
                    blob.upload_from_filename(str(file_path))
                    break
                except Exception as e:
                    if attempt < retry_count - 1:
                        logger.warning(f"Upload attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)
                    else:
                        raise e
                        
            # Update metrics
            self.metrics['files_uploaded'] += 1
            self.metrics['bytes_uploaded'] += file_path.stat().st_size
            
            # Track uploaded file
            self.uploaded_files.add(filename)
            
            logger.info(f"Uploaded {filename} to gs://{self.bucket_name}/{gcs_path}")
            
            # Delete local file if configured
            if self.delete_after_upload:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted local file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete local file {filename}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            raise
            
    def verify_bucket_access(self) -> bool:
        """Verify access to GCS bucket"""
        try:
            # Try to list objects in bucket
            blobs = list(self.bucket.list_blobs(max_results=1))
            logger.info(f"Successfully verified access to bucket: {self.bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to access bucket {self.bucket_name}: {e}")
            return False
            
    def get_metrics(self) -> dict:
        """Get uploader metrics"""
        return {
            **self.metrics,
            'uploaded_files_count': len(self.uploaded_files),
            'pending_files': len(list(self.local_dir.glob("*.parquet"))) - len(self.uploaded_files)
        }

def main():
    """Main entry point"""
    import signal
    import sys
    
    # Check for service account key
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        logger.info("Please set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
        sys.exit(1)
        
    uploader = GCSUploader(
        bucket_name="btc-orderbook-data",
        local_dir="./data/raw",
        upload_interval_minutes=5,
        delete_after_upload=False  # Keep local copies for now
    )
    
    # Verify bucket access
    if not uploader.verify_bucket_access():
        logger.error("Cannot access GCS bucket. Please check credentials and permissions.")
        sys.exit(1)
        
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        uploader.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start uploader
    uploader.start()

if __name__ == "__main__":
    main()