"""Google Cloud Storage uploader for collected BTC/USDT data."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Union

import aiofiles
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)


class GCSUploader:
    """Uploads collected data files to Google Cloud Storage."""
    
    def __init__(
        self,
        bucket_name: str = "btc-orderbook-data",
        project_id: str = "my-project-779482",
        credentials_path: Optional[str] = None,
        local_data_dir: str = "./data/raw",
        gcs_prefix: str = "raw",
        cleanup_after_upload: bool = True,
    ):
        """Initialize GCS uploader.
        
        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
            credentials_path: Path to service account JSON file
            local_data_dir: Local directory containing data files
            gcs_prefix: Prefix for GCS object names
            cleanup_after_upload: Whether to delete local files after successful upload
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.local_data_dir = Path(local_data_dir)
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.cleanup_after_upload = cleanup_after_upload
        
        # Initialize GCS client
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Initialized GCS client for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
            
        # Upload statistics
        self.stats: Dict[str, Union[int, Optional[datetime]]] = {
            "files_uploaded": 0,
            "bytes_uploaded": 0,
            "files_failed": 0,
            "files_deleted": 0,
            "start_time": None,
            "last_upload_time": None,
        }
        
        self._running = False
        self._upload_queue: asyncio.Queue[Path] = asyncio.Queue()
        
    async def _upload_file(self, file_path: Path) -> bool:
        """Upload a single file to GCS.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Generate GCS object name
            file_date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            object_name = f"{self.gcs_prefix}/{file_date}/{file_path.name}"
            
            # Create blob
            blob = self.bucket.blob(object_name)
            
            # Read file content
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
                
            # Upload to GCS (synchronous operation)
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, content
            )
            
            file_size = len(content)
            files_uploaded = self.stats.get("files_uploaded", 0)
            if isinstance(files_uploaded, int):
                self.stats["files_uploaded"] = files_uploaded + 1
            bytes_uploaded = self.stats.get("bytes_uploaded", 0)
            if isinstance(bytes_uploaded, int):
                self.stats["bytes_uploaded"] = bytes_uploaded + file_size
            self.stats["last_upload_time"] = datetime.now(timezone.utc)
            
            logger.info(
                f"Uploaded {file_path.name} to gs://{self.bucket_name}/{object_name} "
                f"({file_size:,} bytes)"
            )
            
            # Clean up local file if requested
            if self.cleanup_after_upload:
                try:
                    file_path.unlink()
                    files_deleted = self.stats.get("files_deleted", 0)
                    if isinstance(files_deleted, int):
                        self.stats["files_deleted"] = files_deleted + 1
                    logger.debug(f"Deleted local file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete local file {file_path}: {e}")
                    
            return True
            
        except Exception as e:
            # Note: In production, we'd catch GoogleCloudError specifically
            # but for testing we catch all exceptions
            if "GoogleCloudError" in str(type(e).__name__):
                logger.error(f"GCS upload error for {file_path}: {e}")
            else:
                logger.error(f"Unexpected error uploading {file_path}: {e}")
            files_failed = self.stats.get("files_failed", 0)
            if isinstance(files_failed, int):
                self.stats["files_failed"] = files_failed + 1
            return False
            
    async def _upload_worker(self) -> None:
        """Worker coroutine that processes the upload queue."""
        while self._running:
            try:
                # Wait for file with timeout
                file_path = await asyncio.wait_for(
                    self._upload_queue.get(), timeout=1.0
                )
                
                # Upload the file
                success = await self._upload_file(file_path)
                
                if not success and not self.cleanup_after_upload:
                    # Re-queue failed uploads if not cleaning up
                    await self._upload_queue.put(file_path)
                    await asyncio.sleep(5)  # Wait before retry
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Upload worker error: {e}")
                await asyncio.sleep(1)
                
    async def _scan_directory(self) -> None:
        """Scan local directory for new files to upload."""
        processed_files = set()
        
        while self._running:
            try:
                # Find all JSONL files
                pattern = "*.jsonl"
                files = list(self.local_data_dir.glob(pattern))
                
                # Queue new files for upload
                for file_path in files:
                    if file_path not in processed_files:
                        await self._upload_queue.put(file_path)
                        processed_files.add(file_path)
                        logger.debug(f"Queued {file_path.name} for upload")
                        
                # Wait before next scan
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Directory scan error: {e}")
                await asyncio.sleep(5)
                
    async def start(self, num_workers: int = 3) -> None:
        """Start the GCS uploader.
        
        Args:
            num_workers: Number of concurrent upload workers
        """
        self._running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        logger.info(f"Starting GCS uploader with {num_workers} workers")
        
        # Create upload workers
        workers = [
            asyncio.create_task(self._upload_worker())
            for _ in range(num_workers)
        ]
        
        # Create directory scanner
        scanner = asyncio.create_task(self._scan_directory())
        
        # Wait for all tasks
        try:
            await asyncio.gather(scanner, *workers)
        except asyncio.CancelledError:
            logger.info("Upload tasks cancelled")
            
    async def stop(self) -> None:
        """Stop the GCS uploader."""
        logger.info("Stopping GCS uploader...")
        self._running = False
        
        # Wait for queue to empty
        while not self._upload_queue.empty():
            await asyncio.sleep(0.1)
            
        # Log statistics
        start_time = self.stats.get("start_time")
        if isinstance(start_time, datetime):
            runtime = datetime.now(timezone.utc) - start_time
        else:
            runtime = None
        logger.info(
            f"Upload stopped. Statistics:\n"
            f"  Runtime: {runtime}\n"
            f"  Files uploaded: {self.stats['files_uploaded']}\n"
            f"  Bytes uploaded: {self.stats['bytes_uploaded']:,}\n"
            f"  Files failed: {self.stats['files_failed']}\n"
            f"  Files deleted: {self.stats['files_deleted']}"
        )
        
    async def upload_file(self, file_path: str) -> bool:
        """Upload a specific file to GCS.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            True if upload successful, False otherwise
        """
        return await self._upload_file(Path(file_path))
        
    def get_stats(self) -> dict:
        """Get current upload statistics."""
        return self.stats.copy()


async def main() -> None:
    """Example usage of GCSUploader."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Use the service account key stored in /tmp
    uploader = GCSUploader(
        bucket_name="btc-orderbook-data",
        project_id="my-project-779482",
        credentials_path="/tmp/gcp_service_account_key.json",
        local_data_dir="./data/raw",
        cleanup_after_upload=False,  # Keep files for testing
    )
    
    try:
        # Run for 5 minutes as a test
        await asyncio.wait_for(uploader.start(num_workers=2), timeout=300)
    except asyncio.TimeoutError:
        await uploader.stop()


if __name__ == "__main__":
    asyncio.run(main())