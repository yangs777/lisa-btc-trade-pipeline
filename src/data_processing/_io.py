"""I/O abstraction layer for data processing modules."""

from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
import pandas as pd


class StorageClient(Protocol):
    """Protocol for storage operations."""
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List blob names with given prefix."""
        ...
    
    def download_blob(self, blob_name: str) -> bytes:
        """Download blob content."""
        ...
    
    def upload_blob(self, blob_name: str, content: bytes) -> None:
        """Upload content to blob."""
        ...
    
    def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists."""
        ...


class GCSStorageClient:
    """Google Cloud Storage implementation."""
    
    def __init__(self, bucket_name: str, project_id: str, credentials_path: Optional[str] = None):
        """Initialize GCS client."""
        from google.cloud import storage
        import os
        
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List blob names with given prefix."""
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    
    def download_blob(self, blob_name: str) -> bytes:
        """Download blob content."""
        blob = self.bucket.blob(blob_name)
        return blob.download_as_bytes()
    
    def upload_blob(self, blob_name: str, content: bytes) -> None:
        """Upload content to blob."""
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(content)
    
    def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists."""
        blob = self.bucket.blob(blob_name)
        return blob.exists()


class MockStorageClient:
    """Mock storage client for testing."""
    
    def __init__(self):
        """Initialize mock storage."""
        self.blobs: Dict[str, bytes] = {}
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List blob names with given prefix."""
        return [name for name in self.blobs.keys() if name.startswith(prefix)]
    
    def download_blob(self, blob_name: str) -> bytes:
        """Download blob content."""
        if blob_name not in self.blobs:
            raise ValueError(f"Blob {blob_name} not found")
        return self.blobs[blob_name]
    
    def upload_blob(self, blob_name: str, content: bytes) -> None:
        """Upload content to blob."""
        self.blobs[blob_name] = content
    
    def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists."""
        return blob_name in self.blobs
    
    def add_mock_blob(self, blob_name: str, content: bytes) -> None:
        """Add a mock blob for testing."""
        self.blobs[blob_name] = content