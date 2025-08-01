"""Tests for I/O abstraction layer."""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path to allow direct module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Always mock google.cloud.storage for testing
HAS_GCS = True

spec = importlib.util.spec_from_file_location("_io", "src/data_processing/_io.py")
if spec and spec.loader:
    _io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_io)
else:
    raise ImportError("Could not load _io module")

StorageClient = _io.StorageClient
GCSStorageClient = _io.GCSStorageClient
MockStorageClient = _io.MockStorageClient


class TestStorageClientProtocol:
    """Test StorageClient protocol."""

    def test_protocol_methods(self):
        """Test that protocol defines required methods."""
        # Protocol should define these methods
        assert hasattr(StorageClient, "list_blobs")
        assert hasattr(StorageClient, "download_blob")
        assert hasattr(StorageClient, "upload_blob")
        assert hasattr(StorageClient, "blob_exists")


class TestMockStorageClient:
    """Test MockStorageClient implementation."""

    @pytest.fixture
    def client(self):
        """Create mock storage client."""
        return MockStorageClient()

    def test_init(self, client):
        """Test initialization."""
        assert isinstance(client.blobs, dict)
        assert len(client.blobs) == 0

    def test_upload_blob(self, client):
        """Test blob upload."""
        blob_name = "test/data.json"
        content = b"test content"

        client.upload_blob(blob_name, content)

        assert blob_name in client.blobs
        assert client.blobs[blob_name] == content

    def test_download_blob(self, client):
        """Test blob download."""
        blob_name = "test/data.json"
        content = b"test content"
        client.blobs[blob_name] = content

        downloaded = client.download_blob(blob_name)

        assert downloaded == content

    def test_download_blob_not_found(self, client):
        """Test downloading non-existent blob."""
        with pytest.raises(ValueError, match="Blob test/missing.json not found"):
            client.download_blob("test/missing.json")

    def test_list_blobs(self, client):
        """Test listing blobs."""
        # Add some blobs
        client.blobs["data/2024/01/01/file1.json"] = b"content1"
        client.blobs["data/2024/01/01/file2.json"] = b"content2"
        client.blobs["data/2024/01/02/file3.json"] = b"content3"
        client.blobs["other/file4.json"] = b"content4"

        # List with prefix
        result = client.list_blobs("data/2024/01/01/")

        assert len(result) == 2
        assert "data/2024/01/01/file1.json" in result
        assert "data/2024/01/01/file2.json" in result
        assert "data/2024/01/02/file3.json" not in result
        assert "other/file4.json" not in result

    def test_list_blobs_empty_prefix(self, client):
        """Test listing all blobs."""
        client.blobs["file1.json"] = b"content1"
        client.blobs["dir/file2.json"] = b"content2"

        result = client.list_blobs("")

        assert len(result) == 2
        assert "file1.json" in result
        assert "dir/file2.json" in result

    def test_blob_exists(self, client):
        """Test blob existence check."""
        client.blobs["exists.json"] = b"content"

        assert client.blob_exists("exists.json") is True
        assert client.blob_exists("not_exists.json") is False

    def test_add_mock_blob(self, client):
        """Test adding mock blob for testing."""
        blob_name = "test/mock.json"
        content = b"mock content"

        client.add_mock_blob(blob_name, content)

        assert blob_name in client.blobs
        assert client.blobs[blob_name] == content
        assert client.blob_exists(blob_name)


@pytest.mark.skipif(not HAS_GCS, reason="google.cloud.storage not available")
class TestGCSStorageClient:
    """Test GCSStorageClient implementation."""

    @pytest.fixture(autouse=True)
    def mock_gcs(self):
        """Mock Google Cloud Storage for all tests in this class."""
        # Create mock storage module
        mock_storage_module = MagicMock()
        mock_client = Mock()
        mock_bucket = Mock()

        # Set up the mock hierarchy
        mock_storage_module.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket

        # Store for use in tests
        self.mock_storage = mock_storage_module
        self.mock_client = mock_client
        self.mock_bucket = mock_bucket

        # Patch google.cloud.storage at import time
        with patch.dict("sys.modules", {"google.cloud.storage": mock_storage_module}):
            yield

    def test_init_without_credentials(self):
        """Test initialization without credentials."""
        client = GCSStorageClient(bucket_name="test-bucket", project_id="test-project")

        assert client.client == self.mock_client
        assert client.bucket == self.mock_bucket
        self.mock_storage.Client.assert_called_once_with(project="test-project")
        self.mock_client.bucket.assert_called_once_with("test-bucket")

    def test_init_with_credentials(self):
        """Test initialization with credentials path."""
        with patch("os.environ", {}):
            client = GCSStorageClient(
                bucket_name="test-bucket",
                project_id="test-project",
                credentials_path="/path/to/creds.json",
            )

            # Should set environment variable
            assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == "/path/to/creds.json"

        assert client.client == self.mock_client
        assert client.bucket == self.mock_bucket

    def test_list_blobs(self):
        """Test listing blobs."""
        # Create mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "data/file1.json"
        mock_blob2 = Mock()
        mock_blob2.name = "data/file2.json"

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        client = GCSStorageClient("test-bucket", "test-project")
        result = client.list_blobs("data/")

        assert len(result) == 2
        assert "data/file1.json" in result
        assert "data/file2.json" in result
        self.mock_bucket.list_blobs.assert_called_once_with(prefix="data/")

    def test_download_blob(self):
        """Test downloading blob."""
        mock_blob = Mock()
        self.mock_bucket.blob.return_value = mock_blob

        content = b"test content"
        mock_blob.download_as_bytes.return_value = content

        client = GCSStorageClient("test-bucket", "test-project")
        result = client.download_blob("test/data.json")

        assert result == content
        self.mock_bucket.blob.assert_called_once_with("test/data.json")
        mock_blob.download_as_bytes.assert_called_once()

    def test_upload_blob(self):
        """Test uploading blob."""
        mock_blob = Mock()
        self.mock_bucket.blob.return_value = mock_blob

        client = GCSStorageClient("test-bucket", "test-project")
        content = b"test content"
        client.upload_blob("test/data.json", content)

        self.mock_bucket.blob.assert_called_once_with("test/data.json")
        mock_blob.upload_from_string.assert_called_once_with(content)

    def test_blob_exists(self):
        """Test checking blob existence."""
        mock_blob = Mock()
        self.mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        client = GCSStorageClient("test-bucket", "test-project")
        result = client.blob_exists("test/data.json")

        assert result is True
        self.mock_bucket.blob.assert_called_once_with("test/data.json")
        mock_blob.exists.assert_called_once()

    def test_blob_not_exists(self):
        """Test checking non-existent blob."""
        mock_blob = Mock()
        self.mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        client = GCSStorageClient("test-bucket", "test-project")
        result = client.blob_exists("test/missing.json")

        assert result is False


class TestIntegration:
    """Test integration scenarios."""

    def test_mock_client_implements_protocol(self):
        """Test that MockStorageClient implements StorageClient protocol."""
        client = MockStorageClient()

        # Should have all protocol methods
        assert callable(client.list_blobs)
        assert callable(client.download_blob)
        assert callable(client.upload_blob)
        assert callable(client.blob_exists)

    @pytest.mark.skipif(not HAS_GCS, reason="google.cloud.storage not available")
    def test_gcs_client_implements_protocol(self):
        """Test that GCSStorageClient implements StorageClient protocol."""
        # Check that GCSStorageClient has all protocol methods without instantiating
        assert hasattr(GCSStorageClient, "list_blobs")
        assert hasattr(GCSStorageClient, "download_blob")
        assert hasattr(GCSStorageClient, "upload_blob")
        assert hasattr(GCSStorageClient, "blob_exists")

    def test_storage_client_type_hints(self):
        """Test that implementations match protocol type hints."""
        # This test mainly ensures the protocol is properly defined
        # and implementations follow it

        # Protocol should define these methods
        assert hasattr(StorageClient, "list_blobs")
        assert hasattr(StorageClient, "download_blob")
        assert hasattr(StorageClient, "upload_blob")
        assert hasattr(StorageClient, "blob_exists")

        # Check that both implementations have the same methods
        mock_client = MockStorageClient()
        for method_name in ["list_blobs", "download_blob", "upload_blob", "blob_exists"]:
            assert hasattr(mock_client, method_name)
            assert callable(getattr(mock_client, method_name))
