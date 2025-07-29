"""Tests for GCS uploader."""
# mypy: ignore-errors

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock google.cloud imports before importing our module
sys.modules['google'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['google.cloud.exceptions'] = MagicMock()

from src.data_collection.gcs_uploader import GCSUploader


@pytest.fixture
def mock_gcs_client():
    """Create mock GCS client."""
    with patch('src.data_collection.gcs_uploader.storage.Client') as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_client, mock_bucket


@pytest.fixture
def uploader(mock_gcs_client):
    """Create a GCSUploader instance with mocked GCS client."""
    _, mock_bucket = mock_gcs_client
    return GCSUploader(
        bucket_name="test-bucket",
        project_id="test-project",
        local_data_dir="./test_data",
        cleanup_after_upload=False,
    )


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "test_data.jsonl"
    file_path.write_text('{"test": "data"}\n{"more": "data"}\n')
    return file_path


@pytest.mark.asyncio
async def test_uploader_initialization(mock_gcs_client) -> None:
    """Test GCS uploader initialization."""
    mock_client, mock_bucket = mock_gcs_client

    uploader = GCSUploader(
        bucket_name="test-bucket",
        project_id="test-project",
    )

    assert uploader.bucket_name == "test-bucket"
    assert uploader.project_id == "test-project"
    assert uploader.stats["files_uploaded"] == 0
    mock_client.assert_called_once_with(project="test-project")


@pytest.mark.asyncio
async def test_upload_file_success(uploader, sample_file, mock_gcs_client) -> None:
    """Test successful file upload."""
    _, mock_bucket = mock_gcs_client
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    # Mock the executor to run the upload synchronously
    async def mock_executor(executor, func, *args):
        # Call the function to simulate upload
        func(*args)
        return None

    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = mock_executor

        success = await uploader._upload_file(sample_file)

        assert success is True
        assert uploader.stats["files_uploaded"] == 1
        assert uploader.stats["bytes_uploaded"] > 0
        mock_bucket.blob.assert_called_once()
        mock_blob.upload_from_string.assert_called_once()


@pytest.mark.asyncio
async def test_upload_file_failure(uploader, sample_file, mock_gcs_client) -> None:
    """Test file upload failure handling."""
    _, mock_bucket = mock_gcs_client
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    # Mock upload to raise exception
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(
            side_effect=Exception("Upload failed")
        )

        success = await uploader._upload_file(sample_file)

        assert success is False
        assert uploader.stats["files_failed"] == 1
        assert uploader.stats["files_uploaded"] == 0


@pytest.mark.asyncio
async def test_cleanup_after_upload(mock_gcs_client, tmp_path) -> None:
    """Test file cleanup after successful upload."""
    _, mock_bucket = mock_gcs_client
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    # Create uploader with cleanup enabled
    uploader = GCSUploader(
        bucket_name="test-bucket",
        project_id="test-project",
        local_data_dir=str(tmp_path),
        cleanup_after_upload=True,
    )

    # Create test file
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"test": "data"}\n')

    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

        success = await uploader._upload_file(test_file)

        assert success is True
        assert not test_file.exists()  # File should be deleted
        assert uploader.stats["files_deleted"] == 1


@pytest.mark.asyncio
async def test_upload_queue_processing(uploader, sample_file) -> None:
    """Test upload queue processing."""
    # Add file to queue
    await uploader._upload_queue.put(sample_file)

    # Mock upload method
    with patch.object(uploader, '_upload_file', new=AsyncMock(return_value=True)):
        uploader._running = True

        # Run worker briefly
        worker_task = asyncio.create_task(uploader._upload_worker())
        await asyncio.sleep(0.1)
        uploader._running = False

        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # Check that upload was attempted
        uploader._upload_file.assert_called_once_with(sample_file)


@pytest.mark.asyncio
async def test_directory_scanning(uploader, tmp_path) -> None:
    """Test directory scanning for new files."""
    # Set up test directory
    uploader.local_data_dir = tmp_path

    # Create some test files
    (tmp_path / "file1.jsonl").write_text('{"data": 1}\n')
    (tmp_path / "file2.jsonl").write_text('{"data": 2}\n')
    (tmp_path / "file3.txt").write_text('not a jsonl file\n')  # Should be ignored

    uploader._running = True

    # Run scanner briefly
    scanner_task = asyncio.create_task(uploader._scan_directory())
    await asyncio.sleep(0.1)
    uploader._running = False

    try:
        await asyncio.wait_for(scanner_task, timeout=1.0)
    except asyncio.TimeoutError:
        pass

    # Check that only JSONL files were queued
    assert uploader._upload_queue.qsize() == 2


@pytest.mark.asyncio
async def test_get_stats(uploader) -> None:
    """Test statistics retrieval."""
    # Modify some stats
    uploader.stats["files_uploaded"] = 10
    uploader.stats["bytes_uploaded"] = 1024 * 1024

    stats = uploader.get_stats()

    assert stats["files_uploaded"] == 10
    assert stats["bytes_uploaded"] == 1024 * 1024
    assert isinstance(stats, dict)

    # Ensure it's a copy
    stats["files_uploaded"] = 20
    assert uploader.stats["files_uploaded"] == 10


@pytest.mark.asyncio
async def test_upload_specific_file(uploader, sample_file, mock_gcs_client) -> None:
    """Test uploading a specific file."""
    _, mock_bucket = mock_gcs_client
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

        success = await uploader.upload_file(str(sample_file))

        assert success is True
        assert uploader.stats["files_uploaded"] == 1
