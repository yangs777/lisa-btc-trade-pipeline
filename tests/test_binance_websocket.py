"""Tests for Binance WebSocket data collector."""
# mypy: ignore-errors

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed

from src.data_collection.binance_websocket import BinanceWebSocketCollector


@pytest.fixture
def collector():
    """Create a BinanceWebSocketCollector instance for testing."""
    return BinanceWebSocketCollector(
        symbol="btcusdt",
        depth_levels=20,
        buffer_size=10,
        output_dir="./test_data"
    )


@pytest.fixture
def sample_orderbook_message():
    """Sample orderbook message from Binance."""
    return {
        "E": 1234567890123,  # Event time
        "s": "BTCUSDT",      # Symbol
        "U": 1234567890,     # First update ID
        "u": 1234567899,     # Final update ID
        "b": [["30000.00", "0.5"], ["29999.00", "1.0"]],  # Bids
        "a": [["30001.00", "0.5"], ["30002.00", "1.0"]]   # Asks
    }


@pytest.fixture
def sample_trade_message():
    """Sample trade message from Binance."""
    return {
        "E": 1234567890123,  # Event time
        "s": "BTCUSDT",      # Symbol
        "t": 123456789,      # Trade ID
        "p": "30000.00",     # Price
        "q": "0.1",          # Quantity
        "b": 88888888,       # Buyer order ID
        "a": 99999999,       # Seller order ID
        "T": 1234567890122,  # Trade time
        "m": False           # Is buyer maker?
    }


@pytest.fixture
def sample_agg_trade_message():
    """Sample aggregated trade message from Binance."""
    return {
        "E": 1234567890123,  # Event time
        "s": "BTCUSDT",      # Symbol
        "a": 123456789,      # Aggregate trade ID
        "p": "30000.00",     # Price
        "q": "1.5",          # Quantity
        "f": 100,            # First trade ID
        "l": 105,            # Last trade ID
        "T": 1234567890122,  # Trade time
        "m": True            # Is buyer maker?
    }


@pytest.mark.asyncio
async def test_collector_initialization(collector) -> None:
    """Test collector initialization."""
    assert collector.symbol == "btcusdt"
    assert collector.depth_levels == 20
    assert collector.buffer_size == 10
    assert collector.output_dir == "./test_data"
    assert len(collector.orderbook_buffer) == 0
    assert len(collector.trade_buffer) == 0
    assert len(collector.agg_trade_buffer) == 0


@pytest.mark.asyncio
async def test_handle_orderbook_message(collector, sample_orderbook_message) -> None:
    """Test orderbook message handling."""
    await collector._handle_orderbook_message(sample_orderbook_message)
    
    assert len(collector.orderbook_buffer) == 1
    assert collector.stats["orderbook_count"] == 1
    
    data = collector.orderbook_buffer[0]
    assert data["event_time"] == sample_orderbook_message["E"]
    assert data["symbol"] == sample_orderbook_message["s"]
    assert data["bids"] == sample_orderbook_message["b"]
    assert data["asks"] == sample_orderbook_message["a"]


@pytest.mark.asyncio
async def test_handle_trade_message(collector, sample_trade_message) -> None:
    """Test trade message handling."""
    await collector._handle_trade_message(sample_trade_message)
    
    assert len(collector.trade_buffer) == 1
    assert collector.stats["trade_count"] == 1
    
    data = collector.trade_buffer[0]
    assert data["event_time"] == sample_trade_message["E"]
    assert data["symbol"] == sample_trade_message["s"]
    assert data["trade_id"] == sample_trade_message["t"]
    assert data["price"] == sample_trade_message["p"]
    assert data["quantity"] == sample_trade_message["q"]


@pytest.mark.asyncio
async def test_handle_agg_trade_message(collector, sample_agg_trade_message) -> None:
    """Test aggregated trade message handling."""
    await collector._handle_agg_trade_message(sample_agg_trade_message)
    
    assert len(collector.agg_trade_buffer) == 1
    assert collector.stats["agg_trade_count"] == 1
    
    data = collector.agg_trade_buffer[0]
    assert data["event_time"] == sample_agg_trade_message["E"]
    assert data["symbol"] == sample_agg_trade_message["s"]
    assert data["agg_trade_id"] == sample_agg_trade_message["a"]
    assert data["price"] == sample_agg_trade_message["p"]
    assert data["quantity"] == sample_agg_trade_message["q"]


@pytest.mark.asyncio
async def test_buffer_flush_on_size_limit(collector, sample_orderbook_message) -> None:
    """Test buffer flushing when size limit is reached."""
    # Create a mock that properly simulates aiofiles context manager
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    
    mock_open = MagicMock()
    mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
    mock_open.return_value.__aexit__ = AsyncMock(return_value=None)
    
    with patch('aiofiles.open', mock_open):
        # Fill buffer to trigger flush (buffer_size=10)
        for _ in range(10):
            await collector._handle_orderbook_message(sample_orderbook_message)
        
        # Check that buffer was flushed
        assert len(collector.orderbook_buffer) == 0
        assert mock_open.called
        assert mock_file.write.called


@pytest.mark.asyncio
async def test_error_handling_in_message_processing(collector) -> None:
    """Test error handling when processing invalid messages."""
    # Mock the buffer flush to raise an exception
    with patch.object(collector, '_flush_orderbook_buffer', side_effect=Exception("Mock error")):
        # Fill buffer to trigger flush which will raise exception
        for _ in range(10):
            await collector._handle_orderbook_message({"E": 1234567890123, "s": "BTCUSDT", "b": [], "a": []})
    
    # Error count should have increased
    assert collector.stats["errors"] >= 1


@pytest.mark.asyncio
async def test_get_stats(collector, sample_orderbook_message, sample_trade_message) -> None:
    """Test statistics retrieval."""
    await collector._handle_orderbook_message(sample_orderbook_message)
    await collector._handle_trade_message(sample_trade_message)
    
    stats = collector.get_stats()
    assert stats["orderbook_count"] == 1
    assert stats["trade_count"] == 1
    assert stats["agg_trade_count"] == 0
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_websocket_reconnection() -> None:
    """Test WebSocket reconnection logic."""
    collector = BinanceWebSocketCollector()
    
    # Track connection attempts
    connect_count = 0
    
    # Patch the websockets.connect function
    with patch('src.data_collection.binance_websocket.websockets.connect') as mock_connect:
        # First call raises ConnectionClosed, second call succeeds but immediately ends
        mock_connect.side_effect = [
            ConnectionClosed(None, None),
            AsyncMock(__aenter__=AsyncMock(return_value=AsyncMock(__aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration))), __aexit__=AsyncMock())
        ]
        
        collector._running = True
        
        # Should handle reconnection without raising
        await collector._websocket_handler("test_stream", AsyncMock())
        
        # Should have attempted to connect at least twice
        assert mock_connect.call_count >= 2


@pytest.mark.asyncio
async def test_stop_collector() -> None:
    """Test stopping the collector."""
    collector = BinanceWebSocketCollector()
    
    # Set start time to avoid NoneType error
    collector.stats["start_time"] = datetime.now(timezone.utc)
    
    # Create mock tasks
    mock_task1 = MagicMock()
    mock_task1.cancel = MagicMock()
    mock_task2 = MagicMock()
    mock_task2.cancel = MagicMock()
    
    collector._tasks = [mock_task1, mock_task2]
    collector._running = True
    
    # Mock gather and _flush_all_buffers to avoid issues
    with patch('asyncio.gather', new=AsyncMock()):
        with patch.object(collector, '_flush_all_buffers', new=AsyncMock()):
            await collector.stop()
    
    assert collector._running is False
    assert mock_task1.cancel.called
    assert mock_task2.cancel.called


@pytest.mark.asyncio
async def test_flush_all_buffers(collector, sample_orderbook_message, sample_trade_message, sample_agg_trade_message) -> None:
    """Test flushing all buffers."""
    # Create a mock that properly simulates aiofiles context manager
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    
    mock_open = MagicMock()
    mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
    mock_open.return_value.__aexit__ = AsyncMock(return_value=None)
    
    with patch('aiofiles.open', mock_open):
        # Add data to all buffers
        await collector._handle_orderbook_message(sample_orderbook_message)
        await collector._handle_trade_message(sample_trade_message)
        await collector._handle_agg_trade_message(sample_agg_trade_message)
        
        # Flush all buffers
        await collector._flush_all_buffers()
        
        # All buffers should be empty
        assert len(collector.orderbook_buffer) == 0
        assert len(collector.trade_buffer) == 0
        assert len(collector.agg_trade_buffer) == 0
        
        # Should have written to 3 files
        assert mock_open.call_count == 3