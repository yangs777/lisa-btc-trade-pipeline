"""Binance WebSocket data collector for BTC/USDT orderbook and trades."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import aiofiles
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class BinanceWebSocketCollector:
    """Collects real-time BTC/USDT data from Binance WebSocket API."""

    def __init__(
        self,
        symbol: str = "btcusdt",
        depth_levels: int = 20,
        buffer_size: int = 1000,
        output_dir: str = "./data/raw",
    ):
        """Initialize Binance WebSocket collector.
        
        Args:
            symbol: Trading pair symbol (default: btcusdt)
            depth_levels: Orderbook depth levels (5, 10, or 20)
            buffer_size: Buffer size before writing to disk
            output_dir: Directory to save collected data
        """
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.buffer_size = buffer_size
        self.output_dir = output_dir
        
        # WebSocket URLs
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.orderbook_stream = f"{self.symbol}@depth{depth_levels}@100ms"
        self.trade_stream = f"{self.symbol}@trade"
        self.agg_trade_stream = f"{self.symbol}@aggTrade"
        
        # Data buffers
        self.orderbook_buffer: list[Dict[str, Any]] = []
        self.trade_buffer: list[Dict[str, Any]] = []
        self.agg_trade_buffer: list[Dict[str, Any]] = []
        
        # Statistics
        self.stats: Dict[str, Union[int, Optional[datetime]]] = {
            "orderbook_count": 0,
            "trade_count": 0,
            "agg_trade_count": 0,
            "last_orderbook_time": None,
            "last_trade_time": None,
            "start_time": None,
            "errors": 0,
        }
        
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def _handle_orderbook_message(self, message: Dict[str, Any]) -> None:
        """Process orderbook depth update message."""
        try:
            data = {
                "timestamp": int(time.time() * 1000),
                "event_time": message.get("E"),
                "symbol": message.get("s"),
                "first_update_id": message.get("U"),
                "final_update_id": message.get("u"),
                "bids": message.get("b", []),
                "asks": message.get("a", []),
            }
            
            self.orderbook_buffer.append(data)
            orderbook_count = self.stats.get("orderbook_count", 0)
            if isinstance(orderbook_count, int):
                self.stats["orderbook_count"] = orderbook_count + 1
            self.stats["last_orderbook_time"] = datetime.now(timezone.utc)
            
            if len(self.orderbook_buffer) >= self.buffer_size:
                await self._flush_orderbook_buffer()
                
        except Exception as e:
            logger.error(f"Error processing orderbook message: {e}")
            errors = self.stats.get("errors", 0)
            if isinstance(errors, int):
                self.stats["errors"] = errors + 1

    async def _handle_trade_message(self, message: Dict[str, Any]) -> None:
        """Process trade message."""
        try:
            data = {
                "timestamp": int(time.time() * 1000),
                "event_time": message.get("E"),
                "symbol": message.get("s"),
                "trade_id": message.get("t"),
                "price": message.get("p"),
                "quantity": message.get("q"),
                "buyer_order_id": message.get("b"),
                "seller_order_id": message.get("a"),
                "trade_time": message.get("T"),
                "is_buyer_maker": message.get("m"),
            }
            
            self.trade_buffer.append(data)
            trade_count = self.stats.get("trade_count", 0)
            if isinstance(trade_count, int):
                self.stats["trade_count"] = trade_count + 1
            self.stats["last_trade_time"] = datetime.now(timezone.utc)
            
            if len(self.trade_buffer) >= self.buffer_size:
                await self._flush_trade_buffer()
                
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
            errors = self.stats.get("errors", 0)
            if isinstance(errors, int):
                self.stats["errors"] = errors + 1

    async def _handle_agg_trade_message(self, message: Dict[str, Any]) -> None:
        """Process aggregated trade message."""
        try:
            data = {
                "timestamp": int(time.time() * 1000),
                "event_time": message.get("E"),
                "symbol": message.get("s"),
                "agg_trade_id": message.get("a"),
                "price": message.get("p"),
                "quantity": message.get("q"),
                "first_trade_id": message.get("f"),
                "last_trade_id": message.get("l"),
                "trade_time": message.get("T"),
                "is_buyer_maker": message.get("m"),
            }
            
            self.agg_trade_buffer.append(data)
            agg_trade_count = self.stats.get("agg_trade_count", 0)
            if isinstance(agg_trade_count, int):
                self.stats["agg_trade_count"] = agg_trade_count + 1
            
            if len(self.agg_trade_buffer) >= self.buffer_size:
                await self._flush_agg_trade_buffer()
                
        except Exception as e:
            logger.error(f"Error processing aggregated trade message: {e}")
            errors = self.stats.get("errors", 0)
            if isinstance(errors, int):
                self.stats["errors"] = errors + 1

    async def _flush_orderbook_buffer(self) -> None:
        """Write orderbook buffer to disk."""
        if not self.orderbook_buffer:
            return
            
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/orderbook_{self.symbol}_{timestamp}.jsonl"
        
        async with aiofiles.open(filename, "w") as f:
            for item in self.orderbook_buffer:
                await f.write(json.dumps(item) + "\n")
                
        logger.info(f"Flushed {len(self.orderbook_buffer)} orderbook entries to {filename}")
        self.orderbook_buffer.clear()

    async def _flush_trade_buffer(self) -> None:
        """Write trade buffer to disk."""
        if not self.trade_buffer:
            return
            
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/trades_{self.symbol}_{timestamp}.jsonl"
        
        async with aiofiles.open(filename, "w") as f:
            for item in self.trade_buffer:
                await f.write(json.dumps(item) + "\n")
                
        logger.info(f"Flushed {len(self.trade_buffer)} trade entries to {filename}")
        self.trade_buffer.clear()

    async def _flush_agg_trade_buffer(self) -> None:
        """Write aggregated trade buffer to disk."""
        if not self.agg_trade_buffer:
            return
            
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/agg_trades_{self.symbol}_{timestamp}.jsonl"
        
        async with aiofiles.open(filename, "w") as f:
            for item in self.agg_trade_buffer:
                await f.write(json.dumps(item) + "\n")
                
        logger.info(f"Flushed {len(self.agg_trade_buffer)} aggregated trade entries to {filename}")
        self.agg_trade_buffer.clear()

    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers to disk."""
        await self._flush_orderbook_buffer()
        await self._flush_trade_buffer()
        await self._flush_agg_trade_buffer()

    async def _websocket_handler(self, stream_name: str, handler_func: Any) -> None:
        """Handle WebSocket connection for a specific stream."""
        url = f"{self.ws_base_url}/{stream_name}"
        retry_count = 0
        max_retries = 5
        
        while self._running and retry_count < max_retries:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info(f"Connected to {stream_name}")
                    retry_count = 0  # Reset on successful connection
                    
                    async for message in websocket:
                        if not self._running:
                            break
                            
                        data = json.loads(message)
                        await handler_func(data)
                        
            except ConnectionClosed:
                logger.warning(f"Connection closed for {stream_name}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    
            except WebSocketException as e:
                logger.error(f"WebSocket error for {stream_name}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)
                    
            except Exception as e:
                logger.error(f"Unexpected error in {stream_name}: {e}")
                errors = self.stats.get("errors", 0)
            if isinstance(errors, int):
                self.stats["errors"] = errors + 1
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)

    async def start(self) -> None:
        """Start collecting data from Binance WebSocket."""
        self._running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        logger.info(f"Starting Binance WebSocket collector for {self.symbol.upper()}")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Start WebSocket handlers
        self._tasks = [
            asyncio.create_task(
                self._websocket_handler(self.orderbook_stream, self._handle_orderbook_message)
            ),
            asyncio.create_task(
                self._websocket_handler(self.trade_stream, self._handle_trade_message)
            ),
            asyncio.create_task(
                self._websocket_handler(self.agg_trade_stream, self._handle_agg_trade_message)
            ),
        ]
        
        # Periodic buffer flush
        async def periodic_flush() -> None:
            while self._running:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_all_buffers()
                
        self._tasks.append(asyncio.create_task(periodic_flush()))
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Collection tasks cancelled")

    async def stop(self) -> None:
        """Stop collecting data and flush buffers."""
        logger.info("Stopping Binance WebSocket collector...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Final buffer flush
        await self._flush_all_buffers()
        
        # Log statistics
        start_time = self.stats.get("start_time")
        if isinstance(start_time, datetime):
            runtime = datetime.now(timezone.utc) - start_time
        else:
            runtime = None
        logger.info(
            f"Collection stopped. Statistics:\n"
            f"  Runtime: {runtime}\n"
            f"  Orderbook updates: {self.stats['orderbook_count']}\n"
            f"  Trades: {self.stats['trade_count']}\n"
            f"  Aggregated trades: {self.stats['agg_trade_count']}\n"
            f"  Errors: {self.stats['errors']}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        return self.stats.copy()


async def main() -> None:
    """Example usage of BinanceWebSocketCollector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    collector = BinanceWebSocketCollector(
        symbol="btcusdt",
        depth_levels=20,
        buffer_size=1000,
        output_dir="./data/raw"
    )
    
    try:
        # Run for 5 minutes as a test
        await asyncio.wait_for(collector.start(), timeout=300)
    except asyncio.TimeoutError:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())