"""
Real-time Orderbook Collector for BTC/USDT Futures
Collects 1-second snapshots with depth20 + trades
"""

import asyncio
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.um_futures import UMFutures
import threading
from collections import deque
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderbookCollector:
    """Collects orderbook depth and trades data from Binance Futures"""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 depth_levels: int = 20,
                 output_dir: str = "./data/raw",
                 rotation_minutes: int = 60):
        
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rotation_minutes = rotation_minutes
        
        # Data buffers
        self.orderbook_buffer = deque(maxlen=3600)  # 1 hour of 1s snapshots
        self.trades_buffer = deque(maxlen=10000)
        
        # Current orderbook state
        self.current_orderbook = {
            'bids': {},
            'asks': {},
            'lastUpdateId': 0
        }
        
        # WebSocket clients
        self.ws_orderbook = None
        self.ws_trades = None
        self.rest_client = UMFutures()
        
        # Control flags
        self.running = False
        self.last_rotation = datetime.now(timezone.utc)
        self.reconnect_delay = 1  # seconds
        self.max_reconnect_delay = 60
        
        # Metrics
        self.metrics = {
            'snapshots_collected': 0,
            'trades_collected': 0,
            'disconnections': 0,
            'last_snapshot_time': None
        }
        
    def start(self):
        """Start collecting orderbook data"""
        self.running = True
        logger.info(f"Starting orderbook collector for {self.symbol}")
        
        # Start WebSocket connections
        self._connect_websockets()
        
        # Start snapshot loop
        snapshot_thread = threading.Thread(target=self._snapshot_loop)
        snapshot_thread.start()
        
        # Start rotation loop
        rotation_thread = threading.Thread(target=self._rotation_loop)
        rotation_thread.start()
        
        # Wait for threads
        try:
            snapshot_thread.join()
            rotation_thread.join()
        except KeyboardInterrupt:
            logger.info("Stopping collector...")
            self.stop()
            
    def stop(self):
        """Stop collecting data"""
        self.running = False
        if self.ws_orderbook:
            self.ws_orderbook.stop()
        if self.ws_trades:
            self.ws_trades.stop()
        
        # Save remaining data
        self._rotate_files(force=True)
        logger.info("Collector stopped")
        
    def _connect_websockets(self):
        """Connect to Binance WebSocket streams"""
        try:
            # Initialize orderbook from REST API
            self._initialize_orderbook()
            
            # Connect to depth stream
            self.ws_orderbook = UMFuturesWebsocketClient(
                on_message=self._on_depth_message,
                on_close=self._on_close,
                on_error=self._on_error
            )
            self.ws_orderbook.partial_book_depth(
                symbol=self.symbol.lower(),
                level=self.depth_levels,
                speed=100  # 100ms updates
            )
            
            # Connect to trades stream
            self.ws_trades = UMFuturesWebsocketClient(
                on_message=self._on_trade_message,
                on_close=self._on_close,
                on_error=self._on_error
            )
            self.ws_trades.agg_trade(symbol=self.symbol.lower())
            
            logger.info("WebSocket connections established")
            self.reconnect_delay = 1  # Reset delay on successful connection
            
        except Exception as e:
            logger.error(f"Failed to connect WebSockets: {e}")
            self._schedule_reconnect()
            
    def _initialize_orderbook(self):
        """Initialize orderbook from REST API snapshot"""
        try:
            snapshot = self.rest_client.depth(
                symbol=self.symbol,
                limit=self.depth_levels
            )
            
            self.current_orderbook['lastUpdateId'] = snapshot['lastUpdateId']
            self.current_orderbook['bids'] = {
                float(price): float(qty) for price, qty in snapshot['bids']
            }
            self.current_orderbook['asks'] = {
                float(price): float(qty) for price, qty in snapshot['asks']
            }
            
            logger.info(f"Orderbook initialized with updateId: {snapshot['lastUpdateId']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize orderbook: {e}")
            
    def _on_depth_message(self, _, message):
        """Handle depth update messages"""
        try:
            data = json.loads(message)
            
            # Update orderbook
            for bid in data.get('b', []):
                price, qty = float(bid[0]), float(bid[1])
                if qty == 0:
                    self.current_orderbook['bids'].pop(price, None)
                else:
                    self.current_orderbook['bids'][price] = qty
                    
            for ask in data.get('a', []):
                price, qty = float(ask[0]), float(ask[1])
                if qty == 0:
                    self.current_orderbook['asks'].pop(price, None)
                else:
                    self.current_orderbook['asks'][price] = qty
                    
            self.current_orderbook['lastUpdateId'] = data.get('u', self.current_orderbook['lastUpdateId'])
            
        except Exception as e:
            logger.error(f"Error processing depth message: {e}")
            
    def _on_trade_message(self, _, message):
        """Handle trade messages"""
        try:
            data = json.loads(message)
            
            trade = {
                'timestamp': data['T'],
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m'],
                'trade_id': data['a']
            }
            
            self.trades_buffer.append(trade)
            self.metrics['trades_collected'] += 1
            
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
            
    def _on_close(self, _):
        """Handle WebSocket close"""
        logger.warning("WebSocket connection closed")
        self.metrics['disconnections'] += 1
        if self.running:
            self._schedule_reconnect()
            
    def _on_error(self, _, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        
    def _schedule_reconnect(self):
        """Schedule WebSocket reconnection with exponential backoff"""
        logger.info(f"Scheduling reconnect in {self.reconnect_delay} seconds")
        threading.Timer(self.reconnect_delay, self._connect_websockets).start()
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
    def _snapshot_loop(self):
        """Take orderbook snapshots every second"""
        while self.running:
            try:
                # Create snapshot
                snapshot = self._create_snapshot()
                if snapshot:
                    self.orderbook_buffer.append(snapshot)
                    self.metrics['snapshots_collected'] += 1
                    self.metrics['last_snapshot_time'] = time.time()
                    
                # Wait for next second
                time.sleep(1.0 - (time.time() % 1.0))
                
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
                time.sleep(1)
                
    def _create_snapshot(self) -> Optional[Dict]:
        """Create orderbook snapshot with features"""
        if not self.current_orderbook['bids'] or not self.current_orderbook['asks']:
            return None
            
        try:
            timestamp = int(time.time() * 1000)
            
            # Sort orderbook
            bids = sorted(self.current_orderbook['bids'].items(), key=lambda x: x[0], reverse=True)[:self.depth_levels]
            asks = sorted(self.current_orderbook['asks'].items(), key=lambda x: x[0])[:self.depth_levels]
            
            # Calculate features
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            spread_pct = spread / mid_price * 100 if mid_price else 0
            
            # Calculate depth imbalance
            bid_volume = sum(qty for _, qty in bids[:5])
            ask_volume = sum(qty for _, qty in asks[:5])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # Get recent trades
            recent_trades = list(self.trades_buffer)
            if recent_trades:
                trades_1s = [t for t in recent_trades if timestamp - t['timestamp'] <= 1000]
                buy_volume = sum(t['quantity'] for t in trades_1s if not t['is_buyer_maker'])
                sell_volume = sum(t['quantity'] for t in trades_1s if t['is_buyer_maker'])
                vwap = sum(t['price'] * t['quantity'] for t in trades_1s) / sum(t['quantity'] for t in trades_1s) if trades_1s else mid_price
            else:
                buy_volume = sell_volume = 0
                vwap = mid_price
                
            snapshot = {
                'timestamp': timestamp,
                'update_id': self.current_orderbook['lastUpdateId'],
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'imbalance_5': imbalance,
                'buy_volume_1s': buy_volume,
                'sell_volume_1s': sell_volume,
                'vwap_1s': vwap,
                'bids': {f'bid_{i}': {'price': p, 'qty': q} for i, (p, q) in enumerate(bids)},
                'asks': {f'ask_{i}': {'price': p, 'qty': q} for i, (p, q) in enumerate(asks)}
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None
            
    def _rotation_loop(self):
        """Rotate output files every hour"""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                if (now - self.last_rotation).total_seconds() >= self.rotation_minutes * 60:
                    self._rotate_files()
                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                
    def _rotate_files(self, force=False):
        """Save buffers to parquet files"""
        if not self.orderbook_buffer and not force:
            return
            
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Save orderbook snapshots
            if self.orderbook_buffer:
                df_snapshots = pd.DataFrame(list(self.orderbook_buffer))
                
                # Flatten nested bid/ask data
                for i in range(self.depth_levels):
                    df_snapshots[f'bid_{i}_price'] = df_snapshots['bids'].apply(
                        lambda x: x.get(f'bid_{i}', {}).get('price', np.nan) if x else np.nan
                    )
                    df_snapshots[f'bid_{i}_qty'] = df_snapshots['bids'].apply(
                        lambda x: x.get(f'bid_{i}', {}).get('qty', np.nan) if x else np.nan
                    )
                    df_snapshots[f'ask_{i}_price'] = df_snapshots['asks'].apply(
                        lambda x: x.get(f'ask_{i}', {}).get('price', np.nan) if x else np.nan
                    )
                    df_snapshots[f'ask_{i}_qty'] = df_snapshots['asks'].apply(
                        lambda x: x.get(f'ask_{i}', {}).get('qty', np.nan) if x else np.nan
                    )
                    
                df_snapshots = df_snapshots.drop(['bids', 'asks'], axis=1)
                
                # Save to parquet
                filename = f"{self.symbol}_orderbook_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
                filepath = self.output_dir / filename
                
                table = pa.Table.from_pandas(df_snapshots)
                pq.write_table(table, filepath, compression='snappy')
                
                logger.info(f"Saved {len(df_snapshots)} snapshots to {filename}")
                self.orderbook_buffer.clear()
                
            # Save trades
            if self.trades_buffer:
                df_trades = pd.DataFrame(list(self.trades_buffer))
                
                filename = f"{self.symbol}_trades_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
                filepath = self.output_dir / filename
                
                table = pa.Table.from_pandas(df_trades)
                pq.write_table(table, filepath, compression='snappy')
                
                logger.info(f"Saved {len(df_trades)} trades to {filename}")
                self.trades_buffer.clear()
                
            self.last_rotation = timestamp
            
        except Exception as e:
            logger.error(f"Error rotating files: {e}")
            
    def get_metrics(self) -> Dict:
        """Get collector metrics"""
        return {
            **self.metrics,
            'buffer_size': len(self.orderbook_buffer),
            'trades_buffer_size': len(self.trades_buffer),
            'running': self.running
        }

def main():
    """Main entry point"""
    collector = OrderbookCollector(
        symbol="BTCUSDT",
        depth_levels=20,
        output_dir="./data/raw",
        rotation_minutes=60
    )
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        collector.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start collecting
    collector.start()

if __name__ == "__main__":
    main()