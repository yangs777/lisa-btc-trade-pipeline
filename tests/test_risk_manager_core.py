"""Core tests for risk management module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.risk_management.risk_manager import RiskManager
from src.risk_management.models.position_sizing import KellyPositionSizer


class TestRiskManager:
    """Test cases for RiskManager."""
    
    @pytest.fixture
    def mock_position_sizer(self):
        """Mock position sizer."""
        sizer = Mock()
        sizer.calculate_position_size.return_value = 0.1  # 0.1 BTC
        return sizer
    
    @pytest.fixture
    def mock_cost_model(self):
        """Mock cost model."""
        model = Mock()
        model.calculate_entry_costs.return_value = {
            'maker_fee': 10.0,
            'taker_fee': 0.0,
            'total_cost': 10.0
        }
        model.calculate_round_trip_costs.return_value = {
            'total_cost': 25.0
        }
        return model
    
    @pytest.fixture
    def risk_manager(self, mock_position_sizer):
        """Create risk manager with mocked dependencies."""
        with patch('src.risk_management.risk_manager.CostModel') as mock_cost_cls, \
             patch('src.risk_management.risk_manager.DrawdownGuard') as mock_dd_cls, \
             patch('src.risk_management.risk_manager.BinanceAPIThrottler') as mock_api_cls:
            
            # Configure mocks
            mock_dd = Mock()
            mock_dd.get_risk_multiplier.return_value = 1.0
            mock_dd.drawdown_triggered = False
            mock_dd_cls.return_value = mock_dd
            
            mock_cost = Mock()
            mock_cost.calculate_entry_costs.return_value = {
                'maker_fee': 10.0,
                'taker_fee': 0.0,
                'total_cost': 10.0
            }
            mock_cost.calculate_round_trip_costs.return_value = {
                'total_cost': 25.0
            }
            mock_cost_cls.return_value = mock_cost
            
            mock_api = Mock()
            mock_api.get_remaining_capacity.return_value = 10
            mock_api_cls.return_value = mock_api
            
            rm = RiskManager(position_sizer=mock_position_sizer)
            rm.cost_model = mock_cost
            rm.drawdown_guard = mock_dd
            rm.api_throttler = mock_api
            
            return rm
    
    def test_init(self):
        """Test initialization."""
        rm = RiskManager(
            max_drawdown=0.15,
            max_daily_loss=0.03,
            max_position_count=5
        )
        
        assert rm.max_daily_loss == 0.03
        assert rm.max_position_count == 5
        assert rm.correlation_limit == 0.7
        assert rm.daily_pnl == 0.0
        assert len(rm.open_positions) == 0
        assert isinstance(rm.position_sizer, KellyPositionSizer)
    
    def test_check_new_position_approved(self, risk_manager):
        """Test approving a new position."""
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is True
        assert result['position_size'] == 0.1
        assert result['notional_value'] == 4000  # 0.1 * 40000
        assert 'entry_costs' in result
        assert result['risk_multiplier'] == 1.0
        assert result['adjusted_confidence'] == 0.8
    
    def test_check_new_position_drawdown_triggered(self, risk_manager):
        """Test rejection when drawdown guard is triggered."""
        risk_manager.drawdown_guard.drawdown_triggered = True
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is False
        assert result['reason'] == 'Drawdown guard triggered'
        assert result['position_size'] == 0.0
    
    def test_check_new_position_daily_loss_exceeded(self, risk_manager):
        """Test rejection when daily loss limit exceeded."""
        risk_manager.daily_pnl = -6000  # 6% loss on 100k portfolio
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is False
        assert result['reason'] == 'Daily loss limit exceeded'
    
    def test_check_new_position_max_positions_reached(self, risk_manager):
        """Test rejection when max positions reached."""
        # Add max positions
        for i in range(3):
            risk_manager.open_positions[f'pos_{i}'] = {
                'symbol': 'BTCUSDT',
                'size': 0.1,
                'entry_price': 40000
            }
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is False
        assert result['reason'] == 'Maximum positions reached'
    
    def test_check_new_position_insufficient_api_capacity(self, risk_manager):
        """Test rejection when API capacity is low."""
        risk_manager.api_throttler.get_remaining_capacity.return_value = 1
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is False
        assert result['reason'] == 'Insufficient API capacity'
    
    def test_check_new_position_below_minimum(self, risk_manager):
        """Test rejection when position size is too small."""
        # Make position sizer return tiny amount
        risk_manager.position_sizer.calculate_position_size.return_value = 0.00001
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert result['approved'] is False
        assert result['reason'] == 'Position size below minimum'
    
    def test_check_new_position_risk_adjustment(self, risk_manager):
        """Test position size adjustment based on risk multiplier."""
        risk_manager.drawdown_guard.get_risk_multiplier.return_value = 0.5
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        # Check that confidence was adjusted
        assert result['risk_multiplier'] == 0.5
        assert result['adjusted_confidence'] == 0.4  # 0.8 * 0.5
        
        # Verify position sizer was called with adjusted confidence
        risk_manager.position_sizer.calculate_position_size.assert_called_with(
            portfolio_value=100000,
            price=40000,
            confidence=0.4
        )
    
    def test_record_position_open(self, risk_manager):
        """Test recording a new position."""
        risk_manager.record_position_open(
            position_id='pos_001',
            symbol='BTCUSDT',
            size=0.1,
            entry_price=40000,
            position_type='long',
            stop_loss=39000,
            take_profit=42000
        )
        
        assert 'pos_001' in risk_manager.open_positions
        position = risk_manager.open_positions['pos_001']
        assert position['symbol'] == 'BTCUSDT'
        assert position['size'] == 0.1
        assert position['entry_price'] == 40000
        assert position['position_type'] == 'long'
        assert position['stop_loss'] == 39000
        assert position['take_profit'] == 42000
        assert position['realized_pnl'] == 0.0
    
    def test_record_position_close_full(self, risk_manager):
        """Test closing a full position."""
        # First open a position
        risk_manager.record_position_open(
            position_id='pos_001',
            symbol='BTCUSDT',
            size=0.1,
            entry_price=40000,
            position_type='long'
        )
        
        # Close with profit
        result = risk_manager.record_position_close(
            position_id='pos_001',
            exit_price=41000
        )
        
        # Check P&L calculation
        assert result['gross_pnl'] == 100  # (41000 - 40000) * 0.1
        assert result['net_pnl'] == 75  # 100 - 25 (costs)
        assert 'pos_001' not in risk_manager.open_positions
        assert risk_manager.daily_pnl == 75
    
    def test_record_position_close_partial(self, risk_manager):
        """Test partial position close."""
        # Open position
        risk_manager.record_position_open(
            position_id='pos_001',
            symbol='BTCUSDT',
            size=0.2,
            entry_price=40000,
            position_type='long'
        )
        
        # Close half
        result = risk_manager.record_position_close(
            position_id='pos_001',
            exit_price=41000,
            exit_size=0.1
        )
        
        assert result['gross_pnl'] == 100
        assert 'pos_001' in risk_manager.open_positions
        assert risk_manager.open_positions['pos_001']['size'] == 0.1
    
    def test_record_position_close_short(self, risk_manager):
        """Test closing a short position."""
        # Open short
        risk_manager.record_position_open(
            position_id='pos_001',
            symbol='BTCUSDT',
            size=0.1,
            entry_price=40000,
            position_type='short'
        )
        
        # Close with profit (price went down)
        result = risk_manager.record_position_close(
            position_id='pos_001',
            exit_price=39000
        )
        
        assert result['gross_pnl'] == 100  # (40000 - 39000) * 0.1
    
    def test_update_portfolio_value(self, risk_manager):
        """Test portfolio value update and metrics calculation."""
        # Add some positions
        risk_manager.open_positions['pos_1'] = {
            'size': 0.1,
            'entry_price': 40000
        }
        risk_manager.open_positions['pos_2'] = {
            'size': 0.05,
            'entry_price': 41000
        }
        risk_manager.daily_pnl = -500
        
        metrics = risk_manager.update_portfolio_value(100000)
        
        assert metrics['portfolio_value'] == 100000
        assert metrics['open_positions'] == 2
        assert metrics['open_exposure'] == 6050  # 0.1*40000 + 0.05*41000
        assert metrics['exposure_pct'] == 6.05
        assert metrics['daily_pnl'] == -500
        assert metrics['daily_return'] == -0.5
    
    def test_daily_reset(self, risk_manager):
        """Test daily P&L reset."""
        risk_manager.daily_pnl = -1000
        risk_manager.last_reset_date = datetime.now().date() - timedelta(days=1)
        
        # Trigger reset by checking new position
        risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.last_reset_date == datetime.now().date()
    
    def test_get_risk_report(self, risk_manager):
        """Test comprehensive risk report generation."""
        # Setup some data
        risk_manager.open_positions['pos_1'] = {
            'size': 0.1,
            'entry_price': 40000
        }
        risk_manager.daily_pnl = -200
        
        # Mock API metrics
        risk_manager.api_throttler.get_metrics.return_value = {
            'requests_used': 100,
            'requests_limit': 1200
        }
        
        # Mock cost estimates
        risk_manager.cost_model.estimate_annual_costs.return_value = {
            'annual_trading_fees': 10000,
            'annual_funding_costs': 5000
        }
        
        report = risk_manager.get_risk_report()
        
        assert 'timestamp' in report
        assert report['portfolio_metrics']['open_positions'] == 1
        assert report['portfolio_metrics']['daily_pnl'] == -200
        assert report['risk_limits']['max_drawdown'] == 0.1
        assert report['risk_limits']['max_daily_loss'] == 0.05
        assert report['risk_limits']['max_positions'] == 3
        assert 'api_metrics' in report
        assert 'cost_estimates' in report
    
    def test_check_new_position_with_volatility_sizer(self, risk_manager):
        """Test position sizing with volatility parity sizer."""
        from src.risk_management.models.position_sizing import VolatilityParityPositionSizer
        
        vol_sizer = Mock(spec=VolatilityParityPositionSizer)
        vol_sizer.calculate_position_size.return_value = 0.08
        risk_manager.position_sizer = vol_sizer
        
        # Add some historical returns
        risk_manager.historical_returns['BTCUSDT'] = [0.01, -0.02, 0.015, -0.01, 0.02]
        
        result = risk_manager.check_new_position(
            symbol='BTCUSDT',
            portfolio_value=100000,
            current_price=40000,
            signal_confidence=0.8
        )
        
        # Check that returns were passed to sizer
        vol_sizer.calculate_position_size.assert_called_with(
            portfolio_value=100000,
            price=40000,
            confidence=0.8,
            returns=risk_manager.historical_returns['BTCUSDT']
        )
        
        assert result['approved'] is True
        assert result['position_size'] == 0.08