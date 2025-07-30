"""Tests for hyperparameter optimization module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.optimization.hyperopt import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test cases for HyperparameterOptimizer."""
    
    def test_init(self):
        """Test initialization."""
        def dummy_objective(params):
            return params['x'] ** 2
        
        search_space = {
            'x': (-10.0, 10.0),
            'y': [1, 2, 3, 4, 5]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=50,
            method='random'
        )
        
        assert optimizer.objective == dummy_objective
        assert optimizer.search_space == search_space
        assert optimizer.n_trials == 50
        assert optimizer.method == 'random'
        assert optimizer.best_params is None
        assert optimizer.best_score == float('inf')
        assert optimizer.history == []
    
    def test_random_search(self):
        """Test random search optimization."""
        # Simple quadratic objective
        def objective(params):
            return (params['x'] - 5) ** 2 + (params['y'] - 3) ** 2
        
        search_space = {
            'x': (0.0, 10.0),
            'y': (0.0, 10.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=100,
            method='random'
        )
        
        best_params = optimizer.optimize()
        
        # Check that optimization ran
        assert best_params is not None
        assert 'x' in best_params
        assert 'y' in best_params
        assert len(optimizer.history) == 100
        
        # Best params should be reasonably close to optimum (5, 3)
        assert 2 <= best_params['x'] <= 8
        assert 1 <= best_params['y'] <= 5
        assert optimizer.best_score < 20  # Should find something decent
    
    def test_grid_search(self):
        """Test grid search optimization."""
        def objective(params):
            # Minimize distance from (2, 'b')
            x_loss = (params['x'] - 2) ** 2
            y_loss = 0 if params['y'] == 'b' else 10
            return x_loss + y_loss
        
        search_space = {
            'x': [1, 2, 3, 4],
            'y': ['a', 'b', 'c']
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=20,  # Will be overridden by grid size
            method='grid'
        )
        
        best_params = optimizer.optimize()
        
        # Should find optimal solution
        assert best_params['x'] == 2
        assert best_params['y'] == 'b'
        assert optimizer.best_score == 0
        
        # Should evaluate all combinations (4 * 3 = 12)
        assert len(optimizer.history) == 12
    
    def test_sample_params(self):
        """Test parameter sampling."""
        search_space = {
            'continuous': (0.0, 1.0),
            'discrete': [10, 20, 30, 40],
            'categorical': ['adam', 'sgd', 'rmsprop']
        }
        
        optimizer = HyperparameterOptimizer(
            objective=lambda x: 0,
            search_space=search_space,
            n_trials=10
        )
        
        # Test multiple samples
        for _ in range(50):
            params = optimizer._sample_params()
            
            # Check continuous parameter
            assert 0.0 <= params['continuous'] <= 1.0
            
            # Check discrete parameter
            assert params['discrete'] in [10, 20, 30, 40]
            
            # Check categorical parameter
            assert params['categorical'] in ['adam', 'sgd', 'rmsprop']
    
    def test_bayesian_search(self):
        """Test Bayesian optimization (basic version)."""
        def objective(params):
            # Rosenbrock function
            x, y = params['x'], params['y']
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        
        search_space = {
            'x': (-2.0, 2.0),
            'y': (-1.0, 3.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=50,
            method='bayesian'
        )
        
        # Bayesian search should at least run without error
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert len(optimizer.history) == 50
        assert optimizer.best_score < 100  # Should find something reasonable
    
    def test_history_tracking(self):
        """Test that optimization history is properly tracked."""
        call_count = 0
        
        def objective(params):
            nonlocal call_count
            call_count += 1
            return call_count * params['x']
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space={'x': (0.0, 1.0)},
            n_trials=10,
            method='random'
        )
        
        optimizer.optimize()
        
        # Check history
        assert len(optimizer.history) == 10
        for i, entry in enumerate(optimizer.history):
            assert 'params' in entry
            assert 'score' in entry
            assert 'trial' in entry
            assert entry['trial'] == i
    
    def test_minimize_vs_maximize(self):
        """Test both minimization and maximization."""
        def objective(params):
            return -params['x'] ** 2  # Maximum at x=0
        
        # Currently minimizes by default
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space={'x': (-5.0, 5.0)},
            n_trials=50,
            method='random'
        )
        
        best_params = optimizer.optimize()
        
        # Should find x near boundaries (±5) for minimum
        assert abs(best_params['x']) > 3
    
    def test_invalid_method(self):
        """Test error handling for invalid optimization method."""
        optimizer = HyperparameterOptimizer(
            objective=lambda x: 0,
            search_space={'x': (0, 1)},
            n_trials=10,
            method='invalid_method'
        )
        
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.optimize()
    
    def test_get_best_n_trials(self):
        """Test getting best N trials from history."""
        def objective(params):
            return params['x'] ** 2
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space={'x': (-10.0, 10.0)},
            n_trials=20,
            method='random'
        )
        
        optimizer.optimize()
        
        # Get best 5 trials
        best_trials = optimizer.get_best_trials(n=5)
        
        assert len(best_trials) == 5
        # Should be sorted by score
        scores = [trial['score'] for trial in best_trials]
        assert scores == sorted(scores)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        def objective(params):
            # Return 0 immediately to trigger early stopping
            return 0.0
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space={'x': (0, 1)},
            n_trials=100,
            method='random'
        )
        
        # Add early stopping logic
        optimizer.early_stopping_patience = 10
        optimizer.early_stopping_threshold = 0.01
        
        optimizer.optimize()
        
        # Should stop early
        assert len(optimizer.history) < 100