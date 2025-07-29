"""Test coverage for hyperparameter optimization module."""

import pytest
from typing import Dict, Any, List
from src.optimization.hyperopt import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return params["x"] ** 2
        
        search_space = {
            "x": (-5.0, 5.0),
            "y": [1, 2, 3, 4]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=50,
            method="random"
        )
        
        assert optimizer.n_trials == 50
        assert optimizer.method == "random"
        assert optimizer.best_params is None
        assert optimizer.best_score == float('inf')
        assert len(optimizer.history) == 0
    
    def test_random_search(self) -> None:
        """Test random search optimization."""
        # Simple quadratic objective
        def objective(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            return (x - 2) ** 2 + (y - 3) ** 2
        
        search_space = {
            "x": (0.0, 5.0),
            "y": (0.0, 5.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=100,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        # Should find something close to optimal (2, 3)
        assert best_params is not None
        assert 1.0 <= best_params["x"] <= 3.0
        assert 2.0 <= best_params["y"] <= 4.0
        assert optimizer.best_score < 2.0  # Should be close to 0
        assert len(optimizer.history) == 100
    
    def test_grid_search(self) -> None:
        """Test grid search optimization."""
        def objective(params: Dict[str, Any]) -> float:
            return abs(params["x"] - 2.5) + abs(params["category"] - 2)
        
        search_space = {
            "x": (0.0, 5.0),
            "category": [1, 2, 3]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=100,
            method="grid"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert best_params["category"] == 2  # Optimal category
        assert 2.0 <= best_params["x"] <= 3.0  # Close to 2.5
        assert optimizer.best_score < 1.0
    
    def test_bayesian_search(self) -> None:
        """Test Bayesian optimization."""
        def objective(params: Dict[str, Any]) -> float:
            # Rosenbrock function
            x = params["x"]
            y = params["y"]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        
        search_space = {
            "x": (-2.0, 2.0),
            "y": (-1.0, 3.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=50,
            method="bayesian"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        # Bayesian should do better than random
        assert optimizer.best_score < 50  # Reasonable bound
        assert len(optimizer.history) == 50
    
    def test_sample_params(self) -> None:
        """Test parameter sampling."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        search_space = {
            "continuous": (0.0, 10.0),
            "integer": (1, 5),
            "categorical": ["a", "b", "c"]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=1,
            method="random"
        )
        
        # Sample multiple times
        for _ in range(100):
            params = optimizer._sample_params()
            
            # Check continuous parameter
            assert 0.0 <= params["continuous"] <= 10.0
            assert isinstance(params["continuous"], float)
            
            # Check integer parameter
            assert 1 <= params["integer"] <= 5
            assert isinstance(params["integer"], int)
            
            # Check categorical parameter
            assert params["categorical"] in ["a", "b", "c"]
    
    def test_generate_grid(self) -> None:
        """Test grid generation."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        search_space = {
            "x": (0.0, 1.0),
            "y": [1, 2, 3]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=100,
            method="grid"
        )
        
        grid = optimizer._generate_grid()
        
        # Should have combinations of x values and y categories
        assert len(grid) >= 30  # At least 10 x values * 3 y values
        
        # Check all y values are represented
        y_values = set(p["y"] for p in grid)
        assert y_values == {1, 2, 3}
        
        # Check x values span the range
        x_values = [p["x"] for p in grid]
        assert min(x_values) == 0.0
        assert max(x_values) == 1.0
    
    def test_sample_near_best(self) -> None:
        """Test sampling near best parameters."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        search_space = {
            "x": (0.0, 10.0),
            "category": ["a", "b", "c"]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=1,
            method="bayesian"
        )
        
        # Set best params
        optimizer.best_params = {"x": 5.0, "category": "b"}
        
        # Sample near best multiple times
        samples = []
        for _ in range(100):
            params = optimizer._sample_near_best()
            samples.append(params)
        
        # Most x values should be near 5.0
        x_values = [s["x"] for s in samples]
        x_mean = sum(x_values) / len(x_values)
        assert 4.0 <= x_mean <= 6.0
        
        # Most categories should be "b"
        b_count = sum(1 for s in samples if s["category"] == "b")
        assert b_count > 50  # More than half
    
    def test_objective_failure_handling(self) -> None:
        """Test handling of objective function failures."""
        call_count = 0
        
        def failing_objective(params: Dict[str, Any]) -> float:
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("Simulated failure")
            return params["x"] ** 2
        
        search_space = {"x": (0.0, 5.0)}
        
        optimizer = HyperparameterOptimizer(
            objective=failing_objective,
            search_space=search_space,
            n_trials=30,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        # Should still find reasonable result despite failures
        assert best_params is not None
        assert optimizer.best_score < 10.0
        # History should have fewer entries due to failures
        assert len(optimizer.history) < 30
    
    def test_invalid_method(self) -> None:
        """Test invalid optimization method."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space={"x": (0.0, 1.0)},
            n_trials=10,
            method="invalid_method"
        )
        
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.optimize()
    
    def test_empty_search_space(self) -> None:
        """Test with empty search space."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space={},
            n_trials=10,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        # Should return empty params
        assert best_params == {}
    
    def test_integer_parameter_handling(self) -> None:
        """Test integer parameter optimization."""
        def objective(params: Dict[str, Any]) -> float:
            # Optimal at x=3
            return abs(params["x"] - 3)
        
        search_space = {"x": (1, 5)}  # Integer range
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=50,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert best_params["x"] == 3  # Should find optimal
        assert isinstance(best_params["x"], int)
        assert optimizer.best_score == 0.0
    
    def test_mixed_parameter_types(self) -> None:
        """Test optimization with mixed parameter types."""
        def objective(params: Dict[str, Any]) -> float:
            score = abs(params["float_param"] - 2.5)
            score += abs(params["int_param"] - 3)
            score += 0 if params["cat_param"] == "optimal" else 1
            return score
        
        search_space = {
            "float_param": (0.0, 5.0),
            "int_param": (1, 5),
            "cat_param": ["bad1", "bad2", "optimal", "bad3"]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=200,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert 2.0 <= best_params["float_param"] <= 3.0
        assert best_params["int_param"] == 3
        assert best_params["cat_param"] == "optimal"
        assert optimizer.best_score < 1.0