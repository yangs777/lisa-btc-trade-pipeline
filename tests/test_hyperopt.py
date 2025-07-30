"""Test coverage for hyperparameter optimization module."""

import pytest
from typing import Dict, Any, List, Tuple, Union
from src.optimization.hyperopt import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return float(params["x"] ** 2)
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
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
            return float((x - 2) ** 2 + (y - 3) ** 2)
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
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
        
        assert best_params is not None
        assert "x" in best_params
        assert "y" in best_params
        assert 0 <= best_params["x"] <= 5
        assert 0 <= best_params["y"] <= 5
        assert optimizer.best_score < 2.0  # Should be close to 0
        assert len(optimizer.history) == 100
    
    def test_grid_search(self) -> None:
        """Test grid search optimization."""
        def objective(params: Dict[str, Any]) -> float:
            return float(abs(params["x"] - 2.5) + abs(params["category"] - 2))
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
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
            return float((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
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
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
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
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
            "x": (0.0, 1.0),
            "y": [1, 2, 3]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=1,
            method="grid"
        )
        
        grid = optimizer._generate_grid()
        
        assert len(grid) > 0
        # Grid should have all combinations
        y_values = set()
        for params in grid:
            assert 0.0 <= params["x"] <= 1.0
            assert params["y"] in [1, 2, 3]
            y_values.add(params["y"])
        
        # Should have all y values
        assert y_values == {1, 2, 3}
    
    def test_invalid_method(self) -> None:
        """Test invalid optimization method."""
        def dummy_objective(params: Dict[str, Any]) -> float:
            return 0.0
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
            "x": (0.0, 1.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=dummy_objective,
            search_space=search_space,
            n_trials=1,
            method="invalid"
        )
        
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.optimize()
    
    def test_mixed_search_space(self) -> None:
        """Test optimization with mixed parameter types."""
        def objective(params: Dict[str, Any]) -> float:
            # Complex objective with multiple parameter types
            x = params["learning_rate"]
            layers = params["n_layers"]
            activation = params["activation"]
            
            # Penalty for certain combinations
            penalty = 0
            if activation == "sigmoid" and layers > 3:
                penalty = 10
            
            return float(abs(x - 0.01) * 100 + abs(layers - 3) * 5 + penalty)
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
            "learning_rate": (0.0001, 0.1),
            "n_layers": [2, 3, 4, 5],
            "activation": ["relu", "tanh", "sigmoid"]
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=200,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert 0.0001 <= best_params["learning_rate"] <= 0.1
        assert best_params["n_layers"] in [2, 3, 4, 5]
        assert best_params["activation"] in ["relu", "tanh", "sigmoid"]
        
        # Should find a reasonable solution
        assert optimizer.best_score < 20
    
    def test_early_stopping(self) -> None:
        """Test early stopping functionality."""
        call_count = 0
        
        def objective(params: Dict[str, Any]) -> float:
            nonlocal call_count
            call_count += 1
            
            # Return 0 on 10th call to trigger early stopping
            if call_count == 10:
                return 0.0
            
            return float(params["x"] ** 2)
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
            "x": (-5.0, 5.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=100,
            method="random"
        )
        
        # Add early stopping (if implemented)
        best_params = optimizer.optimize()
        
        assert best_params is not None
        assert optimizer.best_score == 0.0
        # Should have run all trials (no early stopping by default)
        assert call_count == 100
    
    def test_constraint_handling(self) -> None:
        """Test handling of parameter constraints."""
        def objective(params: Dict[str, Any]) -> float:
            x = params["x"]
            y = params["y"]
            
            # Constraint: x + y <= 5
            if x + y > 5:
                return float('inf')  # Infeasible
            
            return float(-(x * y))  # Maximize x*y
        
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]] = {
            "x": (0.0, 5.0),
            "y": (0.0, 5.0)
        }
        
        optimizer = HyperparameterOptimizer(
            objective=objective,
            search_space=search_space,
            n_trials=500,
            method="random"
        )
        
        best_params = optimizer.optimize()
        
        assert best_params is not None
        # Should satisfy constraint
        assert best_params["x"] + best_params["y"] <= 5.01  # Small tolerance
        # Should be near optimal (x=y=2.5)
        assert abs(best_params["x"] - 2.5) < 0.5
        assert abs(best_params["y"] - 2.5) < 0.5