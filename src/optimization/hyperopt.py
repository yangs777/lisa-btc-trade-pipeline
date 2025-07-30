"""Hyperparameter optimization utilities."""

from typing import Dict, Any, Callable, Union, List, Tuple, Optional
import numpy as np
import random


class HyperparameterOptimizer:
    """Optimize hyperparameters using various methods."""
    
    def __init__(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Union[Tuple[float, float], List[Any]]],
        n_trials: int = 100,
        method: str = "random"
    ):
        """Initialize optimizer.
        
        Args:
            objective: Objective function to minimize
            search_space: Dictionary defining parameter ranges
            n_trials: Number of optimization trials
            method: Optimization method (random, grid, bayesian)
        """
        self.objective = objective
        self.search_space = search_space
        self.n_trials = n_trials
        self.method = method
        
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('inf')
        self.history: List[Dict[str, Any]] = []
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization.
        
        Returns:
            Best parameters found
        """
        if self.method == "random":
            return self._random_search()
        elif self.method == "grid":
            return self._grid_search()
        elif self.method == "bayesian":
            return self._bayesian_search()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _random_search(self) -> Dict[str, Any]:
        """Random search optimization."""
        for trial in range(self.n_trials):
            # Sample parameters
            params = self._sample_params()
            
            # Evaluate
            try:
                score = self.objective(params)
            except Exception as e:
                print(f"Trial {trial} failed: {e}")
                continue
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            # Store history
            self.history.append({
                "trial": trial,
                "params": params,
                "score": score
            })
        
        return self.best_params or {}
    
    def _grid_search(self) -> Dict[str, Any]:
        """Grid search optimization."""
        # Generate grid
        param_grid = self._generate_grid()
        
        # Evaluate all combinations
        for i, params in enumerate(param_grid):
            if i >= self.n_trials:
                break
            
            try:
                score = self.objective(params)
            except Exception as e:
                print(f"Grid point {i} failed: {e}")
                continue
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            # Store history
            self.history.append({
                "trial": i,
                "params": params,
                "score": score
            })
        
        return self.best_params or {}
    
    def _bayesian_search(self) -> Dict[str, Any]:
        """Bayesian optimization (simplified version)."""
        # Start with random search for initial points
        n_random = min(10, self.n_trials // 3)
        
        for trial in range(n_random):
            params = self._sample_params()
            score = self.objective(params)
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            self.history.append({
                "trial": trial,
                "params": params,
                "score": score
            })
        
        # Continue with guided search (mock implementation)
        for trial in range(n_random, self.n_trials):
            # In real implementation, would use Gaussian Process
            # For now, sample near best params
            params = self._sample_near_best()
            
            try:
                score = self.objective(params)
            except Exception:
                continue
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            self.history.append({
                "trial": trial,
                "params": params,
                "score": score
            })
        
        return self.best_params or {}
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params: Dict[str, Any] = {}
        
        for param_name, param_range in self.search_space.items():
            if isinstance(param_range, tuple):
                # Continuous parameter
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = random.randint(low, high)
                else:
                    params[param_name] = float(random.uniform(low, high))
            elif isinstance(param_range, List[Any]):
                # Categorical parameter
                params[param_name] = random.choice(param_range)
            else:
                raise ValueError(f"Invalid range for {param_name}")
        
        return params
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid."""
        import itertools
        
        # Create discrete values for each parameter
        param_values = {}
        for param_name, param_range in self.search_space.items():
            if isinstance(param_range, tuple):
                # Create grid points for continuous
                low, high = param_range
                n_points = min(10, int(np.sqrt(self.n_trials)))
                if isinstance(low, int) and isinstance(high, int):
                    values = np.linspace(low, high, n_points, dtype=int).tolist()
                else:
                    values = np.linspace(low, high, n_points).tolist()
                param_values[param_name] = values
            elif isinstance(param_range, List[Any]):
                param_values[param_name] = param_range
        
        # Generate all combinations
        keys = List[Any](param_values.keys())
        values = [param_values[k] for k in keys]
        
        grid = []
        for combination in itertools.product(*values):
            params = Dict[str, Any](zip(keys, combination))
            grid.append(params)
        
        # Shuffle to randomize order
        random.shuffle(grid)
        
        return grid
    
    def _sample_near_best(self) -> Dict[str, Any]:
        """Sample near best parameters."""
        if self.best_params is None:
            return self._sample_params()
        
        params = {}
        for param_name, param_range in self.search_space.items():
            if isinstance(param_range, tuple):
                # Add noise to best value
                best_value = self.best_params[param_name]
                low, high = param_range
                noise = (high - low) * 0.1 * random.gauss(0, 1)
                new_value = best_value + noise
                new_value = max(low, min(high, new_value))
                
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = int(new_value)
                else:
                    params[param_name] = new_value
            elif isinstance(param_range, List[Any]):
                # Sometimes keep best, sometimes sample
                if random.random() < 0.8:
                    params[param_name] = self.best_params[param_name]
                else:
                    params[param_name] = random.choice(param_range)
        
        return params