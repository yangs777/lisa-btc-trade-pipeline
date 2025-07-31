"""Global test configuration."""

import sys
from unittest.mock import MagicMock

# Import numpy first to avoid conflicts
# Ensure numpy.typing is available
import numpy.typing  # noqa


# Pre-mock problematic modules before any imports
def setup_module_mocks() -> None:
    """Setup mocks for modules that cause import errors."""
    # Mock websockets and all its submodules
    websockets_mock = MagicMock()
    websockets_mock.exceptions = MagicMock()
    websockets_mock.exceptions.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
    websockets_mock.exceptions.WebSocketException = type("WebSocketException", (Exception,), {})
    sys.modules["websockets"] = websockets_mock
    sys.modules["websockets.exceptions"] = websockets_mock.exceptions

    # Mock torch and its submodules
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.Module = type("Module", (), {})
    torch_mock.nn.Linear = MagicMock
    torch_mock.nn.ReLU = MagicMock
    torch_mock.nn.Sequential = MagicMock
    torch_mock.nn.functional = MagicMock()
    torch_mock.optim = MagicMock()
    torch_mock.device = MagicMock(return_value="cpu")
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.Tensor = MagicMock
    torch_mock.tensor = MagicMock
    torch_mock.zeros = MagicMock
    torch_mock.ones = MagicMock
    torch_mock.FloatTensor = MagicMock
    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = torch_mock.nn
    sys.modules["torch.nn.functional"] = torch_mock.nn.functional
    sys.modules["torch.optim"] = torch_mock.optim

    # Mock stable_baselines3
    sb3_mock = MagicMock()
    sb3_mock.SAC = MagicMock
    sb3_mock.PPO = MagicMock
    sb3_mock.DQN = MagicMock
    sb3_mock.common = MagicMock()
    sb3_mock.common.torch_layers = MagicMock()
    sb3_mock.common.torch_layers.BaseFeaturesExtractor = type("BaseFeaturesExtractor", (), {})
    sb3_mock.common.vec_env = MagicMock()
    sb3_mock.common.vec_env.VecNormalize = MagicMock
    sb3_mock.common.vec_env.DummyVecEnv = MagicMock
    sb3_mock.common.callbacks = MagicMock()
    sb3_mock.common.callbacks.BaseCallback = type("BaseCallback", (), {})
    sys.modules["stable_baselines3"] = sb3_mock
    sys.modules["stable_baselines3.common"] = sb3_mock.common
    sys.modules["stable_baselines3.common.torch_layers"] = sb3_mock.common.torch_layers
    sys.modules["stable_baselines3.common.vec_env"] = sb3_mock.common.vec_env
    sys.modules["stable_baselines3.common.callbacks"] = sb3_mock.common.callbacks


# Call this immediately when conftest is loaded
setup_module_mocks()

# Keep any existing pytest fixtures/configuration below this line
import copy
import pytest


@pytest.fixture(autouse=True)
def reset_sys_modules():
    """Reset sys.modules after each test to prevent test pollution."""
    # Take a snapshot of sys.modules before the test
    original_modules = copy.copy(sys.modules)
    
    yield
    
    # After the test, restore original modules and remove any new ones
    # Get keys to remove (ones that were added during the test)
    keys_to_remove = set(sys.modules.keys()) - set(original_modules.keys())
    
    # Remove new modules
    for key in keys_to_remove:
        if key in sys.modules:
            del sys.modules[key]
    
    # Restore original modules that were modified
    for key, value in original_modules.items():
        if key in sys.modules and sys.modules[key] != value:
            sys.modules[key] = value
