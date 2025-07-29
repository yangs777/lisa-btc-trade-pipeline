"""Test version and basic imports."""

import pytest


def test_version() -> None:
    """Test that version is accessible."""
    from src import __version__
    
    assert __version__ == "0.3.0"


def test_imports() -> None:
    """Test that main modules can be imported."""
    import src
    
    assert src.__author__ == "Lisa AI Team"