# Contributing to Bitcoin τ-SAC Trading System

## 🧪 Testing Guidelines

### Test Types

We use a two-tier testing system to balance CI speed with comprehensive coverage:

#### 1. **Light Tests** (Default)
- Run on every push and PR
- Fast execution (<2 minutes)
- Unit tests and light integration tests
- Excludes heavy dependencies (GCS, RL environments)
- Coverage threshold: 35%

```bash
# Run light tests locally
pytest tests/ -m "not heavy" \
  --ignore=tests/test_rl_environment.py \
  --ignore=tests/test_rl_environments.py
```

#### 2. **Full Tests** (Nightly/On-demand)
- Run nightly at 3 AM UTC
- Triggered by `full-tests` label on PR
- Manual trigger via GitHub Actions
- All tests including heavy integration
- Same coverage threshold: 35%

```bash
# Run all tests locally
pytest tests/ --cov=src --cov-report=term-missing
```

### Marking Tests

Mark heavy/slow tests appropriately:

```python
import pytest

@pytest.mark.heavy
def test_gcs_integration():
    """This test requires GCS setup and is slow."""
    # ... test implementation
```

### Coverage Requirements

| Phase | Target | Status | Focus Area |
|-------|--------|--------|------------|
| Phase 1 | 25% | ✅ Complete | Core modules (hyperopt, risk, API) |
| Phase 2 | 50% | ✅ Complete | Business logic (preprocessor, features) |
| Phase 3 | 60% | 🔜 Planned | Integration & E2E tests |
| Phase 4 | 75% | 🔜 Planned | Advanced scenarios & performance |

## 📝 Code Style

- Format: `black src tests`
- Lint: `ruff check src tests`
- Type check: `mypy src tests`
- Security: `bandit -r src tests -ll`

## 🔄 Pull Request Process

1. Create feature branch from `develop`
2. Write tests for new features
3. Ensure all checks pass locally
4. Update documentation if needed
5. Submit PR with clear description
6. Wait for light tests to pass
7. Add `full-tests` label if heavy tests needed

## 🐛 Bug Reports

Include:
- Python version
- Dependencies version
- Steps to reproduce
- Expected vs actual behavior
- Error logs/tracebacks

## 💡 Feature Requests

Open an issue with:
- Use case description
- Proposed implementation
- Impact on existing features
- Performance considerations