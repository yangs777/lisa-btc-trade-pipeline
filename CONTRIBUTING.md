# Contributing to BTC/USDT τ-SAC Trading System

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/unsuperior-ai/lisa-btc-trade-pipeline.git
   cd lisa-btc-trade-pipeline
   ```

2. **Set up Python environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type checker**: Mypy (strict mode)
- **Import sorter**: isort (Black profile)

Run formatting:
```bash
make format
```

## Testing

Run all tests:
```bash
make test
```

Run specific test:
```bash
pytest tests/test_specific.py -v
```

## CI Checks

Before pushing, run all CI checks locally:
```bash
make ci
```

Or use the Python script:
```bash
python scripts/run_ci_checks.py
```

## Commit Guidelines

1. Use conventional commits:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `test:` Tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `ci:` CI/CD changes

2. Example:
   ```
   feat: implement Binance WebSocket data collector
   
   - Add real-time orderbook snapshot collection
   - Implement automatic GCS upload
   - Add connection retry logic
   ```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit

3. Push and create PR:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Ensure all CI checks pass

5. Request review from maintainers

## Project Structure

```
src/
├── data_collection/    # Data collection modules
├── features/          # Feature engineering
├── models/           # ML models and training
├── trading/          # Trading execution
├── monitoring/       # System monitoring
└── utils/           # Utility functions

tests/               # Test files
configs/             # Configuration files
scripts/             # Utility scripts
docs/               # Documentation
```

## Security

- Never commit credentials or API keys
- Use environment variables for sensitive data
- Run security checks: `make security-check`

## Questions?

Open an issue on GitHub for any questions or concerns.