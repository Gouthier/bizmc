# Contributing

Thanks for your interest!

1. Fork and clone the repository, then `pip install -e ".[dev]"`.
2. Create a branch for your change.
3. Add or update tests: every behaviour change needs one.
4. Run `ruff check . && ruff format --check . && pytest` before opening a pull request.

Guidelines:

* Keep the core dependency-free beyond NumPy.
* Keep the separation between observed data, assumptions and outputs.
* Simulations must stay reproducible from a seed.
* Prefer clear, readable models over clever ones: users are expected to read them.
