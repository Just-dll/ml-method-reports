# Publishing

Do not publish automatically. Use these steps when the release is ready.

Build and check locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

Manual release checklist:

- Confirm package metadata in `pyproject.toml`.
- Run tests from a clean environment.
- Create a GitHub release.
- Configure a PyPI API token or GitHub Trusted Publishing.
