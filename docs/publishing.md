# Publishing to PyPI

This project publishes Python distributions through GitHub Actions using PyPI
Trusted Publishing. The normal flow does not require storing a PyPI API token in
GitHub secrets.

## One-time PyPI setup

Create the project on PyPI and TestPyPI, then add GitHub as a trusted publisher
for each index.

Values to set manually in PyPI/TestPyPI:

- Owner: repository owner or organization.
- Repository name: this GitHub repository name.
- Workflow name: `publish-pypi.yml`.
- Environment name for TestPyPI: `testpypi`.
- Environment name for PyPI: `pypi`.

Recommended GitHub environment setup:

- Create an environment named `testpypi`.
- Create an environment named `pypi`.
- Add required reviewers to the `pypi` environment if production publishing
  should require manual approval.

No GitHub secret is required when Trusted Publishing is configured correctly.

## Publish from GitHub Actions

For a test publish:

1. Open GitHub Actions.
2. Select `Publish Python package`.
3. Click `Run workflow`.
4. Set `target` to `testpypi`.
5. Run the workflow.

For a production publish:

1. Update the package version in the project metadata.
2. Commit and push the version change.
3. Create a GitHub release for that version, or run the workflow manually with
   `target` set to `pypi`.

Production publishing uses the `pypi` GitHub environment.

## Manual local publish

Install publishing tools:

```bash
python -m pip install --upgrade build twine
```

Build the package:

```bash
python -m build
```

Validate the generated distributions:

```bash
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

For local `twine upload`, provide either:

- `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<pypi-api-token>`, or
- credentials in `~/.pypirc`.

PyPI does not allow overwriting an already published version. If a publish
fails after the version has reached PyPI, bump the version before retrying.
