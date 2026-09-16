# Publishing `shufflesnap`

The distribution name, import package, native extension destination, repository URLs, and GitHub Actions workflow all use `shufflesnap`. The prepared version is 0.3.0.

## Current release status

The new PyPI project is not published yet. As of 2026-09-15, its public project endpoint returned 404. Publishing a renamed distribution creates a new PyPI project; it does not migrate existing installations or transfer the old project's releases. Keep the previous distribution available for existing users.

This workspace has GitHub authentication but no configured PyPI upload credentials. The repository uses GitHub OIDC trusted publishing, which must be configured separately for the new project name.

## One-time account setup

In [PyPI publishing settings](https://pypi.org/manage/account/publishing/), add a **pending publisher** with:

| Field | Value |
| --- | --- |
| PyPI project name | `shufflesnap` |
| GitHub owner | `kylemcdonald` |
| Repository | `shufflesnap` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

The workflow filename field takes `release.yml`, not the full path. The repository already has the `pypi` environment. See [PyPI's new-project instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Validate locally

From the library repository:

```bash
python -m pip install -U build twine pytest scipy
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python -m pytest -q
```

Test an sdist installation in a separate environment as well. CI checks editable installations on Linux, macOS, and Windows and builds distributions. The release workflow builds wheels for Linux, Windows, and both macOS architectures.

## Publish after account setup and review

1. Push the reviewed source and confirm CI passes.
2. Create and push the release tag `v0.3.0`, or dispatch `release.yml` on the reviewed commit.
3. Confirm all wheel jobs and the PyPI publish job succeed.
4. In a clean environment, run `pip install shufflesnap` and verify `import shufflesnap`.
5. Update any release-status prose and confirm the package link before submitting the paper.

Do not tag solely to test PyPI configuration: the release workflow publishes publicly. `README_PYPI.md` is the package long description; `README.md` also includes the visual example.
