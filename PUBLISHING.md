# Publishing `shufflesnap`

The distribution name, import package, native extension destination, repository URLs, and GitHub Actions workflow all use `shufflesnap`. The current published version is 0.3.0.

## Current release status

[ShuffleSnap 0.3.0](https://pypi.org/project/shufflesnap/0.3.0/) was published on 2026-09-16 from tag `v0.3.0` (commit `29c8f1405cb9b3a50eceef68b904386db86766b8`). The [release workflow](https://github.com/kylemcdonald/shufflesnap/actions/runs/35074346125) passed and published 25 wheels plus the source archive: CPython 3.10–3.14, Linux x86_64 (manylinux and musllinux), Windows x86_64, and macOS Intel/Apple Silicon. A fresh installation from PyPI passed all 25 tests and an assignment smoke test.

```bash
python -m pip install shufflesnap==0.3.0
```

The repository uses GitHub OIDC trusted publishing; no local upload token is needed. The publisher is configured with owner `kylemcdonald`, repository `shufflesnap`, workflow `release.yml`, and environment `pypi`. New installations should use `shufflesnap`; the previous distribution remains available for existing users.

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

## Future releases

1. Increment `version` in `pyproject.toml`, push the reviewed source, and confirm CI passes.
2. Create and push the corresponding new `v*` tag, or dispatch `release.yml` on the reviewed commit. Published versions cannot be overwritten.
3. Confirm all wheel jobs and the PyPI publish job succeed.
4. In a clean environment, run `pip install shufflesnap` and verify `import shufflesnap`.
5. Update any release-status prose and confirm the package link before submitting the paper.

Do not tag solely to test PyPI configuration: the release workflow publishes publicly. `README_PYPI.md` is the package long description; `README.md` also includes the visual example.
