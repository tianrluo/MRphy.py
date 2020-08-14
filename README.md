# MRphy.py

Yet a pytorch based Bloch simulator.
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://tianrluo.github.io/MRphy.py/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://tianrluo.github.io/MRphy.py/dev/index.html)
[![codecov](https://codecov.io/gh/tianrluo/MRphy.py/branch/master/graph/badge.svg?token=83sKL5NADl)](https://codecov.io/gh/tianrluo/MRphy.py)
[![Actions Status](https://github.com/tianrluo/MRphy.py/workflows/Python%20package/badge.svg)](https://github.com/tianrluo/MRphy.py/actions)

## Branches

- `master`: Stable;
- `dev`: Ocassionally `git squash`'d, `git push --force`'d;
- `dev_cache`: Constantly `git push --force`'d.

Developments are mostly done on `dev_cache`; when they have passed local checks, `dev` will be `git rebase`'d to `dev_cache`, and sent for CI tests.
When enough updates have been accumulated, `dev` will be git squashed into one large commit, followed by having `master`-branch `git rebase`'d onto it.

## Installation

The package is not yet sent to `pip` or `conda`.
To use it, navigate to the repo's root directory; then in your command line, type:

```sh
pip install .
```

## Demos

Check out files under `./test`.
After installation, one can quickly play with the tests through:

```sh
pytest -s
```

Only basic demo is available in this early version.
