# MRphy.py

Yet a pytorch based Bloch simulator.
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://tianrluo.github.io/MRphy.py/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://tianrluo.github.io/MRphy.py/dev/index.html)
[![codecov](https://codecov.io/gh/tianrluo/MRphy.py/branch/master/graph/badge.svg?token=83sKL5NADl)](https://codecov.io/gh/tianrluo/MRphy.py)
[![Actions Status](https://github.com/tianrluo/MRphy.py/workflows/Python%20package/badge.svg)](https://github.com/tianrluo/MRphy.py/actions)

Infrastructure of:\
[Joint Design of RF and Gradient Waveforms via Auto-Differentiation for 3D Tailored Exitation in MRI](https://arxiv.org/abs/2008.10594)

cite as:

```bib
@misc{luo2020joint,
  title={Joint Design of RF and gradient waveforms via auto-differentiation for 3D tailored excitation in MRI},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2020},
  eprint={2008.10594},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2008.10594}
}
```

For the `interpT` feature, consider citing:
```bib
@inproceedings{luo2021MultiScale,
  title={Multi-scale Accelerated Auto-differentiable Bloch-simulation based joint design of excitation RF and gradient waveforms},
  booktitle={ISMRM},
  pages={0000},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2021}
}
```

## Branches

- `master`: Stable;
- `dev`: Ocassionally `git squash`'d, `git push --force`'d;
- `dev_cache`: Constantly `git push --force`'d.

Developments are mostly done on `dev_cache`; when they have passed local checks, `dev` will be `git rebase`'d to `dev_cache`, and sent for CI tests.
When enough updates have been accumulated, `dev` will be git squashed into one large commit, followed by having `master`-branch `git rebase`'d onto it.

## Installation

```sh
pip install mrphy
```

(The package is not yet sent to `conda`.)

## Demos

Check out files under `./test`.
After installation, one can quickly play with the tests through:

```sh
pytest -s
```

Only basic demo is available in this early version.
