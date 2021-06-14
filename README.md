# MRphy.py

A pytorch based MR simulator package.
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://tianrluo.github.io/MRphy.py/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://tianrluo.github.io/MRphy.py/dev/index.html)
[![codecov](https://codecov.io/gh/tianrluo/MRphy.py/branch/master/graph/badge.svg?token=83sKL5NADl)](https://codecov.io/gh/tianrluo/MRphy.py)
[![Actions Status](https://github.com/tianrluo/MRphy.py/workflows/Python%20package/badge.svg)](https://github.com/tianrluo/MRphy.py/actions)

Infrastructure of:\
[Joint Design of RF and Gradient Waveforms via Auto-Differentiation for 3D Tailored Exitation in MRI](https://ieeexplore.ieee.org/document/9439482)\
(arXiv: [https://arxiv.org/abs/2008.10594](https://arxiv.org/abs/2008.10594))

cite as:

```bib
@article{luo2021joint,
  author={Luo, Tianrui and Noll, Douglas C. and Fessler, Jeffrey A. and Nielsen, Jon-Fredrik},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Joint Design of RF and gradient waveforms via auto-differentiation for 3D tailored excitation in MRI}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2021.3083104}}
```

For the `interpT` feature, consider citing:
```bib
@inproceedings{luo2021MultiScale,
  title={Multi-scale Accelerated Auto-differentiable Bloch-simulation based joint design of excitation RF and gradient waveforms},
  booktitle={ISMRM},
  pages={3958},
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
