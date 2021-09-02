from copy import deepcopy
import numpy as np
import torch
import pytest
from torch import tensor, cuda

from mrphy import γH, dt0, π, _slice
from mrphy import mobjs

# TODO:
# unit_tests for objects `.to()` methods


def _setup(T1_, T2, γ, device, dtype):
    kw = {'dtype': dtype, 'device': device}
    dt = dt0  # For test coverage, not using self.dt here.

    N, Nd, nT = 1, (3, 3, 3), 512
    nT = 512

    # pulse: Sec; Gauss; Gauss/cm.
    pulse_size = (N, 1, nT)
    t = torch.arange(0, nT, **kw).reshape(pulse_size)
    rf = 10*torch.cat([torch.cos(t/nT*2*π),                # (1,xy, nT)
                       torch.sin(t/nT*2*π)], 1)
    gr = torch.cat([torch.ones(pulse_size, **kw),
                    torch.ones(pulse_size, **kw),
                    10*torch.atan(t - round(nT/2))/π], 1)  # (1,xyz,nT)

    # Pulse
    p = mobjs.Pulse(rf=rf, gr=gr, dt=dt, **kw)
    p = deepcopy(p)  # for pytest coverage
    p = mobjs.Pulse(**(p.asdict(toNumpy=False)))

    # SpinCube (SpinArray is implicitly tested via it)
    shape = (N, *Nd)
    mask = torch.zeros((1,)+Nd, device=device, dtype=torch.bool)
    mask[0, :, 1, :], mask[0, 1, :, :] = True, True
    fov, ofst = tensor([[3., 3., 3.]], **kw), tensor([[0., 0., 1.]], **kw)

    T1_, T2 = T1_.to(**kw), T2.to(**kw)
    cube = mobjs.SpinCube(shape, fov, mask=mask, T1_=T1_, γ=γ, **kw)
    cube = deepcopy(cube)  # for pytest coverage
    cube_dict = cube.asdict(toNumpy=False)
    cube = mobjs.SpinCube(**{k: cube_dict[k] for
                             k in ('shape', 'fov', 'mask', 'T1', 'γ'
                                   )+tuple(kw.keys())})
    cube.ofst = ofst  # separated for test coverage

    cube.M_ = tensor([0., 1., 0.])
    cube.T2 = T2.expand(cube.shape)  # for test coverage

    M001, M100 = tensor([0., 0., 1.], **kw), tensor([1., 0., 0.], **kw)
    crds_100 = cube.crds_([_slice, [0, 1], [1, 0], _slice, _slice])
    cube.M_[crds_100] = M100
    crds_001 = cube.crds_([_slice, [2, 1], [1, 2], _slice, _slice])
    cube.M_[crds_001] = M001

    return cube, p


def to_np(x):
    return x.detach().cpu().numpy()


class Test_mobjs:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # dtype, atol = torch.float32, 1e-4
    dtype, atol = torch.float64, 1e-9
    print(device)

    dkw = {'dtype': dtype, 'device': device}

    γ = γH.to(**dkw)  # Hz/Gauss
    dt = dt0.to(**dkw)  # Sec

    @staticmethod
    def np(x):
        return x.detach().cpu().numpy()

    def test_Examples(self):
        ''' for coverage :/ '''
        assert(isinstance(mobjs.Examples.pulse(), mobjs.Pulse))
        assert(isinstance(mobjs.Examples.spincube(), mobjs.SpinCube))
        assert(isinstance(mobjs.Examples.spincube(), mobjs.SpinArray))
        return

    def test_mobjs(self):
        T1_, T2 = tensor([[1.]]), tensor([[4e-2]])
        cube, p = _setup(T1_, T2, self.γ, device=self.device, dtype=self.dtype)
        assert(p.is_cuda == (p.device.type == 'cuda'))
        assert(cube.is_cuda == (cube.device.type == 'cuda'))
        assert(cube.dim() == len(cube.shape))
        return

    def test_applypulse(self):
        atol = self.atol

        T1_, T2 = tensor([[1.]]), tensor([[4e-2]])
        cube, p = _setup(T1_, T2, self.γ, device=self.device, dtype=self.dtype)

        # gr_x/gr_y == 1 Gauss/cm cancels Δf[1, :, 0]/[1, 0, :] respectively
        cube.Δf = torch.sum(-cube.loc[0:1, :, :, :, 0:2], dim=-1) * cube.γ

        Mres1a = cube.applypulse(p, doEmbed=True)
        cube.applypulse(p, doEmbed=True, doRelax=False, doUpdate=True)
        Mres1b = cube.M

        # assertion
        Mo0a = np.array(
            [[[0.559535641648385,  0.663342640621335, 0.416341441715101],
              [0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222, 0.673391604920576, -0.143262993311057]]])

        Mo0b = np.array(
            [[[0.584337330324116,  0.686096989146395, 0.433382978292808],
              [0.404188676945936,  0.217027890590635, -0.888555236400348],
              [-0.703691265981316, 0.694384487290747, -0.150495136106067]]])

        Mrefa = pytest.approx(Mo0a, abs=atol)
        Mrefb = pytest.approx(Mo0b, abs=atol)

        assert(to_np(Mres1a[0:1, 1, :, 1, :]) == Mrefa)
        assert(to_np(Mres1a[0:1, :, 1, 1, :]) == Mrefa)

        assert(to_np(Mres1b[0:1, 1, :, 1, :]) == Mrefb)
        assert(to_np(Mres1b[0:1, :, 1, 1, :]) == Mrefb)

        return

    def test_freeprec(self):
        dkw, atol = self.dkw, self.atol

        E1, E2 = tensor([[0.5]], **dkw), tensor([[0.5]], **dkw)

        # scalar dur, T1, T2
        dur = torch.tensor(0.5, **dkw)
        T1, T2 = -dur/torch.log(E1), -dur/torch.log(E2)  # ()

        cube, _ = _setup(T1, T2, self.γ, device=self.device, dtype=self.dtype)
        shape = cube.shape
        _, nx, _, nz = shape
        _Δf = tensor([[[1/4/dur], [-1/4/dur], [1]]], **dkw)
        Δf = _Δf.repeat(1, 3, 1, 3)
        cube.Δf = Δf

        Mres1a = cube.freeprec(dur, doEmbed=True)

        # assertion
        Mo0a = np.array([[[0., -0.5, 0.5], [-0.5, 0, 0.5], [0., 0., 1.]]])

        Mrefa = pytest.approx(Mo0a, abs=atol)

        assert(to_np(Mres1a[0:1, 1, :, 1, :]) == Mrefa)

        return

    def test_PulseInterpT(self):
        dkw, atol = self.dkw, self.atol
        dt = dt0
        dt_n = dt*5

        nT, axis = 11, 2
        kind = 'linear'
        pulse_size = (1, 1, nT)

        kw = {'num': nT, 'axis': axis}

        # numpy raw data
        rf = 0.1*np.concatenate([np.linspace([[0.]], 1., **kw),
                                 np.linspace([[1]], 0., **kw)], 1)

        gr = 0.1*np.concatenate([np.linspace([[0.]], 1., **kw),
                                 np.linspace([[1.]], 0., **kw),
                                 np.ones(pulse_size)], 1)

        rf_pt, gr_pt = tensor(rf, **dkw), tensor(gr, **dkw)

        p_old = mobjs.Pulse(rf=rf_pt, gr=gr_pt, dt=dt, **dkw)
        p_new = p_old.interpT(dt=dt_n, kind=kind)

        # reference
        rf_ref = pytest.approx(np.array([[[0.04, 0.09],
                                          [0.06, 0.01]]]), abs=atol)

        gr_ref = pytest.approx(np.array([[[0.04, 0.09],
                                          [0.06, 0.01],
                                          [0.1,   0.1]]]), abs=atol)

        assert(to_np(p_new.rf) == rf_ref)
        assert(to_np(p_new.gr) == gr_ref)

        return


if __name__ == '__main__':
    tmp = Test_mobjs()
    tmp.test_applypulse()
