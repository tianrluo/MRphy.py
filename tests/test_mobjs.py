from pytest import approx
import numpy as np
import torch
from torch import tensor, cuda

from mrphy import γH, dt0, π
from mrphy import mobjs

# TODO:
# unit_tests for objects `.to()` methods


class Test_mobjs:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dtype, atol = torch.float32, 1e-4
    # dtype, atol = torch.float64, 1e-9
    print(device)

    dkw = {'dtype': dtype, 'device': device}

    γ = tensor([[γH]], device=device, dtype=dtype)  # Hz/Gauss
    dt = tensor([[dt0]], device=device, dtype=dtype)   # Sec

    def np(x):
        return x.detach().cpu().numpy()

    def test_mobjs(self):
        kw, atol = self.dkw, self.atol
        γ, dt = self.γ, dt0  # For test coverage, not using self.dt here.

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

        # SpinCube (SpinArray is implicitly tested via it)
        shape = (N, *Nd)
        fov, ofst = tensor([[3., 3., 3.]], **kw), tensor([[0., 0., 1.]], **kw)
        T1, T2 = tensor([[1.]], **kw), tensor([[4e-2]], **kw)

        cube = mobjs.SpinCube(shape, fov, T1=T1, γ=γ, **kw)
        assert(cube.dim() == len(shape))
        cube.T2 = T2  # Separated for testing `setattr`

        cube.M = tensor([[[0., 1., 0.]]], **kw)
        cube.M[:, [0, 1], [1, 0], :, :] = tensor([1., 0., 0.], **kw)
        cube.M[:, [2, 1], [1, 2], :, :] = tensor([0., 0., 1.], **kw)

        cube.ofst = ofst

        # gr_x/gr_y == 1 Gauss/cm cancels Δf[1, :, 0]/[1, 0, :] respectively
        loc = cube.loc

        cube.Δf = torch.sum(-loc[0:1, :, :, :, 0:2], dim=-1) * γ

        Mres1 = cube.applypulse(p)

        # assertion
        Mref = approx(np.array(
            [[[0.559535641648385,  0.663342640621335, 0.416341441715101],
              [0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222, 0.673391604920576, -0.143262993311057]]]),
            abs=atol)

        assert(Test_mobjs.np(Mres1[0:1, 1, :, 1, :]) == Mref)
        assert(Test_mobjs.np(Mres1[0:1, :, 1, 1, :]) == Mref)

        return


if __name__ == '__main__':
    tmp = Test_mobjs()
    tmp.test_mobjs()
