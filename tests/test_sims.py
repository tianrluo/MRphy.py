import pytest
from pytest import approx
import numpy as np
import torch
from torch import tensor
from math import pi as π

import mrphy

class Test_sims:

    def _test_HelloWorld(self):
        print('in test_HelloWorld')
        self.assertEqual('hello'.upper(), 'HELLO')

    def test_blochSim(self):
        dtype = torch.float32

        # spins
        M0 = tensor([[1., 0., 0.],  # (1,nM,xyz)
                     [0., 1., 0.],
                     [0., 0., 1.]], dtype=dtype).reshape((1, 3, 3))
        N, nM, nT = M0.shape[0], M0.shape[1], 512

        # parameters
        fov, γ = tensor([3., 3., 1.], dtype=dtype), 4257.6 # cm; Hz/Gauss
        T1, T2, dt = tensor(1), tensor(4e-2), tensor(4e-6)  # sec

        loc_x = torch.linspace(-1., 1., steps=nM).reshape((1,3))
        loc_y, loc_z = torch.zeros((1,3)), torch.ones((1,3))
        loc = torch.stack([loc_x, loc_y, loc_z], 2)  # (1,nM,xyz)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf
        b1Map = tensor([1, 0])[None, None, ...]

        # pulse
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, dtype=dtype).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),      # (1,xy, nT), Gauss
                           torch.sin(t/nT*2*π)], 1)
        gr = torch.cat([torch.ones(pulse_size),      # (1,xyz,nT), Gauss/cm
                        torch.zeros(pulse_size),
                        10*torch.atan(t - round(nT/2))/π], 1)

        Beff = mrphy.sims.rfgr2B(rf, gr, loc, Δf, b1Map, γ)

        Mo_res = mrphy.sims.blochSim(M0, Beff, T1=T1, T2=T2, γ=γ, dt=dt)

        # assert results
        Mo_res.numpy() == approx(np.array(
            [[[ 0.559535641648385,  0.663342640621335,  0.416341441715101],
              [ 0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222,  0.673391604920576, -0.143262993311057]]]))
