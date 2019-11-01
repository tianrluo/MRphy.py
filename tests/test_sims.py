import pytest
from pytest import approx
import numpy as np
import torch
from torch import tensor, cuda
from math import pi as π

import mrphy

class Test_sims:

    device = 'cuda' if cuda.is_available() else 'cpu'
    dtype = torch.float32
    print(device)

    def _test_HelloWorld(self):
        print('in test_HelloWorld')
        self.assertEqual('hello'.upper(), 'HELLO')

    def test_blochSims(self):

        dtype, device = self.dtype, self.device
        # spins  # (1,nM,xyz)
        M0 = tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    dtype=dtype, device=device).reshape((1, 3, 3))
        N, nM, nT = M0.shape[0], M0.shape[1], 512

        # parameters
        γ, dt = 4257.6, tensor(4e-6, device=device)  # Hz/Gauss; Sec
        T1, T2 = tensor(1, device=device), tensor(4e-2, device=device)  # sec

        loc_x = torch.linspace(-1., 1., steps=nM, device=device).reshape((1,3))
        loc_y, loc_z = (torch.zeros((1,3), device=device),
                        torch.ones((1,3), device=device))
        loc = torch.stack([loc_x, loc_y, loc_z], 2)  # (1,nM,xyz)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf
        b1Map = tensor([1., 0.], device=device)[None, None, ...]


        # pulse
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, dtype=dtype, device=device).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),      # (1,xy, nT), Gauss
                           torch.sin(t/nT*2*π)], 1)
        gr = torch.cat([torch.ones(pulse_size, device=device),      # (1,xyz,nT), Gauss/cm
                        torch.zeros(pulse_size, device=device),
                        10*torch.atan(t - round(nT/2))/π], 1)

        Beff = mrphy.sims.rfgr2B(rf, gr, loc, Δf, b1Map, γ)

        # sim
        Mo_1 = mrphy.sims.blochSim(M0, Beff, T1=T1, T2=T2, γ=γ, dt=dt)

        E1, E2, γ2πdt = torch.exp(-dt/T1), torch.exp(-dt/T2), 2*π*γ*dt
        E1_1 = E1 - 1
        Mo_2, Mo_tmp = M0.clone(), M0.clone()
        for t in range(nT):
            Mo_2, _ = mrphy.sims.blochSim_1Step(Mo_2, Mo_tmp, Beff[...,t],
                                                E1, E1_1, E2, γ2πdt)


        # assert results
        Mo_1.cpu().numpy() == approx(Mo_2.cpu().numpy()) == approx(np.array(
            [[[ 0.559535641648385,  0.663342640621335,  0.416341441715101],
              [ 0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222,  0.673391604920576, -0.143262993311057]]]))
