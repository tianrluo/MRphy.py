import torch
import pytest
from torch import tensor, cuda

from mrphy import γH, dt0, π
from mrphy import beffective, sims, slowsims

import time


class Test_sims:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # dtype, atol = torch.float32, 1e-4
    dtype, atol = torch.float64, 1e-9
    print(device)

    dkw = {'dtype': dtype, 'device': device}

    γ = tensor([[γH]], device=device, dtype=dtype)  # Hz/Gauss
    dt = tensor([[dt0]], device=device, dtype=dtype)   # Sec

    def test_blochsims(self):
        """
        *Note*:
        This test relies on the correctness of `test_slowsims.py`.
        """

        print('\n')
        dkw, atol = self.dkw, self.atol
        γ, dt = self.γ, self.dt

        # spins  # (1,nM,xyz)
        M0 = torch.rand((1, 16*16*2, 3), **dkw)
        # M0 = torch.rand((1, 32*32*20, 3), **dkw)
        N, nM, nT = M0.shape[0], M0.shape[1], 512
        Nd = (nM,)

        M0.requires_grad = True

        # parameters: Sec; cm.
        T1, T2 = tensor([[1.]], **dkw), tensor([[4e-2]], **dkw)

        loc_x = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_y = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_z = torch.ones((N,)+Nd, **dkw)
        loc = torch.stack([loc_x, loc_y, loc_z], 2)  # (1,nM,xyz)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf
        b1Map = tensor([1., 0.], **dkw).reshape((N, 1, 2, 1))

        # pulse: Sec; Gauss; Gauss/cm.
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, **dkw).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),  # (1,xy,nT,nCoils)
                           torch.sin(t/nT*2*π)], 1)[..., None]
        gr = torch.cat([torch.ones(pulse_size, **dkw),
                        torch.zeros(pulse_size, **dkw),
                        10*torch.atan(t - round(nT/2))/π], 1)  # (1,xyz,nT)

        # rf.requires_grad, gr.requires_grad = True, True
        beff = beffective.rfgr2beff(rf, gr, loc, Δf, b1Map, γ)
        beff.requires_grad = True

        # %% sim
        t = time.time()
        Mo_1 = slowsims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: slowsims.blochsim', time.time() - t)

        res1 = torch.sum(Mo_1)
        t = time.time()
        res1.backward()  # keep graph to check `bar.backward()`
        print('backward: slowsims.blochsim', time.time() - t)
        grad_M0_1 = M0.grad.clone().cpu().numpy()
        grad_beff_1 = beff.grad.clone().cpu().numpy()

        M0.grad, beff.grad = None, None

        t = time.time()
        Mo_2 = sims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: sims.blochsim', time.time() - t)

        res2 = torch.sum(Mo_2)
        t = time.time()
        res2.backward()  # keep graph to check `bar.backward()`
        print('backward: sims.blochsim', time.time() - t)
        grad_M0_2 = M0.grad.clone().cpu().numpy()
        grad_beff_2 = beff.grad.clone().cpu().numpy()

        # %% assertion
        assert(pytest.approx(grad_M0_1, abs=atol) == grad_M0_2)
        assert(pytest.approx(grad_beff_1, abs=atol) == grad_beff_2)


if __name__ == '__main__':
    tmp = Test_sims()
    tmp.test_blochsims()
