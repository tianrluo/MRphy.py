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

    dkw = {'dtype': dtype, 'device': device}

    print(device)
    γ = γH.to(**dkw)  # Hz/Gauss
    dt = dt0.to(**dkw)   # Sec

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
        print('\nblochsim tests:')
        t = time.time()
        Mo_1a = slowsims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: slowsims.blochsim', time.time() - t)

        res1a = torch.sum(Mo_1a)
        t = time.time()
        res1a.backward()  # keep graph to check `bar.backward()`
        print('backward: slowsims.blochsim', time.time() - t)
        grad_M0_1a = M0.grad.clone().cpu().numpy()
        grad_beff_1a = beff.grad.clone().cpu().numpy()

        M0.grad, beff.grad = None, None

        t = time.time()
        Mo_2a = sims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: sims.blochsim', time.time() - t)

        res2a = torch.sum(Mo_2a)
        t = time.time()
        res2a.backward()  # keep graph to check `bar.backward()`
        print('backward: sims.blochsim', time.time() - t)
        grad_M0_2a = M0.grad.clone().cpu().numpy()
        grad_beff_2a = beff.grad.clone().cpu().numpy()

        M0.grad, beff.grad = None, None

        # %% assertion
        assert(pytest.approx(grad_M0_1a, abs=atol) == grad_M0_2a)
        assert(pytest.approx(grad_beff_1a, abs=atol) == grad_beff_2a)

        # %% sim w/o relaxations
        print('\nblochsim tests (no relaxations):')
        Mo_1b = slowsims.blochsim(M0, beff, T1=None, T2=None, γ=γ, dt=dt)

        res1b = torch.sum(Mo_1b)
        res1b.backward()  # keep graph to check `bar.backward()`
        grad_M0_1b = M0.grad.clone().cpu().numpy()
        grad_beff_1b = beff.grad.clone().cpu().numpy()

        M0.grad, beff.grad = None, None

        t = time.time()
        Mo_2b = sims.blochsim(M0, beff, T1=None, T2=None, γ=γ, dt=dt)
        print('forward: sims.blochsim', time.time() - t)

        res2b = torch.sum(Mo_2b)
        t = time.time()
        res2b.backward()  # keep graph to check `bar.backward()`
        print('backward: sims.blochsim', time.time() - t)
        grad_M0_2b = M0.grad.clone().cpu().numpy()
        grad_beff_2b = beff.grad.clone().cpu().numpy()

        M0.grad, beff.grad = None, None

        # %% assertion
        assert(pytest.approx(grad_M0_1b, abs=atol) == grad_M0_2b)
        assert(pytest.approx(grad_beff_1b, abs=atol) == grad_beff_2b)



if __name__ == '__main__':
    tmp = Test_sims()
    tmp.test_blochsims()
