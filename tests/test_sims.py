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
        f_t2np = lambda x: x.detach().clone().cpu().numpy()

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
        beff = beffective.rfgr2beff(rf, gr, loc, Δf=Δf, b1Map=b1Map, γ=γ)
        beff.requires_grad = True

        # Check handling of 1-coil `rf`, `b1Map` that omitted the `nCoils` dim
        beff_missing_dim = beffective.rfgr2beff(rf[..., 0], gr, loc, Δf=Δf,
                                                b1Map=b1Map[..., 0], γ=γ)

        # %% sim
        print('\nblochsim tests:')
        t = time.time()
        Mo_1a = slowsims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: slowsims.blochsim', time.time() - t)

        res1a = torch.sum(Mo_1a)
        t = time.time()
        res1a.backward()  # keep graph to check `bar.backward()`
        print('backward: slowsims.blochsim', time.time() - t)
        grad_M0_1a = f_t2np(M0.grad)
        # (..., nT, xyz) → (..., xyz, nT)
        grad_beff_1a = f_t2np(beff.grad)

        M0.grad, beff.grad = None, None

        t = time.time()
        Mo_2a = sims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)
        print('forward: sims.blochsim', time.time() - t)

        res2a = torch.sum(Mo_2a)
        t = time.time()
        res2a.backward()  # keep graph to check `bar.backward()`
        print('backward: sims.blochsim', time.time() - t)
        grad_M0_2a = f_t2np(M0.grad)
        grad_beff_2a = f_t2np(beff.grad)

        M0.grad, beff.grad = None, None

        # %% assertion
        assert(pytest.approx(f_t2np(beff), abs=atol)
               == f_t2np(beff_missing_dim))

        assert(pytest.approx(grad_M0_1a, abs=atol) == grad_M0_2a)
        assert(pytest.approx(grad_beff_1a, abs=atol) == grad_beff_2a)

        # %% sim w/o relaxations
        print('\nblochsim tests (no relaxations):')
        Mo_1b = slowsims.blochsim(M0, beff, T1=None, T2=None, γ=γ, dt=dt)

        res1b = torch.sum(Mo_1b)
        res1b.backward()  # keep graph to check `bar.backward()`
        grad_M0_1b = f_t2np(M0.grad)
        # (..., nT, xyz) → (..., xyz, nT)
        grad_beff_1b = f_t2np(beff.grad)

        # dur_f, dur_b, n_repeat = 0., 0., 1000
        dur_f, dur_b, n_repeat = 0., 0., 1
        for _ in range(n_repeat):
            M0.grad, beff.grad = None, None

            t = time.time()
            Mo_2b = sims.blochsim(M0, beff, T1=None, T2=None, γ=γ, dt=dt)
            dur_f += time.time() - t

            res2b = torch.sum(Mo_2b)

            t = time.time()
            res2b.backward()
            dur_b += time.time() - t

        # print('forward: sims.blochsim', time.time() - t)
        print('forward: sims.blochsim', dur_f/n_repeat)

        print('backward: sims.blochsim', dur_b/n_repeat)
        grad_M0_2b = f_t2np(M0.grad)
        grad_beff_2b = f_t2np(beff.grad)

        M0.grad, beff.grad = None, None

        # %% assertion
        assert(pytest.approx(grad_M0_1b, abs=atol) == grad_M0_2b)
        assert(pytest.approx(grad_beff_1b, abs=atol) == grad_beff_2b)

    def test_freeprec(self):
        """
        *Note*:
        This test relies on the correctness of `test_slowsims.py`.
        """
        f_t2np = lambda x: x.detach().clone().cpu().numpy()

        print('\n')
        dkw, atol = self.dkw, self.atol
        γ = self.γ

        # spins  # (1,nM,xyz)
        M0 = torch.rand((1, 16*16*2, 3), **dkw)
        N, nM = M0.shape[0], M0.shape[1]
        Nd = (nM,)

        M0.requires_grad = True

        # parameters: Sec; cm.
        dur = torch.tensor(0.5, **dkw)
        T1, T2 = tensor([[1.]], **dkw), tensor([[4e-2]], **dkw)

        loc_x = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf

        # %% sim
        print('\nfreeprec tests:')
        t = time.time()
        Mo1 = slowsims.freeprec(M0, dur, T1=T1, T2=T2, Δf=Δf)
        print('forward: slowsims.freeprec', time.time() - t)

        res1 = torch.sum(Mo1)
        t = time.time()
        res1.backward()  # keep graph to check `bar.backward()`
        print('backward: slowsims.freeprec', time.time() - t)
        grad_M0_1 = f_t2np(M0.grad)

        M0.grad = None

        t = time.time()
        Mo2 = sims.freeprec(M0, dur, T1=T1, T2=T2, Δf=Δf)
        print('forward: sims.freeprec', time.time() - t)

        res2 = torch.sum(Mo2)
        t = time.time()
        res2.backward()  # keep graph to check `bar.backward()`
        print('backward: sims.blochsim', time.time() - t)
        grad_M0_2 = f_t2np(M0.grad)

        M0.grad = None

        assert(pytest.approx(grad_M0_1, abs=atol) == grad_M0_2)
        return


if __name__ == '__main__':
    tmp = Test_sims()
    tmp.test_blochsims()
    tmp.test_freeprec()
