from pytest import approx
import numpy as np
import torch
from torch import tensor, cuda

from mrphy import γH, dt0, π
from mrphy import sims


class Test_sims:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dtype, atol = torch.float32, 1e-4
    # dtype, atol = torch.float64, 1e-9
    print(device)

    dkw = {'dtype': dtype, 'device': device}

    γ = tensor([[γH]], device=device, dtype=dtype)  # Hz/Gauss
    dt = tensor([[dt0]], device=device, dtype=dtype)   # Sec

    def test_blochsims(self):

        dkw, atol = self.dkw, self.atol
        γ, dt = self.γ, self.dt

        # spins  # (1,nM,xyz)
        M0 = tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], **dkw)
        N, nM, nT = M0.shape[0], M0.shape[1], 512
        Nd = (nM,)

        # parameters: Sec; cm.
        T1, T2 = tensor([[1.]], **dkw), tensor([[4e-2]], **dkw)

        E1, E2, γ2πdt = torch.exp(-dt/T1), torch.exp(-dt/T2), 2*π*γ*dt
        E1_1 = E1 - 1

        loc_x = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_y = torch.linspace(-1., 1., steps=nM, **dkw).reshape((N,)+Nd)
        loc_z = torch.ones((N,)+Nd, **dkw)
        loc = torch.stack([loc_x, loc_y, loc_z], 2)  # (1,nM,xyz)

        Δf = -loc_x * γ  # gr_x==1 Gauss/cm cancels Δf
        b1Map = tensor([1., 0.], **dkw).reshape((N, 1, 2, 1))

        # pulse: Sec; Gauss; Gauss/cm.
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, **dkw).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),              # (1,xy,nT,nCoils)
                           torch.sin(t/nT*2*π)], 1)[..., None]
        gr = torch.cat([torch.ones(pulse_size, **dkw),
                        torch.zeros(pulse_size, **dkw),
                        10*torch.atan(t - round(nT/2))/π], 1)  # (1,xyz,nT)

        rf.requires_grad, gr.requires_grad = True, True

        beff = sims.rfgr2beff(rf, gr, loc, Δf, b1Map, γ)

        A, B = sims.beff2ab(beff, T1=T1, T2=T2, γ=γ, dt=dt)

        # sim
        Mo_1 = sims.blochsim(M0, beff, T1=T1, T2=T2, γ=γ, dt=dt)

        Mo_2, Mo_tmp = M0.clone(), M0.clone()
        for t in range(nT):
            Mo_2, _ = sims.blochsim_1step(Mo_2, Mo_tmp, beff[..., t],
                                          E1, E1_1, E2, γ2πdt)

        Mo_3 = sims.blochsim_ab(M0, A, B)

        # assertion
        ref = np.array(
            [[[0.559535641648385,  0.663342640621335, 0.416341441715101],
              [0.391994737048090,  0.210182892388552, -0.860954821972489],
              [-0.677062008711222, 0.673391604920576, -0.143262993311057]]])

        assert(approx(Mo_1.detach().cpu().numpy(), abs=atol) == ref)

        assert(approx(Mo_2.detach().cpu().numpy(), abs=atol) == ref)

        assert(approx(Mo_3.detach().cpu().numpy(), abs=atol) == ref)

        # Verify gradients can chain rule back to `rf` and `gr`
        foo = torch.sum(Mo_1)
        foo.backward(retain_graph=True)  # keep graph to check `bar.backward()`
        print(torch.sum(rf.grad), torch.sum(gr.grad))
        rf.grad = gr.grad = None  # clear grads from `foo`

        bar = torch.sum(Mo_3)
        bar.backward()
        print(torch.sum(rf.grad), torch.sum(gr.grad))
