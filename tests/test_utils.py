from pytest import approx
import numpy as np
import torch
from torch import tensor, cuda

from mrphy import γH, dt0
from mrphy import utils

# TODO:
# unit_tests for other utils functions


class Test_utils:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dtype, atol = torch.float32, 1e-4
    # dtype, atol = torch.float64, 1e-9

    dkw = {'dtype': dtype, 'device': device}

    print(device)
    γ = tensor([[γH]], device=device, dtype=dtype)  # Hz/Gauss
    dt = tensor([[dt0]], device=device, dtype=dtype)   # Sec

    def test_ctrSub(self):
        dkw = self.dkw
        x = utils.ctrsub(torch.arange(7, **dkw)).cpu().numpy()
        assert(np.all(x == np.array([0, 0, 1, 1, 2, 2, 3])))
        return

    def test_g2k(self):

        dkw, atol = self.dkw, self.atol
        γ, dt = self.γ, self.dt

        k = tensor([[[1., 2., 3., 4., 0.]]], **dkw)
        gTx, gRx = (utils.k2g(k, True, γ=γ, dt=dt),
                    utils.k2g(k, False, γ=γ, dt=dt))

        assert(utils.g2k(gTx, True, γ=γ, dt=dt).detach().cpu().numpy() ==
               approx(k.detach().cpu().numpy(), abs=atol))

        assert(utils.g2k(gRx, False, γ=γ, dt=dt).detach().cpu().numpy() ==
               approx(k.detach().cpu().numpy(), abs=atol))

        dt1 = tensor([1.], **dkw)
        dt1_np, gTx_np = (dt1.detach().cpu().numpy(),
                          gTx.detach().cpu().numpy())
        assert(utils.g2s(gTx, dt=dt1).detach().cpu().numpy() ==
               approx(np.concatenate((gTx_np[:, :, [0]],
                                      gTx_np[:, :, 1:] - gTx_np[:, :, :-1]),
                                     axis=2)/(dt1_np[..., None]),
                      abs=atol))

        return

    def test_g2s(self):
        return
