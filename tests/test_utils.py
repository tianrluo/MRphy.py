import numpy as np
import torch
import pytest
from torch import tensor, cuda

from mrphy import γH, dt0, rfmax0, smax0, __CUPY_IS_AVAILABLE__
from mrphy import utils
if __CUPY_IS_AVAILABLE__:
    import cupy as cp


# TODO:
# unit_tests for other utils functions
def to_np(x):
    return x.detach().cpu().numpy()


class Test_utils:

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dtype, atol = torch.float32, 1e-4
    # dtype, atol = torch.float64, 1e-9

    dkw = {'dtype': dtype, 'device': device}

    print(device)
    γ = γH.to(**dkw)  # Hz/Gauss
    dt = dt0.to(**dkw)   # Sec

    def test_ctrsub(self):
        dkw = self.dkw
        x = utils.ctrsub(torch.arange(7, **dkw)).cpu().numpy()
        assert(np.all(x == np.array([0, 0, 1, 1, 2, 2, 3])))
        return

    def test_kgs(self):

        dkw, atol = self.dkw, self.atol
        γ, dt = self.γ, self.dt

        k = tensor([[[1., 2., 3., 4., 0.]]], **dkw)
        gTx, gRx = (utils.k2g(k, True, γ=γ, dt=dt),
                    utils.k2g(k, False, γ=γ, dt=dt))
        gTx1 = utils.s2g(utils.g2s(gTx, dt), dt)

        assert(to_np(utils.g2k(gTx, True, γ=γ, dt=dt)) ==
               pytest.approx(to_np(k), abs=atol))

        assert(to_np(utils.g2k(gRx, False, γ=γ, dt=dt)) ==
               pytest.approx(to_np(k), abs=atol))

        assert(to_np(gTx) == pytest.approx(to_np(gTx1), abs=atol))
        return

    def test_rc_rf(self):
        shape, atol = (1, 2, 5), self.atol
        tmp = np.random.rand(*shape)
        rf_r_0_np = tmp.astype(np.double, copy=False)
        rf_r_1_np = utils.rf_c2r(utils.rf_r2c(rf_r_0_np))
        assert(rf_r_0_np == pytest.approx(rf_r_1_np, abs=atol))

        if __CUPY_IS_AVAILABLE__:
            rf_r_0_cp = tmp.astype(cp.double, copy=False)
            rf_r_1_cp = utils.rf_c2r(utils.rf_r2c(rf_r_0_cp))
            assert(cp.asnumpy(rf_r_0_cp) ==
                   pytest.approx(cp.asnumpy(rf_r_1_cp), abs=atol))
        return

    def test_rfclamptan(self):
        shape, rfmax, atol = (1, 2, 10), rfmax0, self.atol
        rf0 = utils.rfclamp(rfmax0*((torch.rand(shape)-0.5)*4), rfmax)
        assert(torch.all(rf0.norm(dim=1) <= rfmax))
        tρ, θ = utils.rf2tρθ(rf0, rfmax)
        rf1 = utils.tρθ2rf(tρ, θ, rfmax)
        assert(to_np(rf0) == pytest.approx(to_np(rf1), abs=atol))
        return

    def test_rfclamplogit(self):
        shape, rfmax, atol = (1, 2, 10), rfmax0, self.atol
        rf0 = utils.rfclamp(rfmax0*((torch.rand(shape)-0.5)*4), rfmax)
        assert(torch.all(rf0.norm(dim=1) <= rfmax))
        lρ, θ = utils.rf2lρθ(rf0, rfmax)
        rf1 = utils.lρθ2rf(lρ, θ, rfmax)
        assert(to_np(rf0) == pytest.approx(to_np(rf1), abs=atol))
        return

    def test_sclamptan(self):
        shape, smax, atol = (1, 3, 10), smax0, self.atol
        s0 = utils.sclamp(smax0*((torch.rand(shape)-0.5)*4), smax)
        assert(torch.all(s0.abs() <= smax))
        s1 = utils.ts2s(utils.s2ts(s0, smax), smax)
        assert(to_np(s0) == pytest.approx(to_np(s1), abs=atol))
        return


if __name__ == '__main__':
    tmp = Test_utils()
    tmp.test_ctrsub()
    tmp.test_kgs()
    tmp.test_rc_rf()
    tmp.test_rfclamptan()
    tmp.test_sclamptan()
