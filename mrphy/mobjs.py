r"""Classes for MRI excitation simulations
"""
import copy
from typing import Optional
import inspect

import numpy as np
from scipy import interpolate
import torch
from torch import tensor, Tensor

from mrphy import γH, dt0, gmax0, smax0, rfmax0, T1G, T2G, π
from mrphy import utils, beffective, sims

__all__ = ['Pulse', 'SpinArray', 'SpinCube', 'Examples']


class Pulse(object):
    r"""Pulse object of RF and GR

    Usage:
        ``pulse = Pulse(rf, gr, *, dt, gmax, smax, rfmax, desc, device,``\
        `` dtype)``

    Inputs:
        - ``rf``: `(N,xy, nT,(nCoils))` "Gauss", ``xy`` for separating real \
          and imag part.
        - ``gr``: `(N,xyz,nT)`, "Gauss/cm"
        - ``dt``: `()` ⊻ `(N ⊻ 1,)`, "Sec", dwell time.
        - ``gmax``: `()` ⊻ `(N ⊻ 1, xyz ⊻ 1)`, "Gauss/cm", max \|gradient\|.
        - ``smax``: `()` ⊻ `(N ⊻ 1, xyz ⊻ 1)`, "Gauss/cm/Sec", max \
          \|slew rate\|.
        - ``rfmax``: `()` ⊻ `(N ⊻ 1,(nCoils))`, "Gauss", max \|RF\|.
        - ``desc``: str, an description of the pulse to be constructed.
        - ``device``: torch.device.
        - ``dtype``: torch.dtype.

    Properties:
        - ``device``
        - ``dtype``
        - ``is_cuda``
        - ``shape``: ``(N,1,nT)``
        - ``gmax``: `(N ⊻ 1, xyz)`, "Gauss/cm", max \|gradient\|.
        - ``smax``: `(N ⊻ 1, xyz)`, "Gauss/cm/Sec", max \|slew rate\|.
        - ``rfmax``: `(N ⊻ 1,(nCoils))`, "Gauss", max \|RF\|.
        - ``rf``: `(N,xy, nT,(nCoils))`, "Gauss", ``xy`` for separating real \
          and imag part.
        - ``gr``: `(N,xyz,nT)`, "Gauss/cm"
        - ``dt``: `(N ⊻ 1,)`, "Sec", dwell time.
        - ``desc``: str, an description of the pulse to be constructed.
    """

    _readonly = ('device', 'dtype', 'is_cuda', 'shape')
    _limits = ('gmax', 'smax', 'rfmax')
    __slots__ = set(_readonly + _limits + ('rf', 'gr', 'dt', 'desc'))

    def __init__(
        self,
        rf: Optional[Tensor] = None, gr: Optional[Tensor] = None, *,
        dt: Tensor = dt0,
        gmax: Tensor = gmax0, smax: Tensor = smax0, rfmax: Tensor = rfmax0,
        desc: str = "generic pulse",
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        assert(isinstance(device, torch.device) and
               isinstance(dtype, torch.dtype))

        # Defaults
        rf_miss, gr_miss = rf is None, gr is None
        assert (not(rf_miss and gr_miss)), "Missing both `rf` and `gr` inputs"

        object.__setattr__(self, 'device', device)
        object.__setattr__(self, 'dtype', dtype)
        object.__setattr__(self, 'is_cuda', self.device.type == 'cuda')

        kw = {'device': self.device, 'dtype': self.dtype}

        if rf_miss:
            N, nT = gr.shape[0], gr.shape[2]
            rf = torch.zeros((N, 2, nT), **kw)
        else:
            N, nT = rf.shape[0], rf.shape[2]
            if gr_miss:
                gr = torch.zeros((N, 3, nT), **kw)
            else:
                assert (N == gr.shape[0] and nT == gr.shape[2])
        object.__setattr__(self, 'shape', torch.Size((N, 1, nT)))

        self.rf, self.gr = rf.to(**kw), gr.to(**kw)
        self.dt, self.gmax, self.smax, self.rfmax = dt, gmax, smax, rfmax
        self.desc = desc
        return

    def __setattr__(self, k, v):
        if 'deepcopy' in (_.function for _ in inspect.stack()):
            # Hack, this enables `deepcopy()` w/o overriding `__deepcopy__()`.
            # Generator is faster than list comprehension.
            object.__setattr__(self, k, v)
            return

        if k in self._readonly:
            raise AttributeError(f"'Pulse' object attribute '{k}'"
                                 " is read-only")

        if k != 'desc':
            kw = {'device': self.device, 'dtype': self.dtype}
            v = (v.to(**kw) if isinstance(v, Tensor) else tensor(v, **kw))

        if k in ('rf', 'gr'):
            assert(v.shape[0] == self.shape[0] and v.shape[2] == self.shape[2])
        elif (k in ('gmax', 'smax')):  # -> (N ⊻ 1, xyz)
            v = v.expand((1 if v.ndim == 0 else v.shape[0], self.gr.shape[1]))
        elif k == 'rfmax':  # -> (N ⊻ 1, (nCoils))
            if v.ndim == 0:
                v = v[None]
            elif v.ndim == 2 and v.shape[1] == 1:
                v = v[:, 0]
        elif k == 'dt':
            if v.ndim == 0:
                v = v[None]
            assert(v.ndim == 1)

        object.__setattr__(self, k, v)
        return

    def asdict(self, *, toNumpy: bool = True) -> dict:
        r"""Convert mrphy.mobjs.Pulse object to dict

        Usage:
            ``d = pulse.asdict(*, toNumpy)``

        Inputs:
            - ``toNumpy``: [T/f], convert Tensor to Numpy arrays.
        Outputs:
            - ``d``: dict, dictionary with detached data identical to the \
              object.
        """
        _ = ('rf', 'gr', 'dt', 'gmax', 'smax', 'rfmax')
        fn_np = ((lambda x: x.detach().cpu().numpy()) if toNumpy else
                 (lambda x: x.detach()))

        d = {k: fn_np(getattr(self, k)) for k in _}
        d.update({k: getattr(self, k) for k in ('desc', 'device', 'dtype')})

        return d

    def beff(
            self, loc: Tensor, *,
            Δf: Optional[Tensor] = None, b1Map: Optional[Tensor] = None,
            γ: Tensor = γH
    ) -> Tensor:
        r"""Compute B-effective of provided location from the pulse

        Usage:
            ``beff = pulse.beff(loc, *, Δf, b1Map, γ)``
        Inputs:
            - ``loc``: `(N,*Nd,xyz)`, "cm", locations.
        Optionals:
            - ``Δf``: `(N,*Nd,)`, "Hz", off-resonance.
            - ``b1Map``: `(N,*Nd,xy,(nCoils))`, a.u., transmit sensitivity.
            - ``γ``: `(N,*Nd)`, "Hz/Gauss", gyro-ratio
        Outputs:
            - ``beff``: `(N,*Nd,xyz,nT)`
        """
        device = self.device
        loc = loc.to(device=device)
        fn = lambda x: None if x is None else x.to(device=device)  # noqa: E731
        Δf, b1Map, γ = (fn(x) for x in (Δf, b1Map, γ))

        return beffective.rfgr2beff(self.rf, self.gr, loc,
                                    Δf=Δf, b1Map=b1Map, γ=γ)

    def interpT(self, dt: Tensor, *, kind: str = 'linear') -> 'Pulse':
        r""" Interpolate pulse of `dt` by `kind`.

        Usage:
            ``new_pulse = pulse.interpT(dt, *, kind)``
        Inputs:
            - ``dt``: `(1,)`, "Sec", new simulation dwell time.
            - ``kind``: str, passed to scipy.interpolate.interp1d.
        Outputs:
            - ``new_pulse``: mrphy.mobjs.Pulse object.

        .. note::
            This method requires both `dt` and `self.dt` to be unique/global,
            i.e., of shape ``(1,)``, which ensures pulse length to be the same
            within a batch after interpolation.
        """
        assert(self.dt.numel() == dt.numel() == 1)

        dt_o_np, dt_n_np = self.dt.item(), dt.item()
        if dt_o_np == dt_n_np:
            return copy.deepcopy(self)

        axis = 2  # Along temporal dimension
        dkw = {'device': self.device, 'dtype': self.dtype}
        kw = {'axis': axis, 'kind': kind, 'copy': False, 'assume_sorted': True}

        f_np = lambda x: x.detach().cpu().numpy()  # noqa: E731
        f_0 = lambda x: np.dstack((np.zeros_like(x[:, :, [0]]),  # noqa: E731
                                   x))

        # convert to np array, then prepend 0's.
        rf_np, gr_np = f_0(f_np(self.rf)), f_0(f_np(self.gr))

        nT = rf_np.shape[axis]

        t_o = np.arange(0, nT)*dt_o_np  # (nT,)
        t_n = np.arange(1, t_o[-1]//dt_n_np + 1)*dt_n_np

        f_rf = interpolate.interp1d(t_o, rf_np, **kw)
        f_gr = interpolate.interp1d(t_o, gr_np, **kw)

        rf_n, gr_n = tensor(f_rf(t_n), **dkw), tensor(f_gr(t_n), **dkw)

        desc = f"{self.desc} + interpT\'ed: dt = {dt_n_np}"
        return Pulse(rf_n, gr_n, dt=dt, desc=desc, **dkw)

    def to(
        self, *,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> 'Pulse':
        r"""Duplicate the object to the prescribed device with dtype

        Usage:
            ``new_pulse = pulse.to(*, device, dtype)``
        Inputs:
            - ``device``: torch.device
            - ``dtype``: torch.dtype
        Outputs:
            - ``new_pulse``: mrphy.mobjs.Pulse object.
        """
        if self.device == device and self.dtype == dtype:
            return self
        return Pulse(self.rf, self.gr, dt=self.dt, desc=self.desc,
                     device=device, dtype=dtype)


class SpinArray(object):
    r"""mrphy.mobjs.SpinArray object

    Usage:
        ``spinarray = SpinArray(shape, mask, *, T1_, T2_, γ_, M_, device,``\
        `` dtype)``
        ``spinarray = SpinArray(shape, mask, *, T1, T2, γ, M, device, dtype)``
    Inputs:
        - ``shape``: tuple, e.g., ``(N, nx, ny, nz)``.
    Optionals:
        - ``mask``: `(1, *Nd)`, where does compact attributes locate in `Nd`.
        - ``T1`` ⊻ ``T1_``: `(N, *Nd ⊻ nM)`, "Sec", T1 relaxation coeff.
        - ``T2`` ⊻ ``T2_``: `(N, *Nd ⊻ nM)`, "Sec", T2 relaxation coeff.
        - ``γ`` ⊻ ``γ_``: `(N, *Nd ⊻ nM)`,  "Hz/Gauss", gyro ratio.
        - ``M`` ⊻ ``M_``: `(N, *Nd ⊻ nM, xyz)`, spins, equilibrium ``[0 0 1]``.
        - ``device``: torch.device.
        - ``dtype``: torch.dtype

    Properties:
        - ``shape``: `(N, *Nd)`.
        - ``mask``: `(1, *Nd)`.
        - ``device``.
        - ``dtype``.
        - ``ndim``: ``len(shape)``
        - ``nM``: ``nM = torch.count_nonzero(mask).item()``.
        - ``T1_``: `(N, nM)`, "Sec", T1 relaxation coeff.
        - ``T2_``: `(N, nM)`, "Sec", T2 relaxation coeff.
        - ``γ_``: `(N, nM)`, "Hz/Gauss", gyro ratio.
        - ``M_``: `(N, nM, xyz)`, spins, equilibrium [0 0 1]

    .. warning::
        - Do NOT modify the ``mask`` of an object, e.g., \
          ``spinarray.mask[0] = True``.
        - Do NOT proceed indexed/masked assignments over any non-compact \
          attribute, e.g., ``spinarray.T1[0] = T1G`` or \
          ``spinarray.T1[mask] = T1G``.
          The underlying compact attributes will **NOT** be updated, since \
          they do not share memory.
          The only exception is when ``torch.all(mask == True)`` and the \
          underlying compact is **contiguous**, where the non-compact is just \
          a ``view((N, *Nd, ...))``.
          Checkout :func:`~mrphy.mobjs.SpinArray.crds_` and \
          :func:`~mrphy.mobjs.SpinArray.mask_` for indexed/masked access to \
          compacts.

    .. tip::
        - ``mask`` is GLOBAL for a batch, in other words, one cannot specify \
          distinct masks w/in a batch. \
          This design is to reduce storage/computations in, e.g., \
          ``applypulse`` (``blochsim``), avoiding extra allocations. \
          For DNN applications where an in-batch variation of ``mask`` may \
          seemingly be of interest, having ``torch.all(mask == True)`` and \
          postponing the variations to eventual losses evaluation can be a \
          better design, which allows reuse of ``M_``, etc., avoiding \
          repetitive allocations.
    """

    _readonly = ('shape', 'mask', 'device', 'dtype', 'is_cuda', 'ndim', 'nM')
    _compact = ('T1_', 'T2_', 'γ_', 'M_')
    __slots__ = set(_readonly + _compact)

    def __init__(
        self, shape: tuple, mask: Optional[Tensor] = None, *,
        T1: Optional[Tensor] = None, T1_: Optional[Tensor] = None,
        T2: Optional[Tensor] = None, T2_: Optional[Tensor] = None,
        γ: Optional[Tensor] = None,  γ_: Optional[Tensor] = None,
        M: Optional[Tensor] = None,  M_: Optional[Tensor] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):

        mask = (torch.ones((1,)+shape[1:], dtype=torch.bool, device=device)
                if mask is None else mask.to(device=device))

        assert(isinstance(device, torch.device) and
               isinstance(dtype, torch.dtype) and
               mask.dtype == torch.bool and
               mask.shape == (1,)+shape[1:])

        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'mask', mask)
        object.__setattr__(self, 'ndim', len(shape))
        object.__setattr__(self, 'nM', torch.count_nonzero(mask).item())
        object.__setattr__(self, 'device', device)
        object.__setattr__(self, 'dtype', dtype)
        object.__setattr__(self, 'is_cuda', self.device.type == 'cuda')

        assert((T1 is None) or (T1_ is None))
        if T1 is None:
            self.T1_ = (T1G if T1_ is None else T1_)
        else:
            self.T1 = T1

        assert((T2 is None) or (T2_ is None))
        if T2 is None:
            self.T2_ = (T2G if T2_ is None else T2_)
        else:
            self.T2 = T2

        assert((γ is None) or (γ_ is None))
        if γ is None:
            self.γ_ = (γH if γ_ is None else γ_)
        else:
            self.γ = γ

        assert((M is None) or (M_ is None))
        if M is None:
            self.M_ = (tensor([0., 0., 1.]) if M_ is None else M_)
        else:
            self.M = M

        return

    def __getattr__(self, k):  # provoked only when `__getattribute__` failed
        if k+'_' not in self._compact:
            raise AttributeError(f"'SpinArray' has no attribute '{k}'")

        v_ = getattr(self, k+'_')
        return (self.embed(v_) if self.nM != np.prod(self.shape[1:]) else
                v_.reshape(self.shape+v_.shape[2:]))  # ``mask`` is all True

    def __setattr__(self, k_, v_):
        if 'deepcopy' in (_.function for _ in inspect.stack()):
            # Hack, this enables `deepcopy()` w/o overriding `__deepcopy__()`.
            # Generator is faster than list comprehension.
            object.__setattr__(self, k_, v_)
            return

        if k_ in self._readonly:
            raise AttributeError(f"'SpinArray' object attribute '{k_}'"
                                 " is read-only")

        # Transfer ``v_`` to ``kw`` before ``extract`
        kw = {'device': self.device, 'dtype': self.dtype}
        v_ = (v_.to(**kw) if isinstance(v_, Tensor) else tensor(v_, **kw))

        shape = self.shape
        if k_+'_' in self._compact:  # enable non-compact assignment
            k_ = k_+'_'
            v_ = self.extract(v_.expand(shape+(3,) if k_ == 'M_' else shape))

        # `tensor.expand(size)` needs `tensor.shape` broadcastable with `size`
        if k_ == 'M_':
            if v_.shape != shape[:1]+(self.nM, 3):  # (N, nM, xyz)
                v_ = v_.expand(shape[:1]+(self.nM, 3)).clone()
        elif k_ in self._compact:  # (T1_, T2_, γ_)
            v_ = v_.expand((self.shape[0], self.nM))  # (N, nM)

        object.__setattr__(self, k_, v_)
        return

    def applypulse(
        self, pulse: Pulse, *,
        doEmbed: bool = False, doRelax: bool = True, doUpdate: bool = False,
        loc: Optional[Tensor] = None, loc_: Optional[Tensor] = None,
        Δf: Optional[Tensor] = None, Δf_: Optional[Tensor] = None,
        b1Map: Optional[Tensor] = None, b1Map_: Optional[Tensor] = None
    ) -> Tensor:
        r"""Apply a pulse to the spinarray object

        Typical usage:
            ``M = spinarray.applypulse(pulse, *, loc, doEmbed=True, doRelax,``\
            `` doUpdate, Δf, b1Map)``
            ``M_ = spinarray.applypulse(pulse, *, loc_, doEmbed=False, `` \
            ``doRelax, doUpdate, Δf_, b1Map_)``
        Inputs:
            - ``pulse``: mrphy.mobjs.Pulse.
            - ``loc`` ⊻ ``loc_``: `(N,*Nd ⊻ nM,xyz)`, "cm", locations.
        Optionals:
            - ``doEmbed``: [t/F], return ``M`` or ``M_``
            - ``doRelax``: [T/f], do relaxation during Bloch simulation.
            - ``doUpdate``: [t/F], update ``self.M_``
            - ``Δf``⊻ ``Δf_``: `(N,*Nd ⊻ nM)`, "Hz", off-resonance.
            - ``b1Map`` ⊻ ``b1Map_``: `(N,*Nd ⊻ nM,xy,(nCoils))`, transmit \
              sensitivity.
        Outputs:
            - ``M`` ⊻ ``M_``: `(N,*Nd ⊻ nM,xyz)`

        .. note::
            When ``doUpdate == True and doEmbed == False``, the output compact
            magnetization Tensor is a reference to ``self.M_``, and needs
            caution when being accessed.
        """
        assert ((loc_ is None) != (loc is None))  # XOR
        loc_ = (loc_ if loc is None else self.extract(loc))

        assert ((Δf_ is None) or (Δf is None))
        Δf_ = (Δf_ if Δf is None else self.extract(Δf))

        assert ((b1Map_ is None) or (b1Map is None))
        b1Map_ = (b1Map_ if b1Map is None else self.extract(b1Map))

        beff_ = self.pulse2beff(pulse, loc_=loc_,
                                Δf_=Δf_, b1Map_=b1Map_, doEmbed=False)

        if doRelax:
            kw_bsim = {'T1': self.T1_, 'T2': self.T2_}
        else:
            kw_bsim = {'T1': None, 'T2': None}

        kw_bsim['γ'] = self.γ_
        kw_bsim['dt'] = pulse.dt

        M_ = sims.blochsim(self.M_, beff_, **kw_bsim)
        if doUpdate:
            self.M_ = M_
        M_ = (self.embed(M_) if doEmbed else M_)
        return M_

    def asdict(self, *, toNumpy: bool = True, doEmbed: bool = True) -> dict:
        r"""Convert mrphy.mobjs.SpinArray object to dict

        Usage:
            ``d = spinarray.asdict(*, toNumpy, doEmbed)``

        Inputs:
            - ``toNumpy``: [T/f], convert ``Tensor`` to Numpy arrays.
            - ``doEmbed``: [T/f], embed compactly stored (nM) data to the \
              mask (\*Nd).
        Outputs:
            - ``d``: dict, dictionary with detached data identical to the \
              object.
        """
        fn_np = ((lambda x: x.detach().cpu().numpy()) if toNumpy else
                 (lambda x: x.detach()))

        _ = (('T1', 'T2', 'γ', 'M') if doEmbed else ('T1_', 'T2_', 'γ_', 'M_'))
        d = {k: fn_np(getattr(self, k)) for k in _}
        d['mask'] = fn_np(getattr(self, 'mask'))

        d.update({k: getattr(self, k) for k in ('shape', 'device', 'dtype')})
        return d

    def crds_(self, crds: list) -> list:
        r"""Compute crds for compact attributes

        Data in a SpinArray object is stored compactly, such that only those
        correspond to ``1`` on the ``spinarray.mask`` is kept.
        This function is provided to facilitate indexing the compact data from
        regular indices, by computing (ix, iy, iz) -> iM

        Usage:
            ``crds_ = spinarray.crds_(crds)``
        Inputs:
            - ``crds``: indices for indexing non-compact attributes.
        Outputs:
            - ``crds_``: list, ``len(crds_) == 2+len(crds)-self.ndim``.

        ``v_[crds_] == v[crds]``, when ``v_[crds_]=new_value`` is effective.
        """
        mask, ndim, nM = self.mask, self.ndim, self.nM
        assert (len(crds) >= ndim)
        crds_ = [crds[i] for i in (0,)+tuple(range(ndim, len(crds)))]
        m = torch.zeros(mask.shape, dtype=tensor(mask.numel()).dtype)-1
        m[mask] = torch.arange(nM)
        inds_ = [ind_ for ind_ in m[[[0]]+crds[1:ndim]].tolist() if ind_ != -1]

        crds_.insert(1, inds_)

        return crds_

    def dim(self) -> int:
        r"""Nd of the spinarray object, syntax sugar for len(spinarray.shape)

        Usage:
            ``Nd = spinarray.dim()``
        """
        return len(self.shape)

    def embed(self, v_: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
        """Embed compact data into the spinarray.mask

        Usage:
            ``out = spinarray.embed(v_, *, out)``
        Inputs:
            - ``v_``: `(N, nM, ...)`, must be contiguous.
        Optionals:
            - ``out``: `(N, *Nd, ...)`, in-place holder.
        Outputs:
            - ``out``: `(N, *Nd, ...)`.
        """
        oshape = self.shape+v_.shape[2:]
        out = (v_.new_full(oshape, float('NaN')) if out is None else out)
        mask = self.mask.expand(self.shape)
        out[mask] = v_.view((-1,)+v_.shape[2:])
        # `v.reshape()` has intermediate alloc, leaving `out` pointless.
        # out[mask] = v_.reshape((-1,)+v_.shape[2:])
        return out

    def extract(self, v: Tensor, *, out_: Optional[Tensor] = None) -> Tensor:
        r"""Extract data with the spinarray.mask, making it compact

        Usage:
            ``out_ = spinarray.extract(v, *, out_)``
        Inputs:
            - ``v``: `(N, *Nd, ...)`.
        Optionals:
            - ``out_``: `(N, nM, ...)`, in-place holder, must be contiguous.
        Outputs:
            - ``out_``: `(N, nM, ...)`.
        """
        oshape = (self.shape[0], self.nM)+v.shape[self.ndim:]
        out_ = (v.new_empty(oshape) if out_ is None else out_)
        mask = self.mask.expand(self.shape)
        # ! do NOT use ``out_.reshape()`; It creats new tensor when should
        # fail instead.
        out_.view((-1,)+v.shape[self.ndim:]).copy_(v[mask])
        # ``v[mask].reshape()`` has intermediate alloc, leaving ``out_``
        # pointless.
        # out_.copy_(v[mask].reshape((-1,)+v.shape[self.ndim:]))
        return out_

    def freeprec(
        self, dur: Tensor, *,
        doEmbed: bool = False, doRelax: bool = True, doUpdate: bool = False,
        Δf: Optional[Tensor] = None, Δf_: Optional[Tensor] = None
    ) -> Tensor:
        r"""Free precession of duration ``dur``

        Typical usage:
            ``M = obj.freeprec(dur, doEmbed=True, doRelax, doUpdate, Δf)``
            ``M_ = obj.applypulse(dur, doEmbed=False, doRelax, doUpdate, Δf_)``
        Inputs:
            - ``dur``: `()` ⊻ `(N ⊻ 1,)`, "Sec", duration of free-precession.
        Optionals:
            - ``doEmbed``: [t/F], return ``M`` or ``M_``
            - ``doRelax``: [T/f], do relaxation during free precession.
            - ``doUpdate``: [t/F], update ``self.M_``
            - ``Δf``⊻ ``Δf_``: `(N ⊻ 1,*Nd ⊻ nM)`, "Hz", off-resonance.
        Outputs:
            - ``M`` ⊻ ``M_``: `(N,*Nd ⊻ nM,xyz)`

        .. note::
            When ``doUpdate == True and doEmbed == False``, the output compact
            magnetization Tensor is a reference to ``self.M_``, and needs
            caution when being accessed.
        """
        assert ((Δf_ is None) or (Δf is None))
        Δf_ = (Δf_ if Δf is None else self.extract(Δf))

        if doRelax:
            kw_bsim = {'T1': self.T1_, 'T2': self.T2_}
        else:
            kw_bsim = {'T1': None, 'T2': None}

        M_ = sims.freeprec(self.M_, dur, **kw_bsim, Δf=Δf_)
        if doUpdate:
            self.M_ = M_
        M_ = (self.embed(M_) if doEmbed else M_)
        return M_

    def mask_(self, *, mask: Tensor) -> Tensor:
        r"""Extract the compact region of an input external ``mask``.

        Usage:
            ``mask_ = spinarray.mask_(mask)``
        Inputs:
            - ``mask``: `(1, *Nd)`.
        Outputs:
            - ``mask_``: `(1, nM)`, ``mask_`` can be used on compact \
              attributes.
        """
        mask_ = mask(self.mask).reshape((1, -1))
        return mask_

    def numel(self) -> int:
        r"""Number of spins for the spinarray object, incompact.

        Syntax sugar of ``spinarray.mask.numel()``, effectively
        ``prod(spinarray.size())``.

        Usage:
            ``res = spinarray.numel()``
        """
        return self.mask.numel()

    def pulse2beff(
        self, pulse: Pulse, *, doEmbed: bool = False,
        loc: Optional[Tensor] = None, loc_: Optional[Tensor] = None,
        Δf: Optional[Tensor] = None, Δf_: Optional[Tensor] = None,
        b1Map: Optional[Tensor] = None, b1Map_: Optional[Tensor] = None
    ) -> Tensor:
        r"""Compute B-effective of ``pulse`` with the spinarray's parameters

        Typical usage:
            ``beff = spinarray.pulse2beff(pulse, *, loc, doEmbed=True, Δf, ``\
            ``b1Map)``
            ``beff_ = spinarray.pulse2beff(pulse, *, loc_, doEmbed=False, ``\
            ``Δf_, b1Map_)``
        Inputs:
            - ``pulse``: mrphy.mobjs.Pulse.
            - ``loc`` ⊻ ``loc_``: `(N,*Nd ⊻ nM,xyz)`, "cm", locations.
        Optionals:
            - ``doEmbed``: [t/F], return ``beff`` or ``beff_``
            - ``Δf`` ⊻ ``Δf_``: `(N,*Nd ⊻ nM)`, "Hz", off-resonance.
            - ``b1Map`` ⊻ ``b1Map_``: `(N,*Nd ⊻ nM,xy,(nCoils))`, transmit \
              sensitivity.
        Outputs:
            - ``beff`` ⊻ ``beff_``: `(N,*Nd ⊻ nM,xyz,nT)`.
        """
        assert ((loc_ is None) != (loc is None))  # XOR
        loc_ = (loc_ if loc is None else self.extract(loc))

        assert ((Δf_ is None) or (Δf is None))
        Δf_ = (Δf_ if Δf is None else self.extract(Δf))

        assert ((b1Map_ is None) or (b1Map is None))
        b1Map_ = (b1Map_ if b1Map is None else self.extract(b1Map))

        pulse = pulse.to(device=self.device, dtype=self.dtype)
        beff_ = pulse.beff(loc_, γ=self.γ_, Δf=Δf_, b1Map=b1Map_)
        beff_ = (self.embed(beff_) if doEmbed else beff_)
        return beff_

    def size(self) -> tuple:
        r"""Size of the spinarray object.

        Syntax sugar of ``spinarray.shape``.

        Usage:
            ``sz = spinarray.size()``
        """
        return self.shape

    def to(
        self, *,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> 'SpinArray':
        r"""Duplicate the object to the prescribed device with dtype

        Usage:
            ``new_spinarray = spinarray.to(*, device, dtype)``
        Inputs:
            - ``device``: torch.device
            - ``dtype``: torch.dtype
        Outputs:
            - ``new_spinarray``: mrphy.mobjs.SpinArray object
        """
        if self.device == device and self.dtype == dtype:
            return self
        return SpinArray(self.shape, self.mask, T1_=self.T1_, T2_=self.T2_,
                         γ_=self.γ_, M_=self.M_, device=device, dtype=dtype)


class SpinCube(SpinArray):
    r"""mrphy.mobjs.SpinCube object

    Usage:
        ``SpinCube(shape, fov, mask, *, ofst, Δf_, T1_, T2_, γ_, M_, device,``\
        `` dtype)``
        ``SpinCube(shape, fov, mask, *, ofst, Δf, T1, T2, γ, M, device,``\
        '' dtype)``
    Inputs:
        - ``shape``: tuple, e.g., ``(N, nx, ny, nz)``.
        - ``fov``: `(N, xyz)`, "cm", field of view.
    Optionals:
        - ``mask``: `(1, *Nd)`, where does compact attributes locate in `Nd`.
        - ``ofst``: `(N, xyz)`, Tensor "cm", fov offset from iso-center.
        - ``Δf`` ⊻ ``Δf_``: `(N, *Nd ⊻ nM)`, "Hz", off-resonance map.
        - ``T1`` ⊻ ``T1_``: `(N, *Nd ⊻ nM)`, "Sec", T1 relaxation coeff.
        - ``T2`` ⊻ ``T2_``: `(N, *Nd ⊻ nM)`, "Sec", T2 relaxation coeff.
        - ``γ`` ⊻ ``γ_``: `(N, *Nd ⊻ nM)`,  "Hz/Gauss", gyro ratio.
        - ``M`` ⊻ ``M_``: `(N, *Nd ⊻ nM, xyz)`, spins, equilibrium ``[0 0 1]``.
        - ``device``: torch.device.
        - ``dtype``: torch.dtype

    Properties:
        - ``spinarray``: SpinArray object.
        - ``Δf_``: `(N, nM)`, "Hz", off-resonance map.
        - ``loc_``: `(N, nM, xyz)`, "cm", location of spins.
        - ``fov``: `(N, xyz)`, "cm", field of view.
        - ``ofst``: `(N, xyz)`, "cm", fov offset from iso-center.
    """

    _readonly = ('spinarray', 'loc_')
    _compact = ('Δf_', 'loc_')  # `loc_` depends on `shape`, `fov` and `ofst`
    __slots__ = set(_readonly+_compact+('fov', 'ofst'))

    def __init__(
        self, shape: tuple, fov: Tensor, *, mask: Optional[Tensor] = None,
        ofst: Tensor = tensor([[0., 0., 0.]]),
        Δf: Optional[Tensor] = None, Δf_: Optional[Tensor] = None,
        T1: Optional[Tensor] = None, T1_: Optional[Tensor] = None,
        T2: Optional[Tensor] = None, T2_: Optional[Tensor] = None,
        γ: Optional[Tensor] = None,  γ_: Optional[Tensor] = None,
        M: Optional[Tensor] = None,  M_: Optional[Tensor] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        # 1) `SpinCube` is subclassed to `SpinArray`.
        # 2) Attribute `spinarray` is included to enable extracting the
        # `SpinArray` part of a `SpinCube()` instance.
        # 3) `SpinCube.__getattr__` is tweaked for immitating super class
        # access to `SpinCube().spinarray`'s attributes.
        # To have these three features, we cannot have `super().__init__()` in
        # `SpinCube`, which will disable the `__getattr__` tweak.
        sp = SpinArray(shape, mask, T1=T1, T1_=T1_, T2=T2, T2_=T2_, γ=γ, γ_=γ_,
                       M=M, M_=M_, device=device, dtype=dtype)
        object.__setattr__(self, 'spinarray', sp)

        kw = {'device': sp.device, 'dtype': sp.dtype}
        # setattr(self, k, v), avoid computing `loc_` w/ `fov` & `ofst` not set
        object.__setattr__(self, 'fov', fov.to(**kw))
        object.__setattr__(self, 'ofst', ofst.to(**kw))
        # Initialize ``loc_`` in memory, reuse it.
        object.__setattr__(self, 'loc_',
                           torch.zeros((sp.shape[0], sp.nM, 3), **kw))
        self._update_loc_()  # compute ``loc_`` from set ``fov`` & ``ofst`

        assert((Δf is None) or (Δf_ is None))
        if Δf is None:
            self.Δf_ = (tensor(0.) if Δf_ is None else Δf_)
        else:
            self.Δf = Δf

        return

    def __getattr__(self, k):  # provoked only when `__getattribute__` failed
        if k+'_' not in self._compact:  # k not in ('Δf_', 'loc')
            # Cannot do `self.spinarray` or `getattr(self, 'spinarray')` here.
            # They cause infinite recursion when `SpinCube().__getattr__()` is
            # queried with `spinarray`, which may happen during `deepcopy` or
            # `pickle`.
            # Therefore, call `object.__getattribute__()` here, and let it fail
            # when it should.
            spinarray = object.__getattribute__(self, 'spinarray')
            try:
                return getattr(spinarray, k)
            except AttributeError:
                raise AttributeError(f"'SpinCube' has no attribute '{k}'")

        v_, sp = getattr(self, k+'_'), self.spinarray
        return (sp.embed(v_) if sp.nM != np.prod(sp.shape[1:]) else
                v_.reshape(sp.shape+v_.shape[2:]))  # `mask` is all True

    def __setattr__(self, k_, v_):
        if 'deepcopy' in (_.function for _ in inspect.stack()):
            # Hack, this enables `deepcopy()` w/o overriding `__deepcopy__()`.
            # Generator is faster than list comprehension.
            object.__setattr__(self, k_, v_)
            return

        if (k_ in self._readonly) or (k_+'_' in self._readonly):
            raise AttributeError(f"'SpinCube' object attribute '{k_}'"
                                 " is read-only")

        sp = self.spinarray
        if k_ in SpinArray.__slots__ or k_+'_' in SpinArray.__slots__:
            setattr(sp, k_, v_)
            return

        kw = {'device': sp.device, 'dtype': sp.dtype}
        v_ = (v_.to(**kw) if isinstance(v_, Tensor) else tensor(v_, **kw))

        shape = sp.shape
        if k_+'_' in self._compact:  # `loc_` excluded by beginning assert
            k_ = k_+'_'
            v_ = self.extract(v_.expand(shape+(3,) if k_ == 'loc_' else shape))

        if k_ == 'Δf_':
            v_ = v_.expand((shape[0], sp.nM))  # (N, nM)
        elif k_ in ('fov', 'ofst'):
            assert(v_.ndim == 2)

        object.__setattr__(self, k_, v_)

        # update `loc_` when needed
        if k_ in ('fov', 'ofst'):
            self._update_loc_()
        return

    def _update_loc_(self):
        r"""Update ``spincube.loc_`` using FOV and offset

        The ``spincube``'s spin locations are computed internally from set FOV
        and offset.

        Usage:
            ``loc_ = spincube._update_loc_()``
        """
        loc_, fov, ofst = self.loc_, self.fov, self.ofst
        sp = self.spinarray
        kw = {'device': sp.device, 'dtype': sp.dtype}

        # locn (1, prod(Nd), xyz)  normalized locations, [-0.5, 0.5)
        shape, mask = sp.shape, sp.mask
        crdn = ((torch.arange(x, **kw)-utils.ctrsub(x))/x for x in shape[1:])
        _locn = torch.meshgrid(*crdn)  # ((*Nd,), (*Nd), (*Nd))

        for i in range(3):  # xyz, (N, nM)
            # According to `memory_profiler`, this does not provoke allocs.
            # `torch.addr`'s `vec2`, _locn[i][mask[0, ...]], provokes alloc.
            loc_[..., i] = (fov[:, None, i]*_locn[i][mask[0, ...]][None, ...]
                            + ofst[:, None, i])

        return

    def applypulse(
        self, pulse: Pulse, *,
        doEmbed: bool = False, doRelax: bool = True, doUpdate: bool = False,
        b1Map: Optional[Tensor] = None, b1Map_: Optional[Tensor] = None
    ) -> Tensor:
        r"""Apply a pulse to the spincube object

        Usage:
            ``M = spincube.applypulse(pulse, *, doEmbed=True, doRelax, b1Map)``
            ``M_ = spincube.applypulse(pulse, *, doEmbed=False, doRelax,``\
            `` b1Map_)``

        Inputs:
            - ``pulse``: mobjs.Pulse object.
        Optionals:
            - ``doEmbed``: [t/F], return ``M`` or ``M_``.
            - ``doRelax``: [T/f], do relaxation during Bloch simulation.
            - ``b1Map`` ⊻ ``b1Map_``: `(N,*Nd ⊻ nM,xy,(nCoils))`, transmit \
              sensitivity.
        Outputs:
            - ``M`` ⊻ ``M_``: `(N,*Nd ⊻ nM,xyz)`.
        """
        assert ((b1Map_ is None) or (b1Map is None))
        b1Map_ = (b1Map_ if b1Map is None else self.extract(b1Map))

        return self.spinarray.applypulse(pulse, doEmbed=doEmbed,
                                         doRelax=doRelax, doUpdate=doUpdate,
                                         Δf_=self.Δf_, loc_=self.loc_,
                                         b1Map_=b1Map_)

    def freeprec(
        self, dur: Tensor, *,
        doEmbed: bool = False, doRelax: bool = True, doUpdate: bool = False
    ) -> Tensor:
        r"""Free precession of duration ``dur``

        Typical usage:
            ``M = obj.freeprec(dur, doEmbed=True, doRelax, doUpdate)``
            ``M_ = obj.applypulse(dur, doEmbed=False, doRelax, doUpdate)``
        Inputs:
            - ``dur``: `()` ⊻ `(N ⊻ 1,)`, "Sec", duration of free-precession.
        Optionals:
            - ``doEmbed``: [t/F], return ``M`` or ``M_``
            - ``doRelax``: [T/f], do relaxation during free precession.
            - ``doUpdate``: [t/F], update ``self.M_``
        Outputs:
            - ``M`` ⊻ ``M_``: `(N,*Nd ⊻ nM,xyz)`

        .. note::
            When ``doUpdate == True and doEmbed == False``, the output compact
            magnetization Tensor is a reference to ``self.M_``, and needs
            caution when being accessed.
        """

        return self.spinarray.freeprec(dur, Δf_=self.Δf_, doEmbed=doEmbed,
                                       doRelax=doRelax, doUpdate=doUpdate)

    def asdict(self, *, toNumpy: bool = True, doEmbed: bool = True) -> dict:
        r"""Convert mrphy.mobjs.SpinCube object to dict

        Usage:
            ``d = spincube.asdict(*, toNumpy, doEmbed)``

        Inputs:
            - ``toNumpy``: [T/f], convert ``Tensor`` to Numpy arrays.
            - ``doEmbed``: [T/f], embed compactly stored (nM) data to the \
              mask `(*Nd)`.
        Outputs:
            - ``d``: dict, dictionary with detached data identical to the \
              object.
        """
        fn_np = ((lambda x: x.detach().cpu().numpy()) if toNumpy else
                 (lambda x: x.detach()))

        _ = (('loc', 'Δf') if doEmbed else ('loc', 'Δf'))
        d = {k: fn_np(getattr(self, k)) for k in _}

        d.update({k: getattr(self, k) for k in ('fov', 'ofst')})

        d.update(self.spinarray.asdict(toNumpy=toNumpy, doEmbed=doEmbed))
        return d

    def pulse2beff(
        self, pulse: Pulse, *,
        doEmbed: bool = False,
        b1Map: Optional[Tensor] = None, b1Map_: Optional[Tensor] = None
    ) -> Tensor:
        r"""Compute B-effective of ``pulse`` with the spincube's parameters

        Typical usage:
            ``beff = spincube.pulse2beff(pulse, *, doEmbed=True, b1Map)``
            ``beff_ = spincube.pulse2beff(pulse, *, doEmbed=False, b1Map_)``
        Inputs:
            - ``pulse``: mrphy.mobjs.Pulse.
        Optionals:
            - ``doEmbed``: [t/F], return ``beff`` or ``beff_``.
            - ``b1Map`` ⊻ ``b1Map_``: `(N,*Nd ⊻ nM,xy,(nCoils))`, transmit \
              sensitivity.
        Outputs:
            - ``beff`` ⊻ ``beff_``: `(N,*Nd ⊻ nM,xyz,nT)`.
        """
        return self.spinarray.pulse2beff(pulse, self.loc_, doEmbed=doEmbed,
                                         Δf_=self.Δf_,
                                         b1Map=b1Map, b1Map_=b1Map_)

    def to(
        self, *,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> 'SpinCube':
        r"""Duplicate the object to the prescribed device with dtype

        Usage:
            ``new_spincube = spincube.to(*, device, dtype)``
        Inputs:
            - ``device``: torch.device.
            - ``dtype``: torch.dtype.
        Outputs:
            - ``new_spincube``: mrphy.mobjs.SpinCube object.
        """
        if self.device == device and self.dtype == dtype:
            return self
        return SpinCube(self.shape, self.fov, ofst=self.ofst, Δf_=self.Δf_,
                        T1_=self.T1_, T2_=self.T2_, γ_=self.γ_, M_=self.M_,
                        device=device, dtype=dtype)


class SpinBolus(SpinArray):
    def __init__(
        self
    ):
        pass
    pass


class Examples(object):
    r"""Class for quickly creating exemplary instances to play around with.
    """
    @staticmethod
    def pulse() -> Pulse:
        r"""Create a mrphy.mobjs.Pulse object.
        """
        device = torch.device('cpu')
        dtype = torch.float32

        kw = {'dtype': dtype, 'device': device}
        N, nT, dt = 1, 512, dt0

        # pulse: Sec; Gauss; Gauss/cm.
        pulse_size = (N, 1, nT)
        t = torch.arange(0, nT, **kw).reshape(pulse_size)
        rf = 10*torch.cat([torch.cos(t/nT*2*π),                # (1,xy, nT)
                           torch.sin(t/nT*2*π)], 1)
        gr = torch.cat([torch.ones(pulse_size, **kw),
                        torch.ones(pulse_size, **kw),
                        10*torch.atan(t - round(nT/2))/π], 1)  # (1,xyz,nT)

        # Pulse
        p = Pulse(rf=rf, gr=gr, dt=dt, **kw)
        return p

    @staticmethod
    def spinarray() -> SpinArray:
        r"""Create a mrphy.mobjs.SpinArray object.
        """
        device = torch.device('cpu')
        dtype = torch.float32
        kw = {'dtype': dtype, 'device': device}

        N, Nd, γ_ = 1, (3, 3, 3), γH
        shape = (N, *Nd)
        mask = torch.zeros((1,)+Nd, device=device, dtype=torch.bool)
        mask[0, :, 1, :], mask[0, 1, :, :] = True, True
        T1_, T2_ = tensor([[1.]], **kw), tensor([[4e-2]], **kw)

        array = SpinArray(shape, mask=mask, T1_=T1_, T2_=T2_, γ_=γ_, **kw)
        return array

    @staticmethod
    def spincube() -> SpinCube:
        r"""Create a mrphy.mobjs.SpinCube object.
        """
        device = torch.device('cpu')
        dtype = torch.float32
        kw = {'dtype': dtype, 'device': device}

        N, Nd, γ_ = 1, (3, 3, 3), γH
        shape = (N, *Nd)
        mask = torch.zeros((1,)+Nd, device=device, dtype=torch.bool)
        mask[0, :, 1, :], mask[0, 1, :, :] = True, True
        fov, ofst = tensor([[3., 3., 3.]], **kw), tensor([[0., 0., 1.]], **kw)
        T1_, T2_ = tensor([[1.]], **kw), tensor([[4e-2]], **kw)

        cube = SpinCube(shape, fov, mask=mask, ofst=ofst,
                        T1_=T1_, T2_=T2_, γ_=γ_, **kw)

        cube.Δf = torch.sum(-cube.loc[0:1, :, :, :, 0:2], dim=-1) * γ_
        return cube
