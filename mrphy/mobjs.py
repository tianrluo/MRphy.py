import torch
from torch import tensor, Tensor
from typing import TypeVar, Type, Union
from numbers import Number

from mrphy import γH, dt0, T1G, T2G
from mrphy import utils, beffective, sims

"""
"""

# TODO:
# - Abstract Class


Pulse = TypeVar('Pulse', bound='Pulse')
SpinArray = TypeVar('SpinArray', bound='SpinArray')
SpinCube = TypeVar('SpinCube', bound='SpinCube')


class Pulse(object):
    """
    # Attributes:
    - `rf` (N,xy, nT,(nCoils)) "Gauss", `xy` for separating real and imag part.
    - `gr` (N,xyz,nT) "Gauss/cm"
    - `dt` (N,1,), "Sec" simulation temporal step size, i.e., dwell time.
    - `desc` str, an description of the pulse to be constructed.
    """

    __slots__ = ('rf', 'gr', 'dt', 'desc', 'device', 'dtype')

    def __init__(
            self,
            rf: Tensor = None, gr: Tensor = None,
            dt: Type[Union[Tensor, Number]] = dt0,
            desc: str = "generic pulse",
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32):

        assert(isinstance(device, torch.device) and
               isinstance(dtype, torch.dtype))

        # Defaults
        rf_miss, gr_miss = rf is None, gr is None
        assert (not(rf_miss and gr_miss)), "Missing both `rf` and `gr` inputs"

        super().__setattr__('device', device)
        super().__setattr__('dtype', dtype)

        dkw = {'device': self.device, 'dtype': self.dtype}

        if rf_miss:
            N, nT = gr.shape[0], gr.shape[2]
            rf = torch.zeros((N, 2, nT), **dkw)

        if gr_miss:
            N, nT = rf.shape[0], rf.shape[2]
            gr = torch.zeros((N, 3, nT), **dkw)

        # super() here, as self.__setattr__() has interdependent sanity check.
        rf, gr = rf.to(**dkw), gr.to(**dkw)
        assert (rf.shape[0] == gr.shape[0] and rf.shape[2] == gr.shape[2])

        super().__setattr__('rf', rf)
        super().__setattr__('gr', gr)

        self.dt, self.desc = dt, desc
        return

    def __setattr__(self, k, v):
        assert (k not in ('device', 'dtype')), "'%s' not writable" % k

        if k in ('rf', 'gr', 'dt'):
            if k == 'dt' and isinstance(v, Number):
                v = tensor(v, device=self.device, dtype=self.dtype)
            else:
                v = v.to(device=self.device, dtype=self.dtype)

        if (k == 'gr'):
            shape = self.rf.shape
            assert (v.shape[0] == shape[0] and v.shape[2] == shape[2])
        if (k == 'rf'):
            shape = self.gr.shape
            assert (v.shape[0] == shape[0] and v.shape[2] == shape[2])

        super().__setattr__(k, v)
        return

    def beff(
            self, loc: Tensor,
            Δf: Tensor = None, b1Map: Tensor = None,
            γ: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `loc`   (N,*Nd,xyz) "cm", locations.
        *OPTIONALS*:
        - `Δf`    (N,*Nd,) "Hz", off-resonance.
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        - `γ`     (N,*Nd) "Hz/Gauss", gyro-ratio
        *OUTPUTS*:
        - `beff`  (N,*Nd,xyz,nT)
        """
        return beffective.rfgr2beff(self.rf, self.gr,
                                    loc.to(device=self.device), Δf=Δf,
                                    b1Map=b1Map, γ=γ)

    def to(self, device: torch.device = torch.device('cpu'),
           dtype: torch.dtype = torch.float32) -> Pulse:
        if (self.device != device) or (self.dtype != dtype):
            return Pulse(self.rf, self.gr, dt=self.dt, desc=self.desc,
                         device=device, dtype=dtype)
        else:
            return self
        return


class SpinArray(object):
    """
        SpinArray(shape; mask, T1, T2, γ, M, device, dtype)
    *INPUTS*:
    - `shape` tuple( (N, nx, (ny, (nz,...))) ).
    *OPTIONALS*:
    - `T1` (self.shape) "Sec", T1 relaxation coeff.
    - `T2` (self.shape) "Sec", T2 relaxation coeff.
    - `γ`  (self.shape) "Hz/Gauss", gyro ratio.
    - `M`  (self.shape)+(xyz,), spins, assumed equilibrium [0 0 1]
    - `device` torch.device; `dtype` torch.dtype
    """

    __slots__ = ('shape', 'mask', 'device', 'dtype', 'T1', 'T2', 'γ', 'M')

    def __init__(
            self, shape: tuple,
            mask: Tensor = None,
            T1: Type[Union[Tensor, Number]] = T1G,
            T2: Type[Union[Tensor, Number]] = T2G,
            γ: Type[Union[Tensor, Number]] = γH,
            M: Tensor = None,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32):

        assert(isinstance(device, torch.device) and
               isinstance(dtype, torch.dtype))

        super().__setattr__('shape', shape)
        super().__setattr__('device', device)
        super().__setattr__('dtype', dtype)

        dkw = {'device': self.device, 'dtype': self.dtype}

        # Defaults
        self.mask = (torch.ones((1, *shape[1:]),
                                dtype=torch.bool, device=device)
                     if (mask is None) else mask)
        self.T1 = T1G if (T1 is None) else T1
        self.T2 = T2G if (T2 is None) else T2
        self.γ = γH if (γ is None) else γ
        self.M = tensor([[[0., 0., 1.]]], **dkw) if (M is None) else M

        return

    @property
    def ndim(self):
        return len(self.shape)

    def __getattr__(self, k):
        raise AttributeError("'SpinArray' object has no attribute '%s'" % k)
        return

    def __setattr__(self, k, v):
        assert (k not in ('shape', 'device', 'dtype')), "'%s' not writable" % k

        if not isinstance(v, Tensor):
            v = tensor(v, device=self.device)
        else:
            v = v.to(device=self.device)

        vs, s, d = v.shape, self.shape, self.ndim

        # tensor.expand is aligned by trailing dimension
        if k in ('T1', 'T2', 'γ', 'M'):
            v.to(dtype=self.dtype)
            if k in ('M') and vs != s+(3,):  # -> self.shape+(3,)
                assert(len(vs) == 3 and vs[1] == 1)
                v = v.reshape(vs[0:1]+(d-1)*(1,)+(3,)).expand(s+(3,)).clone()
            elif k in ('T1', 'T2', 'γ'):  # -> self.shape
                v = v.reshape(vs + (d-v.ndim)*(1,)).expand(s)
        elif k in ('mask'):
            assert(v.dtype == torch.bool)
            assert(vs == (1, *s[1:])), "new %s.shape is incompatible." % k

        super().__setattr__(k, v)

        return

    def applypulse(
            self, p: Pulse, loc: Tensor, doMask: bool = False,
            **kw) -> Tensor:
        """
        *INPUTS*:
        - `p`   (1,) mobjs.Pulse object
        - `loc` (N,*Nd,xyz) "cm", locations.
        *OPTIONALS*:
        - `Δf`
        - `b1Map`
        """
        dkw = {'device': self.device, 'dtype': self.dtype}
        p, loc = p.to(**dkw), loc.to(**dkw)
        beff = self.pulse2beff(p, loc, doMask=doMask, **kw)

        ks = ('T1', 'T2', 'γ')
        if doMask:
            kw_bsim = {k: self.extract(getattr(self, k)) for k in ks}
            M = self.extract(self.M)
        else:
            kw_bsim = {k: getattr(self, k) for k in ks}
            M = self.M

        kw_bsim['dt'] = p.dt

        return sims.blochsim(M, beff, **kw_bsim)

    def applypulse_(
            self, p: Pulse, loc: Tensor, doMask: bool = False,
            **kw) -> Tensor:
        """
        This function does not save allocations but only updates the self.M
        """
        M = self.applypulse(p, loc, doMask=doMask, **kw)
        self.M = (self.embed_(M, self.M) if doMask else M)
        return self.M

    def dim(self) -> int: return len(self.shape)

    def embed(self, v: Tensor, mask: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `v` (N, nM, ...)
        """
        t = v.new_zeros(self.shape+v.shape[2:])
        t = self.embed_(v, t, mask=mask)
        return t

    def embed_(self, v: Tensor, t: Tensor, mask: Tensor = None) -> Tensor:
        mask = (mask if mask is not None else self.mask).expand(self.shape)
        t[mask] = v
        return t

    def extract(self, v: Tensor, mask: Tensor = None) -> Tensor:
        N, ndim = self.shape[0], self.ndim
        mask = (mask if mask is not None else self.mask).expand(self.shape)
        return v[mask].reshape((N, -1,)+v.shape[ndim:])

    def pulse2beff(
            self, p: Pulse, loc: Tensor, doMask: bool = False,
            Δf: Tensor = None, b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `loc`   (N,*Nd,xyz) "cm", locations.
        *OPTIONALS*:
        - `doMask` [t/F], if `True`, return masked `beff`.
        - `Δf`    (N,*Nd,) "Hz", off-resonance.
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        *OUTPUTS*:
        - `beff`  (N,*Nd,xyz,nT)
        """
        dkw = {'device': self.device, 'dtype': self.dtype}
        p, loc = p.to(**dkw), loc.to(**dkw)
        γ = self.γ
        if doMask:
            loc, γ, Δf = (self.extract(x) for x in (loc, γ, Δf))
            b1Map = (self.extract(b1Map) if b1Map else b1Map)

        return p.beff(loc, γ=γ, Δf=Δf, b1Map=b1Map)

    def size(self) -> tuple: return self.shape

    def to(self, device: torch.device = torch.device('cpu'),
           dtype: torch.dtype = torch.float32) -> SpinArray:
        if (self.device != device) or (self.dtype != dtype):
            return SpinArray(self.shape, mask=self.mask,
                             T1=self.T1, T2=self.T2, γ=self.γ, M=self.M,
                             device=device, dtype=dtype)
        else:
            return self
        return


class SpinCube(object):
    """
        SpinCube(shape, fov; ofst, Δf, T1, T2, γ, M, device, dtype)
    *INPUTS*:
    - `shape` tuple( (N, nx, (ny, (nz,...))) ).
    - `fov` (N, xyz,) "cm", field of view.
    *OPTIONALS*:
    - `ofst` (N, xyz,) "cm", fov offset from iso-center.
    - `Δf` (self.shape) "Hz", off-resonance map.
    - `T1` (self.shape) "Sec", T1 relaxation coeff.
    - `T2` (self.shape) "Sec", T2 relaxation coeff.
    - `γ`  (self.shape) "Hz/Gauss", gyro ratio.
    - `M`  (self.shape)+(xyz,), spins, assumed equilibrium [0 0 1]
    - `device` torch.device; `dtype` torch.dtype
    """

    # SpinArray.__slots__ = ('shape', 'device', 'dtype', 'T1', 'T2', 'γ', 'M')
    __slots__ = ('spinarray', 'Δf', 'fov', 'ofst')

    def __init__(
            self, shape: tuple,
            fov: Tensor,
            ofst: Tensor = None,
            mask: Tensor = None,
            Δf: Tensor = None, M=None,
            T1: Type[Union[Tensor, Number]] = T1G,
            T2: Type[Union[Tensor, Number]] = T2G,
            γ: Type[Union[Tensor, Number]] = γH,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32):
        """
        """
        super().__setattr__('spinarray',
                            SpinArray(shape, mask=mask, T1=T1, T2=T2, γ=γ, M=M,
                                      device=device, dtype=dtype))

        dkw = {'device': self.spinarray.device, 'dtype': self.spinarray.dtype}

        self.fov = fov.to(**dkw)
        self.ofst = tensor([[0., 0., 0.]], **dkw) if (ofst is None) else ofst
        self.Δf = tensor([[0.]], **dkw) if (Δf is None) else Δf

        return

    @property
    def loc(self, dtype=torch.float32) -> Tensor:
        """
        - `loc` (self.shape)+(xyz,) "cm", location of spins.
        """
        kw = {'device': self.device, 'dtype': self.dtype}

        # _loc (1, *Nd, len(Nd))
        coord = map(lambda x: (torch.arange(x, **kw)-utils.ctrsub(x))/x,
                    self.shape[1:])
        _loc = torch.stack(torch.meshgrid((*coord,)), dim=-1)[None, ...]

        dim1 = self.ndim-1
        (fov, ofst) = map(lambda x: x.reshape(x.shape[0:1]+dim1*(1,)+(-1,)),
                          (self.fov, self.ofst))  # -> (N,)+dim1*(1,)+(3,)
        loc = fov*_loc + ofst
        return loc  # (self.shape)+(xyz,)

    def __getattr__(self, k):  # provoked only when `__getattribute__` failed
        return getattr(self.spinarray, k)

    def __setattr__(self, k, v):
        assert (k not in ('spinarray')), "'%s' not writable" % k

        if k in SpinArray.__slots__:
            setattr(self.spinarray, k, v)
        else:
            if not isinstance(v, Tensor):
                v = tensor(v, device=self.device, dtype=self.dtype)
            else:
                v = v.to(device=self.device, dtype=self.dtype)

            vs, s = v.shape, self.shape

            if k in ('Δf') and vs != s:
                v = v.reshape(v.shape + (self.ndim-v.ndim)*(1,)).expand(s)
            elif k in ('fov', 'ofst'):
                assert(v.ndim == 2)

            super().__setattr__(k, v)

        return

    def applypulse(
            self, p: Pulse, doMask: bool = False,
            b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `p`   (1,) mobjs.Pulse object
        *OPTIONALS*
        - `b1Map`
        """
        return self.spinarray.applypulse(p, self.loc, doMask=doMask,
                                         Δf=self.Δf, b1Map=b1Map)

    def applypulse_(
            self, p: Pulse, doMask: bool = False,
            b1Map: Tensor = None) -> Tensor:
        """
        This function does not save allocations but only updates the self.M
        """
        return self.spinarray.applypulse_(p, self.loc, doMask=doMask,
                                          Δf=self.Δf, b1Map=b1Map)

    def dim(self) -> int: return self.spinarray.ndim

    def embed(self, v: Tensor, mask: Tensor = None) -> Tensor:
        return self.spinarray.embed(v, mask=mask)

    def embed_(self, v: Tensor, t: Tensor, mask: Tensor = None) -> Tensor:
        return self.spinarray.embed_(v, t, mask=mask)

    def extract(self, v: Tensor, mask: Tensor = None) -> Tensor:
        return self.spinarray.extract(v, mask=mask)

    def pulse2beff(
            self, p: Pulse, doMask: bool = False,
            b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `p` (1,) Pulse object
        *OPTIONALS*:
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        *OUTPUTS*:
        - `beff`  (N,*Nd,xyz,nT)
        """
        return self.spinarray.pulse2beff(p, self.loc, doMask=doMask,
                                         Δf=self.Δf, b1Map=b1Map)

    def size(self) -> tuple: return self.spinarray.shape

    def to(self, device: torch.device = torch.device('cpu'),
           dtype: torch.dtype = torch.float32) -> SpinCube:
        if (self.device != device) or (self.dtype != dtype):
            return SpinCube(self.shape, self.fov, ofst=self.ofst,
                            mask=self.mask, Δf=self.Δf, T1=self.T1, T2=self.T2,
                            γ=self.γ, M=self.M, device=device, dtype=dtype)
        else:
            return self
        return


class SpinBolus(SpinArray):
    def __init__(
            self):
        pass
    pass
