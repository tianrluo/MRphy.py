import torch
from torch import tensor, cuda, Tensor
from typing import TypeVar, Type, Union
from numbers import Number

from mrphy import γH, dt0, T1G, T2G, π
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

    _immutable = ('device', 'dtype')
    __slots__ = _immutable + ('rf', 'gr', 'dt', 'desc')

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
        assert (k not in self._immutable), "'%s' not writable" % k

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

    def asdict(self, toNumpy: bool = True) -> dict:
        tmp = ('rf', 'gr', 'dt')
        fn_np = (lambda x: x.detach().cpu().numpy() if toNumpy else
                 lambda x: x.detach())

        d = {k: fn_np(getattr(self, k)) for k in tmp}
        d.update({k: getattr(self, k) for k in ('desc', 'device', 'dtype')})

        return d

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
        SpinArray(shape; T1, T2, γ, M, device, dtype)
    *INPUTS*:
    - `shape` tuple( (N, nx, (ny, (nz,...))) ).
    *OPTIONALS*:
    - `T1` (self.shape) "Sec", T1 relaxation coeff.
    - `T2` (self.shape) "Sec", T2 relaxation coeff.
    - `γ`  (self.shape) "Hz/Gauss", gyro ratio.
    - `M`  (self.shape)+(xyz,), spins, assumed equilibrium [0 0 1]
    - `device` torch.device; `dtype` torch.dtype
    """

    _immutable = ('shape', 'device', 'dtype', 'ndim')
    __slots__ = _immutable + ('T1', 'T2', 'γ', 'M')

    def __init__(
            self, shape: tuple,
            M: Tensor = None,
            T1: Type[Union[Tensor, Number]] = T1G,
            T2: Type[Union[Tensor, Number]] = T2G,
            γ: Type[Union[Tensor, Number]] = γH,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32):

        assert(isinstance(device, torch.device) and
               isinstance(dtype, torch.dtype))

        super().__setattr__('shape', shape)
        super().__setattr__('ndim', len(shape))
        super().__setattr__('device', device)
        super().__setattr__('dtype', dtype)

        dkw = {'device': self.device, 'dtype': self.dtype}

        # Defaults
        self.T1, self.T2, self.γ = T1, T2, γ
        self.M = tensor([[[0., 0., 1.]]], **dkw) if (M is None) else M

        return

    def __getattr__(self, k):
        raise AttributeError("'SpinArray' object has no attribute '%s'" % k)
        return

    def __setattr__(self, k, v):
        assert (k not in self._immutable), "'%s' not writable" % k

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

        super().__setattr__(k, v)

        return

    def applypulse(self, p: Pulse, loc: Tensor, **kw) -> Tensor:
        """
        *INPUTS*:
        - `p`   (1,) mobjs.Pulse object
        - `loc` (N,*Nd,xyz) "cm", locations.
        *OPTIONALS*:
        - `Δf`    (N,*Nd,) "Hz", off-resonance.
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        """
        dkw = {'device': self.device, 'dtype': self.dtype}
        p, loc = p.to(**dkw), loc.to(**dkw)
        beff = self.pulse2beff(p, loc, **kw)

        ks = ('T1', 'T2', 'γ')
        kw_bsim = {k: getattr(self, k) for k in ks}
        M = self.M

        kw_bsim['dt'] = p.dt

        M = sims.blochsim(M, beff, **kw_bsim)

        return M

    def asdict(self, toNumpy: bool = True) -> dict:
        tmp = ('T1', 'T2', 'γ', 'M')
        fn_np = (lambda x: x.detach().cpu().numpy() if toNumpy else
                 lambda x: x.detach())

        d = {k: fn_np(getattr(self, k)) for k in tmp}
        d.update({k: getattr(self, k) for k in ('shape', 'device', 'dtype')})
        return d

    def dim(self) -> int: return len(self.shape)

    def pulse2beff(
            self, p: Pulse, loc: Tensor,
            Δf: Tensor = None, b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `loc`   (N,*Nd,xyz) "cm", locations.
        *OPTIONALS*:
        - `Δf`    (N,*Nd,) "Hz", off-resonance.
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        *OUTPUTS*:
        - `beff`  (N,*Nd,xyz,nT)
        """
        dkw = {'device': self.device, 'dtype': self.dtype}
        p, loc = p.to(**dkw), loc.to(**dkw)
        γ = self.γ

        return p.beff(loc, γ=γ, Δf=Δf, b1Map=b1Map)

    def size(self) -> tuple: return self.shape

    def to(self, device: torch.device = torch.device('cpu'),
           dtype: torch.dtype = torch.float32) -> SpinArray:
        if (self.device != device) or (self.dtype != dtype):
            return SpinArray(self.shape, T1=self.T1, T2=self.T2, γ=self.γ,
                             M=self.M, device=device, dtype=dtype)
        else:
            return self
        return


class SpinCube(object):
    """
        SpinCube(shape, fov; ofst, Δf, T1, T2, γ, M, device, dtype)
    *INPUTS*:
    - `shape` Tuple `(N, nx, (ny, (nz,...)))`.
    - `fov` (N, xyz,) Tensor "cm", field of view.
    *OPTIONALS*:
    - `ofst` (N, xyz,) Tensor "cm", fov offset from iso-center.
    - `Δf` (self.shape) Tensor "Hz", off-resonance map.
    - `T1` (self.shape) Tensor "Sec", T1 relaxation coeff.
    - `T2` (self.shape) Tensor "Sec", T2 relaxation coeff.
    - `γ`  (self.shape) Tensor "Hz/Gauss", gyro ratio.
    - `M`  (self.shape)+(xyz,) Tensor, spins, assumed equilibrium [0 0 1]
    - `device` torch.device; `dtype` torch.dtype
    """

    # SpinArray.__slots__ = ('shape', 'device', 'dtype', 'T1', 'T2', 'γ', 'M')
    _immutable = ('spinarray',)
    __slots__ = _immutable+('Δf', 'fov', 'ofst')

    def __init__(
            self, shape: tuple, fov: Tensor,
            ofst: Tensor = None, Δf: Tensor = None, M: Tensor = None,
            T1: Type[Union[Tensor, Number]] = T1G,
            T2: Type[Union[Tensor, Number]] = T2G,
            γ: Type[Union[Tensor, Number]] = γH,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32):
        """
        """
        super().__setattr__('spinarray',
                            SpinArray(shape, T1=T1, T2=T2, γ=γ, M=M,
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
        assert (k not in self._immutable), "'%s' not writable" % k

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

    def applypulse(self, p: Pulse, b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `p`   (1,) mobjs.Pulse object
        *OPTIONALS*
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        """
        return self.spinarray.applypulse(p, self.loc, Δf=self.Δf, b1Map=b1Map)

    def asdict(self, toNumpy: bool = True) -> dict:
        tmp = ('Δf', 'fov', 'ofst')
        fn_np = (lambda x: x.detach().cpu().numpy() if toNumpy else
                 lambda x: x.detach())

        d = {k: fn_np(getattr(self, k)) for k in tmp}
        d.update(self.spinarray.asdict(toNumpy=toNumpy))
        return d

    def dim(self) -> int: return self.spinarray.ndim

    def pulse2beff(self, p: Pulse, b1Map: Tensor = None) -> Tensor:
        """
        *INPUTS*:
        - `p` (1,) Pulse object
        *OPTIONALS*:
        - `b1Map` (N,*Nd,xy,(nCoils)) a.u., , transmit sensitivity.
        *OUTPUTS*:
        - `beff`  (N,*Nd,xyz,nT)
        """
        return self.spinarray.pulse2beff(p, self.loc, Δf=self.Δf, b1Map=b1Map)

    def size(self) -> tuple: return self.spinarray.shape

    def to(self, device: torch.device = torch.device('cpu'),
           dtype: torch.dtype = torch.float32) -> SpinCube:
        if (self.device != device) or (self.dtype != dtype):
            return SpinCube(self.shape, self.fov, ofst=self.ofst, Δf=self.Δf,
                            T1=self.T1, T2=self.T2, γ=self.γ, M=self.M,
                            device=device, dtype=dtype)
        else:
            return self
        return


class SpinBolus(SpinArray):
    def __init__(
            self):
        pass
    pass


class Examples(object):
    """
    Just a class quickly creating exemplary instances to play around with.
    """
    @staticmethod
    def pulse() -> Pulse:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')
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
        print('Pulse(rf=rf, gr=gr, dt=gt, device=device, dtype=dtype)')
        return Pulse(rf=rf, gr=gr, dt=dt, **kw)

    @staticmethod
    def spincube() -> SpinCube:
        device = torch.device('cuda' if cuda.is_available() else 'cpu')
        dtype = torch.float32
        kw = {'dtype': dtype, 'device': device}

        N, Nd, γ = 1, (3, 3, 3), γH
        shape = (N, *Nd)
        fov, ofst = tensor([[3., 3., 3.]], **kw), tensor([[0., 0., 1.]], **kw)
        T1, T2 = tensor([[1.]], **kw), tensor([[4e-2]], **kw)

        cube = SpinCube(shape, fov, ofst=ofst, T1=T1, T2=T2, γ=γ, **kw)

        cube.Δf = torch.sum(-cube.loc[0:1, :, :, :, 0:2], dim=-1) * γ
        return cube