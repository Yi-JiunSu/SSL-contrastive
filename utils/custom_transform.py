"""
    Custom Data Transformation
    Yi-Jiun Su 2023/10/19

    x is a tensor [B, C, H, W]
      except 1D numpy array in WindowWraping1D 

    conver to work on different device 'cpu' or 'cuda'
    Y.-J. Su 2023/10/24
"""
import torch
import numpy as np

class FFT(object):
    """
       real part multiple by random number between 0 and 1 of the same array size in REAL(magnitude) part
       The image part (phase) remain unchanged
    """
    def __call__(self, x):
        K = torch.fft.fft2(x).to(device=x.device)
        K.real = K.real*torch.rand(K.shape).to(device=x.device)
        return torch.abs(torch.fft.ifft2(K).to(device=x.device))

class Jitter(object):
    def __call__(self, x):
        std = torch.std(x, dim=3) # size [B, C, H]
        r = 2*torch.rand([x.shape[0],x.shape[1],1,x.shape[3]]).to(device=x.device)-1 # size [B, C, 1, W]
        e = torch.mul(r,std[:,:,:,None]) # size [B, C, H, W]
        return torch.clamp(x + e, min=0., max=1.) 

class Scale(object):
    def __call__(self, x): 
        std = torch.std(x, dim=(1,2,3)) # size [B]
        rn = torch.normal(1, std)
        # torch.norma(mean,std)
        x = x*rn[:,None,None,None]
        return torch.clamp(x, min=0., max=1.)   

def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)
    return m[indicies] * x + b[indicies]

class WindowWrap1D(object):
    def __init__(self, rmn=0.02, rmx=0.1):
        self.rmn = rmn
        self.rmx = rmx

    def __call__(self, x):
        s = len(x)
        d = torch.round(0.5*s*torch.FloatTensor(1).uniform_(self.rmn,self.rmx)).int()
        d2 = d*2
# shink by 1/2 from a randomly selected window
        ind = torch.floor((s-d2)*torch.rand(1)).int()
        a = torch.arange(d2.item())+ind
        af = torch.arange(d.item())+ind
        tmp = torch.zeros(s-d).to(device=x.device)
        tmp[0:a[0]] = x[0:a[0]]
        tmp[af] = x[a[0]:a[d2.item()-1]+1:2]
        tmp[af[d.item()-1]+1:] = x[a[d2.item()-1]+1:]
# expand by 2 from a randomly selected window
        ind = torch.floor((s-d2)*torch.rand(1)).int()
        b = torch.arange(d.item())+ind
        bd = (torch.arange(d2.item())+ind).to(device=x.device)
        new = torch.zeros(s).to(device=x.device)
        new[0:b[0]] = tmp[0:b[0]]
        new[bd] = interp(bd, bd[::2], tmp[b])
        new[bd[d2.item()-1]+1:] = tmp[b[d.item()-1]+1:]
        return new

class WrapHW(object): 
    """
       dim = 3 wraping in Width (Time)
       dim = 2 wraping in Height (Frequency)
    """
    def __init__(self, dim=3, rmn=0.02, rmx=0.1):
        self.dim = dim
        self.rmn = rmn
        self.rmx = rmx

    def __call__(self, x):
        ww = WindowWrap1D(rmn=self.rmn, rmx=self.rmx)
        s = x.shape[self.dim]*2

        if self.dim != 3:
            x = x.permute(0,1,3,2)
        x2 = torch.nn.functional.interpolate(x, size=[x.shape[2],s])

        for B in range(x2.shape[0]):
            for C in range(x2.shape[1]):
                for i in range(x2.shape[2]):
                    x2[B,C,i,:] = ww(x2[B,C,i,:])
#                    x2[B,C,i,:] = torch.from_numpy(ww(x2[B,C,i,:].numpy()))

        x = torch.nn.functional.interpolate(x2, size=[x.shape[2],x.shape[3]])
        if self.dim !=3:
            x = x.permute(0,1,3,2)

        return x 
