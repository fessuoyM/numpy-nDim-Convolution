import numpy as np

class NumpyConv:
    @classmethod
    def conv(cls, h, y, pad=None, padVal=0, nDim=None):
        #N-Dimensional convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the convolution of higher dimensions

        #Ensure smaller array is h, usually the kernal/filter
        if h.shape[-1]>y.shape[-1]: h, y = y, h

        #nDim is calculated if its not passed in.
        if nDim is None:
            nDim=min(y.ndim,h.ndim)

        #Generate and apply default padding of h.shape-1 if a pad variable is not passed in
        if pad is None:
            pad = np.subtract(h.shape[-nDim:],1)
        elif np.isscalar(pad):
            pad = pad*np.ones(nDim, np.int)
        y=np.pad(y,np.append(np.zeros((y.ndim-nDim,2),np.int),(np.ones((2,pad.size),np.int)*pad).T,axis=0),'constant', constant_values=padVal)

        #Flip filter (h)
        h=np.flip(h,tuple(range(h.ndim-nDim,h.ndim)))

        #Generate a window view of y that involves each voxel for convolution
        a=np.lib.stride_tricks.as_strided(y, shape=(np.append(np.append(y.shape[0:-nDim],np.subtract(y.shape[-nDim:], h.shape[-nDim:])+1), h.shape[-nDim:])).astype(np.int),strides=(np.append(y.strides[:-nDim],y.strides[-nDim:]*2)).astype(np.int))

        #generate einsum notation for matrix mult of nDim dimensions
        str1=str2=''
        str3='->'
        o=0
        for i in range(h.ndim-nDim):
            str2+=chr(97+o)
            str3+=chr(97+o)
            o+=1
        for i in range(a.ndim-nDim):
            str1+=chr(97+o)
            str3+=chr(97+o)
            o+=1
        for i in range(nDim):
            str1+=chr(97+i+o)
            str2+=chr(97+i+o)
        str=str1+','+str2+str3

        #Calculate and return convolution
        return np.einsum(str,a,h)

    @classmethod
    def normConv(cls, h, y, pad=None, padVal=0, nDim=None):
            #Normallize the kernal to 0-1 range and call ConvTrunc method
        h=h.astype(np.double)
        if h.min()>=0:
            h = np.interp(h, (h.min(), h.max()), (0, 1))
            h = h/h.sum()
        else:
            p=h>0
            h[p] = np.interp(h[p], (h[p].min(), h[p].max()), (0, 1))
            h[p] = h[p]/h[p].sum()
            p = p==False
            h[p] = np.interp(h[p], (h[p].min(), h[p].max()), (-1, 0))
            h[p] = h[p]/np.abs(h[p].sum())
        return cls.conv(h, y, pad, padVal, nDim)

    @classmethod
    def toInt(cls, a, type):
        infob = np.iinfo(type)
        return np.interp(a,(a.min(), a.max()), (infob.min, infob.max)).astype(type)