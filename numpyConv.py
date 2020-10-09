import numpy as np

class NumpyConv:
    @classmethod
    def conv(cls, h, y, pad=None, padVal=0):
        #N-Dimensional convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the convolution of higher dimensions

        #Call Appriopriate function
        if h.shape[-1]>y.shape[-1]: temp=h; h=y; y=temp
        if h.ndim>=y.ndim:
            return cls.convManyH(h,y,pad,padVal)
        elif y.ndim>h.ndim:
            return cls.convManyY(h,y,pad,padVal)

    @classmethod
    def convManyY(cls, h, y, pad=None, padVal=0):
        #N-Dimensional convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the convolution of higher dimensions

        #Ensure y is h.ndim+1 dimension
        if h.ndim>y.ndim: temp=h; h=y; y=temp
        if h.ndim==y.ndim:
            y=np.reshape(np.ravel(y),(np.append(1,y.shape)))
            yshap = (1)
        elif y.ndim>h.ndim:
            y=np.reshape(np.ravel(y), np.append(np.prod(y.shape[:-h.ndim]),y.shape[-h.ndim:]))
            yshap = y.shape[:-h.ndim]

        #Commented out because y cannot be switched after adding/reshaping the
            #Ensure h is the smaller dimensional array
            # if h.shape[0]>y.shape[1]: temp = h; h=y; y=temp;
            # print(h.shape, y.shape)
            # for i in range(h.ndim):
            #     if h.shape[i]>y.shape[i+1]:
            #         raise Warning("If one array is not larger than the other in ALL dimensions, program may need to generate extra paddings and slow things down. It's suggested to use a filter that is smaller in all dimensions")

        ##Apply specified padding, or add default padding of h.shape-1
        if pad is None:
            pad = np.subtract(h.shape,1)
        elif np.isscalar(pad):
            pad = pad*np.ones(h.ndim, np.int)

        #Flip filter (h) and pad y with specified padding settings
        h=np.flip(h)
        y=np.pad(y,np.append([[0,0]],(np.ones((2,pad.size),np.int)*pad).T,axis=0),'constant', constant_values=padVal)

        #Generate a window view of y that involves each voxel for convolution
        a=np.lib.stride_tricks.as_strided(y, shape=(np.append(np.append(y.shape[0],np.subtract(y[0,...].shape, h.shape)+1), h.shape)),strides=(np.append(y.strides[0],y[0,...].strides*2)))

        #Flatten array y along each voxel regardless of dimension so a dot product with a flattened filter results in the summed elementwise multiplication
        a=np.reshape(np.ravel(a), (a.shape[0]*np.prod(np.subtract(y[0,...].shape, h.shape)+1),np.prod(h.shape)))
        a=np.tensordot(a,np.ravel(h),([1],[0]))

        #Reshape the answer and return the value; if yshap is one, remove the outter shell added for the algorithm generalizability
        if yshap==1:
            return np.reshape(np.ravel(a),np.append(yshap,np.subtract(y.shape[1:], h.shape)+1))[0,...]
        else:
            return np.reshape(np.ravel(a),np.append(yshap,np.subtract(y.shape[1:], h.shape)+1))

    @classmethod
    def convManyH(cls, h, y, pad=None, padVal=0):
        #N-Dimensional convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the convolution of higher dimensions

        #Ensure h is y.ndim+1 dimension
        if y.ndim>h.ndim: temp=h; h=y; y=temp
        if h.ndim==y.ndim:
            h=np.reshape(np.ravel(h),(np.append(1,h.shape)))
            hshap = (1)
        elif h.ndim>y.ndim:
            h=np.reshape(np.ravel(h), np.append(np.prod(h.shape[:-y.ndim]),h.shape[-y.ndim:]))
            hshap = h.shape[:-y.ndim]

        ##Apply specified padding, or add default padding of h.shape-1
        if pad is None:
            pad = np.subtract(h[0,...].shape,1)
        elif np.isscalar(pad):
            pad = pad*np.ones(y.ndim, np.int)

        #Flip filter (h) and pad y with specified padding settings
        h=np.flip(np.flip(h),0)
        y=np.pad(y,(np.ones((2,pad.size),np.int)*pad).T,'constant', constant_values=padVal)

        #Generate a window view of y that involves each voxel for convolution
        a=np.lib.stride_tricks.as_strided(y, shape=(np.append(np.subtract(y.shape, h[0,...].shape)+1, h[0,...].shape)),strides=(y.strides*2))

        #Flatten array y along each voxel regardless of dimension so a dot product with a flattened filter results in the summed elementwise multiplication
        a=np.reshape(np.ravel(a), (np.prod(np.subtract(y.shape, h[0,...].shape)+1),np.prod(h[0,...].shape)))
        a=np.tensordot(a,np.reshape(np.ravel(h),(h.shape[0],np.prod(h[0,...].shape))),([1],[1])).T
        #Reshape the answer and return the value; if yshap is one, remove the outter shell added for the algorithm generalizability
        if hshap==1:
            return np.reshape(np.ravel(a),np.append(hshap,np.subtract(y.shape, h.shape[1:])+1))[0,...]
        else:
            return np.reshape(np.ravel(a),np.append(hshap,np.subtract(y.shape, h.shape[1:])+1))

    @classmethod
    def normConv(cls, h, y, pad=None, pady=None, padVal=0):
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
        return cls.conv(h, y, pad, pady, padVal)

    @classmethod
    def toInt(cls, a, type):
        infob = np.iinfo(type)
        return np.interp(a,(a.min(), a.max()), (infob.min, infob.max)).astype(type)
