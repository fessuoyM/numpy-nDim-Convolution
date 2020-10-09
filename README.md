# numpy-nDim-Convolution
This is a class that can be imported to perform nDim convolution.

The main function if the conv function and takes at least 2 arguments: the 2 signals being convolved. 
pad is an optional paramater that specifies the padding to be applied to the signal during convolution. if pad is a scalar, the pad is applied to all dimensions of the convolution. Otherwise, pad is a one dimensional array with a size equal to the number of dimensions of the signals to be convolved, with each value representing the padding on each axis. padding is applied to both sides of an axis. 
Example:
X =  [[1,2,3],    Y= [[ 1, 1],
      [4,5,6]         [-1,-1]]
      [7,8,9]]
numpyConv.conv(X,Y,2) would lead to X (The Larger Array) being padded as follows
X =  [[0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0],
      [0,0,1,2,3,0,0],   
      [0,0,4,5,6,0,0]         
      [0,0,7,8,9,0,0]
      [0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0]]
      
numpyConv.conv(X,Y,(2,1)) would lead to X (The Larger Array) being padded as follows
X =  [[0,0,0,0,0],
      [0,0,0,0,0],
      [0,1,2,3,0],   
      [0,4,5,6,0]         
      [0,7,8,9,0]
      [0,0,0,0,0],
      [0,0,0,0,0]]
      
If an argument is not passed in for pad, pad is generated as (the smaller array) numpy.subtract(Y.shape,1). In otherwords, it computes the full convultion including edge effects. A value of 0 for pad results in the convlution of only fully overlapping kernels. 

The second optional argument is padVal, and allows for control over what value to padd the image with. This value defaults to 0 when not passed in. 

Finally, if one wishes to perform the convolution of multiple images with one kernal, or one image with multiple kernals, the conv function will allow that. There are 3 acceptable cases here:
1) x.ndim==y.ndim (The condition disscussed up to this point)
2) x.ndim>y.ndim (This will invoke the convManyH method)
3) x.ndim<y.ndim (This will invoke the convManyY method)

convManyY and convManyH both thake the same arguments as conv, but are specical cases when one would like to perfom multiple convolutions at once. The extra dimensions must be added to the front. In other words, X and Y must both have the same ending dimensions. 
Example:
X shape = (a,m) and Y shape = (m) is valid
X shape = (m,a) and Y shape = (m) is NOT valid
X shape = (a,m,n) and Y shape = (m,n) is valid
X shape = (a,m,...,n) and Y shape = (m,...,n) is valid
X shape = (m,n,a) and Y shape = (m,n) is NOT valid
X shape = (a,b,c,m,n) and Y shape = (m,n) is valid
X shape = (a1,...,z1,m,n) and Y shape = (m,n) is valid
X shape = (m,n) and Y shape = (a,m,n) is valid
X shape = (m,n) and Y shape = (a1,...,z1,m,n) is valid
X shape = (m,n) and Y shape = (m,n,a) is NOT valid
X shape = (m,n) and Y shape = (m,n,a,b) is NOT valid
X shape = (m,...,n) and Y shape = (a1,...,z1,m,...,n) is valid

The answer will be return in the same shape where the last dimensions are the image or signal dimensions.

The normConv Function will take Y (the smaller function) and normallize the values such that the sum of that kernal is 1 if the signal only contains positive values  Otherwise if it contains any negative values, the values are simply normallized to a range (-1,1). The method then calls the conv method

Finally, a toInt function exist that converts float arrays back to an integer type of choosing. This is useful for when trying to revert an array to an image using Pillow. This function is not fully bugproof or heavily tested.  

Thank you for stopping by and feel free to leave any comments or notes on ways to make this more professional/better documented as I am trying to learn how to properly set up a github repo for a project like this!
