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

Finally, if one wishes to perform the convolution of multiple images with one kernal, or one image with multiple kernals, the conv function will allow that. The function will always perform the largest nD convolution possible, meaning that the dimension of the convolution will match the smaller smaller dimension between the two inputs. for example:

X.shape=(50,40,30,20,10) and Y = (5, 5) will return a (50,40,30) dimension array in which case each 2d array was convolved with Y. This is useful because it allows the user to run a series of images through the same filter. This is preferable to looping through each image in python and will provide performance enchantment.

Furthermore, one may run a series of images through a series of filters as well, but here the user must explicitly pass in the dimensions of the convolution to be performed. Note that the convolution is always performed on the last dimensions of the array.

X.shape=(50,40,30,20,10) and Y = (10,5, 5) will return a (50,40) dimension array but if conv(x,y,nDim=2), the result will be a (10,50,40). Note that the first dimensions will come from the filters (the array with the smallest shape along the last axis) followed by the dimensions of the image.


The answer will be return in the same shape where the last dimensions are the image or signal dimensions.

The normConv Function will take Y (the smaller function) and normallize the values such that the sum of that kernal is 1 if the signal only contains positive values  Otherwise if it contains any negative values, the values are simply normallized to a range (-1,1). The method then calls the conv method

Finally, a toInt function exist that converts float arrays back to an integer type of choosing. This is useful for when trying to revert an array to an image using Pillow. This function is not fully bugproof or heavily tested.  

Thank you for stopping by and feel free to leave any comments or notes on ways to make this more professional/better documented as I am trying to learn how to properly set up a github repo for a project like this!
