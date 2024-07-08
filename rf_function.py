'''Useful functions to compute the receptive field of a convolutional neural network'''

def conv(n, dilation=1):
    '''Compute the receptive field of the output of a convolutional layer of kernel size 3x3, dilation=dilation stride=1 and padding=dilation'''
    return 2 * dilation + n

def maxpool(n):
    '''Compute the receptive field of the output of a maxpooling layer of kernel size 2x2 and stride=2'''
    return 2 * n

def upsample(n):
    '''Compute the receptive field of the output of a upsampling layer of kernel size 2x2 and stride=2 then two convolutions of kernel size 3x3, stride=1 and padding=1'''
    return 3 + (n//2)

def downsample(n, dilation=1):
    '''Compute the receptive field of the output of a downsampling layer of kernel size 3x3, dilation=dilation and stride=2 then two convolutions of kernel size 3x3, dilation=dilation and stride=1'''
    return conv(conv(maxpool(n), dilation=dilation), dilation=dilation)

