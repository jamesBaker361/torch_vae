import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from models import *

def EncoderTest():
    pass

def UnsharedEncoderTest(
        out_channels=4,
        input_shape=(1,3, 32,32),
        layers=2
        ):
    input_tensor = torch.randn(input_shape)
    unshared=UnsharedEncoder(out_channels,layers=layers)
    output=unshared(input_tensor)
    print(input_shape[-1]//(2**layers),output.shape[-1])
    assert input_shape[-1]//(2**layers)==output.shape[-1]



if __name__ =='__main__':
    UnsharedEncoderTest()