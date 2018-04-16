import random
import math
import numpy as np



def tabulate(dimension):
# Creates a 2D array that tabulate every possible input
# Raises ValueError if input size is too large
    if not isinstance(dimension,int):
        print('Invalid Input')
        raise ValueError
    elif dimension>=10:
        print('Input dimension cannot exceed 10')
        raise ValueError
    else:
        table = np.zeros((dimension,2**dimension))
        for i in range(dimension):
            for j in range(2**dimension):
                table[i,j] = (j//2**(dimension-i-1))%2==1
        return table

def random_samples(dimension, n):
# Creates a 2D array that contains n random samples,
# each sample has the specified dimension. This function is
# recommended for data generation when n << 2**dimension
    table = np.zeros((dimension,n))
    random.seed()
    if not isinstance(dimension,int) or \
       not isinstance(n, int):
        print('Invalid Input')
        raise ValueError
    elif dimension<=32: # can use random.sample for dimension<=32   
        integer_samples = random.sample(range(2**dimension),n)
        #print(samples)
        for i in range(n):
            # converts samples[i] into binary bits, 
            # Big-Endian style
            bits = format(integer_samples[i],'0{}b'.format(dimension))
            # assign the bits to the sample
            table[:,i] = [bit=='1' for bit in bits]
        return table
    else:
        # For very large dimensions, cannot use random.sample to
        # guarantee samples have no duplicates
        # 
        print('Generating Samples:\nDimension: {0}, Sample Size: {1}'.format(dimension,n))
        for i in range(n):
            bits = format(random.getrandbits(dimension),'0{}b'.format(dimension))
            table[:,i] = [bit=='1' for bit in bits]
        return table
        
def get_label(samples):
    # returns the XOR along the 0th axis, e.g. the parity of each row
    return samples.sum(0)%2


def test(): # do not use outside this module

    t = random_samples(64,15)
    print(t)
    print(get_label(t).shape)
    
#test()
