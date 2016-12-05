#activate function

def ReLU(x):
    if (x > 0):
        return x
    else:
        return 0

def dReLU(x):
    if (x > 0):
        return 1.0
    else:
        return 0