import re

import numpy as np
from sklearn.metrics import mean_squared_error

def toSuperscript(number):
    out = ""
    for i in range(0, len(number)):
        curr = number[i]
        if i == 0 and curr == "0":
            continue

        if curr == "1":
            out += chr(int("00B9", 16))
        elif curr == "2" or curr == "3":
            out += chr(int("00B" + curr, 16))
        else:
            out += chr(int("207" + curr, 16))

    return out

def coeffsToEquation(coeffs):
    regex = "(-?)([0-9]*).([0-9]{2})e([-+])([0-9]+)"
    eq = "y = "
    for i in range(0, len(coeffs)):
        currIndex = len(coeffs) - i - 1
        sci = '{:.2e}'.format(float(coeffs[currIndex]))
        match = re.match(regex, sci)

        if i != 0:
            if match[1] != "-":
                eq += " + "
            else:
                eq += " - "

        if int(match[5]) <= 2:
            raw_val = float(match[2] + "." + match[3]) * 10 ** (int(match[5]) * (int(match[4] + "1")))
            trunc_val = '{:.3f}'.format(raw_val)

            while trunc_val[-1] == "0":
                trunc_val = trunc_val[:-1]
            if trunc_val[-1] == ".":
                trunc_val = trunc_val[:-1]

            eq += trunc_val
        else:
            eq += match[2] + "." + match[3] + match[4] + "e" + toSuperscript(match[5])

        if currIndex != 0:
            eq += "x" + toSuperscript(str(currIndex))

    return eq

def timestampToHourFraction(time):
    time_val = time.hour
    time_val = time_val + (time.minute / 60)
    time_val = time_val + (time.second / 6000)

    time_val = round(time_val, 2)

    return time_val

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    yVals = []
    
    for xVal in x:
        y = 0
        for i in range(0, len(coeffs)):
            y += coeffs[i] * (xVal**i)

        yVals.append(y)

    return yVals

def BIC(y, ypred, xpoly):
    # Calculate BIC
    n = len(y)  # Number of observations
    mse = mean_squared_error(y, ypred)
    k = xpoly.shape[1]  # Number of parameters (including intercept)
    
    if(mse == 0 or n == 0):
        return 100000

    bic = (n * np.log(mse) + k * np.log(n)) ** 2
    return bic