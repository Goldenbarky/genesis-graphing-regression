from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from pytz import timezone
import re
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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
    eq = ""
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
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

data_response = (
    supabase.table("data").select("*").execute()
)

users_response = (
    supabase.table("users").select("*").execute()
)

point_dict = {}

for data in users_response.data:
    point_dict[data['id']] = {'x':[], 'y':[]}

for data in data_response.data:
    dt = datetime.fromisoformat(data["created_at"])
    t = dt.astimezone(timezone('US/Eastern'))

    point_dict[data['owner_id']]['x'].append(timestampToHourFraction(t))
    point_dict[data['owner_id']]['y'].append(data["value"])

for users_data in point_dict.keys():
    if(len(point_dict[users_data]['x']) < 1):
        continue

    df = DataFrame({'x': point_dict[users_data]['x'], 'y': point_dict[users_data]['y']}, columns=['x', 'y'])

    x = df[['x']]
    y = df[['y']]

    best = -1

    sols = {
        'Degree':[],
        'BIC':[],
        'YP':[],
        'COEFS':[],
        'LINE':[]
    }

    best_fit_line = []

    for degree in range(1,18):
        pf = PolynomialFeatures(degree=degree)
        xp = pf.fit_transform(x)

        model = sm.OLS(y, xp).fit()
        ypred = model.predict(xp)

        sols['Degree'].append(degree)
        sols['BIC'].append(model.bic ** 2)
        sols['YP'].append(ypred)
        sols['COEFS'].append(model.params)

        if(best == -1 or model.bic ** 2 < sols['BIC'][best]):
            best = len(sols['Degree']) - 1

    polyx = np.linspace(8, 17, 100)
    polyy = PolyCoefficients(polyx, sols['COEFS'][best])

    for i in range(0, 100):
        best_fit_line.append((round(polyx[i], 2), round(polyy[i], 2)))

    coeffs = []
    for i in range(0, len(sols['COEFS'][best])):
        value = 0
        if i == 0:
            value = sols['COEFS'][best]['const']
        else:
            value = sols['COEFS'][best][f'x{i}']
        coeffs.append(value)

    equation = coeffsToEquation(coeffs)

    eq_data = {
        "equation": equation,
        "points": best_fit_line
    }

    response = supabase.table("eod_equations").upsert([{"owner_id":users_data, "eq_data":eq_data}]).execute()