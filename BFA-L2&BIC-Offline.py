def warn(*args, **kwargs):
    pass
from itertools import chain
import warnings
warnings.warn = warn

from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from pytz import timezone

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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
            y = coeffs[i] * xVal**i
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

data_response = (
    supabase.table("data").select("*").execute()
)

users_response = (
    supabase.table("users").select("*").execute()
)

point_dict = {}

for data in users_response.data:
    point_dict[data['id']] = {'name':data['display_name'], 'x':[], 'y':[]}

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

    if (max(point_dict[users_data]['y']) != min(point_dict[users_data]['y'])):
        best = -1

        sols = {
            'Degree':[],
            'BIC':[],
            'ALPHA':[],
            'YP':[],
            'COEFS':[]
        }

        best_fit_line = []

        for degree in range(1,18):
            best_degree_x = {
                'BIC':1000000,
                'ALPHA':0,
                'YP':[],
                'COEFS':[]
            }

            pf = PolynomialFeatures(degree=degree)
            xp = pf.fit_transform(x)

            for a in np.logspace(-4, 4, 100):
                model = Ridge(alpha=a)
                model.fit(xp, y)

                ypred = model.predict(xp)

                bic = BIC(y, ypred, xp)

                if(bic < best_degree_x['BIC']):
                    best_degree_x['BIC'] = bic
                    best_degree_x['ALPHA'] = a
                    best_degree_x['YP'] = ypred
                    best_degree_x['COEFS'] = list(chain.from_iterable(model.coef_))
                    best_degree_x['COEFS'][0] = model.intercept_[0]

                if(bic < 0.001):
                    break
            
            sols['Degree'].append(degree)
            sols['BIC'].append(best_degree_x['BIC'])
            sols['ALPHA'].append(best_degree_x['ALPHA'])
            sols['YP'].append(best_degree_x['YP'])
            sols['COEFS'].append(best_degree_x['COEFS'])

            if(best == -1 or best_degree_x['BIC'] < sols['BIC'][best]):
                best = len(sols['Degree']) - 1

        # polyx = np.linspace(8, 17, 100)
        # print(polyx)
        # polyy = PolyCoefficients(polyx, sols['COEFS'][best][0])

        # for i in range(0, 100):
        #     best_fit_line.append((round(polyx[i], 2), round(polyy[i], 2)))

        figure, axis = plt.subplots(4, 5)
        figure.suptitle(f'{point_dict[users_data]['name']}\'s Data')

        # axis[3, 3].plot(polyx, polyy)
        axis[3, 3].set_xlim([8, 17])
        axis[3, 3].set_ylim([0, 10])

        for i in range(4):
            for j in range(5):
                if ((i*5) + j < len(sols['Degree'])):
                    axis[i, j].scatter(x, y)
                    polyx = np.linspace(8,17,100)

                    # Step 5: Fit the best model
                    poly = PolynomialFeatures(sols['Degree'][(i*5) + j])
                    X_poly = poly.fit_transform(x)
                    model = Ridge(alpha=sols['ALPHA'][(i*5) + j])
                    model.fit(X_poly, y)

                    # Step 6: Generate values for the fitted curve
                    X_fit = np.linspace(8, 17, 100).reshape(-1, 1)
                    X_fit_poly = poly.transform(X_fit)
                    Y_fit = model.predict(X_fit_poly)
                    axis[i, j].plot(X_fit, Y_fit)
                    axis[i, j].set_title('Degree {}'.format(sols['Degree'][(i*5) + j]))
                    axis[i, j].set_xlim([8, 17])
                    axis[i, j].set_ylim([0, 10])

        axis[3,4].plot('Degree', 'BIC', data=sols)

        table = DataFrame(sols, columns=['Degree', 'BIC', 'COEFS'], index=None)
        print(table)
        print(sols['COEFS'][best])
        print(sols['ALPHA'][best])
        print(f'Best Degree: {sols['Degree'][best]}')

        plt.show()
    else:
        plt.scatter(x, y)
        polyX = np.linspace(8, 17, 20)
        polyY = np.linspace(y.min(), y.min(), 20)
        plt.plot(polyX, polyY)
        plt.show()