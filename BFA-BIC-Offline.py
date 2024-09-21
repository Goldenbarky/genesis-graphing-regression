from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from pytz import timezone

import os
from dotenv import load_dotenv
from supabase import create_client, Client

from Helpers import PolyCoefficients, timestampToHourFraction

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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

    best = -1

    sols = {
        'Degree':[],
        'BIC':[],
        'YP':[],
        'COEFS':[]
    }

    best_fit_line = []

    for degree in range(1,18):
        pf = PolynomialFeatures(degree=degree)
        xp = pf.fit_transform(x)

        model = sm.OLS(y, xp).fit()
        ypred = model.predict(xp)

        # plt.scatter(x,y)

        # plt.plot(x, ypred)
        # plt.title('Degree: {}'.format(degree))

        # plt.show()

        sols['Degree'].append(degree)
        sols['BIC'].append(model.bic ** 2)
        sols['YP'].append(ypred)
        sols['COEFS'].append(model.params)

        if(best == -1 or (model.bic ** 2) < sols['BIC'][best]):
            best = len(sols['Degree']) - 1

    polyx = np.linspace(8, 17, 100)
    polyy = PolyCoefficients(polyx, sols['COEFS'][best])

    for i in range(0, 100):
        best_fit_line.append((round(polyx[i], 2), round(polyy[i], 2)))

    figure, axis = plt.subplots(4, 5)
    figure.suptitle(f'{point_dict[users_data]['name']}\'s Data')

    axis[3, 3].plot(polyx, polyy)
    axis[3, 3].set_xlim([8, 17])
    axis[3, 3].set_ylim([0, 10])

    for i in range(4):
        for j in range(5):
            if ((i*5) + j < len(sols['Degree'])):
                axis[i, j].scatter(x, y)
                polyx = np.linspace(8,17,100)
                axis[i, j].plot(polyx, PolyCoefficients(polyx, sols['COEFS'][(i*4) + j]))
                axis[i, j].set_title('Degree {}'.format(sols['Degree'][(i*5) + j]))
                axis[i, j].set_xlim([8, 17])
                axis[i, j].set_ylim([0, 10])

    axis[3,4].plot('Degree', 'BIC', data=sols)

    table = DataFrame(sols, columns=['Degree', 'BIC', 'YP'], index=None)
    print(x)
    print(y)
    print(sols['COEFS'][best])
    print(table)
    print(f'Best {best + 1}')

    plt.show()