from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import Ridge
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
        for i in range(0, len(coeffs)):
            yVals.append(coeffs[i] * xVal**i)

    return yVals


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

    plt.title(f'{point_dict[users_data]['name']}\'s Data')
    plt.scatter(x, y)

    polyx = np.linspace(8, 17, 100)
    alphas = np.logspace(-3, 4, 200)
    bfl = Ridge()

    for a in alphas:
        bfl.set_params(alpha=a).fit(x, y)
        polyy = bfl.predict(polyx.reshape(100, 1))

        plt.plot(polyx, polyy)

    plt.show()