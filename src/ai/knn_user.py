from joblib import load
import numpy as np
from datetime import datetime
import pandas_datareader.data as web
from sklearn import preprocessing
import pathlib

model = load(str(pathlib.Path().absolute()) + '/ai/KNN.joblib')


def predict_today():
    steps = 15

    df = web.DataReader('^GSPC', 'yahoo',
                        start=datetime(1900, 10, 1),
                        end=datetime.now())
    df['Buy'] = (df['Close'] > df['Open']).astype('int')

    for i in range(1, steps):
        df['Close-' + str(i)] = df['Close'].shift(periods=i, fill_value=0.0)
        df['High-' + str(i)] = df['High'].shift(periods=i, fill_value=0.0)
        df['Low-' + str(i)] = df['Low'].shift(periods=i, fill_value=0.0)
    df = df.iloc[steps:]

    columns = ['Close', 'High', 'Low']
    for i in range(1, steps):
        columns += ['Close-' + str(i), 'High-' + str(i), 'Low-' + str(i)]
    x = np.asarray(df.iloc[-1:][columns])
    x = preprocessing.StandardScaler().fit(x).transform(x)

    return int(model.predict(x)[0])
