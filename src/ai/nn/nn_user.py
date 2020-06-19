import os
import yfinance as yf
from datetime import datetime
import numpy as np
from joblib import load
from pandas import read_csv
from tensorflow.python.keras.models import load_model
from ai.nn.nn import models_path, data_path, generate


class NNUser:
    def __init__(self, ticker='^GSPC'):
        self.ticker = ticker
        self.model = load_model(models_path + ticker + '_nn.h5')
        self.scaler = load(models_path + ticker + '_scaler.joblib')
        self.columns = ['High', 'Open', 'Low', 'Close', 'Volume']
        self.steps = 5

    def get_last_ticker_input(self,
                              steps=7,
                              startdate=datetime(1980, 1, 1).strftime('%Y-%m-%d'),
                              enddate=datetime.now().strftime('%Y-%m-%d')):
        ticker_data_path = data_path + self.ticker + '/' + startdate + enddate + '.csv'

        if not os.path.exists(ticker_data_path):
            if not os.path.exists(data_path + self.ticker):
                os.makedirs(data_path + self.ticker)
            df = yf.Ticker(self.ticker).history(start=startdate,
                                                end=enddate)
            df.to_csv(ticker_data_path)
        else:
            df = read_csv(ticker_data_path).set_index('Date')

        columns = sorted(self.columns, key=lambda e: list(df.columns).index(e))
        for s in range(1, steps):
            for c in columns:
                df[c + '-' + str(s)] = df[c].shift(periods=s).dropna()

        result = df.drop(['Dividends', 'Stock Splits'], axis=1).dropna()
        result['Aux'] = df['Close']

        return result.iloc[-1:]

    def predict(self):
        d = np.asarray(self.get_last_ticker_input().reset_index(drop=True)).astype('float32')
        # y = d[-1, -1:].reshape(1, -1)

        ds = self.scaler.transform(d)
        xs = ds[-1, :-1].reshape(1, -1)
        # ys = ds[-1, -1:].reshape(1, -1)

        ys_ = self.model.predict(xs)
        y_ = self.scaler.inverse_transform(np.concatenate((np.zeros(len(xs[0])), ys_[0])).reshape(1, -1))[0, -1:]

        # print('Target (scaler): ' + str(ys))
        # print('Predicted (scaler): ' + str(ys_))
        # print('Target: ' + str(y))
        print('Predicted: ' + str(y_))

        return y_[0]


# generate([7], 1, 4)
nnu = NNUser('^GSPC')
nnu.predict()
