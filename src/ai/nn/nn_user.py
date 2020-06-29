import yfinance as yf
from datetime import datetime
import numpy as np
from joblib import load
from tensorflow.python.keras.models import load_model
from ai.nn.nn import models_path


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
        df = yf.Ticker(self.ticker).history(start=startdate,
                                            end=enddate)

        columns = sorted(self.columns, key=lambda e: list(df.columns).index(e))
        for s in range(1, steps):
            for c in columns:
                df[c + '-' + str(s)] = df[c].shift(periods=s).dropna()

        result = df.drop(['Dividends', 'Stock Splits'], axis=1).dropna()
        result['Target-Close'] = df['Close'].shift(periods=-1).fillna(0)

        return result

    def predict(self):
        d = np.asarray(self
                       .get_last_ticker_input(enddate=datetime.now().replace(day=datetime.now().day + 1).strftime('%Y-%m-%d'))
                       .reset_index(drop=True)).astype('float32')
        y = d[:, -1:]

        ds = self.scaler.transform(d)
        xs = ds[:, :-1]
        # ys = ds[-1, -1:].reshape(1, -1)

        ys_ = self.model.predict(xs)
        y_ = self.scaler.inverse_transform(np.concatenate((np.zeros((len(ys_), xs.shape[1])), ys_), axis=1))[:, -1:]

        # print('Target (scaler): ' + str(ys))
        # print('Predicted (scaler): ' + str(ys_))
        print('Target: ' + str(y))
        print('Predicted: ' + str(y_))

        return y_


nnu = NNUser('^GSPC')
nnu.predict()
