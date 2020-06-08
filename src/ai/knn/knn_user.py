import pandas_datareader as pdr
from datetime import datetime
from joblib import load
from ai.knn.knn_builder import KNNBuilder


class KNNUser:
    def __init__(self, ticker, model_path, x_columns):
        knn = KNNBuilder(ticker, model_path, x_columns)
        knn.start()
        knn.join()
        self.ticker = ticker
        self.model = load(model_path)
        self.x_columns = x_columns

    def get_ticker(self):
        today = datetime.now()
        return pdr.data.DataReader(self.ticker,
                                   data_source='yahoo',
                                   start=today.replace(day=today.day - 5),
                                   end=datetime.now())

    def clean(self, df):
        return (df - df.shift(periods=1)).shift(periods=1).dropna().tail(1)

    def predict(self):
        data = self.get_ticker()
        data = self.clean(data)
        y_ = self.model.predict(data[self.x_columns])
        return y_[0]
