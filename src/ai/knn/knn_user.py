import yfinance as yf
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
        self.batch = 10

    def get_ticker(self):
        today = datetime.now()
        return yf.Ticker(self.ticker).history(start=today.replace(month=today.month - 1))

    def clean(self, df):
        global data
        data = (df - df.shift(periods=1)).shift(periods=1).dropna()
        for i in range(0, self.batch):
            data = (data - data.shift(periods=1)).dropna()
        return data.tail(1)

    def predict(self):
        data = self.get_ticker()
        data = self.clean(data)
        print(data)
        y_ = self.model.predict(data[self.x_columns])
        return y_[0]
