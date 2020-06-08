import threading
import pandas_datareader as pdr
from datetime import datetime
import yfinance as yf
from joblib import dump
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from logger.logger import init_logger

yf.pdr_override()
logger = init_logger(__name__, testing_mode=False)


class KNNBuilder(threading.Thread):
    def __init__(self, ticker, model_path, x_columns):
        super().__init__()
        self.ticker = ticker
        self.model_path = model_path
        self.x_columns = x_columns

    def get_ticker(self):
        return pdr.data.DataReader(self.ticker, data_source='yahoo', start=datetime(1900, 1, 1), end=datetime.now())

    def clean(self, df):
        data = (df - df.shift(periods=1)).shift(periods=1).dropna()
        data['Buy'] = df['Close'] > df['Open']
        return data

    def train(self, x, y, max_ks):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

        best_model = None
        max_accuary = 0
        for k in range(1, max_ks):
            model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
            yhat = model.predict(x_test)
            acc = metrics.accuracy_score(y_test, yhat)

            if acc > max_accuary:
                best_model = model
                max_accuary = acc

        return best_model, max_accuary

    def save(self, model, path):
        dump(model, path)

    def run(self):
        data = self.get_ticker()
        data = self.clean(data)
        best_model, max_accuary = self.train(data[self.x_columns], data['Buy'], 20)
        logger.info('Max accuary = ' + str(max_accuary))
        dump(best_model, self.model_path)
