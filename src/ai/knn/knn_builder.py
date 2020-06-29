import threading
from datetime import datetime
import yfinance as yf
from joblib import dump
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from logger.logger import init_logger

logger = init_logger(__name__, testing_mode=False)


class KNNBuilder(threading.Thread):
    def __init__(self, ticker, model_path, x_columns):
        super().__init__()
        self.ticker = ticker
        self.model_path = model_path
        self.x_columns = x_columns
        self.batch = 5

    def get_ticker(self, ticker):
        return yf.Ticker(ticker).history(start=datetime(1980, 1, 1).strftime('%Y-%m-%d'),
                                              end=datetime(2020, 1, 1).strftime('%Y-%m-%d'))

    def clean(self, df):
        global data
        data = (df - df.shift(periods=1)).shift(periods=1).dropna()
        for i in range(0, self.batch):
            data = (data - data.shift(periods=1)).dropna()
        data['Buy'] = df['Close'] > df['Open']
        return data

    def train(self, x, y, max_ks):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        best_model = None
        max_accuary = 0
        for k in range(1, max_ks):
            model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
            yhat = model.predict(x_test)
            acc = metrics.accuracy_score(y_test.to_numpy(), yhat)

            if acc > max_accuary:
                best_model = model
                max_accuary = acc

        return best_model, max_accuary

    def save(self, model, path):
        dump(model, path)

    def run(self):
        self.batch = 4
        tickers = ['^STOXX50E', '^GSPC', '^IBEX', '^DJI', '^GDAXI', '^FTSE', '^IXIC', '^N225', 'GC=F', '^CMC200',
                   'BTC-EUR', 'SI=F', 'EURGBP=X', 'EURUSD=X', 'BZ=F', 'GBMTRVBO.MX', '^BCOMNG1', 'BCH', '^XSP', 'BTSC',
                   '^JN0U.JO', '^CASE30', '^NZ50', '^KLSE', 'INDD.MC', 'INDS.MC', '^TWII', '^SSMI', '^OMXSPI', '^OSEAX',
                   '^BFX', '^N100', '^BVSP', '^IPSA', '^MXX']
        data = DataFrame()
        for ticker in tickers:
            data = data.append(self.clean(self.get_ticker(ticker)), ignore_index=True)
        best_model, max_accuary = self.train(data[self.x_columns], data['Buy'], 40)
        logger.info('Max accuary = ' + str(max_accuary))
        dump(best_model, self.model_path)


t = KNNBuilder('', './KNN.joblib', ['Open', 'High', 'Low', 'Close'])
t.start()
t.join()
