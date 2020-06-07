from datetime import datetime
import pandas_datareader.data as web
from joblib import load
import pandas as pd

from ai.knn_builder import KNNBuilder


class KNNUser:
    def __init__(self, tickers):
        KNNBuilder(tickers,
                   datetime(2010, 1, 1),
                   datetime.now(), 4, 10,
                   ['High', 'Low', 'Close'], ['Up'],
                   '../resources/models/',
                   'KNN')
        self.tickers = tickers
        self.data = self.read_trickers()

    def get_dataframe(self):
        dates, tickers, columns = [], [], []
        for ticker in self.tickers:
            today = datetime.now()
            df = web.DataReader(ticker, 'yahoo', start=today.replace(day=today.day - 1), end=datetime.now())
            columns = df.columns.values
            dates_values = df.index.values
            tickers.extend([ticker] * len(dates_values))
            dates.extend(dates_values)
        index = pd.MultiIndex.from_arrays([tickers, dates], names=['Ticker', 'Date'])
        return pd.DataFrame(index=index, columns=columns)

    def read_trickers(self):
        df = self.get_dataframe()
        columns = df.columns
        for ticker in self.tickers:
            today = datetime.now()
            df.loc[ticker, columns] = web.DataReader(ticker,
                                                     'yahoo',
                                                     start=today.replace(day=today.day - 1),
                                                     end=datetime.now()).to_numpy()
        print('*** Dataframe read')
        print(df)
        return df

    def predict(self, ticker):
        model = load('../resources/models/' + ticker + 'KNN.joblib')
        x = self.data.loc[ticker].tail(1)[['High', 'Low', 'Close']].to_numpy()
        y = model.predict(x)
        return int(y[0])


KNNUser(['^GSPC', 'AAPL']).predict('AAPL')