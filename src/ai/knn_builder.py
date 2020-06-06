from datetime import datetime
from pathlib import Path

import pandas_datareader.data as web
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import pandas as pd
import numpy as np


class KNNBuilder:
    def __init__(self, tickers, start_date, end_date, max_steps, max_neighbors, x_columns, y_columns, models_path):
        self.models_path = models_path
        self.tickers = tickers
        self.data = self.read_trickers(start_date, end_date)
        self.find_best_models(max_steps, max_neighbors, x_columns, y_columns)

    def get_dataframe(self, start_date, end_date):
        dates, tickers, columns = [], [], []
        for ticker in self.tickers:
            df = web.DataReader(ticker, 'yahoo', start=start_date, end=end_date)
            columns = df.columns.values
            dates_values = df.index.values
            tickers.extend([ticker] * len(dates_values))
            dates.extend(dates_values)
        index = pd.MultiIndex.from_arrays([tickers, dates], names=['Ticker', 'Date'])
        return pd.DataFrame(index=index, columns=columns)

    def read_trickers(self, start_date, end_date):
        df = self.get_dataframe(start_date, end_date)
        columns = df.columns
        df.insert(len(df.columns), "Diff", np.random.randn(len(df.index)), True)
        df.insert(len(df.columns), "Up", np.random.randn(len(df.index)), True)
        for ticker in self.tickers:
            df.loc[ticker, columns] = web.DataReader(ticker, 'yahoo', start=start_date, end=end_date).to_numpy()
            df.loc[ticker, 'Diff'] = (df.loc[ticker, 'Close'] - df.loc[ticker, 'Open']).astype('float').to_numpy()
            df.loc[ticker, 'Up'] = (df.loc[ticker, 'Close'] > df.loc[ticker, 'Open']).astype('int').to_numpy()
        print('*** Dataframe read')
        print(df)
        return df

    def find_best_models(self, max_steps, max_neighbors, x_columns, y_columns):
        self.data = self.extract_historical_data(max_steps, x_columns)
        for ticker in self.tickers:
            print('*** Analyzing ticker ' + str(ticker))
            x = self.data.loc[(ticker, slice(None)), slice(x_columns[0] + '-1', None)]
            y = self.data.loc[(ticker, slice(None)), y_columns]
            mean_accuary, models = self.train(x, y, x_columns, max_steps, max_neighbors)
            best_model = self.select_best(models, mean_accuary, max_steps)
            dump(best_model, self.models_path + '/' + ticker + 'KNN.joblib')
            print()

    def extract_historical_data(self, steps_back, columns):
        for ticker in self.tickers:
            for i in range(1, steps_back):
                for c in columns:
                    self.data.loc[ticker, str(c) + '-' + str(i)] = self.data.loc[ticker][c].shift(periods=i, fill_value=0.0).to_numpy()
        return self.data

    def train(self, x, y, x_columns, steps, ks):
        mean_accuary = []
        models = []

        for s in range(1, steps):
            x_ = x[steps:].to_numpy()[:, :len(x_columns) * s]
            y_ = y[steps:].to_numpy()
            x_ = preprocessing.StandardScaler().fit(x_).transform(x_)
            x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=4)

            mean_accuary.insert(s, [])
            models.insert(s, [])
            for k in range(1, ks):
                model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train.ravel())
                models[s - 1].insert(k - 1, model)

                yhat = model.predict(x_test)

                mean_accuary[s - 1].insert(k - 1, metrics.accuracy_score(y_test, yhat))

        return mean_accuary, models

    def select_best(self, models, mean_accuary, steps):
        max_mean_accuary = []
        for s in range(0, steps-1):
            max_mean_accuary.insert(s, None)
            max_accuary = max(mean_accuary[s])
            k = mean_accuary[s].index(max_accuary)
            max_mean_accuary[s] = (k + 1, max_accuary)
        aux = [x[1] for x in max_mean_accuary]
        max_accuary = max(aux)
        stps = aux.index(max_accuary) + 1
        ks = max_mean_accuary[aux.index(max_accuary)][0]
        print('Max accuary: ' + str(max_accuary))
        print('Neighbors: ' + str(ks))
        print('Steps: ' + str(stps))
        return models[stps-1][ks - 1]


KNNBuilder(['^GSPC', 'AAPL'],
           datetime(1900, 1, 1),
           datetime.now(), 20, 20,
           ['High', 'Low', 'Close'], ['Up'],
           '../resources/modules')
