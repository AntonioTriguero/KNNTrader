import os
from datetime import datetime
from threading import Thread
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from tensorflow.python.keras import regularizers
from joblib import dump

data_path = './data/'
models_path = './models/'


class NNBuilder(Thread):
    def __init__(self, columns: list, ticker: str, steps: int = 5, epochs: int = 100):
        super().__init__()

        self.steps = steps
        self.columns = columns
        self.epochs = epochs
        self.ticker = ticker

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(data_path + self.ticker):
            os.makedirs(data_path + self.ticker)

    def data(self,
             startdate=datetime(1980, 1, 1).strftime('%Y-%m-%d'),
             enddate=datetime.now().strftime('%Y-%m-%d')):
        ticker_data_path = data_path + self.ticker + '/' + startdate + enddate + '.csv'
        data_columns = ['High', 'Open', 'Close', 'Low', 'Volume', 'Dividends', 'Stock Splits']

        if not os.path.exists(ticker_data_path):
            df = yf.Ticker(self.ticker).history(start=startdate,
                                                end=enddate)
            df.to_csv(ticker_data_path)
        else:
            df = read_csv(ticker_data_path).set_index('Date')

        self.columns = sorted(self.columns, key=lambda e: list(df.columns).index(e))
        for s in range(1, self.steps + 1):
            for c in self.columns:
                df[c + '-' + str(s)] = df[c].shift(periods=s).dropna()

        result = df.drop(data_columns, axis=1).dropna()
        result['Close'] = df['Close']

        return result

    def model(self):
        model = models.Sequential()
        model.add(layers.Dense(self.steps,
                               activation='relu',
                               input_shape=(self.steps * len(self.columns),),
                               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                               kernel_initializer='random_uniform',
                               bias_initializer='random_uniform'))
        model.add(layers.Dense(1,
                               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                               kernel_initializer='random_uniform',
                               bias_initializer='random_uniform'))
        model.compile(optimizer='rmsprop',
                      loss='mse')
        return model

    def split_and_normalize_data(self, dataframe, test_size=0.2, val_size=0.2):
        scaler_path = models_path + self.ticker + '_scaler.joblib'

        data = np.asarray(dataframe.reset_index(drop=True)).astype('float32')
        train = data[:int(len(data) * (1 - test_size))]
        test = data[int(len(data) * (1 - test_size)):]

        scaler = MinMaxScaler()
        scaler.fit(train)
        dump(scaler, scaler_path)
        train = scaler.transform(train)
        test = scaler.transform(test)

        x_train = train[:, :-1]
        y_train = train[:, -1:]
        x_test = test[:, :-1]
        y_test = test[:, -1:]

        x_val = x_train[int(len(x_train) * (1 - val_size)):]
        y_val = y_train[int(len(x_train) * (1 - val_size)):]
        partial_x_train = x_train[:int(len(x_train) * (1 - val_size))]
        partial_y_train = y_train[:int(len(x_train) * (1 - val_size))]

        return partial_x_train, partial_y_train, x_val, y_val, x_test, y_test

    def plot_train_loss(self, history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        plt.plot(range(1, self.epochs + 1), loss_values, 'r', label='Training loss')
        plt.plot(range(1, self.epochs + 1), val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss (Steps: ' + str(self.steps) + ')')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_test_loss(self, predicted, target):
        plt.plot(range(len(predicted)),
                 ((target - predicted) ** 2).mean(axis=1))
        plt.title('Test loss (Steps: ' + str(self.steps) + ')')
        plt.show()

    def run(self):
        nn_path = models_path + self.ticker + '_nn.h5'

        df = self.data(enddate=datetime(2020, 1, 1).strftime('%Y-%m-%d'))
        model = self.model()
        partial_x_train, partial_y_train, x_val, y_val, x_test, y_test = self.split_and_normalize_data(df)

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=self.epochs,
                            validation_data=(x_val, y_val),
                            batch_size=self.steps)
        self.plot_train_loss(history)

        results = model.predict(x_test)
        self.plot_test_loss(results, y_test)

        model.save(nn_path)


def generate(steps,
             loops: int,
             epochs: int,
             ticker: str = '^GSPC'):
    for s in steps:
        for i in range(0, loops):
            t = NNBuilder(['High', 'Open', 'Low', 'Close', 'Volume'], ticker, steps=s, epochs=epochs)
            t.start()
            t.join()
