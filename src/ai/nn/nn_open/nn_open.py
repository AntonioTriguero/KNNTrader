from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import layers, optimizers, Input, Model


def to_timesteps(timesteps, data):
    result = []
    for i in range(timesteps, data.shape[0]):
        result.append(data[i-timesteps:i])
    return np.array(result)


startdate = datetime(1980, 1, 1).strftime('%Y-%m-%d')
enddate = datetime.now().strftime('%Y-%m-%d')
ticker = '^GSPC'
columns = ['High', 'Open', 'Low', 'Close', 'Volume']
epochs = 10
batch_size = 10
timesteps = 60

# Prepare the data
data_columns = ['High', 'Open', 'Close', 'Low', 'Volume', 'Dividends', 'Stock Splits']
df = yf.Ticker(ticker).history(start=startdate, end=enddate)
columns = sorted(columns, key=lambda e: list(df.columns).index(e))
data = df.drop(list(set(data_columns) - set(columns)), axis=1).dropna()
data['Target-Close'] = df['Close'].shift(periods=-1)

# Split the data
data = np.asarray(data.reset_index(drop=True)).astype('float32')
test = data[-int(len(data) * 0.2):]
train = data[:-len(test)]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

x_train = train[:, :-1]
y_train = train[:, -1:]
x_test = test[:, :-1]
y_test = test[:, -1:]

x_val = x_train[-int(len(x_train) * 0.2):]
y_val = y_train[-len(x_val):]
partial_x_train = x_train[:-len(x_val)]
partial_y_train = y_train[:-len(x_val)]

x_test = to_timesteps(timesteps, x_test)
x_val = to_timesteps(timesteps, x_val)
partial_x_train = to_timesteps(timesteps, partial_x_train)
y_test = y_test[timesteps:]
y_val = y_val[timesteps:]
partial_y_train = partial_y_train[timesteps:]

# Build the model
input = Input(shape=(timesteps, partial_x_train.shape[2]), dtype='float32')
x1 = layers.LSTM(256)(input)
output = layers.Dense(1)(x1)
model = Model(input, output)
model.compile(optimizer=optimizers.RMSprop(lr=1e-6),
              loss='mse')

# Fit the model
callbacks_list = [
]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    callbacks=callbacks_list)

# Plot train loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(range(1, len(loss_values) + 1), loss_values, 'r', label='Training loss')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot test loss
results = model.predict(x_test)
plt.plot(range(len(results)), ((y_test - results) ** 2).mean(axis=1))
plt.title('Test loss')
plt.xlabel('Samples')
plt.ylabel('Loss')
plt.show()

# Print some examples
print('Target: ' +
      str(scaler.inverse_transform(
          np.concatenate(
              (np.zeros((len(y_test), x_test.shape[2])),
               y_test),
              axis=1))[:, -1:]))
print('Predicted: ' +
      str(scaler.inverse_transform(
          np.concatenate(
              (np.zeros((len(results), x_test.shape[2])),
               results),
              axis=1))[:, -1:]))
