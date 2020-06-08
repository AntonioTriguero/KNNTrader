from datetime import datetime
from random import randint

from flask import Flask, request
from ai.bot.bot import Bot

app = Flask(__name__)
models_path = './models/'
bots = {}


@app.route('/start', methods=['POST'])
def start_bot():
    global models_path, bots

    ticker = request.json['ticker']
    symbol = request.json['symbol']
    start_date = datetime.strptime(request.json['start_hour'], 'T%H:%M:%S')
    end_date = datetime.strptime(request.json['end_hour'], 'T%H:%M:%S')
    columns = request.json['columns']

    bots[ticker] = Bot(ticker, symbol, start_date, end_date, columns, models_path + ticker + 'KNN.joblib')
    bots[ticker].start()

    return app.response_class(status=200)


@app.route('/stop', methods=['DELETE'])
def stop_bot():
    global bots
    ticker = request.json['ticker']

    if ticker not in bots.values():
        return app.response_class(status=404)

    bot = bots[ticker]
    bot.do_run = False
    del bots[ticker]

    return app.response_class(status=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
