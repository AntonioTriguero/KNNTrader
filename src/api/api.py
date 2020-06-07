from ai.knn_user import KNNUser
from api import xAPIConnector


class APIUser:
    def __init__(self, tickers):
        self.user_id = 11172382
        self.password = "Somossiete7"
        self.tickers = tickers
        self.knn = KNNUser(tickers)

    def close(self, order, price):
        client = self.open_connection()

        resp = client.commandExecute('getTrades', arguments={
            "openedOnly": True
        })

        trade = None
        for t in resp['returnData']:
            if t['order2'] == order:
                trade = t

        resp = client.commandExecute('tradeTransaction', arguments={
            'tradeTransInfo': {
                "cmd": 0,
                "order": trade['order'],
                "price": price,
                "symbol": trade['symbol'],
                "type": 2,
                "volume": trade['volume']
            }
        })

        client.disconnect()

        return resp

    def open(self, ticker, price, volume, order_type):
        if ticker not in self.tickers:
            raise ValueError('Ticker unavailable')

        client = self.open_connection()

        cmd = 1 - self.knn.predict(ticker)

        resp = client.commandExecute('tradeTransaction', arguments={
            'tradeTransInfo': {
                "cmd": cmd,
                "price": price,
                "symbol": ticker,
                "type": order_type,
                "volume": volume
            }
        })

        client.disconnect()

        return resp

    def open_connection(self):
        client = xAPIConnector.APIClient()

        login_response = client.execute(xAPIConnector.loginCommand(userId=self.user_id, password=self.password))
        xAPIConnector.logger.info(str(login_response))

        if not login_response['status']:
            print('Login failed. Error code: {0}'.format(login_response['errorCode']))

        return client
