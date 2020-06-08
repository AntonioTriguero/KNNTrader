from api.wrapper import xAPIConnector
from logger.logger import init_logger

logger = init_logger(__name__, testing_mode=False)


class APIUser:
    def __init__(self):
        self.user_id = 11172382
        self.password = 'Somossiete7'

    def close(self, order):
        client = self.open_connection()

        resp = client.commandExecute('getTrades', arguments={
            'openedOnly': True
        })

        trade = None
        for t in resp['returnData']:
            if t['order2'] == order:
                trade = t

        try:
            resp = client.commandExecute('tradeTransaction', arguments={
                'tradeTransInfo': {
                    'cmd': trade['cmd'],
                    'order': trade['order'],
                    'price': 1,
                    'symbol': trade['symbol'],
                    'type': 2,
                    'volume': trade['volume']
                }
            })
        except TypeError:
            logger.error('Order ' + str(order) + ' not found')

        client.disconnect()

        return resp

    def open(self, cmd, symbol):
        client = self.open_connection()

        resp = client.commandExecute('tradeTransaction', arguments={
            'tradeTransInfo': {
                'cmd': cmd,
                'price': 1,
                'symbol': symbol,
                'type': 0,
                'volume': client.commandExecute('getSymbol', arguments={
                    'symbol': symbol
                })['returnData']['lotMin']
            }
        })

        client.commandExecute('tradeTransactionStatus', arguments={
            'order': resp['returnData']['order']
        })

        client.disconnect()

        return resp

    def open_connection(self):
        client = xAPIConnector.APIClient()

        login_response = client.execute(xAPIConnector.loginCommand(userId=self.user_id, password=self.password))
        xAPIConnector.logger.info(str(login_response))

        if not login_response['status']:
            logger.error('Login failed. Error code: {0}'.format(login_response['errorCode']))

        return client
