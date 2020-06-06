from ai import knnuser as knn
from api import xAPIConnector

userId = 11172382
password = "Somossiete7"


def close(order, price):
    client = xAPIConnector.APIClient()

    login_response = client.execute(xAPIConnector.loginCommand(userId=userId, password=password))
    xAPIConnector.logger.info(str(login_response))

    if not login_response['status']:
        print('Login failed. Error code: {0}'.format(login_response['errorCode']))

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


def open(price, volume, type, symbol):
    client = xAPIConnector.APIClient()

    login_response = client.execute(xAPIConnector.loginCommand(userId=userId, password=password))
    xAPIConnector.logger.info(str(login_response))

    if not login_response['status']:
        print('Login failed. Error code: {0}'.format(login_response['errorCode']))

    cmd = 1 - knn.predict_today()

    resp = client.commandExecute('tradeTransaction', arguments={
        'tradeTransInfo': {
            "cmd": cmd,
            "price": price,
            "symbol": symbol,
            "type": type,
            "volume": volume
        }
    })

    client.disconnect()

    return resp
