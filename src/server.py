from datetime import datetime
from pause import sleep
from ai import knnuser as knn
from api import api


def open_trade(open_date: datetime):
    print('[server.py] Opening a trade at ' + str(open_date))
    sleep_to_date(open_date)
    return api.open(0.01, 0.01, 1 - knn.predict_today(), 'US500')


def close_trade(order: int, close_date: datetime):
    print('[server.py] Closing a trade at ' + str(close_date))
    sleep_to_date(close_date)
    return api.close(order, 0.01)


def sleep_to_date(sleep_date: datetime):
    diff = (sleep_date - datetime.now()).total_seconds()
    if diff < 0:
        raise ValueError('sleep_date is less than today')
    # print('[server.py] Sleeping to ' + str(sleep_date))
    sleep(diff)


def main():
    while True:
        open_date = datetime.now().replace(day=datetime.now().day + 1, hour=9, minute=0)
        resp = open_trade(open_date)

        close_date = open_date.replace(hour=15, minute=50)
        close_trade(resp['returnData']['order'], close_date)


if __name__ == "__main__":
    main()
