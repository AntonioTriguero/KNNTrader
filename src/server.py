from datetime import datetime
from pause import sleep
from api.api import APIUser


class Server:
    def __init__(self, symbols):
        self.symbols = symbols
        self.api_user = APIUser(symbols)
        self.resp = []
        self.filename = '[Server]'

    def open_trade(self, open_date: datetime):
        print(self.filename + ' Opening a trade at ' + str(open_date))
        self.sleep_to_date(open_date)

        for ticker in self.symbols.keys():
            self.resp.append(self.api_user.open(ticker))

        return self.resp

    def close_trade(self, close_date: datetime):
        print(self.filename + ' Closing a trade at ' + str(close_date))
        self.sleep_to_date(close_date)

        for r in self.resp:
            self.api_user.close(r['returnData']['order'], 0.01)
        self.resp = []

    def sleep_to_date(self, sleep_date: datetime):
        diff = (sleep_date - datetime.now()).total_seconds()
        if diff < 0:
            raise ValueError(self.filename + ' sleep_date is less than today')
        print(self.filename + ' Sleeping to ' + str(sleep_date))
        sleep(diff)


def main():
    server = Server({'BTC-USD': 'BITCOIN', '^GSPC': 'US500'})

    while True:
        open_date = datetime.now().replace(second=(datetime.now().second + 5) % 60) # datetime.now().replace(day=datetime.now().day + 1, hour=9, minute=0)
        server.open_trade(open_date)
        close_date = open_date.replace(second=(open_date.second + 5) % 60) # open_date.replace(hour=15, minute=50)
        server.close_trade(close_date)


if __name__ == "__main__":
    main()
