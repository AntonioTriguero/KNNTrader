import threading
from datetime import datetime
from pause import sleep
from api.api import APIUser
from logger.logger import init_logger

logger = init_logger(__name__, testing_mode=False)


class Server(threading.Thread):
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.api_user = APIUser(symbols)
        self.resp = []
        self.filename = '[Server]'

    def open_trade(self, open_date: datetime):
        logger.info('Opening a trade at ' + str(open_date))
        self.sleep_to_date(open_date)

        for ticker in self.symbols.keys():
            self.resp.append(self.api_user.open(ticker))

        return self.resp

    def close_trade(self, close_date: datetime):
        logger.info('Closing a trade at ' + str(close_date))
        self.sleep_to_date(close_date)

        for r in self.resp:
            self.api_user.close(r['returnData']['order'], 0.01)
        self.resp = []

    def sleep_to_date(self, sleep_date: datetime):
        diff = (sleep_date - datetime.now()).total_seconds()
        if diff < 0:
            logger.error('sleep_date is less than today')
            raise ValueError('sleep length must be non-negative')
        logger.info('Sleeping to ' + str(sleep_date))
        sleep(diff)

    def run(self):
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            open_date = datetime.now().replace(day=datetime.now().day + 1, hour=9, minute=0)
            self.open_trade(open_date)
            close_date = open_date.replace(hour=15, minute=50)
            self.close_trade(close_date)
