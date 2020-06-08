import threading
from datetime import datetime
from pause import sleep
from ai.knn.knn_user import KNNUser
from api.api import APIUser
from logger.logger import init_logger

logger = init_logger(__name__, testing_mode=False)


class Bot(threading.Thread):
    def __init__(self, ticker, symbol, start_date, end_date, columns, model_path):
        super().__init__()
        self.api_user = APIUser()
        self.knn_user = KNNUser(ticker, model_path, columns)
        self.symbol = symbol
        self.resp = None
        self.start_date = start_date
        self.end_date = end_date

    def open_trade(self, open_date: datetime):
        logger.info('Opening a trade at ' + str(open_date))
        self.sleep_to_date(open_date)
        cmd = int(not self.knn_user.predict())
        self.resp = self.api_user.open(cmd, self.symbol)
        return self.resp

    def close_trade(self, close_date: datetime):
        logger.info('Closing a trade at ' + str(close_date))
        self.sleep_to_date(close_date)
        self.api_user.close(self.resp['returnData']['order'])

    def sleep_to_date(self, sleep_date: datetime):
        diff = (sleep_date - datetime.now()).total_seconds()
        if diff < 0:
            logger.error('sleep_date is less than today')
            raise ValueError('sleep length must be non-negative')
        logger.info('Sleeping to ' + str(sleep_date))
        sleep(diff)

    def run(self):
        self.test()
        # self.prod()

    def prod(self):
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            open_date = datetime.now().replace(day=datetime.now().day + 1,
                                               hour=self.start_date.hour,
                                               minute=self.start_date.minute)
            self.open_trade(open_date)
            close_date = open_date.replace(hour=self.end_date.hour,
                                           minute=self.end_date.minute)
            self.close_trade(close_date)

    def test(self):
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            open_date = datetime.now().replace(second=datetime.now().second + 5)
            self.open_trade(open_date)
            close_date = datetime.now().replace(second=datetime.now().second + 5)
            self.close_trade(close_date)


bot = Bot('^GSPC',
          'US500',
          datetime.now().replace(hour=9, minute=0, second=0),
          datetime.now().replace(hour=16, minute=0, second=0),
          ['Open', 'High', 'Low', 'Close', 'Volume'],
          './^GSPCKNN.joblib')
bot.run()
