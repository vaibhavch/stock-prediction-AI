# Implement a trading strategy based on the trained buy and sell models

import datetime
import yahoo as yf
from encoder.data_point import DataPoint
from encoder.encoder import Encoder
from encoder.stock import StockRecord
from encoder.trainer_file import TrainerFile

BUFFER_TRADE = 30
MIN_TRADE_MINS = datetime.timedelta(minutes=5)
MAX_TRADE_MINS = datetime.timedelta(minutes=30)
MAX_OPEN_TRADES = 4


class Trade(object):
    def __init__(self,
                 start_datetime,
                 start_price,
                 end_datetime,
                 end_price):
        self.start_datetime = start_datetime
        self.start_price = start_price
        self.end_datetime = end_datetime
        self.end_price = end_price
        self.is_open = True

    def get_earnings(self):
        return self.end_price - self.start_price

    def close_trade(self):
        self.is_open = False


class Trader(object):
    def __init__(self, symbol, start_datetime, trade_window):
        self.symbol = symbol
        self.start_datetime = start_datetime
        self.trade_window = trade_window
        self.earnings = 0

    def is_expired(self, day):
        ''' @return true if day is outside the trade_window of the trade '''
        return day > self.start_datetime + self.trade_window

    def get_earnings(self):
        return self.earnings

    def set_earnings(self, earnings):
        self.earnings = earnings

    def start_trading(self):
        encoder = Encoder(self.symbol.replace(".NS",""), loadTp=True,enableLearn=False)
        #encoder = Encoder("lol", loadTp=True,enableLearn=False)
        counter = 0
        score_buffer = []
        open_trades = []
        #while self.start_datetime <= datetime.datetime.now():
        '''
        while self.start_datetime <= self.start_datetime + datetime.timedelta(minutes=40):
            if len(score_buffer) > 30:
                score_buffer.pop(0)

            datasets = self._get_stock_data()
            #print(datasets)
            for index, dataset in enumerate(datasets):
                score = encoder.run_decoder(index, dataset)
                #score_buffer.append({counter: score})
                print score.score

            counter += 1
            self.start_datetime += datetime.timedelta(minutes=1)
        '''
        datasets = self._get_stock_data()
        for index, dataset in enumerate(datasets):
                mean_score, scores = encoder.run_decoder(index, dataset)
                #score_buffer.append({counter: score})
                print scores
                print round(mean_score.score, 2)


    # data from yahoo finance api
    def _get_stock_data(self):

        data = yf.download(self.symbol, interval = "1m", start = self.start_datetime , end= self.start_datetime + datetime.timedelta(minutes=80), auto_adjust = True,)
        #print(data)
        #data = ystockquote.get_historical_prices(self.symbol,
         #                                 datetime.datetime.strftime(self.start_date, '%Y-%m-%d'),
         #                                 datetime.datetime.strftime(self.start_date + self.trade_window, '%Y-%m-%d'))
        datapoints = []
        for index, record in data.iterrows():
            stripkey = str(index).replace("+05:30","")
            datapoint = {
                "datetime": datetime.datetime.strptime(stripkey, '%Y-%m-%d %H:%M:%S'),
                "high": float(record["High"]),
                "low": float(record["Low"]),
                "open": float(record["Open"]),
                "close": float(record["Close"]),
                "volume": int(record["Volume"])
            }

            datapoints.append(DataPoint(datapoint))

        #print(datapoints)
        datapoints = TrainerFile.get_encoded_data_from_array([datapoints,], True)
        #print(datapoints)
        datasets = []
        for dataset in datapoints:
            stock_records = []
            for record in dataset:
                stock_records.append(StockRecord(self.symbol, record))
            datasets.append(stock_records)

        return datasets

    # data from local files
    def _get_local_stock_data(self):
        with open(self.symbol+'', 'rb') as csvfile:
            quote_reader = csv.reader(csvfile, delimiter=',')
            datapoints = []
            for quote_line in quote_reader:
                if not "null" in quote_line:
                  datapoints.append(self.parse_quote_line(quote_line))
            return datapoints