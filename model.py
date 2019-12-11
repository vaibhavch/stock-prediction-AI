#!/usr/bin/python
import datetime
from trader import Trader


TRADE_WINDOW = datetime.timedelta(minutes=30)

def start_prediction():
    #trader = Trader('HDFCBANK.NS', datetime.datetime.now() - datetime.timedelta(minutes=40), TRADE_WINDOW)
    trader = Trader('ACC.NS', datetime.datetime(2019, 8, 2, 12, 23, 00, 00000) - datetime.timedelta(minutes=80), TRADE_WINDOW)
    trader.start_trading()

if __name__ == "__main__":
    start_prediction()

