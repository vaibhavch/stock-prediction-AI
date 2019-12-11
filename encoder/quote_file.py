#!/usr/bin/python
import csv
import datetime
from data_point import DataPoint

__author__ = 'Vaibhav Chiruguri'

##
# Class that encapsulates a csv quote file
# extracts datapoints from the file and converts them
# to a Stock object
##
class QuoteFile:
    def __init__(self, file):
        self.filename = file

    def parse_quote_line(self, quote_line):
        datetime_str, open_str, high_str, low_str, close_str, volume_str = quote_line
        return DataPoint({"datetime": datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S"),
                "open": float(open_str),
                "high": float(high_str),
                "low": float(low_str),
                "close": float(close_str),
                "volume": float(volume_str)})

    def get_datapoints(self):
        with open(self.filename, 'rb') as csvfile:
            quote_reader = csv.reader(csvfile, delimiter=',')
            next(quote_reader, None)
            datapoints = []
            for quote_line in quote_reader:
                if not "null" in quote_line:
                  datapoints.append(self.parse_quote_line(quote_line))
            return datapoints