#!/usr/bin/python
from optparse import OptionParser
import sys
import csv
import time
import multiprocessing as mp
import datetime
import numpy
from encoder.encoder import Encoder
from encoder.utils.utils import print_err
from encoder.trainer_file import TrainerFile
from encoder.stock import Stock, StockRecord
from encoder.trainer import Trainer
import matplotlib.pyplot as plt
from encoder.utils.utils import DotDictify, get_first_average, get_next_average
import os

import sys
sys.setrecursionlimit(10000000)

__author__ = 'Vaibhav Chiruguri'

def print_usage_and_exit():
    print_err("\n" + sys.argv[0] + " --train|--model -q <quote_file_path> -t <train_file>")
    print_err("\nquote_file_path\tpath for the quote file")
    print_err("use_existing_tm\tTrue of False: improve learning")
    exit(1)

def check_command_line_args():
    parser = OptionParser()
    parser.add_option("-q", "--quote_file", dest="quote_file_path",
                      help="parse a quote file", metavar="FILE")
    parser.add_option("--t", "--train",
                      action="store_true", dest="train", default=True,
                      help="train the nupic model")
    parser.add_option("--m", "--model",
                      action="store_false", dest="train", default=True,
                      help="test the nupic model")
    parser.add_option("-f", "--train_file", dest="train_data_file",
                      help="training file with preselected dates and quote file paths")
    parser.add_option("-g", "--test_file", dest="test_data_file",
                      help="testing file with preselected dates and quote file paths")
    parser.add_option("-s", "--symbol_name", dest="symbol_name",
                      help="symbol name for printing")

    return parser.parse_args()

def start_training(quote_file_path, train_data_file, symbol_name, buy=True):

    #encoder = Encoder(loadTp=True)
    if train_data_file is not None and train_data_file != '':

        p = mp.Pool(processes = mp.cpu_count()-2)

        with open(train_data_file, 'rb') as csvfile:
            moments = csv.reader(csvfile, delimiter=',')
            datapoints = []
            head, tail = os.path.split(train_data_file)

            for index, line in enumerate(moments):
                datapoints.append(parse_quote_line(line))

            #parallel_process(train_data_file, datapoints, tail)
            
            datapoints_chunks = list(chunks(datapoints, 10))
            print("chunks ready")
            
            stop = True
            procs = []
            for index, chu in enumerate(datapoints_chunks):
                 stop = True
                 index = index + 1
                
                 proc = mp.Process(target=parallel_process, args=(train_data_file, chu, tail))
                 procs.append(proc)
                 proc.start()

                 if index % 70 == 0:
                    while(stop):
                      if not any(proc.is_alive() for proc in procs):
                        print(str(index * 10) + " Sets Completed...")
                        stop = False
                        del procs[:]
                      time.sleep(3)
                     
            
           
       # trainer_file = TrainerFile(train_data_file, parse=True)
    else:
        stock = Stock(quote_file_path)
        trainer = Trainer(stock)
        trainer_file = trainer.calculate_windows(True)
    '''
    encoded_data = trainer_file.get_encoded_data(buy)
    total_datasets = len(encoded_data)
    total_trained = 0
    for i, dataset in enumerate(encoded_data):
        encoder.run_encoder(dataset)
        total_trained += len(dataset)
        print "Completed " + str(i) + "/" + str(total_datasets) + " sets...Trained on: " + \
              str(total_trained) + " records."
    encoder.output_tm_file()
    '''

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def parallel_process(train_data_file, datapoints, symbol_name, buy=True):
                    #print("lol process")
                    trainer_file = TrainerFile(train_data_file,windows=datapoints)
                    encoded_data = trainer_file.get_encoded_data(buy)
                    encoder = Encoder(symbol_name, loadTp=True)
                    for i, dataset in enumerate(encoded_data):
                        encoder.run_encoder(dataset)
                        #print("kappa")
                    #encoder.output_sp_file()
                    encoder.output_tm_file()
                        


def parse_quote_line(quote_line):
        symbol, start, open_low, open_low_datetime, close_high, close_high_datetime, delta, end = quote_line
        window = {
                "symbol": symbol,
                "start": datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S"),
                "open_low": open_low,
                "open_low_datetime": datetime.datetime.strptime(open_low_datetime, "%Y-%m-%d %H:%M:%S"),
                "close_high": close_high,
                "close_high_datetime": datetime.datetime.strptime(close_high_datetime, "%Y-%m-%d %H:%M:%S"),
                "delta": delta,
                "end": datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
            }
        return DotDictify(window)

class EventScoreGraph(object):
    def __init__(self, event_scores):
        self.event_scores = event_scores

    def display(self):
        pareto = sorted(self.event_scores, key=lambda e: e.score, reverse=True)
        pareto_indices = range(len(pareto))
        pareto_scores = [e.score for e in pareto]
        plt.plot(pareto_indices, pareto_scores, label="Pareto of Avg Anomaly for Test Events")
        plt.show()


def start_modeling(test_data_file, buy=True):
    encoder = Encoder('lol', loadTp=True, enableLearn=False)
    test_file = TrainerFile(test_data_file, parse=True)
    encoded_data = test_file.get_encoded_data(buy)
    total_datasets = len(encoded_data)
    total_trained = 0
    event_scores = []
    for i, dataset in enumerate(encoded_data):
        score = encoder.run_decoder(i, dataset)
        # Calculate the anomaly score for each event in the test set
        event_scores.append(score)
        total_trained += len(dataset)
        
    print "Completed " + str(i) + "/" + str(total_datasets) + " sets...Tested on: " + \
          str(total_trained) + " records."
    for i, score in enumerate(event_scores):
        print score.score
    EventScoreGraph(event_scores).display()


def main(train=True, quote_file_path=None, train_data_file=None, test_data_file=None, symbol_name=None):
    if train:
        start_training(quote_file_path, train_data_file, symbol_name)
    else:
        start_modeling(test_data_file)

if __name__ == "__main__":
    options, args = check_command_line_args()
    main(options.train, options.quote_file_path, options.train_data_file, options.test_data_file, options.symbol_name)