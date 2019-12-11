import numpy
import os
import cPickle as pickle
import capnp
from nupic.algorithms.backtracking_tm_capnp import BacktrackingTMProto
from nupic.algorithms.anomaly import computeRawAnomalyScore
from nupic.algorithms.anomaly import Anomaly
from nupic.encoders import ScalarEncoder, MultiEncoder, PassThroughEncoder, DateEncoder
# Python TP implementation
# C++ TP implementation
#from nupic.research.TP10X2 import TP10X2 as TP
from nupic.algorithms.backtracking_tm import BacktrackingTM as TM
import capnp
from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto
from nupic.algorithms.spatial_pooler import SpatialPooler as SP

__author__ = 'Vaibhav Chiruguri'

# date
COL_MINUTE_OF_HOUR = "minute_of_hour"
COL_HOUR_OF_DAY = "hour_of_day"
COL_DAY_OF_WEEK = "day_of_week"
COL_DAY_OF_MONTH = "day_of_month"
COL_FIRST_LAST_MONTH = "first_last_of_month"
COL_WEEK_OF_MONTH = "week_of_month"
COL_MONTH_OF_YEAR = "month_of_year"
COL_QUART_OF_YEAR = "quart_of_year"
COL_HALF_OF_YEAR = "half_of_year"
COL_YEAR_OF_DECADE = "year_of_decade"
COL_DATE = "date"

#semantics
COL_STOCH_RSI = "stoch_rsi"
COL_SYMBOL = "symbol"
COL_CANDLESTICK = "candlestick"



class Encoder(object):
    def __init__(self, symbol_name, loadTp=True, enableLearn=True):
        self.tm = None
        self.sp = None
        self.symbol_name = symbol_name
        
        if loadTp and os.path.isfile('models_tm/'+self.symbol_name+'.pkl'):
            print "Loading TM"
            '''
            with open('models_sp/'+self.symbol_name+'.pkl','r') as f:

              spserializedMessage = pickle.load(f)
              # Deserialize to a new SP instance
              _TRAVERSAL_LIMIT_IN_WORDS = 1 << 63
              spreader = SpatialPoolerProto.from_bytes_packed(spserializedMessage,traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
              spreaderlol = spreader
              self.tm = TM.read(spreaderlol)
              '''
            with open('models_tm/'+self.symbol_name+'.pkl','r') as f:

              serializedMessage = pickle.load(f)
              # Deserialize to a new TM instance
              _TRAVERSAL_LIMIT_IN_WORDS = 1 << 63
              reader = BacktrackingTMProto.from_bytes_packed(serializedMessage,traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
              readerlol = reader
              self.tm = TM.read(readerlol)
     

        else:
            self.tm = self._init_tm()
            #self.sp = self._init_sp()
        self.enableLearn = enableLearn

    def _get_encoder(self):
        # date encoding
        #date_enc = DateEncoder(name='date', forced=True)
        minute_of_hour_enc = ScalarEncoder(w=21, minval=0, maxval=60, radius=1.5,
                                        periodic=True, name=COL_MINUTE_OF_HOUR, forced=True)
        hour_of_day_enc = ScalarEncoder(w=21, minval=1, maxval=24, radius=1.5,
                                        periodic=True, name=COL_HOUR_OF_DAY, forced=True)
        day_of_week_enc = ScalarEncoder(w=21, minval=0, maxval=7, radius=1.5,
                                        periodic=True, name=COL_DAY_OF_WEEK, forced=True)
        day_of_month_enc = ScalarEncoder(w=21, minval=1, maxval=31, radius=1.5,
                                         periodic=False, name=COL_DAY_OF_MONTH, forced=True)
        first_last_of_month_enc = ScalarEncoder(w=21, minval=0, maxval=2, radius=1, periodic=False,
                                                name=COL_FIRST_LAST_MONTH, forced=True)
        week_of_month_enc = ScalarEncoder(w=21, minval=0, maxval=6, radius=1.5,
                                          periodic=True, name=COL_WEEK_OF_MONTH, forced=True)
        month_of_year_enc = ScalarEncoder(w=21, minval=1, maxval=13, radius=1.5,
                                          periodic=True, name=COL_MONTH_OF_YEAR, forced=True)
        quarter_of_year_enc = ScalarEncoder(w=21, minval=0, maxval=4, radius=1.5,
                                            periodic=True, name=COL_QUART_OF_YEAR, forced=True)
        half_of_year_enc = ScalarEncoder(w=21, minval=0, maxval=2,
                                         radius=1, periodic=True, name=COL_HALF_OF_YEAR, forced=True)
        #year_of_decade_enc = ScalarEncoder(w=21, minval=0, maxval=10, radius=1.5,
        #                                  periodic=True, name=COL_YEAR_OF_DECADE, forced=True)

        # semantics encoder
        stoch_rsi_enc = ScalarEncoder(w=21, minval=0, maxval=1,
                                      radius=0.05, periodic=False, name=COL_STOCH_RSI, forced=True)
        symbol_enc = ScalarEncoder(w=21, minval=0, maxval=1, radius=0.1, periodic=False, name=COL_SYMBOL, forced=True)
        candlestick_enc = PassThroughEncoder(50, name=COL_CANDLESTICK, forced=True)

        encoder = MultiEncoder()
        encoder.addEncoder(minute_of_hour_enc.name, minute_of_hour_enc)
        encoder.addEncoder(hour_of_day_enc.name, hour_of_day_enc)
        encoder.addEncoder(day_of_week_enc.name, day_of_week_enc)
        encoder.addEncoder(day_of_month_enc.name, day_of_month_enc)
        encoder.addEncoder(first_last_of_month_enc.name, first_last_of_month_enc)
        encoder.addEncoder(week_of_month_enc.name, week_of_month_enc)
        #encoder.addEncoder(year_of_decade_enc.name, year_of_decade_enc)
        encoder.addEncoder(month_of_year_enc.name, month_of_year_enc)
        encoder.addEncoder(quarter_of_year_enc.name, quarter_of_year_enc)
        encoder.addEncoder(half_of_year_enc.name, half_of_year_enc)

        encoder.addEncoder(stoch_rsi_enc.name, stoch_rsi_enc)
        encoder.addEncoder(symbol_enc.name, symbol_enc)
        encoder.addEncoder(candlestick_enc.name, candlestick_enc)

        return encoder

    def _init_tm(self):
        return TM(numberOfCols=(self._get_encoder().width), cellsPerColumn=32,
                  initialPerm=0.5, connectedPerm=0.5,
                  minThreshold=30, newSynapseCount=60,
                  permanenceInc=0.1, permanenceDec=0.01,
                  activationThreshold=40,
                  globalDecay=0, 
                  checkSynapseConsistency=False,
                  burnIn=1,
                  seed=1956,
                  pamLength=7)

    def _init_sp(self):
        return SP(inputDimensions=(self._get_encoder().width),
                           columnDimensions=(2048),
                           potentialPct=0.85,
                           globalInhibition=True,
                           localAreaDensity=-1.0,
                           numActiveColumnsPerInhArea=40.0,
                           synPermInactiveDec=0.005,
                           synPermActiveInc=0.04,
                           synPermConnected=0.1,
                           boostStrength=3.0,
                           seed=1956,
                           wrapAround=False)

    def run_encoder(self, stock_records):
        encoder = self._get_encoder()
        input_array = numpy.zeros(encoder.width, dtype="int32")

        for i, record in enumerate(stock_records):
            # Execute Spatial Pooling algorithm over input space.
            encoding = numpy.concatenate(encoder.encodeEachField(record))
            #self.sp.compute(encoding, True, input_array)
            #activeColumnIndices = numpy.nonzero(input_array)[0]
            self.tm.compute(encoding, enableLearn=self.enableLearn)

        self.tm.reset()
        

    def output_sp_file(self):
        print "saving SP"
        with open('models_sp/'+self.symbol_name+'.pkl','w') as f:
           SPbuilder = SpatialPoolerProto.new_message()
           self.sp.write(SPbuilder)
           serializedMessage = SPbuilder.to_bytes_packed()
           pickle.dump(serializedMessage, f)


    def output_tm_file(self):
        print "saving TM"
        with open('models_tm/'+self.symbol_name+'.pkl','w') as f:
           TMbuilder = BacktrackingTMProto.new_message()
           self.tm.write(TMbuilder)
           serializedMessage = TMbuilder.to_bytes_packed()
           pickle.dump(serializedMessage, f)
        

    # Utility routine for printing the input vector
    def formatRow(self, x):
        s = ''
        for c in range(len(x)):
            if c > 0 and c % 10 == 0:
                s += ' '
            s += str(x[c])
        s += ' '
        return s

    def run_decoder(self, i, stock_records, mean=True):
        encoder = self._get_encoder()
        previous_predicted_columns = None
        scores = []
        input_array = numpy.zeros(encoder.width, dtype="int32")
        for i, record in \
                enumerate(stock_records):
            input_array[:] = numpy.concatenate(encoder.encodeEachField(record))
            self.tm.compute(input_array, enableLearn=False)

            active_columns = self.tm.infActiveState['t'].max(axis=1).nonzero()[0].flat
            predicted_columns = self.tm.getPredictedState().max(axis=1).nonzero()[0].flat

            #print active_columns
            #print predicted_columns

            if previous_predicted_columns is not None:
                    scores.append(computeRawAnomalyScore(active_columns,  previous_predicted_columns))
                    event = EventScore(i, computeRawAnomalyScore(active_columns,  previous_predicted_columns),
                                             record.date, record.open_price, record.close_price)
                    #print event
            previous_predicted_columns = predicted_columns

        self.tm.reset()

        
        return EventScore(i, numpy.mean(scores)), scores


class EventScore(object):
    def __init__(self, index, score, date=None,
                 open_price=0, close_price=0):
        self.index = index
        self.score = score
        self.date = date
        self.open_price = open_price
        self.close_price = close_price
