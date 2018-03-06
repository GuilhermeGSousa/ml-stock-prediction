import gdax

import pandas as pd
import numpy as np

import time
import datetime

class GdaxClient:
    
    def __init__(self):
        self.public_client = gdax.PublicClient()
        self.max_dataframe_size = 300
        self.req_per_sec = 2
        
    def get_historical_data(self, begin, end, granularity = 900, pair = 'ETH-USD'):
        """
        Use UTC time
        """
        
        if(end > datetime.datetime.utcnow()):
            raise ValueError("End date can't be set in the future")
        
        dt = datetime.timedelta(minutes=granularity/60 * self.max_dataframe_size)
        current_time = begin
        df_year = pd.DataFrame()
        
        # These transformations must be done due to limitations of the gdax api
        # If the time is not rounded down to the nearest granularity value,
        # the api returns more data than needed (eg. 351 rows for a difference between end and start of the granularity)
        begin = self._round_time(begin, granularity)
        end = self._round_time(end, granularity)
        
        while(current_time < end):
            if(current_time + dt < end):
                data = self.public_client.get_product_historic_rates(pair, 
                                                          start = current_time, 
                                                          end = current_time + dt, 
                                                          granularity=granularity)
                current_time += dt
            elif(current_time + dt >= end):
                data = self.public_client.get_product_historic_rates(pair, 
                                                      start = current_time, 
                                                      end = end, 
                                                      granularity=granularity)
                current_time = end
            if(data and not isinstance(data,dict)):
                df = pd.DataFrame(data, columns=['time','low','high','open', 'close', 'volume'])
                df.time = pd.to_datetime(df['time'], unit='s')
                df=df.iloc[::-1].reset_index(drop=True)
                df_year = df_year.append(df)

            time.sleep(1/self.req_per_sec)
        df_year = df_year.reset_index(drop=True)
        return df_year
    
    def get_market_price(self):
        public_client = gdax.PublicClient()
        data = public_client.get_product_historic_rates('ETH-EUR', granularity=60)
        df = pd.DataFrame(data, columns=['time','low','high','open', 'close', 'volume'])
        return df["close"][0]
    
    def _round_time(self, dt, granularity):
        rounded_time = datetime.datetime(dt.year, dt.month, dt.day, dt.hour,
                                                granularity // 60 *(dt.minute // 15))
        return rounded_time