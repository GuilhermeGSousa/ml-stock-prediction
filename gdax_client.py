import gdax

import pandas as pd
import numpy as np

import time
import datetime

class GdaxClient:
    
    def __init__(self):
        self.public_client = gdax.PublicClient()
        self.max_dataframe_size = 350
        self.req_per_sec = 2
        
    def get_historical_data(self, begin, end, granularity = 900, pair = 'ETH-USD'):
    
        dt = datetime.timedelta(minutes=granularity/60 * self.max_dataframe_size)
        current_time = begin
        df_year = pd.DataFrame()
        
        while(current_time < end):
            data = self.public_client.get_product_historic_rates(pair, 
                                                          start = current_time, 
                                                          end = current_time + dt, 
                                                          granularity=granularity)
            if(data):
                df = pd.DataFrame(data, columns=['time','low','high','open', 'close', 'volume'])
                df.time = pd.to_datetime(df['time'], unit='s')
                df=df.iloc[::-1].reset_index(drop=True)
                df_year = df_year.append(df)
            if(current_time + dt > end):
                current_time = end - dt #Added to finish precisely on the end date
            else:
                current_time += dt
            time.sleep(1/self.req_per_sec)
            print(current_time)
        return df_year