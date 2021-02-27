# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:31:19 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
from MyTools.MongoDB_api import MongoDB_Engine
from datetime import datetime, timedelta
from MyTools.BlackScholes import OptPricingEngine
import pickle

class MarketDataEngine():
    
    def __init__ (self, underlying, RfRateName, StartDate, EndDate,
                  ColumnsAdd = ['strike','put_call','exp_date','listed_date']):
        '''
        underlying: str, underlying of option
        RfRateName: str, risk-free rate name
        StartDate: str
        EndDate: str
        ColumnsAdd: list, columns need to copy from option information table
        '''        
        self.undl = underlying
        self.RfRateName = RfRateName
        self.StartDate = pd.to_datetime(StartDate)
        self.EndDate = pd.to_datetime(EndDate)
        self.ColumnsAdd = ColumnsAdd
        
        self._get_raw_data()
        #self.integrate_market_data()
        
        
    def _get_raw_data(self):
        '''
        extract option price, option information, underlying price, risk-free rate from MongoDB
        '''
        db_engine = MongoDB_Engine('OptionDaily')
        self.OptClose_df = db_engine.query_mongo_data(self.undl, {'date': {'$gte': self.StartDate, '$lte': self.EndDate}})
        
        db_engine = MongoDB_Engine('OptionInfo')
        self.OptInfo_df = db_engine.query_mongo_data(self.undl)
        self.OptInfo_df.wind_code = self.OptInfo_df.wind_code.astype(str)

        db_engine = MongoDB_Engine('UnderlyingDaily')
        UndlSpot_series = db_engine.query_mongo_data(self.undl, {}, 'date').close.rename('undl_spot')
        #UndlSpot_series = db_engine.query_mongo_data(self.undl).set_index(['date']).close.rename('undl_spot')
        UndlSpot_series.index = pd.to_datetime(UndlSpot_series.index)
        self.UndlSpot_series = UndlSpot_series[((UndlSpot_series.index >= self.StartDate) &
                                                (UndlSpot_series.index <= self.EndDate))]
        #self.UndlSpot_series.rename(columns={'close': 'undl_spot'}, inplace = True)
        
        db_engine = MongoDB_Engine('RiskFreeRate')
        rf_series = db_engine.query_mongo_data(self.RfRateName, {}, 'date').close.rename('rf_rate')
        rf_series.index = pd.to_datetime(rf_series.index)
        self.rf_series = rf_series[((rf_series.index >= self.StartDate) & (rf_series.index <= self.EndDate))]
        #self.rf_series.rename(columns={'closeRate': 'rf_rate'}, inplace = True)

        
    def get_rebal_calendar(self, RebalFreq = 1):
        '''
        :param RebalFreq: int, 1 stand for 1 month
        :return: list, rebalancing dates
        '''
        
        calendar = list(pd.to_datetime(self.OptInfo_df.exp_date.unique()))
        calendar.append(self.StartDate)
        calendar.sort()
        
        RebalCalendar = [date for date in calendar if ((self.StartDate <= date) and (self.EndDate >= date))]
        
        if RebalFreq != 1:
            #iteration = int(rebal_freq[:-1])
            RebalCalendar = [RebalCalendar[i] for i in range(RebalFreq-1, len(RebalCalendar), RebalFreq)]
        
        # if the first rebalancing date is too close to trade date, than delete the first rebalancing date
        if (RebalCalendar[1] - RebalCalendar[0]).days < 20:
            del RebalCalendar[1]
        
        return RebalCalendar
               

    def integrate_market_data (self):
        '''
        combine option price, option information, underlying price, risk-free rate
        calculate tenor, opposite premium, forward, implied volatility
        '''
        self.data = self.OptClose_df
        
        self.combine_opt_info()
        print('OptInfo_df combined')
        # make sure each option pair (same strike, expiration date, trading date) is on the neighbour line
        self.data = self.data.sort_values(by = ['date','listed_date','exp_date','strike','put_call'])
        
        self.combine_rf_rate()
        print('rf combined')
        self.combine_undl_spot()
        print('underlying combined')
        
        self.data['tenor'] = (self.data.exp_date - self.data.date) / timedelta(days=365)

        self.data['opposite_prem'] = self.get_opposite_prem()       
        self.data['forward'] = self.get_pc_parity_fwd()
        
        BSparams = {'spot': np.array(self.data.undl_spot),
                    'strike': np.array(self.data.strike),
                    'tenor':np.array(self.data.tenor),
                    'rf': np.array(self.data.rf_rate),
                    'put_call': np.array(self.data.put_call),
                    'prem': np.array(self.data.price),
                    'forward': np.array(self.data.forward)}

        pricing_engine = OptPricingEngine(BSparams)
        self.data['implied_vol'] = pricing_engine.sigma
        self.data['delta'] = pricing_engine.calc_delta()
        self.data['gamma'] = pricing_engine.calc_gamma()
        self.data['vega'] = pricing_engine.calc_vega()
        self.data['theta'] = pricing_engine.calc_theta()
        

    def combine_opt_info(self):
        '''
        combine each option informaiton with option closing price
        '''
        
        for feature in self.ColumnsAdd:
            
            self.data[feature] = np.nan
            
            for contract in self.data.contract.unique():
                print (f'{feature}: {contract}')
                self.data[feature].loc[self.data.contract == contract] = self.OptInfo_df[feature][self.OptInfo_df.wind_code == contract].values[0]
        
        self.data.date = self.data.date.apply(lambda x: pd.to_datetime(str(x)))
        self.data.exp_date = pd.to_datetime(self.data.exp_date)
      
    
    def combine_rf_rate(self):
        '''
        add risk free rate into DataFrame
        '''
        
        self.data['rf_rate'] = np.nan
        for date in self.data.date.unique():
            self.data.rf_rate.loc[self.data.date == date] = (self.rf_series[self.rf_series.index == date].values[0])
    
    
    def combine_undl_spot(self):
        '''
        add underlying spot into DataFrame
        '''
        
        self.data['undl_spot'] = np.nan
        for date in self.data.date.unique():
            self.data.undl_spot.loc[self.data.date == date] = self.UndlSpot_series[self.UndlSpot_series.index == date].values[0]

    
    def get_opposite_prem(self):
        '''
        switch option premium of each pair
        '''
        price_lag = self.data.price.shift(1).fillna(0)
        price_preceed = self.data.price.shift(-1).fillna(0)
        
        put_call_flag = (self.data.put_call == 'C')
        
        # call option: use the price on next line
        # put option: use the price on above line
        opposite_prem = put_call_flag * price_preceed + (1 - put_call_flag) * price_lag
        
        return opposite_prem
    
    
    def get_pc_parity_fwd(self):
        '''
        find the forward based on put-call parity
        '''
        put_call_flag = (self.data.put_call == 'C') * 2 - 1    # call为1，put为-1
        
        # call: (c-p)*exp(rT) + K
        # put: (p-c)*exp(rT) + K
        forward = (self.data.price - self.data.opposite_prem) * put_call_flag * np.exp(self.data.tenor * self.data.rf_rate) + self.data.strike
        
        return forward


if __name__ == '__main__':
    underlying = '510050'
    RfRateName = 'GC001'
    StartDate = '20180102'
    EndDate = '20181231'
    
    MktData_engine = MarketDataEngine(underlying, RfRateName, StartDate, EndDate)
    MktData_engine.integrate_market_data()
    data = MktData_engine.data
    
    pickle.dump(data, open('G:\Python projects\myproject\derivatives_strategy\option_data.pkl','wb'))





'''
BSparams = {'spot': np.array(data.undl_spot),
            'strike': np.array(data.strike),
            'tenor':np.array(data.tenor),
            'rf': np.array(data.rf_rate),
            'put_call': np.array(data.put_call),
            'prem': np.array(data.price),
            'forward': np.array(data.forward)}

'''

'''

model_prem = bs.calc_option_prem()

test3=pd.DataFrame(opt_data.price)
test3['model_prem']=model_prem
test3['iv']=iv


        self.spot = BSparams.get('spot')                  # np.array
        self.strike = BSparams.get('strike')                  # np.array
        self.tenor = BSparams.get('tenor')                  # np.array
        self.rf = BSparams.get('rf')                  # np.array         
        self.put_call = BSparams.get('put_call') 
        self.prem = BSparams.get('prem', np.nan)   # np.array
        self.sigma = BSparams.get('sigma', np.nan)          # np.array
        self.forward = BSparams.get('forward', np.nan) 
        self.div = BSparams.get('div', np.nan) 

'''


'''
und_minute = pd.read_csv('510050minute.csv', encoding = 'gb2312').iloc[:-3,:].fillna(method='ffill') 
db_engine = MongoDB_Engine('IndexDaily')
db_engine.upload_df(und_minute, '510050')

und_minute.index = pd.to_datetime(und_minute.DateTime)
und_daily=und_minute.resample('D').last().dropna(axis=0, how='all')
db_engine.upload_df(undl_spot, '510050')

'''


'''

test1=opt_data.date.apply(lambda x: str(x.date()))
test2=opt_data.exp_date.apply(lambda x: str(x.date()))
test3=opt_data.strike.apply(lambda x: str(x))

test = opt_data.listed_date + test1+test2+test3

count_list=[]
test_list=list(test)
for i in range(len(test)):
    print(i)
    count_list.append(test_list.count(test_list[i]))

np.unique(count_list)
sum((count_list))

same_list=[]
for i in range(len(test)-1):
    same_list.append(test_list[i] == test_list[i+1])

same_list2=[]
for i in range(len(same_list)-1):
    same_list2.append(same_list[i] == same_list[i+1])

sum(same_list2)

PC_list = list(opt_data.put_call)
same_list3=[]
for i in range(len(PC_list)-1):
    same_list3.append(PC_list[i] == PC_list[i+1])

sum(same_list3)

'''


