# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 07:17:23 2020

@author: Administrator
"""



from MyTools.Operators import seriesNearest, vecMax
import pandas as pd
import pickle
#data = pd.read_pickle('G:\Python projects\myproject\derivatives_strategy\option_data.pkl')

class StrategyEngine():
    '''
    This is a container to store each component class
    '''
    def __init__(self, undl_cls = None, options_cls = None):
        '''
        undl_cls: class, which contains underlying's name, long/short
        options_clslist: list, which contains each option class
        '''
        self.undl_cls = undl_cls
        self.options_clslist = options_cls
        
    @classmethod
    def UnderWriting(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        '''
        underlying: str, underlying name
        SelectionMode: str, select options by 'forward' or 'delta' or 'spot'
        TgtLevel: float, target feature level
        QueryFunc: function to handle the issue of no exact same TgtLevel is available
        ExpTenor: int, T of target option
        '''
        # short put option
        put_cls = PutOption('Short', SelectionMode, TgtLevel, QueryFunc, ExpTenor)
        
        return cls(None, [put_cls])

    @classmethod
    def OverWriting(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        # long equity + short call opton
        undl_cls = Underlying(underlying, 'Long')
        call_cls = CallOption('Short', SelectionMode, TgtLevel, QueryFunc, ExpTenor)
        
        return cls(undl_cls, [call_cls])

    @classmethod
    def Collar(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        # long equity + short call opton + long put option
        undl_cls = Underlying(underlying, 'Long')
        call_cls = CallOption('Short', SelectionMode, TgtLevel[0], QueryFunc, ExpTenor)
        put_cls = PutOption('Long', SelectionMode, TgtLevel[1], QueryFunc, ExpTenor)
    
        return cls(undl_cls, [call_cls, put_cls])

    @classmethod
    def PutOverlay(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        # long equity + long put option
        undl_cls = Underlying(underlying, 'Long')
        put_cls = PutOption('Long', SelectionMode, TgtLevel, QueryFunc, ExpTenor)

        return cls(undl_cls, [put_cls])

    @classmethod
    def PutSpreadOverlay(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        # long equity + long put option(high strike) + short put option(low strike)
        undl_cls = Underlying(underlying, 'Long')
        PutLow_cls = PutOption('Short', SelectionMode, TgtLevel[0], QueryFunc, ExpTenor)
        PutHigh_cls = PutOption('Long', SelectionMode, TgtLevel[1], QueryFunc, ExpTenor)

        return cls(undl_cls, [PutLow_cls, PutHigh_cls])

    @classmethod
    def PutSpreadCollarOverlay(cls, underlying, SelectionMode, TgtLevel, QueryFunc = seriesNearest, ExpTenor = 1):
        # long equity + short call option (high strike) + long put option(mid strike) + short put option(low strike)
        undl_cls = Underlying(underlying, 'Long')
        call_cls = CallOption('Short', SelectionMode, TgtLevel[0], QueryFunc, ExpTenor)
        PutLow_cls = PutOption('Short', SelectionMode, TgtLevel[1], QueryFunc, ExpTenor)
        PutHigh_cls = PutOption('Long', SelectionMode, TgtLevel[2], QueryFunc, ExpTenor)

        return cls(undl_cls, [call_cls, PutLow_cls, PutHigh_cls])

    def __repr__(self):

        headline = 'Strategy Components:\n'

        undlline = ''
        if self.undl_cls != None:
            undlline = str(self.undl_cls) + '\n'

        optionlines = ''
        for option in self.options_clslist:
            optionlines = optionlines + str(option) + '\n'

            #print (option)

        return headline + undlline + optionlines
        #return f'{self.undl_cls}{self.call}{self.put}'
    

class OptionSelection():
    
    def __init__ (self, SelectionMode, TgtLevel, QueryFunc, ExpTenor):
        
        self.SelectionMode = SelectionMode         # str: 'forward' or 'delta' or 'spot'
        self.TgtLevel = TgtLevel                   # float: target feature level
        self.QueryFunc = QueryFunc   # function to handle the issue of no exact same TgtLevel is available
        self.ExpTenor = ExpTenor
        #self._initialize()
    '''        
    def _initialize(self):
        if self.SelectionMode == 'forward':
            self.selection_func = self.FwdSelection
    '''
    def get_tgt_opt(self, data):

        data = data[data.date < data.ExpDate]
        #curr_date = data.date.iloc[0]   # timestamp
       
        TgtExp = self.get_TgtExpiry_date(data.ExpDate)
        # filter the options by CallPut and expiration date
        for date in data.date.unique():
            TgtOpt_df = data[(data.date == pd.to_datetime(date)) &
                             (data.ExpDate == TgtExp) &
                             (data.CallPut == self.CallPut)]

            if TgtOpt_df.shape[0] > 0:
                break
            else:
                print (f'target option is not available on {str(date)[:10]}')

        if self.SelectionMode == 'forward':
            params = TgtOpt_df.strike/ TgtOpt_df.forward

        if self.SelectionMode == 'delta':
            params = TgtOpt_df.delta

        if self.SelectionMode == 'spot':
            params = TgtOpt_df.strike/TgtOpt_df.UndlSpot

        # transfer option information of DataFrame into dictionary
        TgtOpt_dict = TgtOpt_df[self.QueryFunc(params, self.TgtLevel)].to_dict(orient='record')[0]
        TgtOpt_dict['LongShort'] = self.LongShort
        TgtOpt_dict['CallPut'] = self.CallPut
        TgtOpt_dict['payoff_func'] = self.get_option_payoff

        return PackageOptionDetails(TgtOpt_dict)

       
    def get_TgtExpiry_date(self, expiry_dates):
        '''
        expiry_dates: pd.Series, used to find the target expiration date        
        return: TimeStamp, expiration date that new option need to trade
        '''
        expiry = expiry_dates.unique()
        #expiry_month = expiry.astype('datetime64[M]').astype(int)%12+1
        expiry.sort()
        #[date for date in expiry_dates if date > curr_date][self.ExpTenor - 1]
        
        return expiry[self.ExpTenor - 1]
        

class CallOption(OptionSelection):
    
    def __init__ (self, LongShort, SelectionMode, TgtLevel, QueryFunc, ExpTenor):
        super(CallOption, self).__init__(SelectionMode, TgtLevel, QueryFunc, ExpTenor)
        
        self.LongShort = 1 if LongShort == 'Long' else -1
        self.CallPut = 'C'
           
    def get_option_payoff(self, spot, strike):
        '''
        spot: np.array/pd.Serise
        strike: np.array/pd.Serise
        '''        
        return self.LongShort * vecMax(spot - strike, 0)
                
    def __repr__(self):
        LongShort = 'Long' if self.LongShort == 1 else 'Short'
        
        return f'Call Option: {LongShort}, Option select with {self.SelectionMode}: {self.TgtLevel}'


class PutOption(OptionSelection):
    
    def __init__ (self, LongShort, SelectionMode, TgtLevel, QueryFunc, ExpTenor):
        super(PutOption, self).__init__(SelectionMode, TgtLevel, QueryFunc, ExpTenor)
        
        self.LongShort = 1 if LongShort == 'Long' else -1
        self.CallPut = 'P'


    def get_option_payoff(self, spot, strike):
        '''
        spot: np.array/pd.Serise
        strike: np.array/pd.Serise
        '''
        return self.LongShort * vecMax(strike - spot, 0)
        
    def __repr__(self):
        LongShort = 'Long' if self.LongShort == 1 else 'Short'
        
        return f'Put Option: {LongShort}, Option select with {self.SelectionMode}: {self.TgtLevel}'


class Underlying():
    
    def __init__ (self, underlying, LongShort = 'No position'):
        self.underlying = underlying
        self.LongShort = 1 if LongShort == 'Long' else -1
        
    def __repr__ (self):
        LongShort = 'Long' if self.LongShort == 1 else 'Short'
        
        return f'Underlying: {self.underlying}, {LongShort}'



class PackageOptionDetails():
    '''
    This class is to put dictinary data as class
    '''
    def __init__(self, option_dict):
        self.option_dict = option_dict
        self._initialize()
        # self.LongShort = option_dict['LongShort']
        # self.CallPut = option_dict['CallPut']
        # self.contract = option_dict['contract']
        # self.price = option_dict['price']
        # self.opposite_prem = option_dict['opposite_prem']
        # self.date = option_dict['date']
        # self.ExpDate = option_dict['ExpDate']
        # self.listed_date = option_dict['listed_date']
        # self.UndlSpot = option_dict['UndlSpot']
        # self.strike = option_dict['strike']
        # self.tenor = option_dict['tenor']
        # self.rf_rate = option_dict['rf_rate']
        # self.forward = option_dict['forward']
        # self.implied_vol = option_dict['implied_vol']
        # self.delta = option_dict['delta']
        # self.gamma = option_dict['gamma']
        # self.vega = option_dict['vega']
        # self.theta = option_dict['theta']
        # self.payoff_func = option_dict['payoff_func']
        
        #self._quantify_variables()

    def _initialize(self):
        attr_list = list(self.option_dict.keys())
        for attr in attr_list:
            setattr(self, attr, self.option_dict[attr])
        
    #def _quantify_variables(self):
        
        #self.LongShort = 1 if self.LongShort == 'long' else -1
        #self.CallPut = 1 if self.CallPut == 'C' else -1

        
    def __repr__ (self):
        
        LongShort = 'Long' if self.LongShort == 1 else 'Short'
        CallPut = 'call' if self.CallPut == 'C' else 'put'
            
        #date = self.date.date()
        ExpDate = self.ExpDate.date()
        StrkToSpot = round(self.strike/self.UndlSpot, 4)
        strike = round(self.strike, 4)
        
        #line1 = f"{date} is a rebalancing date, {self.LongShort} {CallPut} option:\n"
        #line2 = f"premium: {self.price}, forward to strike: {fwd_to_strike}, expiry date: {ExpDate}\n"
        
        return f"{LongShort} {CallPut}, strike: {strike}, strike to forward: {StrkToSpot}, expiry date: {ExpDate}"


'''
undl = Underlying('510050', 'Long')
call = CallOption('Short', 'select_with_fwd',1.05, '1M')
put = PutOption('Long', 'select_with_fwd',0.95, '1M')

call.FwdSelection(data, seriesNearest)

strategy_params = Strategy(undl, call, put)

print (strategy_params)

'''
