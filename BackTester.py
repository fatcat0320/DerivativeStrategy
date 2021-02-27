# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:14:22 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from operator import methodcaller
from MyTools.Operators import seriesNearest
from MyTools.StaticParams import StockParams, OptionParams
from DerivativesPackage.CacheOptionData import MarketDataEngine
from DerivativesPackage.StrategyHandler import StrategyEngine, CallOption, PutOption, Underlying
from DerivativesPackage.PerfHandler import PerfEngine


class BackTestEngine():
    def __init__(self, strategy, data, UndlSpot, rf, EqTransCostRate=0.003,
                 OptTransCostRate=0.0003, InitNotional = 100000):
        self.strategy_cls = strategy                    # class, contains underlying class and option class
        self.data = data                            # DataFrame, contains
        self.UndlSpot_series = UndlSpot             # pd.Series, underlying closing price
        self.rf_series = rf                         # pd.Series, risk-free rate
        self.EqTransCostRate = EqTransCostRate      # float, equity trading fee rate
        self.OptTransCostRate = OptTransCostRate    # float, equity trading fee rate
        self.InitNotional = InitNotional
        
        self._initialize()


    def _initialize(self):

        self.ExpTenor = self.strategy_cls.options_clslist[0].ExpTenor
        self.StratCalendar = self.UndlSpot_series.index
        self.RebalFreq = 1
        self.RebalCalendar = self._get_RebalCalendar()
        
        try:
            # try to find the position of underlying
            self.UndlLongShort = self.strategy_cls.undl_cls.LongShort
        except:
            # in this case, no underlying input
            self.UndlLongShort = 0


    def _get_RebalCalendar(self):
        '''
        return: list, the expiration dates between start date and end date (include the first backtest date)
        '''
        # find all the expiration dates and remove duplicates
        calendar = list(pd.to_datetime(self.data.ExpDate.unique()))
        start = pd.to_datetime(self.data.date.values[0])
        end = pd.to_datetime(self.data.date.values[-1])
        
        calendar.append(start)  # add the first back test date
        calendar.sort()        
        # find monthly expiration dates between start date and end date
        RebalCalendar = [date for date in calendar if ((start <= date) and (end >= date))]
        
        if self.RebalFreq != 1:
            # rebalancing freqency is larger than 1 month: 
            # pick the dates from RebalCalendar, step by RebalFreq
            RebalCalendar = [RebalCalendar[i] for i in range(self.RebalFreq-1, len(RebalCalendar), self.RebalFreq)]
        
        # if the first rebalancing date is too close to trade date, than delete the first rebalancing date
        if (RebalCalendar[1] - RebalCalendar[0]).days < 20:
            del RebalCalendar[1]    
            
        return RebalCalendar

    def create_container(self, init_val):

        container = []
        for i in range(len(self.strategy_cls.options_clslist)):
            #if tenor == 1:
                #container.append(init_val)
            #else:
                container.append([init_val] * self.ExpTenor)
        return container

    def RunBackTest(self):

        #holding_opt_num = len(self.strategy_cls.options_clslist) * self.ExpTenor
        CashPile_array = np.array([]) 
        # [array([], shape=(5, 0), dtype=float64), array([], shape=(5, 0), dtype=float64)]
        OptCalc_matlist = self.create_container(np.vstack(([np.array([])] * 5)))
        # array([], shape=(5, 0), dtype=float64)
        StratCalc_mat = np.vstack(([np.array([])] * 5))

        SelectOpt_clslist = self.create_container(None)
        
        # calculate payoff periodically
        for RebalDateInd in range(len(self.RebalCalendar)):            
            print(f"\n{self.RebalCalendar[RebalDateInd].date()} is a rebalancing date.")
            
            # find the calendar between 2 rebalancing dates (both include)
            try:
                period_series = self.StratCalendar[(self.StratCalendar >= self.RebalCalendar[RebalDateInd]) &
                                             (self.StratCalendar <= self.RebalCalendar[RebalDateInd+1])]
            except:
                # period after the last rebalancing date
                period_series = self.StratCalendar[self.StratCalendar >= self.RebalCalendar[-1]]
        
            # find the data between 2 rebalancing date 
            PeriodData_df = self.data[self.data.date.isin(period_series)]
                
            # find strat_notional(t-1)
            try:
                StratNotionalOpen = StratCalc_mat[4][-1]
            except:
                # if this is first period, use cost-deducted InitNotional
                StratNotionalOpen = self.InitNotional * (1 - self.UndlLongShort * self.EqTransCostRate)
        
            # query underling spot,  between two rebalancing dates
            UndlSpot_array = self.UndlSpot_series[(self.UndlSpot_series.index >= period_series.values[0]) &
                                                    (self.UndlSpot_series.index <= period_series.values[-1])]
            # equities number/options contract number
            unit = StratNotionalOpen/UndlSpot_array[0]/self.ExpTenor
        
            TotalOptPnl_array = 0  # total opt_pnl of all the options in this period
            TotalPrem = 0        # total option premium on each rebalancing date
            TotalIntrinsicVal = 0  # total option intrinsic value on each expiration date
            OptTransCost = 0       # # total option trading cost on each rebalancing date            
            # loop all the options in strategy
            for OptTypeID in range(len(self.strategy_cls.options_clslist)):
                for optID in range(self.ExpTenor):
                    if (optID + RebalDateInd) % 2==0 or self.ExpTenor==1:
                    # select the option based on the data on the first day of the period
                    # SelectOpt_cls: class, contains selected option's price, strike, long/short, greeks....
                        SelectOpt_cls = self.strategy_cls.options_clslist[OptTypeID].get_tgt_opt(PeriodData_df)
                        print(SelectOpt_cls)
                        SelectOpt_clslist[OptTypeID][optID] = SelectOpt_cls
                        OptCalc_mat = self.get_OptCalc_mat(SelectOpt_cls, PeriodData_df, unit)
                        premium = OptCalc_mat[3][0]
                        if self.ExpTenor==1:
                            # when ExpTenor is 1, contract would be switched on each rebalancing date
                            TotalIntrinsicVal += SelectOpt_cls.payoff_func(UndlSpot_array[-1], SelectOpt_cls.strike) * unit
                    else:
                        SelectOpt_cls = SelectOpt_clslist[OptTypeID][optID]
                        premium = 0
                        #OptCalc_mat = self.get_OptCalc_mat(SelectOpt_cls, PeriodData_df, PrevUnit)
                        # get the intrinsic value of all the options on expiration date
                        if SelectOpt_cls is not None:
                            TotalIntrinsicVal += SelectOpt_cls.payoff_func(UndlSpot_array[-1], SelectOpt_cls.strike) * PrevUnit
                            OptCalc_mat = self.get_OptCalc_mat(SelectOpt_cls, PeriodData_df, PrevUnit)
                        else:
                            OptCalc_mat = np.zeros((5, len(period_series) - 1))
                    # query the data of selected option between two rebalancing dates (both include)
                    #opt_data_period = PeriodData_df[PeriodData_df.contract == SelectOpt_cls.contract]

                    # if SelectOpt_cls is not None:
                    #     # np.array() * 5: Option price, strike, unit, total outstanding value, pnl
                    #     OptCalc_mat = self.get_OptCalc_mat(SelectOpt_cls, PeriodData_df, unit)
                    # else:
                    #     OptCalc_mat = np.zeros((5,len(period)-1))


                    # opt_pnl of all the options in this period
                    TotalOptPnl_array += OptCalc_mat[4]

                    #premium = OptCalc_mat[3][0]
                    OptTransCost += abs(premium) * self.OptTransCostRate
                    TotalPrem += premium
                    # get the intrinsic value of all the options on expiration date
                    #TotalIntrinsicVal += SelectOpt_cls.payoff_func(UndlSpot_array[-1], SelectOpt_cls.strike) * unit

                    OptCalc_matlist[OptTypeID][optID] = np.hstack((OptCalc_matlist[OptTypeID][optID], OptCalc_mat))
                
            #calculate cash_pile (np.array)    
            try: 
                #total_prem = CashPile_array[-1] - total_opt_val_period.values[0]
                CashPile_array[-1] = CashPile_array[-1] - TotalPrem
            except:
                # first period, no cash pile(t-1)
                CashPile_array = np.append(CashPile_array, (1-self.UndlLongShort)*self.InitNotional - TotalPrem)
            
            CashPilePeriod_array = self.get_cash_pile(CashPile_array[-1], period_series.date)
            # account the final payoff of option on rebalancing date
            CashPilePeriod_array[-1] += TotalIntrinsicVal            
            CashPile_array = np.hstack((CashPile_array, CashPilePeriod_array))
            
            # combine data of past period and the data of new period
            # np.array() * 5: equity unit, equity pnl, total option pnl, total option transaction cost, strategy notional
            StratCalc_mat = np.hstack((StratCalc_mat, 
                                       self.get_StratCalc_mat(StratNotionalOpen, UndlSpot_array, unit, TotalOptPnl_array, OptTransCost)))

            PrevUnit = unit
        # integrate data into DataFrame
        self.BackTestResult = self.integrate_calc_data(OptCalc_matlist, StratCalc_mat, CashPile_array)

    
    def get_OptCalc_mat(self, opt_cls, data, unit):
        '''
        Calculate the lastest Option price, strike, unit, total outstanding value, pnl

        opt_cls: class, contains selected option's price, strike, long/short, greeks...
        data: DataFrame, options data in current period
        unit: float: option contract number
        
        return: np.array()*5: Option price, strike, unit, total outstanding value, pnl
        '''
        # query data for select option
        SelectOpt_df = data[data.contract == opt_cls.contract]
        
        # query option price and option strike between two rebalancing dates

        price_array = SelectOpt_df.prem.values
        if min(SelectOpt_df.date) != min(data.date):
            # 合约到期日的后一天才会出2个月到期的期权
            price_array = np.hstack((price_array[0], price_array))
        strike_array = opt_cls.strike * np.ones(len(price_array)-1)
        #print (f"{data.date.values[0]}: {price_array[0]}")
            
        OptUnit = opt_cls.LongShort * unit
        
        # total outstanding_opt_val = opt_cont_num * opt_premium     
        OptVal_array = OptUnit * price_array  #(np.array)
    
        # opt_pnl = outstanding_opt_val - outstanding_opt_val(t-1)
        # on rebalancing date: opt_pnl is the pnl of old option         
        OptPnl_array = np.diff(OptVal_array)    #(np.array)
        
        calc_mat = np.vstack((price_array[:-1],
                              strike_array,
                              OptUnit * np.ones(len(strike_array)),
                              OptVal_array[:-1],
                              OptPnl_array))
        return calc_mat

            
    def get_cash_pile(self, CashValOpen, ObsPeriod_array):
        '''
        cash_value(t) = cash_value(t-1) * (1+rf/360)^daycount
        
        CashValOpen: float
        ObsPeriod: np.array, period that cash value need to be calculate
        
        return: np.array
        '''    
        rf_series = self.rf_series[self.rf_series.index.isin(ObsPeriod_array[:-1])]   #pd.Series
        daycount_array = np.diff(ObsPeriod_array)/timedelta(1)   #np.array
        
        # cash_value(t) = cash_value(t-1) * (rf/360+1)^daycount
        CashVal = ((rf_series.values/360 + 1)**daycount_array).cumprod() * CashValOpen
        
        return CashVal
        
    
    def get_StratCalc_mat(self, StratNotionalOpen, UndlSpot_array, unit, OptPnl_array, OptTransCost):
        '''
        Calculate the lastest equity unit, equity pnl, total option pnl, total option transaction cost, strategy notional

        StratNotionalOpen: float, strategy notional at the beginning of the period
        UndlSpot_array: np.array, underlying spot in current period
        unit: float, equity number
        OptPnl_array: np.array, sum of options pnl
        OptTransCost: float, options tranaction cost
        
        return: np.array()*5: equity unit, equity pnl, total option pnl, total option transaction cost, strategy notional 
        '''
        EquityUnit = self.UndlLongShort * unit * self.ExpTenor
        EquityPnl = np.diff(UndlSpot_array) * EquityUnit

        # strategy_notional = strategy_notional(t-1) + opt_pnl + spot_pnl + OptTransCost     (np.array)        
        StratNotional = StratNotionalOpen + OptPnl_array.cumsum() + EquityPnl.cumsum() - OptTransCost
        
        calc_mat = np.vstack((EquityUnit * np.ones(len(EquityPnl)),
                              EquityPnl,
                              OptPnl_array,
                              np.hstack((OptTransCost, np.zeros(len(EquityPnl)-1))),
                              StratNotional))
        return calc_mat


    def integrate_calc_data(self, OptCalc_matlist, StratCalc_mat, CashPile_array):
        '''
        Summarize the calculation into dataframe.

        OptCalc_matlist: list, [np.array()*5, np.array()*5, .....]
        StratCalc_mat: np.array()*5
        CashPile_array: np.array
        
        return: DataFrame
        '''
        #self.OptCalc_matlist_test = OptCalc_matlist
        BTresult_df = pd.DataFrame(self.UndlSpot_series[:-1])
        # loop all the strategy components (except underlying)
        for OptTypeID in range(len(OptCalc_matlist)):
            for optID in range(len(OptCalc_matlist[0])):

                opt_df = pd.DataFrame(OptCalc_matlist[OptTypeID][optID].T, index = self.StratCalendar[:-1])
                # define columns name
                opt_df.columns = ['opt_price'+ str(OptTypeID+1) + str(optID+1),
                                  'strike'+ str(OptTypeID+1) + str(optID+1),
                                  'opt_unit'+ str(OptTypeID+1) + str(optID+1),
                                  'outstanding_opt_val'+ str(OptTypeID+1) + str(optID+1),
                                  'opt_pnl'+ str(OptTypeID+1) + str(optID+1)]

                BTresult_df = pd.concat([BTresult_df, opt_df],axis=1)
                BTresult_df['opt_pnl'+ str(OptTypeID+1) + str(optID+1)] = BTresult_df['opt_pnl'+ str(OptTypeID+1) + str(optID+1)].shift(1).fillna(0)
                            
        strat_df = pd.DataFrame(StratCalc_mat.T, index = self.StratCalendar[:-1])
        strat_df.columns = ['equity_unit', 'equity_pnl','total_opt_pnl', 'OptTransCost','strategy_notional']
        BTresult_df = pd.concat([BTresult_df, strat_df],axis=1)
        
        BTresult_df.equity_pnl = BTresult_df.equity_pnl.shift(1).fillna(0)
        BTresult_df.total_opt_pnl = BTresult_df.total_opt_pnl.shift(1).fillna(0)
        # shift one line down and put the first line as initial notional
        BTresult_df.strategy_notional = BTresult_df.strategy_notional.shift(1).fillna(self.InitNotional)
        
        BTresult_df['rf'] = self.rf_series.values[:-1]
        BTresult_df['cash_pile'] = CashPile_array[:-1]
        
        return BTresult_df

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    