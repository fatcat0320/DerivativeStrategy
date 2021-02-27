
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PerfEngine():
    def __init__(self, BT_dict, window):
        self.BT_dict = BT_dict
        self.window = window
        self._initialize()

    def _initialize(self):
        self.InitNotional = list(self.BT_dict.values())[0].strategy_notional.iloc[0]
        year_list = list(np.unique(list(self.BT_dict.values())[0].index.year))
        self.perf_dict = dict((k, v) for (k, v) in [(k, pd.DataFrame()) for k in year_list])
        for StratName in self.BT_dict.keys():
            notional = self.BT_dict[StratName].strategy_notional
            notional.index = pd.to_datetime(notional.index)

            stats_dict = self.get_stats(notional, self.window)

            for year in stats_dict.keys():
                self.perf_dict[year] = self.perf_dict[year].append(stats_dict[year].T)

        for year in year_list:
            self.perf_dict[year].index = self.BT_dict.keys()


    def get_stats(self, notional, window):
        drawdown_series = self.get_drawdown(notional, window)
        DailyRet_series = notional.pct_change(1)
        DailyVar_series = self.get_daily_var(notional)

        year_list = list(np.unique(notional.index.year))

        stats_dict = {}
        OpenNotional = notional.values[0]
        for year in year_list:
            print(year)

            mdd_list = []
            ret_list = []
            rv_list = []
            SR_list = []

            mdd_list.append(min(drawdown_series[drawdown_series.index.year == year]))

            CloseNotional = notional[notional.index.year == year].values[-1]
            ret_list.append(CloseNotional/OpenNotional - 1)
            OpenNotional = CloseNotional

            # average daily_var * 252
            rv_list.append(np.sqrt(np.mean(DailyVar_series[DailyVar_series.index.year == year])*252))

            SR_list.append(ret_list[-1]/rv_list[-1])

            stats_dict[year] = pd.DataFrame([ret_list, rv_list, SR_list, mdd_list],
                                             index = ['Return', 'Realised_Vol', 'Sharpe_Ratio', 'Max_Drawdown'])

        # stats_df = pd.DataFrame([ret_list, rv_list, SR_list, mdd_list], columns = year_list,
        #                         index = ['Return', 'Realised_vol', 'Return_to_RV', 'Max_Drawdown']).T
        return stats_dict

    def get_drawdown(self, notional, window):
        '''
        :param notional: pd.Series
        :param window: int, days of lookback
        :return: pd.Series
        '''
        dd_list = []
        for DateInd in range(window-1, len(notional)):
            dd_list.append(min(0, min(notional.values[DateInd - window + 1: DateInd + 1])/notional.values[DateInd - window + 1]-1))
        return pd.Series(dd_list, index = notional.index[window-1: len(notional)])

    def get_daily_var(self, notional):
        '''
        (ln(notional return + 1))^2

        :param notional: pd.Series
        :return: pd.Series
        '''
        DailyVar_series = (np.log(notional.pct_change(1)+1))**2
        DailyVar_series = DailyVar_series.fillna(0)
        return DailyVar_series

    def plot_notional(self):
        '''
        Plot notional of each strategy in one picture
        '''
        plt.figure(1)
        plt.figure(figsize=(10, 7.5))
        count = 0
        Colors_list = ['green', 'red', 'blue', 'purple', 'black', 'yellow', 'skyblue', 'orange', 'brown']
        plt.title('Strategy Notional')
        plt.ylabel('Notional')
        plt.xlabel('Date')
        for StratName in self.BT_dict.keys():
            plt.plot(self.BT_dict[StratName].index, self.BT_dict[StratName].strategy_notional,
                     color = Colors_list[count], label = StratName)
            count += 1
        plt.legend()
        plt.show()
















