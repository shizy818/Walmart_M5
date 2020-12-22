#!/usr/bin/python

import os
import pandas as pd
import numpy as np
import math
import logging
import datetime
from pathlib import Path

from common import RESULT_DIR, WORK_DIR
from util import merge_by_concat, reduce_mem_usage

class DataPreProcess():
    '''
    A class to process raw dataframe
    '''

    def __init__(self, datawarehouse, config):
        self._DW = datawarehouse
        self.config = config
        self.target = self.config['target']
        self.main_index_list = self.config['main_index_list']

    @property
    def DW(self):
        return self._DW

    def generate_grid_base(self, grid_base_path, holdout_path, end_train_day_x, predict_horizon):
        train_df = self.DW.train_df
        prices_df = self.DW.prices_df
        calendar_df = self.DW.calendar_df

        logging.info('generate_grid_base')
        logging.info('melt')
        index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        grid_df = pd.melt(train_df, id_vars=index_columns, var_name='d', value_name=self.target)

        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id        item_id    dept_id   cat_id store_id state_id       d  sales
        # 0       HOBBIES_1_025_CA_1_evaluation  HOBBIES_1_025  HOBBIES_1  HOBBIES     CA_1       CA     d_1      0
        # 1       HOBBIES_1_052_CA_1_evaluation  HOBBIES_1_052  HOBBIES_1  HOBBIES     CA_1       CA     d_1      0
        # 590057    FOODS_3_194_WI_3_evaluation    FOODS_3_194    FOODS_3    FOODS     WI_3       WI  d_1941      1
        # 590058    FOODS_3_282_WI_3_evaluation    FOODS_3_282    FOODS_3    FOODS     WI_3       WI  d_1941     40

        logging.info('remove days before end_train_day_x / generate holdout')
        num_before = grid_df.shape[0]
        grid_df['d_org'] = grid_df['d']
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        # (1941, 1969]
        holdout_df = grid_df[(grid_df['d'] > end_train_day_x) & \
                             (grid_df['d'] <= end_train_day_x + predict_horizon)][
            self.main_index_list + [self.target]
            ]
        logging.info('holdout_df.shape: {}'.format(holdout_df.shape))
        logging.info('holdout_df - {}'.format(holdout_df.head()))
        holdout_df.to_csv(holdout_path, index=False)

        # grid_df['d'] <= end_train_day_x 
        grid_df = grid_df[grid_df['d'] <= end_train_day_x]
        grid_df['d'] = grid_df['d_org']
        grid_df = grid_df.drop('d_org', axis=1)
        num_after = grid_df.shape[0]
        logging.info('{} --> {}'.format(num_before, num_after))

        logging.info('add test days')
        add_grid = pd.DataFrame()
        for i in range(predict_horizon):
            temp_df = train_df[index_columns]
            temp_df = temp_df.drop_duplicates()
            temp_df['d'] = 'd_' + str(end_train_day_x + i + 1)
            temp_df[self.target] = np.nan
            add_grid = pd.concat([add_grid, temp_df])

        grid_df = pd.concat([grid_df, add_grid])
        grid_df = grid_df.reset_index(drop=True)

        del temp_df, add_grid
        del train_df

        logging.info('convert to category')
        for col in index_columns:
            grid_df[col] = grid_df[col].astype('category')

        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id        item_id    dept_id   cat_id store_id state_id       d  sales
        # 0       HOBBIES_1_025_CA_1_evaluation  HOBBIES_1_025  HOBBIES_1  HOBBIES     CA_1       CA     d_1    0.0
        # 1       HOBBIES_1_052_CA_1_evaluation  HOBBIES_1_052  HOBBIES_1  HOBBIES     CA_1       CA     d_1    0.0
        # 592182    FOODS_2_342_WI_3_evaluation    FOODS_2_342    FOODS_2    FOODS     WI_3       WI  d_1948    NaN
        # 592183    FOODS_3_033_WI_3_evaluation    FOODS_3_033    FOODS_3    FOODS     WI_3       WI  d_1948    NaN

        logging.info('calc release week')
        release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id', 'item_id', 'release']
        #     store_id      item_id  release
        # 0     CA_1  FOODS_1_001    11101
        # 1     CA_1  FOODS_1_002    11101
        # 2     CA_1  FOODS_1_003    11101
        # 3     CA_1  FOODS_1_004    11206
        # 4     CA_1  FOODS_1_005    11101

        grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
        del release_df
        
        grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])
        grid_df = grid_df.reset_index(drop=True)

        logging.info('convert release to int16')
        grid_df['release'] = grid_df['release'] - grid_df['release'].min()
        grid_df['release'] = grid_df['release'].astype(np.int16)

        logging.info('save grid_base')
        grid_df.to_pickle(grid_base_path)

        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id        item_id    dept_id   cat_id store_id state_id       d  sales  release  wm_yr_wk
        # 592182    FOODS_2_342_WI_3_evaluation    FOODS_2_342    FOODS_2    FOODS     WI_3       WI  d_1948    NaN        0     11618
        # 592183    FOODS_3_033_WI_3_evaluation    FOODS_3_033    FOODS_3    FOODS     WI_3       WI  d_1948    NaN      252     11618

        return

    def generate_grid_price(self, grid_base_path, grid_price_path):
        prices_df = self.DW.prices_df
        calendar_df = self.DW.calendar_df

        logging.info('generate_grid_price')
        logging.info('load grid_base')
        grid_df = pd.read_pickle(grid_base_path)

        prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
        prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
        prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
        prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
        prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']
        prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
        prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

        calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
        calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
        prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
        del calendar_prices

        prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
            'sell_price'].transform(lambda x: x.shift(1))
        prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
            'sell_price'].transform('mean')
        prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
            'sell_price'].transform('mean')

        prices_df['sell_price_cent'] = [math.modf(p)[0] for p in prices_df['sell_price']]
        prices_df['price_max_cent'] = [math.modf(p)[0] for p in prices_df['price_max']]
        prices_df['price_min_cent'] = [math.modf(p)[0] for p in prices_df['price_min']]

        del prices_df['month'], prices_df['year']

        # logging.info('prices_df.columns: {}'.format(prices_df.columns))
        # logging.info('prices_df.shape: {}'.format(prices_df.shape))
        # logging.info('prices_df - {}'.format(prices_df.head(-5)))
        #        store_id        item_id  wm_yr_wk  sell_price  price_max  ...  price_momentum_m  price_momentum_y  sell_price_cent  price_max_cent  price_min_cent
        # 0          CA_1  HOBBIES_1_002     11121        3.97       3.97  ...               1.0               1.0             0.97            0.97            0.97
        # 1          CA_1  HOBBIES_1_002     11122        3.97       3.97  ...               1.0               1.0             0.97            0.97            0.97
        # 656812     WI_3    FOODS_3_827     11615        1.00       1.00  ...               1.0               1.0             0.00            0.00            0.00
        # 656813     WI_3    FOODS_3_827     11616        1.00       1.00  ...               1.0               1.0             0.00            0.00            0.00

        logging.info('merge prices')
        original_columns = list(grid_df)
        grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        keep_columns = [col for col in list(grid_df) if col not in original_columns]
        grid_df = grid_df[self.main_index_list + keep_columns]
        grid_df = reduce_mem_usage(grid_df)

        # logging.info('grid_df.columns: {}'.format(grid_df.columns))
        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id       d  sell_price  price_max  ...  price_momentum_y  sell_price_cent  price_max_cent  price_min_cent
        # 0       HOBBIES_1_025_CA_1_evaluation     d_1         NaN        NaN  ...               NaN              NaN             NaN             NaN
        # 1       HOBBIES_1_052_CA_1_evaluation     d_1         NaN        NaN  ...               NaN              NaN             NaN             NaN
        # 592185    FOODS_3_194_WI_3_evaluation  d_1948         NaN        NaN  ...               NaN              NaN             NaN             NaN
        # 592186    FOODS_3_282_WI_3_evaluation  d_1948         NaN        NaN  ...               NaN              NaN             NaN             NaN

        logging.info('save grid_price')
        grid_df.to_pickle(grid_price_path)
        del prices_df
        return

    def generate_grid_calendar(self, grid_base_path, grid_calendar_path):
        calendar_df = self.DW.calendar_df

        logging.info('generate_grid_calendar')
        grid_df = pd.read_pickle(grid_base_path)

        grid_df = grid_df[self.main_index_list]

        import math, decimal
        dec = decimal.Decimal

        def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
            diff = datetime.datetime.strptime(d, '%Y-%m-%d') - datetime.datetime(2001, 1, 1)
            days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
            lunations = dec("0.20439731") + (days * dec("0.03386319269"))
            phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
            return int(phase_index) & 7

        calendar_df['moon'] = calendar_df.date.apply(get_moon_phase)

        # Merge calendar partly
        icols = ['date',
                 'd',
                 'event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI',
                 'moon',
                 ]

        grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

        icols = ['event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']
        for col in icols:
            grid_df[col] = grid_df[col].astype('category')

        grid_df['date'] = pd.to_datetime(grid_df['date'])

        grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
        grid_df['tm_w'] = grid_df['date'].dt.isocalendar().week.astype(np.int8)
        grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['tm_y'] = grid_df['date'].dt.year
        grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
        grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: math.ceil(x / 7)).astype(np.int8)

        grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
        grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)
        del grid_df['date']

        # logging.info('grid_df.columns: {}'.format(grid_df.columns))
        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id       d event_name_1 event_type_1 event_name_2 event_type_2 snap_CA snap_TX snap_WI  moon  tm_d  tm_w  tm_m  tm_y  tm_wm  tm_dw  tm_w_end
        # 0       HOBBIES_1_025_CA_1_evaluation     d_1          NaN          NaN          NaN          NaN       0       0       0     7    29     4     1     0      5      5         1
        # 1       HOBBIES_1_052_CA_1_evaluation     d_1          NaN          NaN          NaN          NaN       0       0       0     7    29     4     1     0      5      5         1
        # 592185    FOODS_3_194_WI_3_evaluation  d_1948          NaN          NaN          NaN          NaN       0       0       0     6    29    21     5     5      5      6         1
        # 592186    FOODS_3_282_WI_3_evaluation  d_1948          NaN          NaN          NaN          NaN       0       0       0     6    29    21     5     5      5      6         1

        grid_df.to_pickle(grid_calendar_path)

        del calendar_df
        del grid_df

        return

    def modify_grid_base(self, grid_base_path):
        logging.info('modify_grid_base')
        grid_df = pd.read_pickle(grid_base_path)
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        del grid_df['wm_yr_wk']

        logging.info('grid_df.shape: {}'.format(grid_df.shape))
        logging.info('grid_df - {}'.format(grid_df.head(-5)))

        grid_df.to_pickle(grid_base_path)

        del grid_df
        return

    def generate_grid_full(self, grid_base_path, grid_price_path, grid_calendar_path, holdout_path, end_train_day_x, predict_horizon):
        self.generate_grid_base(grid_base_path, holdout_path, end_train_day_x, predict_horizon)
        self.generate_grid_price(grid_base_path, grid_price_path)
        self.generate_grid_calendar(grid_base_path, grid_calendar_path)
        self.modify_grid_base(grid_base_path)
        return

    def process(self):
        end_train_day_x_list = self.config['fold_id_list_csv']
        prediction_horizon_list = self.config['prediction_horizon_list_csv']
        logging.info('end_train_day_x_list - {}'.format(end_train_day_x_list))
        logging.info('prediction_horizon_list - {}'.format(prediction_horizon_list))
            
        for end_train_day_x in end_train_day_x_list:
            WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
            Path(WORK_DIR_DAY_X).mkdir(parents=True, exist_ok=True)
            RESULT_DIR_DAY_X = os.path.join(RESULT_DIR, str(end_train_day_x))
            Path(RESULT_DIR_DAY_X).mkdir(parents=True, exist_ok=True)
            holdout_path = os.path.join(RESULT_DIR_DAY_X, f'holdout.csv')

            for predict_horizon in prediction_horizon_list:
                logging.info('----------------- fold_id {}, predict_horizon {}'.format(end_train_day_x, predict_horizon))

                grid_base_path = os.path.join(WORK_DIR_DAY_X, f'grid_base_{predict_horizon}.pkl')
                grid_price_path = os.path.join(WORK_DIR_DAY_X, f'grid_price_{predict_horizon}.pkl')
                grid_calendar_path = os.path.join(WORK_DIR_DAY_X, f'grid_calendar_{predict_horizon}.pkl')
                
                self.generate_grid_full(grid_base_path, grid_price_path, grid_calendar_path, holdout_path, end_train_day_x, predict_horizon)
