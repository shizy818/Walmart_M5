import os
import logging
import pandas as pd
import numpy as np
import time

from util import set_seed
from pathlib import Path
from common import WORK_DIR

class FeatureGenerator(object):
    '''
    A class to extract feature from data.
    '''
    def __init__(self, config):
        self.config = config
        self.target = self.config['target']
        self.num_lag_day = 15

    def generate_lag_feature(self, grid_base_path, lag_feature_path, predict_horizon):
        logging.info('generate_lag')
        logging.info('load gird_base')
        grid_df = pd.read_pickle(grid_base_path)

        grid_df = grid_df[['id', 'd', 'sales']]

        start_time = time.time()
        logging.info('create lags')

        num_lag_day_list = [*range(predict_horizon, predict_horizon + self.num_lag_day)]
        grid_df = grid_df.assign(**{
            '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
            for l in num_lag_day_list
            for col in [self.target]
        })

        for col in list(grid_df):
            if 'lag' in col:
                grid_df[col] = grid_df[col].astype(np.float16)

        start_time = time.time()
        logging.info('create rolling aggs')

        for num_rolling_day in self.config['num_rolling_day_list']:
            logging.info('rolling period {}'.format(num_rolling_day))
            grid_df['rolling_mean_' + str(num_rolling_day)] = grid_df.groupby(['id'])[self.target].transform(
                lambda x: x.shift(predict_horizon).rolling(num_rolling_day).mean()).astype(np.float16)
            grid_df['rolling_std_' + str(num_rolling_day)] = grid_df.groupby(['id'])[self.target].transform(
                lambda x: x.shift(predict_horizon).rolling(num_rolling_day).std()).astype(np.float16)

        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id     d  ...  rolling_mean_180  rolling_std_180
        # 592182    FOODS_2_357_WI_3_evaluation  1948  ...          4.128906         2.826172
        # 592183    FOODS_2_365_WI_3_evaluation  1948  ...          0.444336         0.726562

        logging.info('save lag_feature')
        grid_df.to_pickle(lag_feature_path)

        return

    def generate_target_encoding_feature(self, grid_base_path, target_encoding_feature_path, end_train_day_x, predict_horizon):
        set_seed(self.config['seed'])

        grid_df = pd.read_pickle(grid_base_path)
        grid_df[self.target][
            grid_df['d'] > (end_train_day_x - predict_horizon)] = np.nan
        base_cols = list(grid_df)

        icols = [
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        ]

        for col in icols:
            logging.info('encoding {}'.format(col))
            col_name = '_' + '_'.join(col) + '_'
            grid_df['enc' + col_name + 'mean'] = grid_df.groupby(col)[self.target].transform('mean').astype(
                np.float16)
            grid_df['enc' + col_name + 'std'] = grid_df.groupby(col)[self.target].transform('std').astype(
                np.float16)

        keep_cols = [col for col in list(grid_df) if col not in base_cols]
        grid_df = grid_df[['id', 'd'] + keep_cols]

        # logging.info('grid_df.shape: {}'.format(grid_df.shape))
        # logging.info('grid_df - {}'.format(grid_df.head(-5)))
        #                                    id  ...  enc_item_id_store_id_std
        # 0       HOBBIES_1_051_CA_1_evaluation  ...                  0.400146
        # 1       HOBBIES_1_102_CA_1_evaluation  ...                  0.354492

        logging.info('save target_encoding_feature')
        grid_df.to_pickle(target_encoding_feature_path)
        return

    def process(self):
        end_train_day_x_list = self.config['fold_id_list_csv']
        prediction_horizon_list = self.config['prediction_horizon_list_csv']
        logging.info('end_train_day_x_list - {}'.format(end_train_day_x_list))
        logging.info('prediction_horizon_list - {}'.format(prediction_horizon_list))
            
        for end_train_day_x in end_train_day_x_list:
            WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
            # Path(WORK_DIR_DAY_X).mkdir(parents=True, exist_ok=True)

            for predict_horizon in prediction_horizon_list:
                logging.info('----------------- fold_id {}, predict_horizon {}'.format(end_train_day_x, predict_horizon))

                grid_base_path = os.path.join(WORK_DIR_DAY_X, f'grid_base_{predict_horizon}.pkl')
                lag_feature_path = os.path.join(WORK_DIR_DAY_X, f'lag_feature_{predict_horizon}.pkl')
                target_encoding_feature_path = os.path.join(WORK_DIR_DAY_X, f'target_encoding_{predict_horizon}.pkl')

                self.generate_lag_feature(grid_base_path, lag_feature_path, predict_horizon)
                self.generate_target_encoding_feature(grid_base_path, target_encoding_feature_path, end_train_day_x, predict_horizon)