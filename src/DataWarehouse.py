import os
import pandas as pd
import numpy as np
import logging

# customized setting
from common import DATA_TRAIN, DATA_PRICES, DATA_CALENDAR, DATA_SAMPLE_SUBMISSION
from common import SUBMISSION_FOLDER

class DataWarehouse():
    """A class to handle data IO
    """

    def __init__(self, config):
        self.sampling_rate = config['sampling_rate']

    def read_data(self):
        self.train_df = pd.read_csv(DATA_TRAIN)
        self.prices_df = pd.read_csv(DATA_PRICES)
        self.calendar_df = pd.read_csv(DATA_CALENDAR)
        self.submission_df = pd.read_csv(DATA_SAMPLE_SUBMISSION)

        # 抽样
        logging.info('sampling_rate: {}'.format(self.sampling_rate))
        if self.sampling_rate < 1.0:
            logging.info('sampling start')
            id_list = self.train_df['id'].unique().tolist()
            item_id_store_id_list = [
                (id.replace('_evaluation', '')[:-5],
                 id.replace('_evaluation', '')[-4:])
                for id in id_list
            ]
            train_sampled_df = pd.DataFrame()
            prices_sampled_df = pd.DataFrame()
            submission_sampled_df = pd.DataFrame()
            num_samples = int(len(item_id_store_id_list) * self.sampling_rate)
            logging.info('#samples: {}'.format(num_samples))

            sample_id_header_list = []
            sample_store_id_list = []
            sample_item_id_list = []
            for sample_index in sorted(np.random.permutation(len(item_id_store_id_list))[:num_samples]):
                sample_item_id = item_id_store_id_list[sample_index][0]
                sample_store_id = item_id_store_id_list[sample_index][1]
                sample_id_header = f'{sample_item_id}_{sample_store_id}'
                logging.debug('sample_index: {}, sample_id_header: {}'.format(sample_index, sample_id_header))
                sample_id_header_list.append(sample_id_header)
                sample_store_id_list.append(sample_store_id)
                sample_item_id_list.append(sample_item_id)
            
            train_sampled_df = pd.concat(
                [train_sampled_df,
                 self.train_df[self.train_df['id'].str.contains('|'.join(sample_id_header_list))]])

            prices_sampled_df = pd.concat(
                [prices_sampled_df,
                 self.prices_df[(self.prices_df['store_id'].str.contains('|'.join(sample_store_id_list))) & \
                           (self.prices_df['item_id'].str.contains('|'.join(sample_item_id_list)))]])

            submission_sampled_df = pd.concat(
                [submission_sampled_df,
                 self.submission_df[self.submission_df['id'].str.contains('|'.join(sample_id_header_list))]])

            # 重置索引
            self.train_df = train_sampled_df.reset_index(drop=True)
            self.prices_df = prices_sampled_df.reset_index(drop=True)
            self.submission_df = submission_sampled_df.reset_index(drop=True)

            logging.info('sampling end')
        
        logging.info('train_df.shape: {}'.format(self.train_df.shape))
        logging.info('train_df - {}'.format(self.train_df.head()))
        #                               id        item_id  ... d_1940 d_1941
        # 0  HOBBIES_1_052_CA_1_evaluation  HOBBIES_1_052  ...      0      0
        # 1  HOBBIES_1_060_CA_1_evaluation  HOBBIES_1_060  ...      0      0

        logging.info('prices_df.shape: {}'.format(self.prices_df.shape))
        logging.info('prices_df - {}'.format(self.prices_df.head()))
        #   store_id        item_id  wm_yr_wk  sell_price
        # 0     CA_1  HOBBIES_1_004     11106        4.34
        # 1     CA_1  HOBBIES_1_004     11107        4.34

        logging.info('calendar_df.shape: {}'.format(self.calendar_df.shape))
        logging.info('calendar_df - {}'.format(self.calendar_df.head()))
        #          date  wm_yr_wk    weekday  ...  snap_CA  snap_TX  snap_WI
        # 0  2011-01-29     11101   Saturday  ...        0        0        0
        # 1  2011-01-30     11101     Sunday  ...        0        0        0


    def generate_submission(self, ypred):
        """
        Generate submission given predicted output
        :param ypred: predicted output corresponding to test_id
        :return: none
        """
        out_df = pd.DataFrame(ypred)
        # out_df.columns = [DATA_OUT_FEATURE]
        # out_df[SUBMIT_ID] = self.test_id
        out_df.to_csv(os.path.join(SUBMISSION_FOLDER, "submission.csv"), index=False)