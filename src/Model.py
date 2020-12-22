import logging
import gc
import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from util import set_seed

from pathlib import Path
from common import MODEL_DIR, WORK_DIR, RESULT_DIR, DATA_SAMPLE_SUBMISSION
from WrmsseEvaluator import WrmsseEvaluator

class Model():
    def __init__(self, datawarehouse, config):
        self._DW = datawarehouse
        self.store_id_set_list = list(datawarehouse.train_df['store_id'].unique())
        self.config = config
        self.target = self.config['target']
        self.main_index_list = self.config['main_index_list']
        self.lgb_params = self.config['lgb_params']
        self.mean_features = self.config['mean_features']
        self.remove_features = self.config['remove_features']
        self.start_train_day_x = 1
        self.end_train_day_default = 1941
        self.predict_horizon_default = 28

    @property
    def DW(self):
        return self._DW

    def load_grid_full(self, end_train_day_x, predict_horizon):
        logging.info('load_grid_full')
        WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
        grid_base_path = os.path.join(WORK_DIR_DAY_X, f'grid_base_{predict_horizon}.pkl')
        grid_price_path = os.path.join(WORK_DIR_DAY_X, f'grid_price_{predict_horizon}.pkl')
        grid_calendar_path = os.path.join(WORK_DIR_DAY_X, f'grid_calendar_{predict_horizon}.pkl')

        grid_df = pd.concat([pd.read_pickle(grid_base_path),
                             pd.read_pickle(grid_price_path).iloc[:, 2:],
                             pd.read_pickle(grid_calendar_path).iloc[:, 2:]],
                            axis=1)
        return grid_df

    def load_grid_by_store(self, df, store_id, target_encoding_feature_path, lag_feature_path):
        if store_id != 'all':
            df1 = df[df['store_id'] == store_id]

        df2 = pd.read_pickle(target_encoding_feature_path)[self.mean_features]
        df2 = df2[df2.index.isin(df1.index)]

        df3 = pd.read_pickle(lag_feature_path).iloc[:, 3:]
        df3 = df3[df3.index.isin(df1.index)]

        df1 = pd.concat([df1, df2], axis=1)
        del df2

        df1 = pd.concat([df1, df3], axis=1)
        del df3

        enable_features = [col for col in list(df1) if col not in self.remove_features]
        df1 = df1[['id', 'd', self.target] + enable_features]

        df1 = df1[df1['d'] >= self.start_train_day_x].reset_index(drop=True)

        return df1, enable_features

    def load_base_test(self, end_train_day_x, predict_horizon):
        WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
        base_test = pd.DataFrame()

        for store_id in self.store_id_set_list:
            temp_df = pd.read_pickle(
                os.path.join(WORK_DIR_DAY_X, f'test_{store_id}_{predict_horizon}.pkl'))
            temp_df['store_id'] = store_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
        return base_test

    def train(self, end_train_day_x, predict_horizon):
        RESULT_DIR_DAY_X = os.path.join(RESULT_DIR, str(end_train_day_x))
        WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
        MODEL_DIR_DAY_X = os.path.join(MODEL_DIR, str(end_train_day_x))
        Path(MODEL_DIR_DAY_X).mkdir(parents=True, exist_ok=True)
        
        lag_feature_path = os.path.join(WORK_DIR_DAY_X, f'lag_feature_{predict_horizon}.pkl')
        target_encoding_feature_path = os.path.join(WORK_DIR_DAY_X, f'target_encoding_{predict_horizon}.pkl')

        prediction_horizon_list = self.config['prediction_horizon_list_csv']

        set_seed(self.lgb_params['seed'])
        df = self.load_grid_full(end_train_day_x, predict_horizon)

        feature_importance_all_df = pd.DataFrame()
        for store_index, store_id in enumerate(self.store_id_set_list):
            logging.info('train {}'.format(store_id))

            grid_df, enable_features = self.load_grid_by_store(
                df, store_id, target_encoding_feature_path, lag_feature_path)
            self.enable_features = enable_features

            train_mask = grid_df['d'] <= end_train_day_x
            valid_mask = train_mask & (grid_df['d'] > (end_train_day_x - predict_horizon))
            preds_mask = grid_df['d'] > (end_train_day_x - 100)

            logging.info('[{3} - {4}] train {0}/{1} {2}'.format(
                store_index + 1, len(self.store_id_set_list), store_id, end_train_day_x, predict_horizon))

            if self.config['export_all_flag']:
                logging.info('export train')
                grid_df[train_mask].to_csv(os.path.join(RESULT_DIR_DAY_X, f'exp_train_{store_id}.csv'), index=False)

            train_data = lgb.Dataset(grid_df[train_mask][enable_features],
                                     label=grid_df[train_mask][self.target])

            if self.config['export_all_flag']:
                logging.info('export valid')
                grid_df[valid_mask].to_csv(os.path.join(RESULT_DIR_DAY_X, f'exp_valid_{store_id}.csv'), index=False)

            valid_data = lgb.Dataset(grid_df[valid_mask][enable_features],
                                     label=grid_df[valid_mask][self.target])

            if self.config['export_all_flag']:
                logging.info('export test')
                grid_df[preds_mask].to_csv(os.path.join(RESULT_DIR_DAY_X, f'exp_test_{store_id}.csv'), index=False)

            if self.config['export_all_flag']:
                logging.info('export train_valid_test')
                grid_df[train_mask | valid_mask | preds_mask].to_csv(
                    os.path.join(RESULT_DIR_DAY_X, f'exp_train_valid_test_{store_id}.csv'), index=False)

            # Saving part of the dataset for later predictions
            # Removing features that we need to calculate recursively
            grid_df = grid_df[preds_mask].reset_index(drop=True)
            grid_df.to_pickle(os.path.join(WORK_DIR_DAY_X, f'test_{store_id}_{predict_horizon}.pkl'))
            del grid_df

            set_seed(self.lgb_params['seed'])
            estimator = lgb.train(self.lgb_params,
                                  train_data,
                                  valid_sets=[valid_data],
                                  verbose_eval=False,
                                  # callbacks=[self.log.log_evaluation(period=100)],
                                  )

            # feature importance fore store
            feature_importance_store_df = pd.DataFrame(sorted(zip(enable_features, estimator.feature_importance())),
                                                       columns=['feature_name', 'importance'])
            feature_importance_store_df = feature_importance_store_df.sort_values('importance', ascending=False)
            feature_importance_store_df['store_id'] = store_id
            feature_importance_store_df.to_csv(
                os.path.join(RESULT_DIR_DAY_X, f'feature_importance_{store_id}_{predict_horizon}.csv'), index=False)
            feature_importance_all_df = pd.concat([feature_importance_all_df, feature_importance_store_df])
            
            model_name = os.path.join(MODEL_DIR_DAY_X, f'lgb_model_{store_id}_{predict_horizon}.bin')
            pickle.dump(estimator, open(model_name, 'wb'))

            del train_data, valid_data, estimator
            gc.collect()

        feature_importance_all_df.to_csv(
            os.path.join(RESULT_DIR_DAY_X, f'feature_importance_all_{predict_horizon}.csv'), index=False)
        
    def predict(self, end_train_day_x, predict_horizon, predict_horizon_prev):
        RESULT_DIR_DAY_X = os.path.join(RESULT_DIR, str(end_train_day_x))
        WORK_DIR_DAY_X = os.path.join(WORK_DIR, str(end_train_day_x))
        MODEL_DIR_DAY_X = os.path.join(MODEL_DIR, str(end_train_day_x))

        prediction_horizon_list = self.config['prediction_horizon_list_csv']

        logging.info('aggregate feature importance')
        feature_importance_all_df = pd.read_csv(
            os.path.join(RESULT_DIR_DAY_X, f'feature_importance_all_{predict_horizon}.csv'))
        feature_importance_agg_df = feature_importance_all_df.groupby(
            'feature_name')['importance'].agg(['mean', 'std']).reset_index()
        feature_importance_agg_df.columns = ['feature_name', 'importance_mean', 'importance_std']
        feature_importance_agg_df = feature_importance_agg_df.sort_values('importance_mean', ascending=False)
        feature_importance_agg_df.to_csv(
            os.path.join(RESULT_DIR_DAY_X, f'feature_importance_agg_{predict_horizon}.csv'), index=False)

        logging.info('load base_test')
        base_test = self.load_base_test(end_train_day_x, predict_horizon)
        if self.config['export_all_flag']:
            base_test.to_csv(
                os.path.join(RESULT_DIR_DAY_X, f'exp_base_test_{predict_horizon}_a.csv'), index=False)
        
        if predict_horizon_prev > 0:
            pred_v_prev_df = None
            for ph in prediction_horizon_list:
                if ph <= predict_horizon_prev:
                    pred_v_temp_df = pd.read_csv(os.path.join(RESULT_DIR_DAY_X, f'pred_v_{ph}.csv'))
                    pred_v_prev_df = pd.concat([pred_v_prev_df, pred_v_temp_df])
            for predict_day in range(1, predict_horizon_prev + 1):
                base_test[self.target][base_test['d'] == (end_train_day_x + predict_day)] = \
                    pred_v_prev_df[self.target][
                        pred_v_prev_df['d'] == (end_train_day_x + predict_day)].values
        
        if self.config['export_all_flag']:
            base_test.to_csv(
                os.path.join(RESULT_DIR_DAY_X, f'exp_base_test_{predict_horizon}_b.csv'), index=False)

        # main_time = time.time()
        pred_h_df = pd.DataFrame()
        for predict_day in range(predict_horizon_prev + 1, predict_horizon + 1):
            logging.info('predict day{:02d}'.format(predict_day))
            # start_time = time.time()
            grid_df = base_test.copy()

            day_mask = base_test['d'] == (end_train_day_x + predict_day)
            for store_index, store_id in enumerate(self.store_id_set_list):
                logging.info('[{3} - {4}] predict {0}/{1} {2} day {5}'.format(
                    store_index + 1, len(self.store_id_set_list), store_id, end_train_day_x, predict_horizon, predict_day))

                model_path = str(
                    os.path.join(MODEL_DIR_DAY_X, f'lgb_model_{store_id}_{predict_horizon}.bin'))

                estimator = pickle.load(open(model_path, 'rb'))
                if store_id != 'all':
                    store_mask = base_test['store_id'] == store_id
                    mask = (day_mask) & (store_mask)
                else:
                    mask = day_mask

                if self.config['export_all_flag']:
                    logging.info('export pred')
                    grid_df[mask].to_csv(
                        os.path.join(RESULT_DIR_DAY_X, f'exp_pred_{store_id}_day_{predict_day}.csv'), index=False)
                base_test[self.target][mask] = estimator.predict(grid_df[mask][self.enable_features])

            temp_df = base_test[day_mask][['id', self.target]]
            temp_df.columns = ['id', 'F' + str(predict_day)]
            if 'id' in list(pred_h_df):
                pred_h_df = pred_h_df.merge(temp_df, on=['id'], how='left')
            else:
                pred_h_df = temp_df.copy()

            del temp_df
        
        if self.config['export_all_flag']:
            base_test.to_csv(
                os.path.join(RESULT_DIR_DAY_X, f'exp_base_test_{predict_horizon}_c.csv'), index=False)
        
        pred_h_df.to_csv(
            os.path.join(RESULT_DIR_DAY_X, f'pred_h_{predict_horizon}.csv'), index=False)

        pred_v_df = base_test[
            (base_test['d'] >= end_train_day_x + predict_horizon_prev + 1) *
            (base_test['d'] < end_train_day_x + predict_horizon + 1)
            ][
            self.main_index_list + [self.target]
            ]
        pred_v_df.to_csv(os.path.join(RESULT_DIR_DAY_X, f'pred_v_{predict_horizon}.csv'),
                         index=False)

        return pred_h_df, pred_v_df

    def calc_wrmsse(self, end_train_day_x, all_preds):
        logging.info('calc wrmsse')
        temp_df = self.DW.train_df
        logging.info('adjust end of train period')
        num_before = self.DW.train_df.shape
        num_diff_days = self.end_train_day_default - end_train_day_x - self.predict_horizon_default
        if num_diff_days > 0:
            temp_df = self.DW.train_df.iloc[:, :-1 * num_diff_days]
        num_after = temp_df.shape
        logging.info(f'{num_before} --> {num_after}')

        train_fold_df = temp_df.iloc[:, :-28]
        valid_fold_df = temp_df.iloc[:, -28:].copy()

        valid_preds = self.DW.submission_df[self.DW.submission_df['id'].str.contains('evaluation')][['id']]
        valid_preds = valid_preds.merge(all_preds, on=['id'], how='left').fillna(0)
        valid_preds = valid_preds.drop('id', axis=1)
        valid_preds.columns = valid_fold_df.columns
        
        # train_fold_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'eval_wrmsse_train.csv'), index=False)
        # valid_fold_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'eval_wrmsse_test.csv'), index=False)
        # valid_preds.to_csv(os.path.join(RESULT_DIR_DAY_X, 'eval_wrmsse_pred.csv'), index=False)

        evaluator = WrmsseEvaluator(train_fold_df, valid_fold_df, self.DW.calendar_df, self.DW.prices_df)
        wrmsse = evaluator.score(valid_preds)
        logging.info(f'wrmsse {wrmsse}')

        return wrmsse

    def process(self):
        end_train_day_x_list = self.config['fold_id_list_csv']
        prediction_horizon_list = self.config['prediction_horizon_list_csv']
        logging.info('end_train_day_x_list - {}'.format(end_train_day_x_list))
        logging.info('prediction_horizon_list - {}'.format(prediction_horizon_list))
        
        result_summary_all_df = pd.DataFrame()
        for end_train_day_x in end_train_day_x_list:
            RESULT_DIR_DAY_X = os.path.join(RESULT_DIR, str(end_train_day_x))
            Path(RESULT_DIR_DAY_X).mkdir(parents=True, exist_ok=True)

            pred_h_all_df = pd.DataFrame()
            pred_v_all_df = pd.DataFrame()
            predict_horizon_prev = 0
            for predict_horizon in prediction_horizon_list:
                logging.info('----------------- fold_id {}, predict_horizon {}'.format(end_train_day_x, predict_horizon))

                # 训练
                self.train(end_train_day_x, predict_horizon)

                # 预测
                pred_h_df, pred_v_df = self.predict(end_train_day_x, predict_horizon, predict_horizon_prev)
                if pred_h_all_df.shape[1] == 0:
                    pred_h_all_df = pred_h_df
                else:
                    pred_h_all_df = pred_h_all_df.merge(pred_h_df, on='id')
                pred_v_all_df = pd.concat([pred_v_all_df, pred_v_df], axis=0)

                #  更新predict_horizon_prev
                predict_horizon_prev = predict_horizon

            pred_h_all_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'pred_h_all.csv'), index=False)
            pred_v_all_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'pred_v_all.csv'), index=False)

            holdout_df = pd.read_csv(os.path.join(RESULT_DIR_DAY_X, f'holdout.csv'))
            logging.info('holdout_df.shape {}'.format(holdout_df.shape))
            logging.info('pred_v_all_df.shape {}'.format(pred_v_all_df.shape))

            if holdout_df.shape[0] == 0:
                logging.info('no holdout')
                logging.info('generate submission')
                pred_h_all_df = pred_h_all_df.reset_index(drop=True)
                submission = pd.read_csv(DATA_SAMPLE_SUBMISSION)[['id']]
                submission = submission.merge(pred_h_all_df, on=['id'], how='left').fillna(0)
                submission.to_csv(os.path.join(RESULT_DIR_DAY_X, 'submission.csv'), index=False)
                result_summary_df = None
            else:
                logging.info('calc metrics')
                result_df = holdout_df.merge(pred_v_all_df, on=['id', 'd'], how='inner')
                result_df.columns = ['id', 'd', 'y_test', 'y_pred']
                logging.info('result_df.shape'.format(pred_v_all_df.shape))
                result_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'result.csv'), index=False)

                wrmsse = self.calc_wrmsse(end_train_day_x, pred_h_all_df)
                rmse = np.sqrt(mean_squared_error(result_df['y_test'], result_df['y_pred']))

                result_summary_df = pd.DataFrame(
                    [
                        [end_train_day_x, 'wrmsse', wrmsse],
                        [end_train_day_x, 'rmse', rmse],
                    ],
                    columns=['fold_id', 'metric_name', 'metric_value'])
                logging.info(result_summary_df)
                result_summary_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'result_summary.csv'), index=False)
                result_summary_all_df = pd.concat([result_summary_all_df, result_summary_df])
        
        if result_summary_all_df.shape[0] == 0:
            pass
        else:
            logging.info(result_summary_all_df)
            logging.info(result_summary_all_df.groupby('metric_name')['metric_value'].agg(['mean', 'median']))
            result_summary_all_df.to_csv(os.path.join(RESULT_DIR_DAY_X, 'result_summary_all.csv'), index=False)
