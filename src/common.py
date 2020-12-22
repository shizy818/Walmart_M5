# training/testing file path
import os, logging, yaml
from pathlib import Path

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SRC_PATH, '../data/')
# DATA_DIR="/Users/shizy/kaggle/Walmart_M5_Forecasting/data"

# dirs
RESULT_DIR = os.path.join(DATA_DIR, 'output/result/')
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
WORK_DIR = os.path.join(DATA_DIR, 'output/work/')
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = os.path.join(DATA_DIR, 'output/model/')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# define data csv files
DATA_TRAIN = os.path.join(DATA_DIR, 'sales_train_evaluation.csv')
DATA_PRICES = os.path.join(DATA_DIR, 'sell_prices.csv')
DATA_CALENDAR = os.path.join(DATA_DIR, 'calendar.csv')
DATA_SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'sample_submission.csv')
# DATA_TEST = os.path.join(DATA_DIR, 'test.csv')

# define submission
SUBMISSION_FOLDER = os.path.join(SRC_PATH, '../submission/')
SUBMIT_ID = 'PassengerId'

# logging and debug setting
LOG_FILE = 'record.log'
IS_DEBUG_MODE = 0

def set_logging_config():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=LOG_FILE,
                        filemode='w')

    # output logging info to screen as well (from python official website)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# function for parsing yml file
def get_model_config(config_file, loader=yaml.Loader):
    # read in config for related models
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=loader)
    return config

if __name__ == "__main__":
    # read in configuration
    config = get_model_config('config/basic.yml')
    print(config)