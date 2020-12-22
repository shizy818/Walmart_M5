# load infrastructure
from DataWarehouse import DataWarehouse
from DataPreProcess import DataPreProcess
from FeatureGenerator import FeatureGenerator
from Model import Model
# from Validator import Validator
import os

# load config
from common import set_logging_config, get_model_config
from common import WORK_DIR

# set logging config
set_logging_config()

from absl import app, flags
 
FLAGS = flags.FLAGS
flags.DEFINE_boolean("pre_process", False, "Data Proprocessing")
flags.DEFINE_boolean("gen_features", False, "Generate features")
flags.DEFINE_boolean("train", False, "Train and predict")

def main(argv):
    del argv

    # read in configuration
    config = get_model_config('config/basic.yml')

    # read in data
    MyData = DataWarehouse(config)
    MyData.read_data()

    # preprocessing
    if FLAGS.pre_process:
        MyPreprocess = DataPreProcess(MyData, config)
        MyPreprocess.process()

    # generate feature
    if FLAGS.gen_features:
        MyFeature = FeatureGenerator(config)
        MyFeature.process()

    # train & predict
    if FLAGS.train:
        model = Model(MyData, config)

        #---------Generating outputs ---------------
        model.process()
        # ypred = model.predict()
        # MyData.generate_submission(ypred)

    try:
        # logging.info('clear work_dir')
        shutil.rmtree(WORK_DIR)
    except Exception as e:
        pass

if __name__ == "__main__":
    app.run(main) 