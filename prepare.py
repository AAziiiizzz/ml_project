# prepare.py
# small script used by Makefile to call prepare_data (it does not persist anything)
from model_pipeline import prepare_data
TRAIN = "data/train_u6lujuX_CVtuZ9i.csv"
TEST = "data/test_Y3wMUE5_7gLdaTN.csv"
_ = prepare_data(TRAIN, TEST)
print("prepare_data executed successfully.")
