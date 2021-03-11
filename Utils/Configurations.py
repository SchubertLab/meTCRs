import os


# Paths to different datasets
PATH_DATA = os.path.join(os.path.dirname(__file__), '../data', )
PATH_DATA_TRAIN = os.path.join(PATH_DATA, 'distanceLearning_train.csv')
PATH_DATA_VAL = os.path.join(PATH_DATA, 'full_val.csv')  # 'dl_vdj_val.csv')
PATH_DATA_TEST = os.path.join(PATH_DATA, 'dl_tcrmatch_test.csv')  # dl_vdj_test.csv')

PATH_BLOSUM_45 = os.path.join(os.path.dirname(__file__), 'blosum_45.txt')
