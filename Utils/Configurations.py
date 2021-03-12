import os


# Paths to different datasets
PATH_DATA = os.path.join(os.path.dirname(__file__), '../data', )

PATH_DATA_TRAIN = {
    'TcrMatch_old': os.path.join(PATH_DATA, 'full_train.csv'),
    'TcrMatch_new': os.path.join(PATH_DATA, 'dl_tcrmatch_train.csv'),
    'IEDB': os.path.join(PATH_DATA, 'dl_iedb_train.csv'),
    'VDJ': os.path.join(PATH_DATA, 'dl_vdj_train.csv'),
    '10x': os.path.join(PATH_DATA, 'dl_10x_train.csv'),
    'All': os.path.join(PATH_DATA, 'distanceLearning_train.csv'),
}
PATH_DATA_VAL = {
    'TcrMatch_old': os.path.join(PATH_DATA, 'full_val.csv'),
    'TcrMatch_new': os.path.join(PATH_DATA, 'dl_tcrmatch_val.csv'),
    'IEDB': os.path.join(PATH_DATA, 'dl_iedb_val.csv'),
    'VDJ': os.path.join(PATH_DATA, 'dl_vdj_val.csv'),
    '10x': os.path.join(PATH_DATA, 'dl_10x_val.csv'),
    'All': os.path.join(PATH_DATA, 'distanceLearning_val.csv'),
}
PATH_DATA_TEST = {
    'TcrMatch_old': os.path.join(PATH_DATA, 'full_test.csv'),
    'TcrMatch_new': os.path.join(PATH_DATA, 'dl_tcrmatch_test.csv'),
    'IEDB': os.path.join(PATH_DATA, 'dl_iedb_test.csv'),
    'VDJ': os.path.join(PATH_DATA, 'dl_vdj_test.csv'),
    '10x': os.path.join(PATH_DATA, 'dl_10x_test.csv'),
    'All': os.path.join(PATH_DATA, 'distanceLearning_test.csv'),
}


PATH_BLOSUM_45 = os.path.join(os.path.dirname(__file__), 'blosum_45.txt')

PATH_MODEL_MULTI_BATCH = os.path.join(os.path.dirname(__file__), '../trained_models/batch_model.h5')
PATH_MODEL_PAIRED = os.path.join(os.path.dirname(__file__), '../trained_models/paired_model.h5')
PATH_MODEL_AE = os.path.join(os.path.dirname(__file__), '../trained_models/ae_model.h5')

PATH_LOGS = os.path.join(os.path.dirname(__file__), '../logs')
