# config.py - Configuration parameters

#data parms
TEST_SIZE = 0.2
RANDOM_SEED = 42

#model parm
N_ESTIMATORS = 100
MAX_DEPTH = 10
BATCH_SIZE = 1000

# augmentation parms
AUGMENTATION_RATE = 0.4  #generate anx % additional samples
BIAS_THRESHOLD = 0.1  #minimum bias difference to trigger augmentation