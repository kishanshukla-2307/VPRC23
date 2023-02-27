from trainer import Trainer

TRAIN_CSV_PATH = '../data/train.csv'
TEST_CSV_PATH = '../data/test.csv'
IMG_PATH = '../data/train/'
TEST_IMG_PATH = '/content/test/'
CHECKPOINT_PATH = '/content/drive/MyDrive/model_checkpoint'
MAP_SCORES_PATH = '/content/drive/MyDrive/'
MODEL_NAME = 'mobilenet'
NUM_CLASSES = 9691   ## 361
EMB_DIM = 768
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.0001
MODEL_PATH = '/content/drive/MyDrive/model_checkpoint/model_triplet_c&g_LR_imbalance_handled_0.0001_0002.pth'

trainer = Trainer(MODEL_NAME, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LR, 
    TRAIN_CSV_PATH, TEST_CSV_PATH, CHECKPOINT_PATH, IMG_PATH, TEST_IMG_PATH, MAP_SCORES_PATH, EMB_DIM)

trainer.run_training_loop()