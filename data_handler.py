import string
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

class DataHandler():
    def __init__(self, train_csv_path: string,
                 test_csv_path: string):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.train = None
        self.test1 = None
        self.test2 = None
        self.validation = None
        self.gallery_for_val = None
        self.class_to_cat = None
        self.transform = None
    
    def load_data(self):
        self.train = pd.read_csv(self.train_csv_path)
        self.test = pd.read_csv(self.test_csv_path)
        self.test = self.test.groupby('class', group_keys=False).apply(lambda x: x.sample(n=1, random_state=42) if (len(x) == 1) else x.sample(n=len(x)//6 + 1 , random_state=42))
        self.class_to_cat = dict(self.train[['class', 'group']].values)

    def split(self, ratio: float):
        if self.train is None:
            raise Exception("data not loaded yet!")
        # self.train, self.validation = train_test_split(self.train, test_size=ratio, stratify=self.train['class'])

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
        # train_idx, val_idx = next(sss.split(self.train.drop('class', axis=1), self.train['class']))

        # self.validation = self.train.iloc[val_idx]
        # self.train = self.train.iloc[train_idx]

        # self.validation = self.train.groupby('class', group_keys=False).apply(lambda x: x.sample(n=1, random_state=42))
        self.validation = self.train.groupby('class', group_keys=False).apply(lambda x: x.sample(n=1, random_state=42) if (len(x) == 1) else x.sample(n=len(x)//10 + 1 , random_state=42))
        self.gallery_for_val = train_test_split(self.train, test_size=0.1, stratify=self.train['class'])[1]

    def set_transformation(self):
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.RandomRotation((0, 180)),
            T.RandomPerspective(distortion_scale=0.6),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # T.RandomEqualize(),
            # T.RandomGrayscale(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])