import math
from torch.utils.data import Dataset
from PIL import Image


from torch.utils.data import Dataset

class Product10kDataset(Dataset):
    def __init__(self, img_names, class_ids, group, img_path, transforms=None, offline_strategy=False, batch_size=64):
        self.img_names = img_names
        self.class_ids = class_ids
        self.group = group
        self.img_path = img_path
        self.transforms = transforms
        self.offline_strategy = offline_strategy
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.img_names)
        # return self.batch_size
    
    def __getitem__(self, idx):
        # return np.random.randint(low=0, high=10, size=1)
        # return idx
        if self.offline_strategy:
            found = False;
            while not found:
                anchor_img = self.img_names[idx]
                anchor_img = Image.open(self.img_path + anchor_img)
                anchor_class = self.class_ids[idx]
                anchor_group = self.group[idx]

                same_class = np.where(self.class_ids == anchor_class)[0]
                same_group = np.where(self.group == anchor_group)[0]

                # print(type(same_class), same_class)

                if len(same_class) == len(same_group):
                    idx += 1
                    continue

                positive_idx = np.random.choice(same_class)
                while (positive_idx == idx):
                    positive_idx = np.random.choice(same_class)
                
                rel_ids = []
                for id in same_group:
                    if self.class_ids[id] != anchor_class:
                        rel_ids.append(id)

                negative_idx = np.random.choice(np.array(rel_ids))
                
                positive_img = self.img_names[positive_idx]
                positive_img = Image.open(self.img_path + positive_img)

                negative_img = self.img_names[negative_idx]
                negative_img = Image.open(self.img_path + negative_img)

                if self.transforms is not None:
                    anchor_img = self.transforms(anchor_img)
                    positive_img = self.transforms(positive_img)
                    negative_img = self.transforms(negative_img)
                
                
                label = self.class_ids[idx]

                return anchor_img, positive_img, negative_img, label
        
        else:
            # print(idx)
            idx = math.floor(idx)
            anchor_img = self.img_names[idx]
            anchor_img = Image.open(self.img_path + anchor_img)
            if self.transforms is not None:
                anchor_img = self.transforms(anchor_img)
            
            class_id = self.class_ids[idx]
            group_id = self.group[idx]

            return anchor_img, class_id, group_id