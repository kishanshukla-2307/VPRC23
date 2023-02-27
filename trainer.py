from collections import OrderedDict
import os
import timm
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoImageProcessor, BeitModel
from dataset import Product10kDataset
from samplers import Batch_Sampler
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from data_handler import DataHandler
from loss import BatchAllTtripletLoss
from models.mobile_net import MobileNet

# from torch.nn.functional import normalize

class Trainer:
    def __init__(self, model_name, num_classes, batch_size, num_epochs, lr, train_csv_path, test_csv_path, checkpoint_path, img_path, test_img_path, map_scores_path, emb_dim=128):
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.checkpoint_path = checkpoint_path
        self.img_path = img_path
        self.test_img_path = test_img_path
        self.map_scores_path = map_scores_path
        self.emb_dim = emb_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.feature_extract = False
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.data_handler = None

    def load_model(self):
        
        # if self.model_name == 'mobile_net':
        #     self.model = MobileNet(self.emb_dim)
        
        # elif self.modal_name == 'beit':
            

        # self.model = MyBEiT(BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k'))
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k')
        self.model.to(self.device)

        return

        if False:
          self.model = timm.create_model(self.model_name, pretrained=True)
          # for param in self.model.parameters():
          #     param.requires_grad = False
          self.model.classifier = nn.Linear(1280, self.emb_dim)
          # for param in self.model.blocks[4:7].parameters():
          #     param.requires_grad = True
          # for param in self.model.conv_head.parameters():
          #     param.requires_grad = True
          self.model.to('cuda')

          # self.model = timm.create_model(self.model_name, pretrained=True)
          # for param in self.model.parameters():
          #     param.requires_grad = False
          # self.model.classifier = nn.Linear(1280, self.num_classes)
          
          # checkpoint = torch.load('/content/drive/MyDrive/model_checkpoint/model_class_0000.pth',
          #                         map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['state_dict']

          # new_state_dict = OrderedDict()
          # for k, v in checkpoint.items():
          #     name = k.replace("module.", "")
          #     new_state_dict[name] = v

          # self.model.load_state_dict(new_state_dict)
          # for param in self.model.blocks[5:7].parameters():
          #     param.requires_grad = True
          # for param in self.model.conv_head.parameters():
          #     param.requires_grad = True
          # self.model.to('cuda')
        else:
          self.model = timm.create_model('mobilenetv3_large_100', pretrained=False)
          self.model.classifier = nn.Linear(1280, self.emb_dim)
          # self.model_with_classifier = timm.create_model('mobilenetv3_large_100', pretrained=False)
          # self.model_with_classifier.classifier = nn.Linear(1280, 9691)


          checkpoint = torch.load(MODEL_PATH,
                                  map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['state_dict']

          new_state_dict = OrderedDict()
          for k, v in checkpoint.items():
              name = k.replace("module.", "")
              new_state_dict[name] = v

          self.model.load_state_dict(new_state_dict)
          self.model.to(self.device)

    def set_data_handler(self):
        self.data_handler = DataHandler(self.train_csv_path, self.test_csv_path)
        self.data_handler.load_data()
        self.data_handler.split(0.15)
        self.data_handler.set_transformation()

    def set_criterion(self):
        # self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
        # self.criterion = TripletLoss().to('cuda')
        self.criterion = BatchAllTtripletLoss().to('cuda')
    
    def set_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0001)
    
    def get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return scheduler
    
    def set_train_loader(self, train_set):
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=self.batch_sampler)

    def set_validation_loader(self, validation_set):
        self.val_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
    
    def set_test_loader(self, test_set):
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        
    def train(self, model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epoch) -> None:

        model.train()


        train_iter = tqdm(self.train_loader, desc='Train', dynamic_ncols=True, position=2)

        running_loss = []

        for step, crr_batch in enumerate(train_iter):

            imgs, class_ids, group_ids = crr_batch

            np_imgs = imgs.numpy()
            imgs = np.split(np_imgs, imgs.shape[0], axis=0)
            imgs = [np.squeeze(x, axis=0) for x in imgs]
            # print(imgs.shape)
            imgs = self.image_processor(imgs, return_tensors="pt")
            # inputs = image_processor(image, return_tensors="pt")
            # print(imgs.shape)
            imgs = imgs.to(self.device)
            class_ids = class_ids.to(self.device)
            group_ids = group_ids.to(self.device)
            


            optimizer.zero_grad()

            embeddings = model(**imgs)

            loss = criterion(embeddings.pooler_output, class_ids, group_ids)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())

            if step % 1 == 0 and not step == 0:
                print('Epoch: {}; step: {}; loss: {:.4f}'.format(epoch, step, running_loss[-1]))

        print('Train process of epoch: {} is done; \n loss: {:.4f}'.format(epoch, np.mean(running_loss)))

    def validation(self, model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:

        with torch.no_grad():
            model.eval()

            val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)
            running_loss = []

            for step, crr_batch in enumerate(val_iter):
                imgs, class_ids, group_ids = crr_batch

                imgs = imgs.to(self.device)
                class_ids = class_ids.to(self.device)
                group_ids = group_ids.to(self.device)

                # out = model(x.cuda().to(memory_format=torch.contiguous_format))
                embeddings = model(imgs)

                # loss = criterion(out, y.cuda())
                loss = criterion(embeddings, class_ids, group_ids)

                running_loss.append(loss.cpu().detach().numpy())

                if step % 1 == 0 and not step == 0:
                    print('Epoch: {}; step: {}; loss: {:.4f}'.format(epoch, step, running_loss[-1]))

        print('Validation of epoch: {} is done; \n loss: {:.4f}'.format(epoch, np.mean(running_loss)))

        return np.mean(running_loss)


    def test(self, gallery_set, test_set):

        with torch.no_grad():
            self.model.eval()

            gallery_loader = self.val_loader
            gallery_embeddings = np.zeros((len(gallery_set), self.emb_dim))
            print(len(gallery_set))

            gal_iter = tqdm(gallery_loader, desc='Gallery', dynamic_ncols=True, position=2)

            for step, crr_batch in enumerate(gal_iter):
                imgs, class_ids, group_ids = crr_batch

                imgs = imgs.to(self.device)
                # class_ids = class_ids.to(self.device)
                # group_ids = group_ids.to(self.device)
                # print(group_ids)

                embeddings = self.model(imgs)

                gallery_embeddings[
                    step*self.batch_size:(step*self.batch_size + self.batch_size), :
                ] = embeddings.data.cpu().numpy()

            test_iter = tqdm(self.test_loader, desc='Test', dynamic_ncols=True, position=2)
            test_embeddings = np.zeros((len(test_set), self.emb_dim))
            running_loss = []

            for step, crr_batch in enumerate(test_iter):
                imgs, class_ids, group_ids = crr_batch

                imgs = imgs.to(self.device)
                # class_ids = class_ids.to(self.device)
                # group_ids = group_ids.to(self.device)

                # out = model(x.cuda().to(memory_format=torch.contiguous_format))
                embeddings = self.model(imgs)

                test_embeddings[
                    step*self.batch_size:(step*self.batch_size + self.batch_size), :
                ] = embeddings.data.cpu().numpy()

        # calculate_mAP(test_embeddings, test_labels, gallery_embeddings, gallery_labels)

        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(test_embeddings)
        distances = pairwise_distances(query_embeddings, gallery_embeddings)
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]

        return self.calculate_map(sorted_distances, test_set.class_ids, gallery_set.class_ids)

    
    def compute_precision_at_k(self, ranked_targets: np.ndarray,
                           k: int) -> float:
        assert k >= 1
        assert ranked_targets.size >= k, ValueError('Relevance score length < k')
        return np.mean(ranked_targets[:k])


    def compute_average_precision(self, ranked_targets: np.ndarray,
                                  gtp: int) -> float:
        assert gtp >= 1
        # compute precision at rank only for positive targets
        out = [self.compute_precision_at_k(ranked_targets, k + 1) for k in range(ranked_targets.size) if ranked_targets[k]]
        if len(out) == 0:
            # no relevant targets in top1000 results
            return 0.0
        else:
            return np.sum(out) / gtp


    def calculate_map(self, ranked_retrieval_results: np.ndarray,
                      query_labels: np.ndarray,
                      gallery_labels: np.ndarray) -> float:
        """
        Calculates the mean average precision.
        Args:
            ranked_retrieval_results: A 2D array of ranked retrieval results (shape: n_queries x 1000), because we use
                                    top1000 retrieval results.
            query_labels: A 1D array of query class labels (shape: n_queries).
            gallery_labels: A 1D array of gallery class labels (shape: n_gallery_items).
        Returns:
            The mean average precision.
        """
        assert ranked_retrieval_results.ndim == 2
        assert ranked_retrieval_results.shape[1] == 1000

        class_average_precisions = []

        class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
        class_id2quantity_dict = dict(zip(class_ids, class_counts))
        for gallery_indices, query_class_id in tqdm(
                                zip(ranked_retrieval_results, query_labels),
                                total=len(query_labels)):
            # Checking that no image is repeated in the retrival results
            assert len(np.unique(gallery_indices)) == len(gallery_indices), \
                        ValueError('Repeated images in retrieval results')

            current_retrieval = gallery_labels[gallery_indices] == query_class_id
            gpt = class_id2quantity_dict[query_class_id]

            class_average_precisions.append(
                self.compute_average_precision(current_retrieval, gpt)
            )

        with open(self.map_scores_path + "q_labels.txt", 'w') as f:
          for line in query_labels:
              f.write(f"{line}\n")
        
        with open(self.map_scores_path + "class_map.txt", 'w') as f:
          for line in class_average_precisions:
              f.write(f"{line}\n")

        # df = pd.DataFrame(list(zip(query_labels, class_average_precisions)))
        # DF = df
        # df.to_csv('map_scores.csv')
        # df.to_csv(self.map_scores_path)
        # file = open('map_scores.csv', 'w', newline ='')
    
        # # writing the data into the file
        # with file:   
        #     write = csv.writer(file)
        #     write.writerows(class_average_precisions)
        # with open("map_scores", "w") as outfile:
        #     outfile.write("\n".join(str(item) for item in class_average_precisions))
        
        mean_average_precision = np.mean(class_average_precisions)
        return mean_average_precision
        

    def run_training_loop(self):
        # data_handler = DataHandler(self.train_csv_path, self.test_csv_path)
        # data_handler.load_data()
        # data_handler.split(0.15)
        # data_handler.set_transformation()
        self.set_data_handler()

        train_X = self.data_handler.train['name'].to_numpy()
        train_Y = self.data_handler.train['class'].to_numpy()
        # one_hot_category = np.zeros((train_Y.size, train_Y.max() + 1))
        # one_hot_category[np.arange(train_Y.size), train_Y] = 1
        # train_Y = one_hot_category
        train_set = Product10kDataset(train_X, train_Y, self.data_handler.train['group'].to_numpy(), self.img_path, self.data_handler.transform, offline_strategy=False)

        val_X = self.data_handler.validation['name'].to_numpy()
        val_Y = self.data_handler.validation['class'].to_numpy()
        # one_hot_category = np.zeros((val_Y.size, val_Y.max() + 1))
        # one_hot_category[np.arange(val_Y.size), val_Y] = 1
        # val_Y = one_hot_category
        validation_set = Product10kDataset(val_X, val_Y, self.data_handler.train['group'].to_numpy(), self.img_path, self.data_handler.transform, offline_strategy=False)

        test_X = self.data_handler.test['name'].to_numpy()
        test_Y = self.data_handler.test['class'].to_numpy()
        # one_hot_category = np.zeros((val_Y.size, val_Y.max() + 1))
        # one_hot_category[np.arange(val_Y.size), val_Y] = 1
        # val_Y = one_hot_category
        test_set = Product10kDataset(test_X, test_Y, self.data_handler.test['Usage'].to_numpy(), self.test_img_path, self.data_handler.transform, offline_strategy=False)

        self.batch_sampler = Batch_Sampler(self.data_handler.train['class'].to_numpy(), self.data_handler.train['group'].to_numpy(), self.batch_size)
        self.load_model()
        self.set_optimizer()
        self.set_criterion()
        self.set_train_loader(train_set)
        self.set_validation_loader(validation_set)
        self.set_test_loader(test_set)

        train_epoch = tqdm(range(self.num_epochs), dynamic_ncols=True, 
                       desc='Epochs', position=0)
        
        scheduler = self.get_scheduler(self.optimizer)

        best_acc = 0.0
        for epoch in train_epoch:
            self.train(self.model, self.train_loader, self.criterion, self.optimizer, epoch)
            self.save_checkpoint(self.model, self.optimizer, scheduler, epoch, self.checkpoint_path)
            # epoch_avg_acc = self.validation(self.model, self.val_loader, self.criterion, epoch)
            # if epoch_avg_acc >= best_acc:
            #     self.save_checkpoint(self.model, self.optimizer, scheduler, epoch, self.checkpoint_path)
            #     best_acc = epoch_avg_acc
            scheduler.step()

        # print("best accuracy: ", best_acc)

        print("test mAP: ", self.test(validation_set, test_set))
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, outdir):
        """Saves checkpoint to drive"""

        filename = "model_triplet_c&g_imbalance_handled_beit_0.0001_{:04d}.pth".format(epoch)
        directory = outdir
        filename = os.path.join(directory, filename)
        weights = model.state_dict()
        state = OrderedDict([
            ('state_dict', weights),
            ('optimizer', optimizer.state_dict()),
            ('scheduler', scheduler.state_dict()),
            ('epoch', epoch),
        ])

        torch.save(state, filename)
