import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import timm
import torch
from tqdm import tqdm

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(config, logger, 
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch) -> None:

    model.train()

    train_iter = tqdm(train_loader)

    running_loss = AverageMeter()

    for step, crr_batch in enumerate(train_iter):

        imgs, class_ids, group_ids = crr_batch

        imgs = imgs.to(config['system']['device'])
        class_ids = class_ids.to(config['system']['device'])
        group_ids = group_ids.to(config['system']['device'])
        
        optimizer.zero_grad()

        logits = model(imgs)

        # print(class_ids)
        class_ids = torch.nn.functional.one_hot(class_ids, 7907)
        loss = criterion(logits, class_ids)

        loss.backward()
        optimizer.step()

        running_loss.update(loss.cpu().detach().numpy(), imgs.size(0))
        
        # train_iter.set_description('L_c: %.4f, L_a: %.4f' % (running_loss.val, running_loss.avg))

        if step % 10 == 0 and not step == 0:
            # print('Epoch: {}; step: {}; loss: {:.4f}'.format(epoch, step, running_loss[-1]))
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('Epoch: %d | Iter: [%d/%d], Memory_used: %.0fMB, loss_cur: %.5f, loss_avg: %.5f' % (epoch, step, len(train_loader), mu, running_loss.val, running_loss.avg))


    print('Train process of epoch: {} is done; \n loss: {:f}'.format(epoch, running_loss.avg))

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

    print('Validation of epoch: {} is done; \n loss: {:.4f}'.format(epoch, running_loss.avg))

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

