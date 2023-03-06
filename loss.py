import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchAllTripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss
  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1.):
    super().__init__()
    self.margin = margin
    self.eps = 1e-8
    
  def forward(self, embeddings, labels, groups):
    """computes loss value.
    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
    Returns:
      Scalar loss value.
    """
    # step 1 - get distance matrix
    # shape: (batch_size, batch_size)
    distance_matrix = self.euclidean_distance_matrix(embeddings)

    # print(distance_matrix)

    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

    # print(triplet_loss)

    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = self.get_triplet_mask(labels, groups)
    triplet_loss *= mask
    # print(mask)
    # easy triplets have negative loss values
    triplet_loss = F.relu(triplet_loss)
    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > self.eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + self.eps)

    return triplet_loss

  def get_triplet_mask(self, labels, groups):
    """compute a mask for valid triplets
    Args:
      labels: Batch of integer labels. shape: (batch_size,)
    Returns:
      Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
      A triplet is valid if:
      `labels[i] == labels[j] and labels[i] != labels[k]`
      and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices

    # shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # print(distinct_indices)

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # shape: (batch_size, batch_size, batch_size)
    valid_classes = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    groups_equal = groups.unsqueeze(0) == groups.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = groups_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = groups_equal.unsqueeze(1)
    # shape: (batch_size, batch_size, batch_size)
    valid_groups = torch.logical_and(i_equal_j, i_equal_k)

    # print(valid_indices)

    # step 3 - combine two masks
    mask = torch.logical_and(torch.logical_and(distinct_indices, valid_classes), valid_groups)

    return mask


  def euclidean_distance_matrix(self, x):
    """Efficient computation of Euclidean distance matrix
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)
      
    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

    # get rid of negative distances due to numerical instabilities
    distance_matrix2 = F.relu(distance_matrix)

    # step 4 - compute the non-squared distances
    
    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix2 == 0.0).float()

    # use this mask to set indices with a value of 0 to eps
    distance_matrix3 = distance_matrix2 + mask * self.eps

    # now it is safe to get the square root
    distance_matrix4 = torch.sqrt(distance_matrix3)

    # undo the trick for numerical stability
    distance_matrix5 = distance_matrix4 * (1.0 - mask)

    return distance_matrix5


class TripletLoss(nn.Module):
  def __init__(self, margin=0.5):
    super(TripletLoss, self).__init__()
    self.margin = margin
      
  def calc_euclidean(self, x1, x2):
    return (x1 - x2).pow(2).sum(1)
  
  def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    distance_positive = self.calc_euclidean(anchor, positive)
    distance_negative = self.calc_euclidean(anchor, negative)
    losses = torch.relu(distance_positive - distance_negative + self.margin)
    # print(distance_positive.shape, distance_negative.shape)
    # print(distance_positive, distance_negative)
    return losses.mean()

class Smoth_CE_Loss(nn.Module):
    def __init__(self, ls_=0.9):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(reduction="none")  
        self.ls_ = ls_

    def forward(self, logits, labels):
        labels *= self.ls_
        return self.crit(logits, labels)

# class DenseCrossEntropy(nn.Module):
#     def forward(self, x, target):
#         logprobs = F.log_softmax(x, dim=1)
#         loss = -logprobs * target
#         loss = loss.sum(dim=1)
#         return loss

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.3, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        # cosine = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float())).float()
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        # return output
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    
class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, stride=0.1, max_m=0.8, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.m = m
        self.m_s = stride
        self.max_m = max_m
        self.last_epoch = 1
        self.reduction = reduction
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    def update(self, c_epoch):
        self.m = min(self.m+self.m_s*(c_epoch-self.last_epoch), self.max_m)
        self.last_epoch = c_epoch
        # logger.info('Update margin----')
        # logger.info(f'Curent Epoch: {c_epoch}, Curent Margin: {self.m:.2f}')
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    def forward(self, logits, labels, mode='train', c_epoch=1):
        if c_epoch!=self.last_epoch and mode=='train':
            self.update(c_epoch)
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class Contrastive_Arc_Loss(nn.Module):
    def __init__(self, n_views=5, s=45.0, m=0.1, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.n = n_views
        if crit == "ce":
            self.crit = DenseCrossEntropy()   
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, features):
        labels = torch.cat([torch.arange(features.shape[0]//self.n) for i in range(self.n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        labels_positives = torch.ones_like(positives)
        labels_negatives = torch.zeros_like(negatives)
        cosine = torch.cat([positives, negatives], dim=1)
        labels = torch.cat([labels_positives, labels_negatives], dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class DenseCrossEntropyLoss(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()