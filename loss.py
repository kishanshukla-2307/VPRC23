import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchAllTtripletLoss(nn.Module):
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