import math
import numpy as np
from torch.utils.data import Sampler

class Batch_Sampler(Sampler):
  def __init__(self, class_ids, group, batch_size, total_samples=141931, num_classes=9691, num_group=361):
    self.class_ids = class_ids
    self.group = group
    self.total_samples = total_samples
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.num_group = num_group

  def __iter__(self):
    sample_iter = Sample_Iterator(1, math.floor(self.total_samples/self.batch_size), self.class_ids, 
                                  self.group, self.batch_size, self.num_classes, self.num_group)
    return sample_iter
  
  def __len__():
    return 3


class Sample_Iterator:
  def __init__(self, low, high, class_ids, group, batch_size, tot_classes, tot_groups):
    self.current = low - 1
    self.high = high
    self.class_ids = class_ids
    self.group = group
    self.batch_size = batch_size
    self.tot_classes = tot_classes
    self.tot_groups = tot_groups
    self.preprocess()

  def preprocess(self):
    groups = np.arange(self.tot_groups)
    self.group_to_class = [np.unique(self.class_ids[np.where(self.group == grp)[0]], return_counts=True) for grp in groups]

  def __iter__(self):
      return self

  def __next__(self): # Python 2: def next(self)
    self.current += 1
    if self.current < self.high:
        return self.group_based_sampling(1)

    raise StopIteration

  def group_based_sampling(self, num_groups):
    if True:
      group_nos = self.uniform_sampling_of_group(num_groups)
      return self.uniform_sampling_of_class_within_grp(group_nos)

    else:
      ids = []
      while len(ids) < self.batch_size:
        group_nos = np.random.randint(0, 361, num_groups)
        
        ids = np.empty(0)
        for grp in group_nos:
            ids = np.append(ids, np.where(self.group == grp))
                    
      return np.random.choice(ids, size=self.batch_size)
  
  def uniform_sampling_of_group(self, num_groups):
    p0 = 1/self.tot_classes  ## 1/num_classes
    p = np.zeros(self.tot_groups)

    ## to-do: vectorize the loop
    for i in range(self.tot_groups):
      p[i] = p0 * len(self.group_to_class[i][0])
    
    return np.random.choice(np.arange(self.tot_groups), size=num_groups, p=p)

  def uniform_sampling_of_class_within_grp(self, grp_no):
    grp_no = math.floor(grp_no)
    ids = np.where(self.group == grp_no)[0]
    
    class_probs = {}
    
    clss_ids = self.group_to_class[grp_no][0]
    counts = self.group_to_class[grp_no][1]
    p0 = 1/np.sum(1/counts)

    for clss_id, cnt in zip(clss_ids, counts):
      class_probs[clss_id] = p0 * (1/cnt)
    
    p = [class_probs[self.class_ids[id]] for id in ids]
    p = p / np.sum(p)
    return np.random.choice(ids, size=self.batch_size, p=p)