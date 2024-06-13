import numpy as np
class ReplayBuffer():
    "Buffer to store environment transitions"
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, experience:tuple):
        self.buffer[self.idx] = experience
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size=24):    
        indices = np.random.randint(0, self.capacity if self.full else self.idx,
                                   size=batch_size)
        batch = self.buffer[indices]
        return batch

    def __getitem__(self, index):
        if index >= 0 and index < self.capacity if self.full else self.idx:
            return self.buffer[index]
        else:
            raise ValueError('Index is out of range')

    def __len__(self):
        return self.capacity if self.full else self.idx


#######
## Sum-Tree Class
#####
class SumTree(object):
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.data_pointer = 0
        self.full = False   # indicates if the buffer is full

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary tree (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # non-leaf or Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)  # contains priorities

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # data is stored at the leaf of the tree from index: n-1 to 2*n-1
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
            self.full = True

    def __len__(self):  # returns the size of data buffer only
        return self.capacity if self.full else self.data_pointer

    def __getitem__(self, index):
        # return data and priority at index i
        if index >= 0 and index < self.capacity \
                    if self.full else self.data_pointer:
            tree_idx = index + self.capacity - 1
            return self.data[index], self.tree[tree_idx]
        else:
            raise ValueError('index out of range')

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node


#############
## Sum-Tree Replay Buffer
##########
class STBuffer(object):
    # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    def add(self, experience):
        # Find the max priority of leaf nodes
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the priority for new experience

    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)
        priorities = np.empty((n,), dtype=np.float32)  # store sample priorities
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index
            priorities[i] = priority   # experimental 
            minibatch.append([data[0],data[1],data[2],data[3],data[4]])

        return b_idx, minibatch, 

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper) 
        
        # stochastic prioritization
        ps = np.power(clipped_errors, self.PER_a)  # values between 0 and 1
        
        # convert priorities into probabilities
        prob = ps / np.sum(ps) # experimental
        
        # importance sampling weights: iw = [1 / ( N * P)]^b
        is_wts = np.power(len(prob) * prob, -self.PER_b) # experimental
        
        for ti, p, iw in zip(tree_idx, ps, is_wts):
            new_p = p * iw
            self.tree.update(ti, new_p)
            
        # gradually increase PER_b for more focus on high-error experience
        self.PER_b = min(1.0, self.PER_b + self.PER_b_increment_per_sampling)

    def __len__(self):
        return len(self.tree) 

    def __getitem__(self, index):
        return self.tree[index]
