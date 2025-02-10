import numpy as np
from torch.utils.data import Sampler



class EqualLengthsBatchSampler(Sampler):

    def __init__(self, batch_size, inputs, targets):
        
        # Throw an error if the number of inputs and targets don't match
        if targets is not None:
            if len(inputs) != len(targets):
                raise Exception("[EqualLengthsBatchSampler] inputs and targets have different sizes")
        
        # Remember batch size and number of samples
        self.batch_size, self.num_samples = batch_size, len(inputs)
        
        self.unique_length_pairs = set()
        self.lengths_to_samples = {}
        
        for i in range(0, len(inputs)):
            len_input = len(inputs[i])
            try:
                # Fails if targets[i] is not a sequence but a scalar (e.g., a class label)
                len_target = len(targets[i])
            except:
                # In case of failure, we just the length to 1 (value doesn't matter, it only needs to be a constant)
                len_target = 1

            # Add length pair to set of all seen pairs
            self.unique_length_pairs.add((len_input, len_target))
        
            # For each lengths pair, keep track of which sample indices for this pair
            # E.g.: self.lengths_to_sample = { (4,5): [3,5,11], (5,5): [1,2,9], ...}
            if (len_input, len_target) in self.lengths_to_samples:
                self.lengths_to_samples[(len_input, len_target)].append(i)
            else:
                self.lengths_to_samples[(len_input, len_target)] = [i]
        
        # Convert set of unique length pairs to a list so we can shuffle it later
        self.unique_length_pairs = list(self.unique_length_pairs)
        
        
    def __len__(self):
        return self.num_samples
    
    def __iter__(self):

        # Shuffle list of unique length pairs
        np.random.shuffle(self.unique_length_pairs)
        
        # Iterate over all possible sentence length pairs
        for length_pair in self.unique_length_pairs:
            
            # Get indices of all samples for the current length pairs
            # for example, all indices with a lenght pair of (8,7)
            sequence_indices = self.lengths_to_samples[length_pair]
            sequence_indices = np.array(sequence_indices)
            
            # Shuffle array of sequence indices
            np.random.shuffle(sequence_indices)

            # Compute the number of batches
            num_batches = np.ceil(len(sequence_indices) / self.batch_size)

            # Loop over all possible batches
            for batch_indices in np.array_split(sequence_indices, num_batches):
                yield np.asarray(batch_indices)