from random import random

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_index += self.batch_size
        return batch