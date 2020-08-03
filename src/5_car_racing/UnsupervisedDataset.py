import numpy
import torch

class UnsupervisedDataset:
    def __init__(self, max_size):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.items = []

    def add(self, item):
        if self.is_full() == False:
            self.items.append(item.copy())

    def is_full(self):
        if len(self.items) > self.max_size:
            return True
        else:
            return False


    def get_random_batch(self, batch_size = 32, device = "cpu"):
        item_shape  = self.items[0].shape
        result      = torch.zeros((batch_size, ) + item_shape)

        for i in range(batch_size):
            n = numpy.random.randint(len(self.items))
            result[i] = torch.from_numpy(self.items[n])

        result = result.to(device)

        return result
