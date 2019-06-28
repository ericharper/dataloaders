import numpy as np

from pytorch.batch_dataloader.batch_dataset import TensorBatchDataset
from pytorch.batch_dataloader.parquet_batch_dataset import ParquetBatchDataset


class ParquetBatchDataLoader:
    """Parquet data loader.

    Iterates through larger than memory parquet dataset.

    Arguments:
        parquet_dataset (ParquetBatchDataset): larger than memory dataset to iterate through.
        total_samples (int): total number of samples to iterate through
        shuffle (bool, optional): set to ``True`` to shuffle each in memory dataset. (default: ``False``)
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """
    def __init__(self, parquet_dataset, total_samples, shuffle=False, drop_last=False):
        self.parquet_dataset = parquet_dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.drop_last:
            raise NotImplementedError("Drop last not implemented yet")
        self.total_samples = total_samples

        self.batch_size = parquet_dataset.batch_size

    def __len__(self):
        num_batches = int(np.ceil(self.total_samples / self.batch_size))
        if self.drop_last:
            return num_batches - 1
        else:
            return num_batches

    def __iter__(self):
        self.batch_idx = 0  # batch index across all chunks
        self.chunk_idx = 0
        self._load_tensor_batch_dataset(self.chunk_idx)
        return self

    def __next__(self):
        if self.batch_idx >= len(self):
            raise StopIteration

        if self.chunk_batch_idx > len(self.tensor_batch_dataset) - 1:
            self.chunk_idx += 1
            self._load_tensor_batch_dataset(self.chunk_idx)
        batch = self.tensor_batch_dataset[self.chunk_batch_idx]
        self.chunk_batch_idx += 1
        self.batch_idx += 1
        return batch

    def _load_tensor_batch_dataset(self, chunk_idx):
        self.chunk = self.parquet_dataset.load_chunk_as_tensor(chunk_idx)
        self.chunk_batch_idx = 0
        self.tensor_batch_dataset = TensorBatchDataset([self.chunk], batch_size=self.batch_size)
        if self.shuffle:
            self.tensor_batch_dataset.shuffle()
