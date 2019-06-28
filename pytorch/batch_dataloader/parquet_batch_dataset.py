import numpy as np
import pyarrow.parquet as pq
import torch


class ParquetBatchDataset:
    """Dataset from parquet files.

    Arguments:
        files (list): List of parquet files.
        num_files_per_chunk (int): Number of files per chunk. Chunks are data to be loaded into memory at one time.
        batch_size (int): The size of the batch to return.
    """
    def __init__(self, files, num_files_per_chunk, batch_size):
        self.files = files
        self.num_files_per_chunks = num_files_per_chunk
        self.batch_size = batch_size

    def __len__(self):
        # number of chunks to be loaded into memory
        return int(np.ceil(len(self.files) / self.num_files_per_chunks))

    def __getitem__(self, chunk_idx):
        file_idx = chunk_idx * self.num_files_per_chunks
        return self.files[file_idx:file_idx + self.num_files_per_chunks]

    def load_chunk_as_tensor(self, chunk_idx):
        print('Loading chunk %d from disk.' % chunk_idx)
        chunk = pq.ParquetDataset(self[chunk_idx]).read_pandas()
        chunk = chunk.to_pandas()
        chunk = torch.from_numpy(chunk.values)
        return chunk

    def compute_samples_per_chunk(self):
        samples_per_chunk = []
        for idx in range(len(self)):
            data = pq.ParquetDataset(self[idx]).read_pandas()
            num_samples = data.shape[0]
            samples_per_chunk += [num_samples]
        return samples_per_chunk
