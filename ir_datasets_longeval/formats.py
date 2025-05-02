from ir_datasets.datasets.base import Dataset


class MetaDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self._datasets = datasets

    def get_datasets(self):
        return self._datasets
