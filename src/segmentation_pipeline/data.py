import torch
from torch.utils.data import Dataset


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, n=16, image_size=64, num_classes=4, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, image_size, image_size, generator=g)
        self.masks = torch.randint(0, num_classes, (n, image_size, image_size), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]
