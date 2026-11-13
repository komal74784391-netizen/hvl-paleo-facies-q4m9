from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CoreImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(Path(root).glob("*.jpg"))
        self.transform = transform
        self.labels = [int(path.stem.split("_")[0]) for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
