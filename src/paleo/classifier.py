import torch


class PrototypeClassifier:
    def __init__(self, embedding_dim=128, num_classes=5):
        self.prototypes = torch.zeros(num_classes, embedding_dim)
        self.counts = torch.zeros(num_classes)

    def fit(self, embeddings, labels):
        for class_id in range(self.prototypes.shape[0]):
            mask = labels == class_id
            if mask.any():
                self.prototypes[class_id] = embeddings[mask].mean(dim=0)
                self.counts[class_id] = mask.sum()
        return self

    def predict(self, embeddings):
        distances = torch.cdist(embeddings, self.prototypes)
        return distances.argmin(dim=1)
