import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from paleo.classifier import PrototypeClassifier
from paleo.dataset import CoreImageDataset
from paleo.features import CoreFeatureExtractor


def main(config_path):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = CoreImageDataset(config["data_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=16)
    encoder = CoreFeatureExtractor(config["embedding_dim"]).eval()
    classifier = PrototypeClassifier(config["embedding_dim"], config["num_classes"])
    for images, labels in loader:
        with torch.no_grad():
            embeddings = encoder(images)
        classifier.fit(embeddings, labels)
    torch.save({"prototypes": classifier.prototypes}, config["checkpoint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
