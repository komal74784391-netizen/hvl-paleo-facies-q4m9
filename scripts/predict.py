import argparse
import yaml
import torch
from torchvision import transforms

from paleo.classifier import PrototypeClassifier
from paleo.dataset import CoreImageDataset
from paleo.features import CoreFeatureExtractor


def main(config_path):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    state = torch.load(config["checkpoint"])
    classifier = PrototypeClassifier(config["embedding_dim"], config["num_classes"])
    classifier.prototypes = state["prototypes"]
    encoder = CoreFeatureExtractor(config["embedding_dim"]).eval()
    dataset = CoreImageDataset(args.images, transform=transforms.ToTensor())
    predictions = []
    for image, _ in dataset:
        with torch.no_grad():
            embedding = encoder(image.unsqueeze(0))
        predictions.append(int(classifier.predict(embedding)[0]))
    print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images", required=True)
    args = parser.parse_args()
    main(args.config)
