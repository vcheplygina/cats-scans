

from .task2vec import Task2Vec
from .models import get_model
from .datasets import get_dataset
import pickle
import hydra
import logging
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig):
    logging.info(cfg.pretty())
    train_dataset, test_dataset = get_dataset(cfg.dataset.root, cfg.dataset)
    probe_network = get_model('resnet50', pretrained=True, num_classes=7)
    embedding = Task2Vec(probe_network).embed(train_dataset)
    with open(f'{cfg.dataset.root}/embedding_isic2018.p', 'wb') as f:
        pickle.dump(embedding, f)


if __name__ == "__main__":
    main()


