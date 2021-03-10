# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


#!/usr/bin/env python3.6
import pickle

import hydra
import logging

from .task2vec import Task2Vec
from .models import get_model
from .datasets import get_dataset
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig):
    logging.info(cfg.pretty())
    # run the embedding creation code 100 times (for 100 different subsets)
    for i in range(100):
        print(f'creating embedding for subset {i}')
        # get a random value that is used as random state for the sample creation
        random = i
        train_dataset = get_dataset(cfg.dataset.root, cfg.dataset, rand_int=random)
        if hasattr(train_dataset, 'task_name'):
            print(f"======= Embedding for task: {train_dataset.task_name} =======")
        probe_network = get_model(cfg.model.arch, pretrained=cfg.model.pretrained,
                                  num_classes=train_dataset.num_classes).cuda()
        embedding = Task2Vec(probe_network).embed(train_dataset)
        embedding.meta = OmegaConf.to_container(cfg, resolve=True)
        # embedding.meta['task_name'] = getattr(train_dataset, 'task_name', None)
        with open(f'{cfg.dataset.root}/embedding_pcam-middle_subset{i}.p', 'wb') as f:
            pickle.dump(embedding, f)


if __name__ == "__main__":
    main()
