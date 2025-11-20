import os.path as osp
import math
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class IndexableBuffer:
    def __init__(self, data):
        self.data = data
        self.num_embeddings = len(data)
        self.embedding_dim = data.shape[1] if data.ndim > 1 else None

    def __getitem__(self, index):
        return self.data[index]

    @property
    def device(self):
        return self.data.device
    
    def __setitem__(self, index, val):
        self.data[index] = val

    def __call__(self, index=None):
        return self.__getitem__(index)


class DecorecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.img_size = config['img_size'] if hasattr(config, 'img_size') else config['plm_size']
        self.plm_suffix = config['plm_suffix']
        self.img_suffix = config['img_suffix']
        # self.init_mapper()
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight, self.plm_size)
        img_embedding_weight = self.load_img_embedding()
        self.img_embedding = self.weight2emb(img_embedding_weight, self.img_size)

    def init_mapper(self):
        self.iid2id = {}
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            self.iid2id[int(token)] = i

        self.uid2id = {}
        for i, token in enumerate(self.field2id_token['user_id']):
            if token == '[PAD]':
                continue
            self.uid2id[int(token)] = i

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def load_img_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.img_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.img_size)

        mapped_feat = np.zeros((self.item_num, self.img_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]':
                continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight, emd_size):
        # plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding = nn.Embedding(self.item_num, emd_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

