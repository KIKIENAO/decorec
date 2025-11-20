import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from model_utils import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import scipy.sparse as sp
import cupy as cp
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans as cuKMeans

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by the number of heads"

        # Linear layers to generate Q, K, V
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Apply linear transformations to get Q, K, V
        values = self.values(value)  # shape: (N, value_len, embed_size)
        keys = self.keys(key)        # shape: (N, key_len, embed_size)
        queries = self.queries(query) # shape: (N, query_len, embed_size)

        # Split the embedding into multiple heads
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        # Transpose to get dimensions: (batch_size, heads, seq_len, head_dim)
        values = values.permute(0, 2, 1, 3)  # (N, heads, value_len, head_dim)
        keys = keys.permute(0, 2, 1, 3)      # (N, heads, key_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (N, heads, query_len, head_dim)

        # Calculate the attention scores (energy)
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        # (batch_size, heads, query_len, key_len)

        # Scale energy scores and apply softmax
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        # (batch_size, heads, query_len, key_len)

        # Multiply attention scores with values
        out = torch.einsum("nhqk,nhvd->nhqd", [attention, values])
        # (batch_size, heads, query_len, head_dim)

        # Reshape and combine heads
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        # (batch_size, query_len, heads * head_dim)

        # Final linear layer
        out = self.fc_out(out)
        return out
class Decorec(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config=config
        self.datasetname=config["dataset"]

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.gamma = config['gamma']
        self.modal_type = config['modal_type']
        self.id_type = config['id_type']
        self.seq_mm_fusion = config['seq_mm_fusion'] # 'add' | 'contextual'
        self.n_items=len(dataset.field2token_id["item_id"])

        self.item_mm_fusion = config['item_mm_fusion'] # 'static' | 'dynamic_shared' | 'dynamic_instance'
        # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
        # assert self.item_mm_fusion in ['static', 'dynamic_shared', 'dynamic_instance']



        assert self.train_stage in [
            'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
            all_num_embeddings = 0
            if 'text' in self.modal_type:
                self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
                self.register_buffer('plm_embedding_empty_mask', (~self.plm_embedding.weight.data.sum(-1).bool()))

            if 'img' in self.modal_type:
                self.img_embedding = copy.deepcopy(dataset.img_embedding)
                self.register_buffer('img_embedding_empty_mask', (~self.img_embedding.weight.data.sum(-1).bool()))

        if 'text' in self.modal_type:
            self.text_adaptor = nn.Sequential(
                nn.Linear(config['plm_size'], config['hidden_size']),
                nn.ReLU(),
                nn.Linear(config['hidden_size'], config['hidden_size']),
            )


        if 'img' in self.modal_type:
            self.img_adaptor = nn.Sequential(
                nn.Linear(config['img_size'], config['hidden_size']),
                nn.ReLU(),
                nn.Linear(config['hidden_size'], config['hidden_size']),
            )
        # self.adapted_modal_weight = torch.cat([self.text_adaptor(self.plm_embedding.weight.detach()), self.img_adaptor(self.img_embedding.weight.detach())], dim=0)
        self.interaction_matrix=torch.load(f"interaction_matrix_{self.datasetname}.pth")
        self.interaction_matrix.requires_grad_(False)




        # self.interaction_matrix=self.get_norm_adj()
        # self.interaction_matrix=self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).to_dense()
        # #
        # self.register_buffer("similarity_matrix",torch.cosine_similarity( self.modal_weight.unsqueeze(1),self.modal_weight.unsqueeze(0)))
        # modal_weight = torch.cat([self.plm_embedding.weight, self.img_embedding.weight], dim=0)
        # batch_size = 1000  # 设定每次计算的批量大小
        # #
        # # # 创建一个空的矩阵来存储相似度
        # similarity_matrix = torch.zeros(modal_weight.size(0), modal_weight.size(0), device=modal_weight.device)
        #
        # # 计算相似度矩阵
        # for i in range(0, modal_weight.size(0), batch_size):
        #     start_i = i
        #     end_i = min(i + batch_size, modal_weight.size(0))
        #
        #     for j in range(0, modal_weight.size(0), batch_size):
        #         start_j = j
        #         end_j = min(j + batch_size, modal_weight.size(0))
        #
        #         # 计算一个小块的余弦相似度
        #         similarity_matrix[start_i:end_i, start_j:end_j] = torch.cosine_similarity(
        #             modal_weight[start_i:end_i].unsqueeze(1),
        #             modal_weight[start_j:end_j].unsqueeze(0),
        #             dim=-1
        #         )
        #
        # _, self.indices = torch.topk(self.similarity_matrix, 10, dim=1, largest=True, sorted=False)


        self.mapping,self.interest_embedding=self.get_collabrative_interest_embedding(0.1)
        self.mapping_img, self.interest_embedding_img = self.get_collabrative_interest_embedding(
            ratio=0.1, modality="img")
    def  get_collabrative_interest_embedding(self,ratio,modality="text"):
        device="cuda" if torch.cuda.is_available() else "cpu"

        similarity_matrix = self.interaction_matrix.to(device)

        # 确保 similarity_matrix 是稀疏格式
        if not similarity_matrix.is_sparse:
            similarity_matrix = similarity_matrix.to_sparse()

        # 根据 item_seq 创建一个掩码，表示有效项的位置

        # 获取 item_embedding 权重
        if modality=="text":
            agg_embedding =torch.cat( [self.plm_embedding.weight.to(device)])
        elif modality == "img":
            agg_embedding = torch.cat([self.img_embedding.weight.to(device)])
        layers = [agg_embedding]
        num_layers=1
        weight=[0.01 for i in range(num_layers)]

        # 使用稀疏矩阵乘法
        for i in range(num_layers):
            agg_embedding = weight[i]*self.sparse_multiply(similarity_matrix, agg_embedding)  # 稀疏矩阵乘法
            layers.append(agg_embedding)

        # 堆叠所有的聚合嵌入
        agg_embeddings = torch.stack(layers, dim=0)
        agg_embedding = agg_embeddings.sum(dim=0)
        adapted_modal_weight = torch.cat(
            [agg_embedding], dim=0)

        modal_weight_gpu = adapted_modal_weight.to(device).detach()

        k_clusters = max(1, int(len(modal_weight_gpu) * ratio))  # 1/5 的聚类数，至少为 1

        print("start_train")
        kmeans = cuKMeans(n_clusters=k_clusters)  # 使用 cuML 的 KMeans
        kmeans.fit(modal_weight_gpu)  # 在 GPU 上训练
        print("train_end")

        # 获取聚类中心
        center_points = kmeans.cluster_centers_

        print("start_train3")

        # 将中心点转为 Tensor 并移到 GPU
        center_points_tensor = torch.tensor(center_points.get(), dtype=torch.float32).to(device)

        print("start_train4")

        # 找出每个嵌入的中心点索引
        # 通过计算每个嵌入到所有中心点的距离，找到最近的中心点
        batch_size = 1000  # You can adjust this based on your GPU memory
        n_batches = (modal_weight_gpu.shape[0] + batch_size - 1) // batch_size

        center_indices = cp.empty(modal_weight_gpu.shape[0], dtype=cp.int32)

        for i in range(n_batches):
            print(f"{i}/{n_batches}")
            start = i * batch_size
            end = min((i + 1) * batch_size, modal_weight_gpu.shape[0])
            differences = modal_weight_gpu[start:end, None, :] - center_points[None, :, :]
            distances = cp.linalg.norm(differences, axis=2)
            center_indices[start:end] = cp.argmin(distances, axis=1)

        print("start_train6")

        # 将数据放到字典中
        mapping = {
            "center_points": center_points_tensor,  # 使用 KMeans 获得的中心点坐标
            "center_indices": torch.tensor(center_indices.get(), dtype=torch.long).to(device)  # 每个嵌入的中心点索引
        }
        data = center_indices.get()  # 获取 indices 数据
        indices_tensor = torch.tensor(data, dtype=torch.long).to(device)

        # 统计每个不同的 indice 出现的频率
        unique_indices, counts = torch.unique(indices_tensor, return_counts=True)

        # 将 counts 和 unique_indices 结合在一起，便于排序
        frequency_pairs = list(zip(unique_indices.cpu().numpy(), counts.cpu().numpy()))

        # 按照频率（count）降序排序，获取频率前十的 indices
        sorted_frequency_pairs = sorted(frequency_pairs, key=lambda x: x[1], reverse=True)

        # 提取前十个频率最高的 indices 和对应的频率
        top_10_indices = [pair[0] for pair in sorted_frequency_pairs[:10]]
        top_10_counts = [pair[1] for pair in sorted_frequency_pairs[:10]]

        # 打印结果
        print("前十个频率最高的 indices：", top_10_indices)
        print("对应的频率：", top_10_counts)

        # 创建兴趣嵌入并将其转到 GPU
        interest_embedding = nn.Embedding(num_embeddings=k_clusters, embedding_dim=self.config['hidden_size'])
        interest_embedding.weight.data = torch.zeros_like(interest_embedding.weight.data)


        return mapping,interest_embedding



    def sparse_multiply(self, sparse_adj, embedding, layer=1):
        import torch
        """
        Function to multiply a sparse adjacency matrix with an embedding matrix multiple times.
        The adjacency matrix is normalized using left and right multiplication.

        Args:
            sparse_adj (torch.sparse.FloatTensor): 输入的稀疏邻接矩阵。
            embedding (torch.FloatTensor): 输入的嵌入矩阵。
            layer (int, optional): 进行矩阵乘法的次数，默认为 1.

        Returns:
            torch.FloatTensor: 与稀疏矩阵多次相乘后的嵌入矩阵。
        """
        # 检查是否支持 CUDA，并将张量移动到相应的设备
        device = 'cuda'
        sparse_adj = sparse_adj.to(device)
        embedding = embedding.to(device)

        # 确保稀疏矩阵和嵌入矩阵数据类型一致，强制转换为 float32
        sparse_adj = sparse_adj.to(torch.float32)
        embedding = embedding.to(torch.float32)

        # 对稀疏邻接矩阵进行归一化处理（左乘和右乘）
        import torch

        def normalize_sparse_adj(adj):
            # 计算每一行的元素之和
            row_sum = adj.sum(dim=1)  # 输出 shape 是 [N, 1] 或 [N]，取决于 PyTorch 版本

            # 检查 row_sum 是否为一维张量，如果不是，则调整其形状
            if row_sum.dim() == 2 and row_sum.size(1) == 1:
                row_sum = row_sum.squeeze(1)  # 转换为一维张量 [N]

            # 将稀疏张量转换为稠密张量
            row_sum_dense = row_sum.to_dense()

            # 避免除以零，使用一个小常数避免除零错误
            row_sum_dense = row_sum_dense + 1e-7

            # 构造对角矩阵 D_row^{-1}（左乘归一化）
            D_row_values = 1.0 / row_sum_dense
            D_row_indices = torch.stack([torch.arange(len(D_row_values)), torch.arange(len(D_row_values))])

            # 创建稀疏的对角矩阵 D_row
            D_row = torch.sparse_coo_tensor(D_row_indices, D_row_values, adj.size(), device=adj.device)

            # 计算每一列的元素之和
            col_sum = adj.sum(dim=0)  # 输出 shape 是 [1, N] 或 [N]，取决于 PyTorch 版本

            # 检查 col_sum 是否为一维张量，如果不是，则调整其形状
            if col_sum.dim() == 2 and col_sum.size(0) == 1:
                col_sum = col_sum.squeeze(0)  # 转换为一维张量 [N]

            # 将稀疏张量转换为稠密张量
            col_sum_dense = col_sum.to_dense()

            # 避免除以零，使用一个小常数避免除零错误
            col_sum_dense = col_sum_dense + 1e-7

            # 构造对角矩阵 D_col^{-1}（右乘归一化）
            D_col_values = 1.0 / col_sum_dense
            D_col_indices = torch.stack([torch.arange(len(D_col_values)), torch.arange(len(D_col_values))])

            # 创建稀疏的对角矩阵 D_col
            D_col = torch.sparse_coo_tensor(D_col_indices, D_col_values, adj.size(), device=adj.device)

            # 归一化后的邻接矩阵 A_norm = D_row^{-1} @ A @ D_col^{-1}
            return torch.sparse.mm(torch.sparse.mm(D_row, adj), D_col)

        # 构造稀疏对角矩阵

        normalized_adj = normalize_sparse_adj(sparse_adj)

        # 多层矩阵乘法
        for _ in range(layer):
            embedding = torch.sparse.mm(normalized_adj, embedding)

        return embedding
    def get_colabrative_item_embedding(self, item_seq,modality):
        # batch_size, seq_length = item_seq.size()
        #
        # # 获取当前设备
        # device = item_seq.device
        device="cuda"
        similarity_matrix = self.interaction_matrix.to(device)

        # 确保 similarity_matrix 是稀疏格式
        if not similarity_matrix.is_sparse:
            similarity_matrix = similarity_matrix.to_sparse()

        # 获取 item_embedding 权重
        if modality=="text":
            item_embeddings = self.item_embedding.weight.to(device)
        elif modality=="img":
            item_embeddings = self.item_embedding_img.weight.to(device)
        # item_embeddings=F.dropout(item_embeddings,p=0.3)


        agg_embedding = item_embeddings
        layers = [agg_embedding]

        # 使用稀疏矩阵乘法
        for i in range(2):
            agg_embedding = self.sparse_multiply(similarity_matrix,agg_embedding) # 稀疏矩阵乘法
            layers.append(agg_embedding)

        # 堆叠所有的聚合嵌入
        agg_embeddings = torch.stack(layers, dim=0)
        agg_embedding = agg_embeddings.mean(dim=0)
        return agg_embedding
    def get_colabrative_pool_embedding(self,item_seq=None,modality="text"):
            colabrative_item_embedding=self.get_colabrative_item_embedding(item_seq,modality)
            if item_seq==None:
                return colabrative_item_embedding
            batchsize,seq_length=item_seq.shape
            item_seq_flatten=item_seq.flatten()
            item_emb=colabrative_item_embedding[item_seq_flatten]
            item_emb=item_emb.reshape(batchsize,seq_length,-1)
            return colabrative_item_embedding,item_emb
    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        """memory_mask: [BxL], dec_input_seq: [BxNq]"""
        key_padding_mask = (dec_input_seq == 0)  # binary, [BxNq], Nq=L
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf'), device=dec_input_seq.device),
                               diagonal=1) if is_casual else None
        return attn_mask, key_padding_mask

    def get_decoder_attention_mask(self, enc_input_seq, item_modal_empty_mask, is_casual=True):
        # enc_input_seq: [BxL]
        # item_modal_empty_mask: [BxMxL]
        assert enc_input_seq.size(0) == item_modal_empty_mask.size(0)
        assert enc_input_seq.size(-1) == item_modal_empty_mask.size(-1)
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape  # M
        if self.seq_mm_fusion == 'add':
            key_padding_mask = (enc_input_seq == 0)  # binary, [BxL]
        else:
            # binary, [Bx1xL] | [BxMxL] => [BxMxL]
            key_padding_mask = torch.logical_or((enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask)
            key_padding_mask = key_padding_mask.flatten(1)  # [BxMxL] => [Bx(M*L)]
        if is_casual:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=enc_input_seq.device),
                                   diagonal=1)  # [LxL]
            if self.seq_mm_fusion != 'add':
                attn_mask = torch.tile(attn_mask, (num_modality, num_modality))  # [(M*L)x(M*L)]
        else:
            attn_mask = None
        cross_attn_mask = None  # Full mask
        return attn_mask, cross_attn_mask, key_padding_mask




    def get_norm_adj(self):
        self.interaction_matrix=sp.lil_matrix(self.interaction_matrix.numpy())
        adj_mat = sp.dok_matrix(( self.n_items, self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:,:] = R
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_items, self.n_items:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)






    def get_pool_item_embedding(self, item_seq):
        batch_size, seq_length = item_seq.size()

        # 获取当前设备
        device = item_seq.device

        # 将 indices 和 similarity_matrix 迁移到与 item_seq 相同的设备
        indices = self.indices.to(device)
        similarity_matrix = self.similarity_matrix.to(device)




        # 创建一个与 item_seq 形状相同的 tensor 用于存储结果
        pool_item_embedding = torch.zeros_like(item_seq, device=device).unsqueeze(-1).expand(-1, -1,
                                                                                             self.config['hidden_size'])

        # 根据 item_seq 创建一个掩码，表示有效项的位置
        mask = item_seq != 0


        # 获取 item_embedding 权重
        item_embeddings = self.item_embedding.weight.to(device)  # Ensure item_embedding is also on the correct device

        # 获取每个 item 对应的前 10 个最相似的元素的索引
        top_indices = indices[item_seq]

        # shape: [batch_size, seq_length, top_k]

        # 获取每个 item 对应的相似度矩阵 (相似度的索引应该是 top_indices 中的每个元素与 similarity_matrix 中的对应元素)
        # 扩展 batch_indices 和 top_indices

        batch_indices = item_seq.unsqueeze(-1).expand(-1, -1, 10)  # [batch_size, seq_length]
        batch_indices = batch_indices.reshape(-1)  # Flatten to [batch_size * seq_length]

        top_indices = top_indices.reshape(-1)  # Flatten to [batch_size * seq_length * top_k]

        # 获取相似度矩阵（确保正确的广播）
        similarity_scores = similarity_matrix[batch_indices, top_indices].reshape(batch_size, seq_length,
                                                                                  -1)
        similarity_scores=torch.softmax(similarity_scores,dim=-1)
        similarity_scores[:,:,0]+=1
        similarity_scores=similarity_scores/2

        # [batch_size, seq_length, top_k]

        # 获取每个 item 对应的 top_embeddings
        top_embeddings = item_embeddings[top_indices].reshape(batch_size, seq_length, -1, self.config[
            'hidden_size'])  # [batch_size, seq_length, top_k, hidden_size]

        # 将相似度加权应用到 top_embeddings
        weighted_embeddings = top_embeddings * similarity_scores.unsqueeze(
            -1)  # shape: [batch_size, seq_length, top_k, hidden_size]

        # 对 top_k 维度进行加权求和
        pooled_embeddings = weighted_embeddings.sum(dim=2)  # shape: [batch_size, seq_length, hidden_size]

        # 仅保留有效的项的聚合结果
        pool_item_embedding = pooled_embeddings * mask.unsqueeze(-1)  # shape: [batch_size, seq_length, hidden_size]

        return pool_item_embedding

    def normalize_sparse_adj(adj):
        # 计算每一行的元素之和
        row_sum = adj.sum(dim=1)  # 输出 shape 是 [N, 1] 或 [N]，取决于 PyTorch 版本

        # 检查 row_sum 是否为一维张量，如果不是，则调整其形状
        if row_sum.dim() == 2 and row_sum.size(1) == 1:
            row_sum = row_sum.squeeze(1)  # 转换为一维张量 [N]

        # 将稀疏张量转换为稠密张量
        row_sum_dense = row_sum.to_dense()

        # 避免除以零，使用一个小常数避免除零错误
        row_sum_dense = row_sum_dense + 1e-9

        # 构造对角矩阵 D_row^{-1}（左乘归一化）
        D_row_values = torch.pow(row_sum_dense, -1)
        D_row_indices = torch.stack([torch.arange(len(D_row_values)), torch.arange(len(D_row_values))])

        # 创建稀疏的对角矩阵 D_row
        D_row = torch.sparse_coo_tensor(D_row_indices, D_row_values, adj.size(), device=adj.device)

        # 计算每一列的元素之和
        col_sum = adj.sum(dim=0)  # 输出 shape 是 [1, N] 或 [N]，取决于 PyTorch 版本

        # 检查 col_sum 是否为一维张量，如果不是，则调整其形状
        if col_sum.dim() == 2 and col_sum.size(0) == 1:
            col_sum = col_sum.squeeze(0)  # 转换为一维张量 [N]

        # 将稀疏张量转换为稠密张量
        col_sum_dense = col_sum.to_dense()

        # 避免除以零，使用一个小常数避免除零错误
        col_sum_dense = torch.pow(col_sum_dense + 1e-9, -0.5)
        print(col_sum_dense[:20], D_row_values[:20])

        # 构造对角矩阵 D_col^{-1}（右乘归一化）
        D_col_values = col_sum_dense
        D_col_indices = torch.stack([torch.arange(len(D_col_values)), torch.arange(len(D_col_values))])

        # 创建稀疏的对角矩阵 D_col
        D_col = torch.sparse_coo_tensor(D_col_indices, D_col_values, adj.size(), device=adj.device)

        # 归一化后的邻接矩阵 A_norm = D_row^{-1} @ A @ D_col^{-1}
        return torch.sparse.mm(D_row, adj)

    def contrastive_loss_fn(self,data, data_aug, margin=1.0):
        """
        计算对比损失函数，确保同一批次的样本增强版本相似，不同批次的样本不相似。

        参数:
        - text_output (torch.Tensor): 原始文本输出, shape: [batch_size, features]
        - text_aug_output (torch.Tensor): 文本增强输出, shape: [batch_size, features]
        - margin (float): 不相似样本的最小距离, 默认为1.0

        返回:
        - loss (torch.Tensor): 计算的对比损失
        """
        # 计算同一批次内每对样本的余弦相似度
        cos_sim = F.cosine_similarity(data, data_aug, dim=-1)  # 形状 [batch_size]

        # 对于相似样本，目标是最大化相似度，标签为1，计算正样本的损失
        positive_loss = torch.mean((1 - cos_sim) ** 2)

        # 计算负样本的损失
        batch_size = data_aug.size(0)

        # 计算所有样本之间的余弦相似度 (大小为 [batch_size, batch_size])
        cos_sim_matrix = F.cosine_similarity(data.unsqueeze(1),data_aug.unsqueeze(0), dim=-1)

        # 创建一个负样本的损失矩阵：确保对角线（相似的样本对）不会被计算为负样本
        mask = torch.eye(batch_size, device=data.device).bool()

        # 负样本损失为：如果两个样本不同，损失值为 max(0, margin - cos_sim)
        negative_loss = torch.clamp(margin - cos_sim_matrix.masked_fill(mask, float('-inf')), min=0)

        # 负样本损失的平均值
        negative_loss = negative_loss.pow(2).sum() / (batch_size * (batch_size - 1))

        # 总损失为正样本损失和负样本损失的和
        total_loss = positive_loss + negative_loss
        return total_loss





    def calculate_loss(self, interaction):
        if self.train_stage=="pretrain":
            return self.pretrain(interaction)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]


        test_item_emb = 0
        item_modal_empty_mask_list = []
        item_size=self.plm_embedding.weight.shape[0]
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))
            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]

        if 'text' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)

            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
            item_modal_empty_mask_list.append(text_emb_empty_mask)

        if 'img' in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]
            item_modal_empty_mask_list.append(img_emb_empty_mask)
        batch_size, sequence_length = item_seq.shape
        test_emb=torch.cat([test_text_emb,test_img_emb],dim=0)
        if self.train_stage=="transductive_ft":
            test_id_emb, id_emb = self.get_colabrative_pool_embedding(item_seq)
            test_id_emb = torch.cat([test_id_emb,test_id_emb])
            self.mapping["center_indices"].to(item_seq.device)
            test_collabrative_interest_embedding_text = self.interest_embedding(
                self.mapping["center_indices"]
            )
            self.mapping_img["center_indices"].to(item_seq.device)
            test_collabrative_interest_embedding_img = self.interest_embedding_img(
                self.mapping_img["center_indices"]
            )
            test_collabrative_interest_embedding = torch.cat(
                [test_collabrative_interest_embedding_text, test_collabrative_interest_embedding_img])

            test_id_emb = test_id_emb + test_collabrative_interest_embedding

            test_emb = test_emb + test_id_emb

            expanded_center_indices = self.mapping["center_indices"].unsqueeze(0).expand(batch_size, -1)
            text_interest_embedding = self.interest_embedding(
                torch.gather(expanded_center_indices, dim=-1, index=item_seq)
            )
            expanded_center_indices = self.mapping_img["center_indices"].unsqueeze(0).expand(batch_size, -1)
            img_interest_embedding = self.interest_embedding_img(
                torch.gather(expanded_center_indices, dim=-1, index=item_seq)
            )
            img_emb = img_emb + id_emb + img_interest_embedding
            text_emb = text_emb + id_emb + text_interest_embedding


        batch_size,sequence_length=item_seq.shape
        mask=torch.randint(0,2,(batch_size,sequence_length),device=text_emb.device)
        mask=mask.unsqueeze(-1).expand_as(text_emb)
        aug_emb=text_emb*mask+img_emb*(1-mask)
        aug_emb = self.LayerNorm(aug_emb)
        aug_emb= self.dropout(aug_emb)




        text_emb = self.LayerNorm(text_emb)
        text_emb = self.dropout(text_emb)
        img_emb = self.LayerNorm(img_emb)
        img_emb = self.dropout(img_emb)


        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(item_seq, is_casual=True)
        text_id = self.trm_model.encoder(src=text_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)
        img_id = self.trm_model.encoder(src=img_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)
        text_output = self.gather_indexes(text_id, item_seq_len - 1)

        img_output = self.gather_indexes(img_id, item_seq_len - 1)
        temperature=2
        logits_text=torch.exp(text_output@test_emb.transpose(1,0)/temperature)
        pos_items = interaction[self.POS_ITEM_ID]

        targets=[[pos_items[i],pos_items[i]+item_size]for i in range(len(pos_items))]
        targets=torch.tensor(targets,device=pos_items.device)
        pos_logits=torch.gather(logits_text,dim=-1,index=targets)/temperature
        loss_text=-torch.log(torch.sum(pos_logits,dim=-1)/(torch.sum(logits_text,dim=-1)))
        img_logits =torch.exp(img_output @ test_emb.transpose(1, 0) / temperature)
        pos_items = interaction[self.POS_ITEM_ID]
        targets = [[pos_items[i],pos_items[i]+item_size] for i in range(len(pos_items))]
        targets = torch.tensor(targets, device=pos_items.device)
        pos_logits = torch.gather(img_logits, dim=-1, index=targets) / temperature
        loss_img =-torch.log( torch.sum(pos_logits, dim=-1) /torch.sum(img_logits, dim=-1))

        return (loss_text).mean()+loss_img.mean()


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        test_item_emb = 0
        item_modal_empty_mask_list = []
        item_size=self.plm_embedding.weight.shape[0]
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))
            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]

        if 'text' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)

            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
            item_modal_empty_mask_list.append(text_emb_empty_mask)

        if 'img' in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]
            item_modal_empty_mask_list.append(img_emb_empty_mask)
        batch_size,_=item_seq.shape

        test_emb = torch.cat([test_text_emb, test_img_emb], dim=0)
        if self.train_stage == "transductive_ft":
            test_id_emb, id_emb = self.get_colabrative_pool_embedding(item_seq)
            test_id_emb = torch.cat([test_id_emb, test_id_emb])
            self.mapping["center_indices"].to(item_seq.device)
            test_collabrative_interest_embedding_text = self.interest_embedding(
                self.mapping["center_indices"]
            )
            self.mapping_img["center_indices"].to(item_seq.device)
            test_collabrative_interest_embedding_img = self.interest_embedding_img(
                self.mapping_img["center_indices"]
            )
            test_collabrative_interest_embedding = torch.cat(
                [test_collabrative_interest_embedding_text, test_collabrative_interest_embedding_img])

            test_id_emb = test_id_emb + test_collabrative_interest_embedding

            test_emb = test_emb + test_id_emb

            expanded_center_indices = self.mapping["center_indices"].unsqueeze(0).expand(batch_size, -1)
            text_interest_embedding = self.interest_embedding(
                torch.gather(expanded_center_indices, dim=-1, index=item_seq)
            )
            expanded_center_indices = self.mapping_img["center_indices"].unsqueeze(0).expand(batch_size, -1)
            img_interest_embedding = self.interest_embedding_img(
                torch.gather(expanded_center_indices, dim=-1, index=item_seq)
            )
            img_emb = img_emb + id_emb + img_interest_embedding
            text_emb = text_emb + id_emb + text_interest_embedding


        batch_size,sequence_length=item_seq.shape
        mask=torch.randint(0,2,(batch_size,sequence_length),device=text_emb.device)

        text_emb = self.LayerNorm(text_emb)
        text_emb = self.dropout(text_emb)
        img_emb = self.LayerNorm(img_emb)
        img_emb = self.dropout(img_emb)

        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(item_seq, is_casual=True)
        text_id = self.trm_model.encoder(src=text_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)
        img_id = self.trm_model.encoder(src=img_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)

        text_output = self.gather_indexes(text_id, item_seq_len - 1)

        img_output = self.gather_indexes(img_id, item_seq_len - 1)
        temperature=2
        logits_text=torch.exp(text_output@test_emb.transpose(1,0)/temperature)/torch.sum(torch.exp(text_output@test_emb.transpose(1,0)/temperature))
        img_logits =torch.exp(img_output @ test_emb.transpose(1, 0) / temperature)/torch.sum(torch.exp(text_output@test_emb.transpose(1,0)/temperature))
        modal_output=logits_text+img_logits
        modal_output=modal_output[:,:item_size]+modal_output[:,item_size:]
        return modal_output