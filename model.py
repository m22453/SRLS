import copy
import torch
from sklearn.neighbors import NearestNeighbors
from torch.nn import Parameter


class OurModel(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, n_classes, batch_size,  
                 drop_rate, init_mode, train_mode='sf', beta=0.001,
                 fea_select=1.0, ck=10, n_hetero_layer=2, divide=False, n_step=5) :
        super(OurModel, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.init_mode = init_mode
        self.train_mode = train_mode 
        self.beta = beta
        self.fea_select = fea_select
        self.divide = divide
        self.n_step = n_step

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]


        self.sample_hereo = nn.ModuleList([HeteroGNNLayer(self.feature_dim[0], ck=ck) for _ in range(n_hetero_layer)])
        if self.fea_select == 1.0:
            self.fcl = FeatureSelection(self.feature_dim[0])
        else:
            self.fcl = PartialFeatureCorrelationLoss(self.feature_dim[0], split_ratio=self.fea_select)

        
        # for end for end cluster
        self.cluster_layer = Parameter(torch.Tensor(n_classes, sum(self.feature_dim)))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.loss_function = nn.KLDivLoss(size_average=False)

        self.v = 1.0


    def forward(self, data, momentum, warm_up):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]

        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp = torch.eye(z[0].shape[0]).cuda()
            mp = [mp for _ in range(self.n_views)] 

            # w/o warm up
            # H_list = []
            # for v in range(self.n_views):
            #     z_others = [z_t[u] for u in range(self.n_views) if u != v]
            #     h_v = z_t[v]
            #     for layer in self.sample_hereo:
            #         h_v = layer(h_v, z_others, mode='sample')
            #     H_list.append(h_v)
            
            # mp = [self.kernel_affinity(H_list[i]) for i in range(self.n_views)]

        else:
            H_list = []
            for v in range(self.n_views):
                z_others = [z_t[u] for u in range(self.n_views) if u != v]
                h_v = z_t[v]
                for layer in self.sample_hereo:
                    h_v = layer(h_v, z_others, mode='sample')
                H_list.append(h_v)

            # F_list = []
            # for v in range(self.n_views):
            #     z_others = [z_t[u].t() for u in range(self.n_views) if u != v]
            #     f_v = z_t[v].t() 
            #     for layer in self.feature_hereo:
            #         f_v = layer(f_v, z_others, mode='feature')
            #     F_list.append(f_v.t())


            # tmp = [self.glu(torch.cat((F_list[i],H_list[i]), dim=-1)) for i in range(self.n_views)]
            # H_final = torch.cat([torch.max(F_list[i],H_list[i]) for i in range(self.n_views)], dim=1)  
            # H_final = self.glu(torch.cat((identify, x), dim=-1)) + identify
            # H_final = torch.cat([torch.max(H_list[i],F_list[i]) for i in range(self.n_views)], dim=1)  

            if self.divide:
                mp = [self.kernel_affinity(z_t[i], step=self.n_step) for i in range(self.n_views)]
            else:
                H_final = torch.cat(H_list, dim=1)
                mp = [self.kernel_affinity(H_list[i], step=self.n_step) for i in range(self.n_views)]
                # mp = [self.multi_view_knn(H_list) for _ in range(self.n_views)]

            # l_inter = (self.cl(p[0], z_t[1], mp[1]) + self.cl(p[1], z_t[0], mp[0])) / 2
            # l_intra = (self.cl(z[0], z_t[0], mp[0]) + self.cl(z[1], z_t[1], mp[1])) / 2
            
            # l_cross_corr = (self.fcl(p[0], z_t[1]) + self.fcl(p[1], z_t[0]) )/2 + (self.fcl(z[0], z_t[0]) + self.fcl(z[1], z_t[1])) / 2


        # intra view          
        l_intra = 0 
        l_f_intra = 0
        for i in range(self.n_views):
            l_intra += self.cl(z[i], z_t[i], mp[i]) 
            l_f_intra += self.fcl(z[i], z_t[i]) 

            # target encoder ablation
            # l_intra += self.cl(z[i], z[i], mp[i]) 
            # l_f_intra += self.fcl(z[i], z[i]) 


            l_intra /= 2
            l_f_intra /= 2

        # inter view 
        l_inter = 0
        l_f_inter = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                if i == j:
                    continue

                l_inter += self.cl(p[j], z_t[i], mp[i])
                l_f_inter += self.fcl(p[j], z_t[i])
                # target encoder ablation
                # l_inter += self.cl(p[j], z[i], mp[i])
                # l_f_inter += self.fcl(p[j], z[i])

            l_inter /= 2
            l_f_inter /= 2

        
        l_ = l_intra + l_inter
        l_f = l_f_intra + l_f_inter

        if self.divide: # degrad to divide
            loss = l_ + 0 * l_f
            return loss

        if self.train_mode == 'sf': 
            if self.init_mode == 'f' and warm_up:
                loss = l_f
            elif self.init_mode == 's' and warm_up:
                loss = l_
            else:
                loss = l_ + self.beta * l_f
        elif self.train_mode == 's':
            if warm_up:
                loss = l_
            else:
                loss = l_
        elif self.train_mode == 'f':
            if warm_up:
                loss = l_f
            else:
                loss = l_f

        return loss
    
    def fine_tuning(self, data, momentum):
        self._update_target_branch(momentum)
        
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        z_p = torch.cat([self.cross_view_decoder[i](z[i]) for i in range(self.n_views)], dim=1)
        z_p = L2norm(z_p)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        weight = (q ** 2) / torch.sum(q, 0)
        p = (weight.t() / torch.sum(weight, 1)).t() 

        
        kl_loss = self.loss_function(q.log(), p) / q.shape[0]
        return kl_loss


    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        z_t = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]

        z = [L2norm(z_t[i]) for i in range(self.n_views)]


        return z
    
    @torch.no_grad()
    def extract_q(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        z_p = torch.cat([self.cross_view_decoder[i](z[i]) for i in range(self.n_views)], dim=1)
        z_p = L2norm(z_p)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_p.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q


import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FeatureSelection(nn.Module):
    def __init__(self, hidden_dim, lambd=0.0051) -> None:
        super(FeatureSelection, self).__init__()

        self.lambd = lambd
    
        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, x1, x2):
        
        batch_size = x1.size(0)
        
        # empirical cross-correlation matrix
        c = self.bn(x1).T @ self.bn(x2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss



class PartialFeatureCorrelationLoss(nn.Module):
    def __init__(self, hidden_dim, lambd=0.0051, split_ratio=0.9) -> None:
        super().__init__()
        self.lambd = lambd
        self.split_idx = int(hidden_dim * split_ratio) 

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # x1 = self.projector(x1)
        # x2 = self.projector(x2)
        
        # 归一化特征
        x1 = self.bn(x1)
        x2 = self.bn(x2)
        
        # 交叉相关矩阵
        c = x1.T @ x2 / batch_size
        
        # 仅约束前 split_idx 个对角线元素
        diag_elements = torch.diagonal(c)
        on_diag = diag_elements[:self.split_idx].add(-1).pow(2).sum()
        
        # 非对角线约束（全部）
        off_diag = off_diagonal(c).pow(2).sum()
        
        loss = on_diag + self.lambd * off_diag
        return loss


class HeteroGNNLayer(nn.Module):
    def __init__(self, latent_dim, ck=10):
        super(HeteroGNNLayer, self).__init__()
        # W1, W2, W3 for self, intra-view, inter-view
        self.W1 = nn.Linear(latent_dim, latent_dim)
        self.W2 = nn.Linear(latent_dim, latent_dim)
        self.W3 = nn.Linear(latent_dim, latent_dim)
        self.k = ck
        # self.W4 = nn.Linear(latent_dim, latent_dim)

    
    # @torch.no_grad()
    # def idx_highorder_knn(self, z, temperature=0.1, step: int = 5, k: int = 5):
    #     z = L2norm(z)
    #     G = (2 - 2 * (z @ z.t())).clamp(min=0.)  
    #     G = torch.exp(-G / temperature)         
    #     G = G / G.sum(dim=1, keepdim=True)     
        
    #     G = torch.matrix_power(G, step)         
        
    #     _, topk_indices = torch.topk(G, k=k+1, dim=1)
        
    #     return topk_indices  #  [N, k+1] 
    
    # @torch.no_grad()
    # def idx_correlation_knn(self, z_v: torch.Tensor, z_u: torch.Tensor, k: int = 5) -> torch.Tensor:
    #     num_nodes = z_v.size(0)
    #     k = min(k, num_nodes - 1)
        
    #     z_v_centered = z_v - z_v.mean(dim=1, keepdim=True)  # [n, d]
    #     z_u_centered = z_u - z_u.mean(dim=1, keepdim=True)  # [n, d]
        
    #     norm_v = torch.norm(z_v_centered, p=2, dim=1, keepdim=True)  # [n, 1]
    #     norm_u = torch.norm(z_u_centered, p=2, dim=1, keepdim=True)  # [n, 1]
    #     corr_matrix = (z_v_centered @ z_u_centered.T) / (norm_v @ norm_u.T).clamp(min=1e-8)
        
    #     _, indices = torch.topk(corr_matrix, k=k+1, dim=1)  # [n, k+1]
        
    #     return indices
    
    # @torch.no_grad()
    # def idx_faiss_knn(self, query_feats, other_feats, k=10):
    #     query = query_feats.cpu().detach().numpy().astype('float32').copy()
    #     db = other_feats.cpu().detach().numpy().astype('float32').copy()

    #     # cosine similarity
    #     faiss.normalize_L2(query)
    #     faiss.normalize_L2(db)

    #     # index = faiss.IndexFlatIP(query.shape[1])  # Cosine similarity via inner product
    #     index = faiss.IndexFlatL2(query.shape[1])
    #     # nlist = 5  # 聚类中心数量
    #     # index = faiss.IndexIVFFlat(index, query.shape[1], faiss.METRIC_L2)
    #     # index.nlist = nlist  # 手动设置聚类中心数量
    #     # index.train(db)  # 必须先训练索引
    #     index.add(db)        

    #     _, indices = index.search(query, k)  # [n, k]

    #     # # top-k feature
    #     # neighbors = []
    #     # for i in range(query.shape[0]):
    #     #     neighbors.append(other_feats[indices[i]])  # each [k, d]
    #     # neighbors = torch.stack(neighbors, dim=0)  # [n, k, d]
    #     # return neighbors
    #     indices = torch.from_numpy(indices).long().to(query_feats.device)
        
    #     return indices #  [N, k] 

    
    @torch.no_grad()
    def idx_affinity_knn(self, z: torch.Tensor) -> torch.Tensor:
        num_nodes = z.size(0)
        k = min(self.k, num_nodes - 1) 
        z_np = z.cpu().numpy() if z.is_cuda else z.numpy()
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(z_np)
        _, indices = nbrs.kneighbors(z_np)
        
        indices = torch.from_numpy(indices).long().to(z.device)
        
        return indices #  [N, k+1] 


    @torch.no_grad()
    def indices_to_weighted_adj(
        self,
        indices: torch.Tensor, 
        values = None,  # Shape [num_nodes, k], e.g., similarity/correlation values
        symmetric: bool = False,
        row_normalize: bool = True  
    ) -> torch.Tensor:
        num_nodes, k = indices.shape
        device = indices.device
        
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        
        row_indices = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k)

        if values == None:
            adj[row_indices.flatten(), indices.flatten()] = 1
            if symmetric:
                adj = (adj + adj.T).clamp(max=1)  # Ensure values are 0 or 1
        else:

            adj[row_indices.flatten(), indices.flatten()] = values.flatten()

            if symmetric:
                adj = (adj + adj.T) / 2
        
        if row_normalize:
            row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8) 
            adj = adj / row_sums
        
        return adj

    def forward(self, z_v, z_others, mode):
        # z_v: [n, d], z_others: list of [n, d], mode='feature' or 'sample'
        # self feature
        h_self = self.W1(z_v)  # [n, d]

        if mode == 'sample':

            # intra-view 
            adj_v = self.indices_to_weighted_adj(self.idx_affinity_knn(z_v))
            agg_intra = adj_v @ z_v
            h_intra = self.W2(agg_intra)

            # inter-view mean of same instance
            if len(z_others) > 0:
                
                agg_inter = torch.zeros_like(agg_intra)
                for z_v in z_others:
                    adj_v = self.indices_to_weighted_adj(self.idx_affinity_knn(z_v))
                    agg_inter += adj_v @ z_v
                h_inter = self.W3(agg_inter / len(z_others))


            else:
                h_inter = torch.zeros_like(h_self)

        h = h_self +  h_intra +  h_inter 
        return F.relu(h)


class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=True, act_layer=None, drop_out=0.0, norm_last_layer=False):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            if norm_layer:
                layers.append(nn.BatchNorm1d(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
    
class MultiViewIntegration(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * feature_dim, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, H_list, F_list):
        # 计算均值
        h_mean = torch.cat(F_list, dim=1)  
        f_mean = torch.cat(H_list, dim=1)  
        
        # 融合
        combined = torch.cat([h_mean, f_mean], dim=1)
        return self.fusion(combined)