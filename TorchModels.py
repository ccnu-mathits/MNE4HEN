# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2021/10/26 16:59
    
--------------------------------------

"""
import torch
import math


# storage of nodeembeddings
# measuring the gaps between standard embeddings and predicted embeddings
class NodeEmbedding_model(torch.nn.Module):
    def __init__(self,node_num,embedding_size,nodes,device,dropout,UQ_samples,rho,proj,mask):
        super(NodeEmbedding_model,self).__init__()
        self.node_embeddings = torch.nn.Parameter(torch.randn(node_num,embedding_size)*0.01)
        self.GNN = GNN(node_num,embedding_size,nodes,device,dropout,UQ_samples,proj)
        self.rho = rho
        self.mask = mask

    def forward(self,node_batchs, node_feq, var_type):
        smooth_factor = 1e-3
        emb_loss = 0
        for n_t in node_batchs:
            batch_index = node_batchs[n_t]
            pre_embs_mean,pre_embs_var = self.GNN.forward(batch_index,self.mask[batch_index,:])
            st_embs = self.node_embeddings[batch_index]
            if var_type == 'anisotropy':
                noise_var = (pre_embs_var + smooth_factor).__pow__(self.rho)
            elif var_type == 'isotropy':
                noise_var = ((pre_embs_var.mean(1) + smooth_factor).__pow__(self.rho))
            else:
                print('Wrong var_type, chose from ["isotropy","anisotropy"]!!!')
                quit()
            # print(self.var_weight.detach().cpu().numpy(),(1/noise_var).detach().mean().cpu().numpy())
            noise = st_embs-pre_embs_mean
            if var_type == 'anisotropy':
                n_loss1 = 0.5*(((1/noise_var)*noise*noise).mean(1).sum())
            elif var_type == 'isotropy':
                n_loss1 = 0.5*(((1/noise_var)*((noise*noise).mean(1))).sum())
            n_loss = n_loss1
            emb_loss += n_loss * node_feq[n_t]
        return emb_loss

    def get_q_embedding(self,q_index):
        return self.node_embeddings[q_index,:]

# the model generating embeddings
class GNN(torch.nn.Module):
    def __init__(self, node_num, embedding_size,nodes,device,dropout,UQ_samples,proj):
        super(GNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.proj = proj  # [affine,  hyperplane]
        # Input feature H0: {node_type: feature_matrix}
        self.H0 = {}
        # Projection matrix proj_m:{node_type: proj_matrix}
        self.proj_m = {}
        for n_t in nodes:
            self.H0[n_t] = torch.nn.Parameter(torch.randn(nodes[n_t].size()[0], embedding_size) * 0.01)
            self.register_parameter(n_t+'H0', self.H0[n_t])
            if self.proj == 'affine':
                self.proj_m[n_t] = torch.nn.Parameter(torch.randn(embedding_size, embedding_size) * 0.01)
                self.register_parameter(n_t + 'proj', self.proj_m[n_t])
            elif self.proj == 'hyperplane':
                self.proj_m[n_t] = torch.nn.Parameter(torch.randn(embedding_size,1) * 0.01)
                self.register_parameter(n_t + 'proj', self.proj_m[n_t])
            elif self.proj == 'MLP':
                self.proj_m[n_t] = torch.nn.Sequential(
                    torch.nn.Linear(embedding_size, embedding_size),
                    torch.nn.ReLU(True),
                    torch.nn.Linear(embedding_size, embedding_size),
                ).to(device)
            else:
                self.proj_m[n_t] = 0


        self.UQ_samples = UQ_samples
        self.att_w1 = torch.nn.Parameter(torch.randn(embedding_size, 1) * 0.01)
        self.att_w2 = torch.nn.Parameter(torch.randn(embedding_size, 1) * 0.01)

    def forward(self, batch, mask):
        H_out_batch = []

        H0_p_set = []
        for n_t in self.H0:
            if self.proj == 'affine':
                H0_n_p = self.H0[n_t].mm(self.proj_m[n_t])
            elif self.proj == 'hyperplane':
                H0_n_p = self.H0[n_t] - (self.H0[n_t].mm(self.proj_m[n_t])).mm(self.proj_m[n_t].t())
            elif self.proj == 'MLP':
                H0_n_p = self.proj_m[n_t](self.H0[n_t])
            else:
                H0_n_p = self.H0[n_t]
            H0_p_set.append(H0_n_p)
        H0_p = torch.cat(H0_p_set,0)
        H0_batch_p = H0_p[batch]

        score_1 = H0_batch_p.mm(self.att_w1)
        score_2 = H0_p.mm(self.att_w2)
        score = score_1 + score_2.t()
        mask1 = (-1 / mask) + 1
        att_score = score + mask1
        att = torch.softmax(att_score, 1)

        for i in range(self.UQ_samples):
            Hatt_out_batch = torch.mm(att,self.dropout(H0_p))
            H_out_batch.append(Hatt_out_batch+H0_batch_p)

        H_out_batch = torch.stack(H_out_batch)
        H_out_batch_mean = H_out_batch.mean(0)
        H_out_batch_var = H_out_batch.var(0, unbiased=False).detach()

        return H_out_batch_mean,H_out_batch_var



# compute loss for all downstream tasks according nodeembeddings
# graph reconstruction and evolved edge weight regression
class TargetForecast_model(torch.nn.Module):
    def __init__(self,edge_type,embedding_size,maxlength,device,q_squences,c_squences):
        super(TargetForecast_model,self).__init__()

        # tasks {task:weight}
        self.tasks_sd = {}
        # graph reconstruction tasks
        for e_t in edge_type:
            self.tasks_sd[e_t] = torch.nn.Parameter(torch.tensor(1.0))
            self.register_parameter(e_t + '_weight', self.tasks_sd[e_t])
        self.tasks_sd['squences'] = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter('squences_weight', self.tasks_sd['squences'])

        self.Transformer = Trasnsformer_model(
            EMBEDDING_SIZE = embedding_size,
            N_HEADS = 1,
            DROPOUT = 0,
            MAX_LENGTH = maxlength,
            WEIGHT_AMOUNT = 2,
            DEVICE = device,
        )
        self.q_squences = q_squences
        self.c_squences = c_squences

    def forward(self,edge_completeness,pos_batchs,neg_batchs,embeddings,neg_ms, sequences_batch):
        # edge_completeness: {'u_q':True,'q_s':False,'u_class':True,'class_teacher':True}
        # pos_batchs: {edge_typr: tensor(edges)}
        # neg_batchs: {edge_typr: tensor(edges)}
        # embeddings: {node_type: tensor(node_num,embedding_size)}
        # neg_ms: neg_samples times
        # sequences_batch: [u,...]

        tf_loss = 0

        # loss for graph reconstruction
        for e_t in pos_batchs:
            pos_head_embedding = embeddings[pos_batchs[e_t][:, 0], :]
            pos_tail_embedding = embeddings[pos_batchs[e_t][:, 1], :]
            neg_head_embedding = embeddings[neg_batchs[e_t][:, 0], :]
            neg_tail_embedding = embeddings[neg_batchs[e_t][:, 1], :]
            if edge_completeness[e_t]:
                pos_score = (pos_head_embedding * pos_tail_embedding).sum(1)
                neg_score = (neg_head_embedding * neg_tail_embedding).sum(1)
                loss = -torch.log(torch.sigmoid(pos_score)+1e-8).sum() \
                       - torch.log(torch.sigmoid(-neg_score)+1e-8).sum()
                p_t = pos_head_embedding.size()[0] + neg_head_embedding.size()[0]
            else:
                pos_head_embedding_e = pos_head_embedding.repeat(neg_ms, 1)
                pos_tail_embedding_e = pos_tail_embedding.repeat(neg_ms, 1)
                pos_score = (pos_head_embedding_e * pos_tail_embedding_e).sum(1)
                neg_score_h = (neg_head_embedding * pos_tail_embedding_e).sum(1)
                neg_score_t = (pos_head_embedding_e * neg_tail_embedding).sum(1)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score_h)+1e-8).sum() \
                       - torch.log(torch.sigmoid(pos_score - neg_score_t)+1e-8).sum()
                p_t = neg_head_embedding.size()[0] * 2
            tf_loss += loss * (1/(self.tasks_sd[e_t].__pow__(2)+1e-8)) + p_t * torch.log(self.tasks_sd[e_t]+1e-8)

        # loss for evolved edge weight regression
        sequences_batch = sequences_batch.cpu().tolist()
        u_state_batch = []
        q_state_batch = []
        c_s_batch = []
        for u in sequences_batch:
            q_s = self.q_squences[u]
            c_s = self.c_squences[u]
            u_state,q_state,_ = self.Transformer.forward(q_s, c_s, embeddings.t())
            u_state_batch.append(u_state)
            q_state_batch.append(q_state)
            c_s_batch.append(c_s)
        u_state_batch = torch.cat(u_state_batch,0)
        q_state_batch = torch.cat(q_state_batch,0)
        c_s_batch = torch.cat(c_s_batch,0).float()
        pred_batch = torch.sigmoid(torch.sum(u_state_batch * q_state_batch, 1))
        loss = -(c_s_batch*torch.log(pred_batch+1e-8)+(1-c_s_batch)*torch.log(1-pred_batch+1e-8)).sum()
        p_t = q_s.size()[0]
        tf_loss += loss * (1 / (self.tasks_sd['squences'].__pow__(2)+1e-8)) + p_t * torch.log(self.tasks_sd['squences']+1e-8)

        return tf_loss


    def envaluate_student_state(self,u,embeddings):
        _,_,student_state = self.Transformer.forward(self.q_squences[u], self.c_squences[u], embeddings.t())
        return student_state.detach()


# model for evolved edge weight regression
class Trasnsformer_model(torch.nn.Module):
    def __init__(self,EMBEDDING_SIZE,N_HEADS,DROPOUT,MAX_LENGTH,WEIGHT_AMOUNT,DEVICE):
        super(Trasnsformer_model, self).__init__()

        self.user_init_representation = torch.nn.Parameter(torch.randn(1,EMBEDDING_SIZE)*0.01)
        self.weight_representations = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE,WEIGHT_AMOUNT)*0.01)

        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.N_HEADS = N_HEADS
        self.W_q = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE*N_HEADS,EMBEDDING_SIZE)*0.01)
        self.W_k = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE*N_HEADS,EMBEDDING_SIZE)*0.01)
        self.W_v = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE*N_HEADS,EMBEDDING_SIZE)*0.01)
        self.Mask = (-1/torch.triu(-torch.ones(MAX_LENGTH,MAX_LENGTH))-1).to(DEVICE)
        self.reduction = 1/torch.sqrt(torch.tensor(EMBEDDING_SIZE))
        self.Head_agg = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE,N_HEADS*EMBEDDING_SIZE)*0.01)

        # None
        self.position_encoding = torch.zeros(EMBEDDING_SIZE, MAX_LENGTH).to(DEVICE)

        # Fully learnable
        # self.position_encoding = torch.nn.Parameter(torch.randn(EMBEDDING_SIZE,MAX_LENGTH)*0.01)

        # Fixed sinusoidal
        # pe = torch.zeros(MAX_LENGTH,EMBEDDING_SIZE)
        # position = torch.arange(0,MAX_LENGTH).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0,EMBEDDING_SIZE,2)*-(math.log(10000)/EMBEDDING_SIZE))
        # pe[:,0::2] = torch.sin(position*div_term)
        # pe[:,1::2] = torch.cos(position*div_term)
        # self.position_encoding = pe.t().to(DEVICE)

        # Learnable sinusoidal
        # pe = torch.zeros(MAX_LENGTH, EMBEDDING_SIZE).to(DEVICE)
        # position = torch.arange(0, MAX_LENGTH).unsqueeze(1).to(DEVICE)
        # self.div_term = torch.nn.Parameter(torch.randn(int(EMBEDDING_SIZE / 2)) * 0.01).to(DEVICE)
        # pe[:, 0::2] = torch.sin(position * self.div_term).to(DEVICE)
        # pe[:, 1::2] = torch.cos(position * self.div_term).to(DEVICE)
        # self.position_encoding = pe.t().to(DEVICE)


    def forward(self, sq, cq, item_representations):
        length = sq.size(0)
        I_w = self.weight_representations[:,cq]
        I_q = item_representations[:,sq]
        I = I_w + I_q

        q_flat = self.W_q@I
        k_flat = self.W_k@I
        v_flat = self.W_v@I
        q = q_flat.reshape(self.N_HEADS,self.EMBEDDING_SIZE,length)
        k = k_flat.reshape(self.N_HEADS,self.EMBEDDING_SIZE,length)
        v = v_flat.reshape(self.N_HEADS,self.EMBEDDING_SIZE,length)

        mask = self.Mask[:length,:length]
        score = torch.bmm(k.transpose(1,2),q)*self.reduction
        s = score+mask

        # att: N_HEADS*length*length
        att = torch.softmax(s,1)
        # b: N_HEADS*EMBEDDING_SIZE*length
        b = torch.bmm(v,att)
        # o: EMBEDDING_SIZE*length
        o = (self.Head_agg)@(b.reshape(self.N_HEADS*self.EMBEDDING_SIZE,length))

        # Residual connection
        # o = o + I

        # state: length*EMBEDDING_SIZE
        state = o.t()
        u_state = torch.cat([self.user_init_representation,state],0)

        return u_state[:-1,:], I_q.t(), u_state[-1,:]
