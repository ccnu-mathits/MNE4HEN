# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen

    Date:   2021/10/26 16:59

--------------------------------------

"""
import torch
from Tools import read_data,get_sequences,Data_loader
import torch.optim as opt
import random
from TorchModels import NodeEmbedding_model,TargetForecast_model
from sklearn.metrics import roc_auc_score,accuracy_score
from tqdm import trange
import os

class MNE:
    def __init__(self,
                 dataset='assistment09',
                 epoch=1000,
                 m_lr=0.001,
                 embedding_size = 128,
                 neg_ms = 3,
                 dropout = 0.5,
                 UQ_samples = 5,
                 batch_slices=100,
                 earlystop = 10,
                 rho = 1.0,
                 var='anisotropy',
                 proj = 'affine',
                 topk_k=[1, 5, 10, 20, 50],
                 ns_strategy = 'global',
                 device="cuda:0" if torch.cuda.is_available() else 'cpu',
                 ):
        self.embedding_size = embedding_size
        self.device = device
        self.dropout = dropout
        self.batch_slices = batch_slices
        self.m_lr = m_lr
        self.proj = proj
        self.epoch = epoch
        self.neg_ms = neg_ms
        self.UQ_samples = UQ_samples
        self.topk_k = topk_k
        self.earlystop = earlystop
        self.ns_strategy = ns_strategy
        self.rho = rho
        self.var = var

        self.graph = {}
        if dataset == 'assistment09' or dataset == 'assistment12':
            self.graph['name'] = dataset
            self.graph['node_type'] = ['u','q','s','class','teacher']
            self.graph['edge_type'] = ['u_q','q_s','u_class','class_teacher']
            self.graph['node_freq'] = {'u':2,'q':3,'s':1,'class':2,'teacher':1}
            self.graph['edge_completeness'] = {'u_q':True,'q_s':False,'u_class':True,'class_teacher':True}
        elif dataset == 'assistment17':
            self.graph['name'] = dataset
            self.graph['node_type'] = ['u','q','s','school']
            self.graph['edge_type'] = ['u_q','q_s','u_school']
            self.graph['node_freq'] = {'u':2,'q':3,'s':1,'school':1}
            self.graph['edge_completeness'] = {'u_q':True,'q_s':False,'u_school':True}
        self.graph['node_index'],self.graph['edges_list'],self.graph['q_s_test'],self.graph['u_q_test'],\
        self.graph['node_num'], self.graph['edge_index'], self.graph['nodes'] = read_data(self.graph)

        self.graph['q_squences'],self.graph['c_squences'],self.graph['u_squences'],self.graph['squences_maxlength'] \
            = get_sequences(self.graph['edges_list']['u_q'])

        # put data to tensor
        self.graph['edges'] = {}
        for e in self.graph['edges_list']:
            self.graph['edges'][e] = torch.tensor(self.graph['edges_list'][e]).to(self.device)
        for n in self.graph['nodes']:
            self.graph['nodes'][n] = torch.tensor(self.graph['nodes'][n]).to(self.device)
        for u in self.graph['q_squences']:
            self.graph['q_squences'][u] = torch.tensor(self.graph['q_squences'][u]).to(self.device)
            self.graph['c_squences'][u] = torch.tensor(self.graph['c_squences'][u]).to(self.device)
        self.graph['u_q_test'] = torch.tensor(self.graph['u_q_test']).to(self.device)
        # get adj_matrix
        self.graph['adj_matrix'] = torch.eye(self.graph['node_num'],dtype=torch.int8).to(self.device)
        for e in self.graph['edges_list']:
            for edge in self.graph['edges_list'][e]:
                self.graph['adj_matrix'][edge[0], edge[1]] = 1
                self.graph['adj_matrix'][edge[1], edge[0]] = 1



    def train(self):
        print('#' * 17, 'building model', '#' * 17)

        data_loaders_tasks = {}
        # dataloaders for graph reconstruction
        for e_t in self.graph['edge_type']:
            data_loaders_tasks[e_t] = Data_loader(
                dataset=self.graph['edges'][e_t],
                batch_size=int(self.graph['edges'][e_t].size()[0]/(self.batch_slices)),
            )
        # dataloaders for evolved edge weight regression
        data_loaders_tasks['squences'] = Data_loader(
                dataset=torch.tensor(self.graph['u_squences']).to(self.device),
                batch_size=int(self.graph['u_squences'].__len__()/(self.batch_slices)),
            )

        data_loaders_nodes = {}
        # dataloaders for node embeddings
        for n_t in self.graph['node_type']:
            data_loaders_nodes[n_t] = Data_loader(
                dataset=self.graph['nodes'][n_t],
                batch_size=int(self.graph['nodes'][n_t].size()[0]/(self.batch_slices)),
            )

        # models for NodeEmbedding_Model
        self.NodeEmbedding_Model = NodeEmbedding_model(
            node_num = self.graph['node_num'],
            embedding_size = self.embedding_size,
            nodes = self.graph['nodes'],
            device=self.device,
            dropout = self.dropout,
            UQ_samples = self.UQ_samples,
            rho = self.rho,
            proj = self.proj,
            mask = self.graph['adj_matrix'],
        ).to(self.device)
        # models for TargetForecast_Model
        self.TargetForecast_Model = TargetForecast_model(
            edge_type=self.graph['edge_type'],
            embedding_size=self.embedding_size,
            maxlength=self.graph['squences_maxlength'],
            device=self.device,
            q_squences = self.graph['q_squences'],
            c_squences = self.graph['c_squences'],
        ).to(self.device)

        params = [{'params':self.TargetForecast_Model.parameters()},
                  {'params':self.NodeEmbedding_Model.parameters()}]
        optimizer = opt.Adam(params, lr=self.m_lr)

        self.bestACC = 0
        self.bestAUC = 0

        self.bestPrecision = {}
        self.bestRecall = {}
        self.bestF1 = {}
        for i in self.topk_k:
            self.bestPrecision[i] = 0
            self.bestRecall[i] = 0
            self.bestF1[i] = 0

        # tag for early stop
        stop = 0

        # training process...
        print('#' * 17, 'start training', '#' * 17)
        for e in range(self.epoch):
            train_tfloss = 0
            train_embloss = 0

            # train with batches
            for i in trange(self.batch_slices):

                ##### Downstream Tasks #####
                pos_batchs = {}
                for e_t in self.graph['edge_type']:
                    pos_batchs[e_t] = data_loaders_tasks[e_t].get_batch()
                neg_batchs = self.negative_sampling(pos_batchs,self.neg_ms,self.ns_strategy)
                sequences_batch = data_loaders_tasks['squences'].get_batch()

                # get all losses for downstream tasks
                tf_loss = self.TargetForecast_Model.forward(
                    self.graph['edge_completeness'],
                    pos_batchs,
                    neg_batchs,
                    self.NodeEmbedding_Model.node_embeddings,
                    self.neg_ms,
                    sequences_batch,
                )
                ##### Embedding Tasks #####
                node_batchs = {}
                for n_t in self.graph['node_type']:
                    node_batchs[n_t] = data_loaders_nodes[n_t].get_batch()

                # get all losses for embeddings
                emb_loss = self.NodeEmbedding_Model.forward(
                    node_batchs,
                    self.graph['node_freq'],
                    self.var,
                )
                model_loss = tf_loss + emb_loss
                ##### Optimizing #####
                optimizer.zero_grad()
                model_loss.backward()
                optimizer.step()
                train_tfloss += tf_loss
                train_embloss += emb_loss

            # report results for each epoch
            print('-------- Epoch',e,'--------')
            print('tf_loss: %.2f, emb_loss: %.2f' % (train_tfloss,train_embloss))

            print('-------- Evaluation --------')
            # evaluation for Automatic Labeling for Learning Resources
            print('Automatic Labeling for Learning Resources:')
            testPrecision, testRecall, testF1 = self.Evaluation_AL()
            for k in self.topk_k:
                if testF1[k] > self.bestF1[k]:
                    self.bestPrecision[k] = testPrecision[k]
                    self.bestRecall[k] = testRecall[k]
                    self.bestF1[k] = testF1[k]
                    stop = 0
                print('Top %d -- P: %.4f, R: %.4f, F1: %.4f || Best P: %.4f, R: %.4f, F1: %.4f' %
                      (k,testPrecision[k],testRecall[k],testF1[k],self.bestPrecision[k],self.bestRecall[k],self.bestF1[k]))

            # evaluation for Student Performance Prediction
            print('Student Performance Prediction:')
            ACC, AUC = self.Evaluation_SPP()
            if AUC > self.bestAUC:
                self.bestAUC = AUC
                self.bestACC = ACC
                stop = 0
            print('ACC: %.4f, AUC: %.4f || Best ACC: %.4f, Best AUC: %.4f' % (ACC,AUC,self.bestACC,self.bestAUC))
            stop = stop + 1
            if stop >= self.earlystop:
                break


    def Evaluation_AL(self):
        testPrecision = {}
        testRecall = {}
        testF1 = {}
        for k in self.topk_k:
            testPrecision[k] = 0
            testRecall[k] = 0
            testF1[k] = 0

        for q in self.graph['q_s_test']:
            s_set = self.graph['q_s_test'][q]
            q_embedding = self.NodeEmbedding_Model.node_embeddings[q, :]
            candidate_embedding = self.NodeEmbedding_Model.node_embeddings[self.graph['nodes']['s'], :]
            score = (candidate_embedding * q_embedding).sum(1)
            topk_index_max = torch.topk(score, max(self.topk_k)).indices
            topk_candicates_max = self.graph['nodes']['s'][topk_index_max]
            for k in self.topk_k:
                topk_candicates = topk_candicates_max[:k].cpu().tolist()
                right = set(topk_candicates).intersection(set(s_set)).__len__()
                Precision = right / k
                Recall = right / (s_set.__len__())
                F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-8)
                testPrecision[k] += Precision
                testRecall[k] += Recall
                testF1[k] += F1
        test_size = self.graph['q_s_test'].__len__()
        for k in self.topk_k:
            testPrecision[k] /= test_size
            testRecall[k] /= test_size
            testF1[k] /= test_size
        return testPrecision,testRecall,testF1

    def Evaluation_SPP(self):
        u_test = self.graph['u_q_test'][:, 0]
        q_test = self.graph['u_q_test'][:, 1]
        c_test = self.graph['u_q_test'][:, 2]

        u_state_test = []
        for u in u_test.tolist():
            if self.graph['u_squences'].__contains__(u):
                u_state_test.append(
                    self.TargetForecast_Model.envaluate_student_state(u, self.NodeEmbedding_Model.node_embeddings)
                )
            else:
                u_state_test.append(self.TargetForecast_Model.Transformer.user_init_representation.squeeze())

        u_state_test = torch.stack(u_state_test)
        q_state_test = self.NodeEmbedding_Model.node_embeddings[q_test, :]

        pred = torch.sigmoid(torch.sum(u_state_test * q_state_test, 1))
        test_pred_all = pred.detach().cpu().numpy()
        test_pred01_all = pred.ge(0.5).float().detach().cpu().numpy()
        test_y_all = c_test.cpu().numpy()

        ACC = accuracy_score(test_y_all, test_pred01_all)
        AUC = roc_auc_score(test_y_all, test_pred_all)

        return ACC,AUC

    def negative_sampling(self,pos_batchs,ns_num,strategy="global"):
        ns_batchs = {}
        if strategy=="global":
            sample_ratio = [self.graph['edges'][e_t].size()[0] for e_t in self.graph['edge_type']]

        for e_t in pos_batchs:
            batch_size = pos_batchs[e_t].size()[0]
            ns_batch = []
            while ns_batch.__len__() < batch_size*ns_num:
                if strategy=="global":
                    ne_t = random.choices(self.graph['edge_type'],sample_ratio)[0]
                else:
                    ne_t = e_t
                r_head = random.randrange(self.graph['edges_list'][ne_t].__len__())
                r_tail = random.randrange(self.graph['edges_list'][ne_t].__len__())
                neg_head = self.graph['edges_list'][ne_t][r_head][0]
                neg_tail = self.graph['edges_list'][ne_t][r_tail][1]
                ne_index = str(neg_head)+'_'+str(neg_tail)
                if not self.graph['edge_index'][ne_t].__contains__(ne_index):
                    ns_batch.append([neg_head, neg_tail])
            ns_batchs[e_t] = torch.tensor(ns_batch)
        return ns_batchs

    def log_result(self):
        filename = os.path.split(__file__)[-1].split(".")[0]
        f = open("./Results/" + filename + "-" + self.graph['name'] + ".txt", "a+")
        f.write("datasets = " + self.graph['name'] + "\n")
        f.write("embedding_k = " + str(self.embedding_size) + ' proj = '+ str(self.proj) +"\n")
        f.write("ns_strategy = " + str(self.ns_strategy) + " neg_ms = " + str(self.neg_ms) + "\n")
        f.write("dropout = " + str(self.dropout) + " rho = " + str(self.rho) + "\n")
        f.write(" UQ_samples = " + str(self.UQ_samples) + " var = " + str(self.var) + "\n")
        f.write('Automatic Labeling for Learning Resources:'+"\n")
        for k in self.topk_k:
            f.write('Top %d-- Best P: %.4f, R: %.4f, F1: %.4f '%(k,self.bestPrecision[k],self.bestRecall[k],self.bestF1[k]) + "\n")
        f.write('Student Performance Prediction:'+"\n")
        f.write('Best ACC: %.4f, AUC: %.4f' % (self.bestACC,self.bestAUC) + "\n")
        f.write("\n")
        f.write("\n")
        print("The results are logged!!!")
        f.close()

if __name__ == '__main__':
    a = MNE(
        dataset='assistment09',
        embedding_size=128,
        ns_strategy='local',  # [global,local]
        neg_ms=10,  # number of negative samples
        dropout=0.3,  # uncertainty generator in node embedding
        rho=0.5,  # exponent of noises in node embedding prediction
        var='isotropy',  # the var computing type including ['isotropy','anisotropy']
        UQ_samples=10,  # number of samples in UQ
        proj="hyperplane",  # [affine,  hyperplane, MLP, none]
        batch_slices=100,
        m_lr=0.001,
        earlystop=20,
        device='cuda:0',
    )
    a.train()
    a.log_result()



