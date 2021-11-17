# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2021/10/26 16:59
    
--------------------------------------

"""
import torch

def read_data(graph):
    name = graph['name']     #'assistment2009'
    node_type = graph['node_type']     #['u','q','s','class','teacher']
    edge_type = graph['edge_type']     #['u_q','q_s','u_class','class_teacher']
    print('Dataset is loading...')

    # load all nodes
    # node_index = {n_type:{node_id:index}}
    # nodes = {n_type: [n1,n2,...]}
    node_index = {}
    nodes = {}
    node_number = 0
    for n_type in node_type:
        node_index[n_type] = {}
        nodes[n_type] = []
    for n_type in node_type:
        n_path = str('./data/'+name+'/'+n_type+'.txt')
        lines = open(n_path).readlines()
        for line in lines:
            node = line.strip()
            if not node_index[n_type].__contains__(node):
                node_index[n_type][node] = node_number
                nodes[n_type].append(node_number)
                node_number += 1
    # load all edges
    # edges = {e_type:[head_index,tail_index,weight(*),time(*)]}
    edges = {}
    for e_type in edge_type:
        edges[e_type] = []
    for e_type in edge_type:
        e_path = str('./data/' + name + '/' + e_type + '.txt')
        head,tail = e_type.split('_')
        head_index = node_index[head]
        tail_index = node_index[tail]
        lines = open(e_path).readlines()
        for line in lines:
            data = line.strip().split('|')
            if e_type == 'q_s':
                skills = data[1].split(',')
                for skill in skills:
                    edges[e_type].append([head_index[data[0]],tail_index[skill]])
            elif e_type == 'u_q':
                edges[e_type].append([head_index[data[0]], tail_index[data[1]], int(data[2]), int(data[3])])
            else:
                edges[e_type].append([head_index[data[0]],tail_index[data[1]]])

    edge_index = {}
    for e_t in edges:
        edge_index[e_t] = set()
        for edge in edges[e_t]:
            edge_index[e_t].add(str(edge[0])+'_'+str(edge[1]))

    ###########################################################
    # load testsets:
    # TEST1: q_s_test
    # q_s_test = {q_index:[s_index,..]} one question for many skills
    q_s_test = {}
    lines = open('./data/'+name+'/q_s_test.txt').readlines()
    for line in lines:
        data = line.strip().split('|')
        q_index = node_index['q'][data[0]]
        skills = data[1].split(',')
        skills_index = []
        for skill in skills:
            skills_index.append(node_index['s'][skill])
        q_s_test[q_index]=skills_index

    # TEST2: u_q_test
    # u_q_test = [u_index,q_index,correct,time]
    u_q_test = []
    lines = open('./data/' + name + '/u_q_test.txt').readlines()
    for line in lines:
        data = line.strip().split('|')
        u_q_test.append([node_index['u'][data[0]], node_index['q'][data[1]], int(data[2]), int(data[3])])

    print("#" * 50)
    print('Dataset',name, 'is loaded...')
    print('-----which has nodes:')
    for n in node_index:
        print(n,':',node_index[n].__len__())
    print('-----which has edges:')
    for e in edges:
        print(e, ':', edges[e].__len__())

    return node_index,edges,q_s_test,u_q_test,node_number,edge_index,nodes

def get_sequences(list):
    # q_squences: {u_id,[q1,q2,...]}
    # c_squences: {u_id,[c1,c2,...]}
    # u_squences：[u1,u2,...]
    q_squences = {}
    c_squences = {}
    u_squences = []
    for log in list:
        u = log[0]
        i = log[1]
        c = log[2]
        if q_squences.__contains__(u):
            q_squences[u].append(i)
            c_squences[u].append(c)
        else:
            q_squences[u] = [i]
            c_squences[u] = [c]
    for u in q_squences:
        u_squences.append(u)
    max_length = max([q_squences[u].__len__() for u in q_squences])
    return q_squences,c_squences,u_squences,max_length


class Data_loader:
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.datasize = dataset.size()[0]


    def get_batch(self):
        index = torch.randint(self.datasize,[self.batch_size])
        return self.dataset[index]
