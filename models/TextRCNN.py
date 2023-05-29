# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/re_data/temp_train.json'                                # 训练集
        self.dev_path = dataset + '/re_data/temp_dev.json'                                    # 验证集
        self.test_path = dataset + '/re_data/temp_test.json'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/labels.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.device = torch.device('cpu')

        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 5                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                         # lstm隐藏层
        self.num_layers = 1                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)# + config.embed, config.num_classes)


    def forward(self, x):
        # x, _ = x    #torch.Size([128, 32])
        # embed = self.embedding(x)   # [batch_size, sentence_len, embeding]=[128, 32, 300]
        # out, _ = self.lstm(embed)   # [batch_size, sentence_len, hidden_size]=[128, 32, 512]
        # # out = torch.cat((embed, out), 2)    #[batch_size, sentence_len, hidden_size*2]= [128, 32, 812]
        # out = F.relu(out)   #torch.Size([128, 32, 812])
        # out = out.permute(0, 2, 1)  #torch.Size([128, 812, 32])
        # out = self.maxpool(out).squeeze()   #torch.Size([128, 812]) 最大池
        # out = self.fc(out)#torch.Size([128, 10])
        # return out
        cluster_x, _ = x
        new_cluster_x = [n for a in cluster_x for n in a]
        new_cluster_x = torch.LongTensor(new_cluster_x)
        embed = self.embedding(new_cluster_x)
        out, _ = self.lstm(embed)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
        # outs = []
        # for x in cluster_x:
        #     x = torch.LongTensor(x) #[num_clusters, cluster_len]
        #     embed = self.embedding(x)#[num_clusters, cluster_len, embedding]
        #     out, _ = self.lstm(embed)#[num_clusters, cluster_len, hidden_size*2]
        #     out = F.relu(out)
        #     out = out.permute(0,2,1)#[num_clusters, hidden_size*2, cluster_len]
        #     out = self.maxpool(out).squeeze()#[num_clusters, hidden_size*2]
        #     out = self.fc(out)#[num_clusters, num_classes]
        #     outs.append(out)
        # return outs#[batch_size, num_clusters, num_classes]
