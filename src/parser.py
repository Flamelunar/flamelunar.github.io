import sys

from sympy import true
sys.path.append("/home/ljj/3-biaffine-taketurn/srcmambaformer")
sys.path.append("/home/ljj/3-biaffine-taketurn/src+mambaformer/norm.py")
sys.path.append("/home/ljj/3-biaffine-taketurn/src-matchingloss-review-3lxiugai/ContraNorm.py")
from pytest import param
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random
from config import *
from optimizer import *
from classifiermodel import *
from nn_modules import *
import time
from instance import *
from CPMLSTM import *
from vocab import *
from dataset import *
import shutil
import os
import warnings
from classifier import *
from kan import KAN, KANLinear


from bertembed import *
from bertvocab import *

from torch import Tensor

from mymamba import MambaBlock, MambaModel
from torch_multi_head_attention import MultiHeadAttention
from norm import myLayerNorm
from mambaformer import Mambaformer, Configs
from ContraNorm import ContraNorm




class Parser(object):
    def __init__(self, conf):
        self._conf = conf
        self._source_out = Tensor
        self._torch_device = torch.device(self._conf.device)  # device(type='cuda', index=1)
        self._use_cuda, self._cuda_device = ('cuda' == self._torch_device.type, self._torch_device.index) #True, 1
        if self._use_cuda:
            assert 0 <= self._cuda_device < 8
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_device)
            self._cuda_device = self._cuda_device
        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []
        self._unlabel_train_datasets = []
        self._word_dict = VocabDict('words')
        self._tag_dict = VocabDict('postags')
        # there may be more than one label dictionaries in the multi-task learning scenario
        self._label_dict = VocabDict('labels')
        self._ext_word_dict = VocabDict('ext_words')
        if self._conf.is_charlstm:
            self._char_dict = VocabDict('chars')
        # self._charlstm_layer = []
        self._ext_word_emb_np = None

        self._all_params_requires_grad = []
        self._all_params = []
        self._all_layers = []
        self._input_layer = None
        self.use_meta: bool
        self.match_only: bool
        self.is_diff_loss: bool
        
        if self._conf.is_use_lstm:       
            if self._conf.is_shared_lstm:
                self._lstm_layer = []
                self._gate_lstm = []
            else:
                self._lstm_layer = None
        else:
            self._lstm_layer = None
            
        if self._conf.is_use_mamba:
            self._mambaformer_layer = []
            self._lstm_layer = []
        else:
            self._mambaformer_layer = None
            
        self._proj_layer = []   
             
        self._normlayer1 = []
        self._normlayer2 = []
        
        self._second_lstm_layer = []
        
        if self._conf.is_adversary:
            # 1.初始化条件：
            # - 代码首先使用“if ”语句检查两个条件：
            # - 'self._conf.is_adrivalry'：此条件检查模型配置中是否启用了对抗学习。如果已启用（“True”），它将继续初始化对抗性组件。
            # - 'self._conf.is_shared_lstm'：此条件检查是否使用了共享LSTM图层。根据此条件，将不同的层添加到模型中。
            self._classficationD = []
            self._linear = []
            
        self._kan_layer = []
        
        self._multihead_attention_layer = []
               
        self._mlp_layer = []
            
        self._bi_affine_layer_arc = []
        self._bi_affine_layer_label = []

        self._eval_metrics = EvalMetrics()

        self._domain_batch = torch.arange(5).cuda(self._cuda_device)
        
        if self._conf.is_meta:
            self.use_meta,self.match_only =True,False
            self._loss_weight_type = 'relu6'
            self._loss_weight_init = 1.0
            self._meta_params_requires_grad = []
            self._meta_params = []
            self._meta_layers = []
            self._feature_match_layer = []
            self._meta_optimizer = None
            self._meta_optimizer_type = 'sgd'
            self._pairs = []
            self._source_model_list = []
            self._target_model_list = []
            for i in range(self._conf.lstm_layer_num):
                self._source_model_list.append(2 * self._conf.lstm_hidden_dim)
                self._target_model_list.append(2 * self._conf.lstm_hidden_dim)
                for j in range(self._conf.lstm_layer_num):
                    self._pairs.append((i,j))


    # self._domain_batch = torch.arange(5).cuda(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def add_lstm(self, i, lstm_input_size, lstm_layer_num):
        self._lstm_layer.append(MyLSTM('lstm_' + str(i), \
                                       input_size=lstm_input_size, hidden_size=self._conf.lstm_hidden_dim, \
                                       num_layers=lstm_layer_num, bidirectional=True, \
                                       dropout_in=self._conf.lstm_input_dropout_ratio, \
                                       dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp,
                                       is_fine_tune=True))

    def add_second_lstm(self, i, mamba_input_size, lstm_layer_num):
        self._second_lstm_layer.append(MyLSTM('secondlstm_' + str(i), \
                                       input_size=mamba_input_size, hidden_size=self._conf.lstm_hidden_dim, \
                                       num_layers=lstm_layer_num, bidirectional=True, \
                                       dropout_in=self._conf.lstm_input_dropout_ratio, \
                                       dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp,
                                       is_fine_tune=True))
    
    def add_lstm_cpm(self, lstm_input_size):
        self._lstm_layer.append(UniCPM_LSTM('CPMbilstm', input_size=lstm_input_size, \
                                            hidden_size=self._conf.lstm_hidden_dim,
                                            task_dim_size=self._conf.domain_emb_dim, \
                                            num_layers=1, batch_first=True, bidirectional=True, \
                                            dropout_in=self._conf.lstm_input_dropout_ratio, \
                                            dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp))
    
    def add_mambaformer(self, i, configs):

        self._mambaformer_layer.append(Mambaformer('mambaformer_' + str(i), configs)) #输入输出维度不变
        
    def add_proj1(self, i, mlp_input_size, mlp_output_size):
        
        # self._proj_layer.append(MLPLayer('proj_mlp' + str(i + 1), activation=nn.LeakyReLU(0.1), input_size=mlp_input_size, \
        #                                 hidden_size=mlp_output_size))
        
        self._proj_layer.append(KAN('kan' + str(i), layers_hidden=[mlp_input_size, mlp_output_size]))
   
            
    
    def add_norm1(self, i, dim):
        self._normlayer1.append(myLayerNorm('layernorm1'+ str(i), dim).cuda(self._cuda_device))
          
    def add_multi_head_attention(self):
        # 使用多头注意力
        self._multihead_attention_layer.append(MultiHeadAttention('MultiHeadAttention', in_features=self._conf.lstm_hidden_dim * 2, \
                head_num=self._conf.attention_head_num, bias=True, activation=F.relu).cuda(self._cuda_device))
    
    def add_norm2(self, dim):
        self._normlayer2.append(ContraNorm("contraNorm", dim=dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False))
        
    def add_mlp_biaffine(self, i, mlp_input_size):
        # 使用kan
        if self._conf.is_use_kan:
            self._kan_layer.append(KAN('kan' + str(i + 1), layers_hidden=[mlp_input_size, 2 * (
                                                    self._conf.mlp_output_dim_arc + self._conf.mlp_output_dim_rel)]).cuda(self._cuda_device))
        # 使用mlp
        if self._conf.is_use_mlp:
            self._mlp_layer.append(MLPLayer('mlp' + str(i + 1), activation=nn.LeakyReLU(0.1), input_size=mlp_input_size, \
                                        hidden_size=2 * (
                                                self._conf.mlp_output_dim_arc + self._conf.mlp_output_dim_rel)))
        
        self._bi_affine_layer_arc.append(BiAffineLayer('biaffine-arc' + str(i + 1), self._conf.mlp_output_dim_arc, \
                                                       self._conf.mlp_output_dim_arc, 1, bias_dim=(1, 0)))
        self._bi_affine_layer_label.append(BiAffineLayer('biaffine-label' + str(i + 1), self._conf.mlp_output_dim_rel, \
                                                         self._conf.mlp_output_dim_rel, self._label_dict.size(),
                                                         bias_dim=(2, 2)))

    # create and init all the models needed according to config
    def init_models(self):
        self.use_meta = True
        self.match_only = True
        self.is_diff_loss = True
        bert_path = "/home/ljj/xlm-roberta-base/"
        bert_dim, bert_layer = 768, 4
        # 和bertembed中的bert_outs = bert_outs[len(bert_outs) - self.bert_layer: len(bert_outs)]
        # 获取后四层的表示，这个是一个list，里面存放了所有层的表示  12-4=8，获取 8，9，10，11层 下标是从0开始到11共12层
        assert self._ext_word_dict.size() > 0 and self._ext_word_emb_np is not None and self._word_dict.size() > 0
        self._input_layer = InputLayer('input', self._conf, self._word_dict.size(), self._ext_word_dict.size(), \
                                       self._char_dict.size(), self._tag_dict.size(), self._ext_word_emb_np, \
                                       bert_path, bert_dim, bert_layer)

        bert_vocab_path = "/home/ljj/xlm-roberta-base/"
        # bert_vocab_path = "/home/liying/xlm-roberta-base/tokenizer.json"
        self.bertvocab = Vocab(bert_vocab_path)
        
        lstm_input_size = self._conf.word_emb_dim + self._conf.tag_emb_dim
        mamba_input_size = self._conf.word_emb_dim + self._conf.tag_emb_dim  #200
        #lstm_input_size_domain = self._conf.word_emb_dim + self._conf.tag_emb_dim + self._conf.domain_emb_dim
        
        """初始化一个线性层"""
        hidden_size, input_size = 2 * self._conf.lstm_hidden_dim, mamba_input_size #800，200
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size).cuda(self._cuda_device) #200 -> 800
        weights = orthonormal_initializer(hidden_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights).cuda(self._cuda_device)
        self.linear.weight.requires_grad = True
        b = np.zeros(hidden_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b).cuda(self._cuda_device)
        self.linear.bias.requires_grad = True
        
        self.norm1 = nn.LayerNorm(2 * self._conf.lstm_hidden_dim).cuda(self._cuda_device) 
        self.dropout = nn.Dropout(self._conf.dropout_1).cuda(self._cuda_device) 
        configs = Configs(
            d_model=mamba_input_size, #200
            n_heads=8,
            d_layers=3,
            dropout=0.5,
            output_attention=False,
            c_out=mamba_input_size,   #200
            factor=5
        ) 
        
        
        """仅使用lstm，mlp的维度变成800"""
        if self._conf.is_use_lstm and self._conf.is_use_mamba is False:
            #"""share lstm"""
            if  self._conf.is_shared_lstm:
                self.add_lstm(0, lstm_input_size, self._conf.lstm_layer_num)  # self._conf.lstm_layer_num)
            else:
                self.add_lstm(1, lstm_input_size)
                
            #"""注意力机制融合特征"""
            if self._conf.is_use_multi_head_attention:
                self.add_multi_head_attention()
            
            #"""multi or single"""    
            if self._conf.is_multi:
                for i in range(self._conf.domain_size):
                    self.add_mlp_biaffine(i, 2 * self._conf.lstm_hidden_dim)
            else:
                self.add_mlp_biaffine(1, 2 * self._conf.lstm_hidden_dim)
                print("1 is running")   
                
            #"""adversary"""    
            if self._conf.is_adversary:
                self._classficationD.append(ClassificationD('classficationd', activation=nn.ReLU(), \
                                                            input_size=2 * self._conf.lstm_hidden_dim,
                                                            hidden_size=self._conf.domain_size + 1))
                print("2 is running")
                #self._all_layers.append(self._classficationD[0])
                print("3 is running")
                #self._all_layers_adv.append(self._classficationD[0])   
                
                 
        """仅使用mamba，mlp的维度变成200"""   
        if self._conf.is_use_mamba and self._conf.is_use_lstm is False: 
            self.add_mambaformer(0, configs)
            self.add_proj1(0, mamba_input_size, 2 * self._conf.lstm_hidden_dim)
            self.add_norm1(0,2 * self._conf.lstm_hidden_dim)
            #self.add_norm2(2 * self._conf.lstm_hidden_dim)  
          
            if self._conf.is_shared_lstm:
                for i  in range(self._conf.domain_size+1): # 0,1,2  bilstm(src) , bilstm(tgt), bilstm(share)
                    
                    self.add_lstm(i, lstm_input_size, self._conf.lstm_layer_num)
            else:
                self.add_lstm(1, lstm_input_size, self._conf.lstm_layer_num)    
 
            #"""注意力机制融合特征"""
            if self._conf.is_use_multi_head_attention:
                self.add_multi_head_attention()
            
            #"""multi or single"""    
            if self._conf.is_multi:
                for i in range(self._conf.domain_size):
                    self.add_mlp_biaffine(i, 2 * self._conf.lstm_hidden_dim)
            else:
                self.add_mlp_biaffine(1, 2 * self._conf.lstm_hidden_dim)
                print("1 is running")   
                
            #"""adversary"""    
            if self._conf.is_adversary:
                self._classficationD.append(ClassificationD('classficationd', activation=nn.ReLU(), \
                                                            input_size=mamba_input_size,
                                                            hidden_size=self._conf.domain_size + 1))
                print("2 is running")
                #self._all_layers.append(self._classficationD[0])
                print("3 is running")
                #self._all_layers_adv.append(self._classficationD[0])   
                
        
        """同时使用lstm和mamba，mlp的维度变成统一成800   ，改成串联""" 
        if self._conf.is_use_mamba and self._conf.is_use_lstm: 
            #"""share lstm"""
            if  self._conf.is_shared_lstm:
                self.add_lstm(0, lstm_input_size, self._conf.lstm_layer_num)  # self._conf.lstm_layer_num)
            else:
                self.add_lstm(1, lstm_input_size)
                
             #"""mamba"""
            self.add_mambaformer(0, configs)
            
            #"""add"""
            self.add_norm1(2 * self._conf.lstm_hidden_dim)
            
            # #"""second share lstm"""
            # if  self._conf.is_shared_lstm:
            #     self.add_second_lstm(0, mamba_input_size, self._conf.lstm_layer_num)  # self._conf.lstm_layer_num) 800 -> 800
            # else:
            #     self.add_second_lstm(1, mamba_input_size)
                           
            #"""注意力机制融合特征"""
            if self._conf.is_use_multi_head_attention:
                self.add_multi_head_attention()  
                self.add_norm2(2 * self._conf.lstm_hidden_dim)    
           
            #"""multi or single"""    
            if self._conf.is_multi:
                for i in range(self._conf.domain_size):
                    self.add_mlp_biaffine(i, 2 * self._conf.lstm_hidden_dim)
            else:
                self.add_mlp_biaffine(1, 2 * self._conf.lstm_hidden_dim)
                print("1 is running")   
                
            #"""adversary"""    
            if self._conf.is_adversary:
                self._classficationD.append(ClassificationD('classficationd', activation=nn.ReLU(), \
                                                            input_size=2 * self._conf.lstm_hidden_dim,
                                                            hidden_size=self._conf.domain_size + 1))
                print("2 is running")
                #self._all_layers.append(self._classficationD[0])
                print("3 is running")
                #self._all_layers_adv.append(self._classficationD[0])  
                     
        assert ([] == self._all_layers)
        
        if self._conf.is_use_lstm and self._conf.is_use_mamba is False:
            if self._conf.is_shared_lstm:
                """
                2. **层的初始化**：
                基于上述条件，将不同的层添加到“self._all_layers”列表中，该列表可能代表神经网络的层。
                如果“self._conf.is_shared_lstm”为“True”，则会添加各种层，包括输入层、LSTM、MLP、BiAffine层，以及用于对抗学习的潜在分类和线性层。
                如果“self._conf.is_shared_lstm”为“False”，则会添加一组不同的层，不包括共享的 LSTM 层
                """
                if self._conf.is_adversary:
                    for one_layer in [self._input_layer] + self._lstm_layer \
                                    + self._kan_layer + self._multihead_attention_layer + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label \
                                    + self._classficationD + self._linear + self._gate_lstm:
                        self._all_layers.append(one_layer)
                else:
                    for one_layer in [self._input_layer] + self._lstm_layer \
                                    + self._kan_layer + self._multihead_attention_layer + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label \
                                    + self._gate_lstm:
                        self._all_layers.append(one_layer)    
                                
            else:
                for one_layer in [self._input_layer, self._lstm_layer] \
                                + self._kan_layer + self._multihead_attention_layer + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label:
                    self._all_layers.append(one_layer)
                if self._conf.is_adversary:
                    self._all_layers.append(self._classficationD[0])
                    
        if self._conf.is_use_mamba and self._conf.is_use_lstm is False:       
                """
                2. **层的初始化**：
                基于上述条件，将不同的层添加到“self._all_layers”列表中，该列表可能代表神经网络的层。
                如果“self._conf.is_shared_lstm”为“True”，则会添加各种层，包括输入层、LSTM、MLP、BiAffine层，以及用于对抗学习的潜在分类和线性层。
                如果“self._conf.is_shared_lstm”为“False”，则会添加一组不同的层，不包括共享的 LSTM 层
                """
                if self._conf.is_adversary:
                    for one_layer in [self._input_layer] + self._mambaformer_layer + self._lstm_layer \
                                    + self._kan_layer + self._multihead_attention_layer + self._proj_layer + self._normlayer1 + self._normlayer2 + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label \
                                    + self._classficationD + self._linear:
                        self._all_layers.append(one_layer)
                else:
                    for one_layer in [self._input_layer] + self._mambaformer_layer + self._lstm_layer \
                                    + self._kan_layer + self._multihead_attention_layer + self._proj_layer + self._normlayer1 + self._normlayer2 + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label:
                        self._all_layers.append(one_layer)    
        
        if self._conf.is_use_mamba and self._conf.is_use_lstm:
            if self._conf.is_shared_lstm:
                """
                2. **层的初始化**：
                基于上述条件，将不同的层添加到“self._all_layers”列表中，该列表可能代表神经网络的层。
                如果“self._conf.is_shared_lstm”为“True”，则会添加各种层，包括输入层、LSTM、MLP、BiAffine层，以及用于对抗学习的潜在分类和线性层。
                如果“self._conf.is_shared_lstm”为“False”，则会添加一组不同的层，不包括共享的 LSTM 层
                """
                if self._conf.is_adversary:
                    for one_layer in [self._input_layer] + self._lstm_layer + self._mambaformer_layer + self._normlayer1 + self._second_lstm_layer\
                                    + self._kan_layer + self._multihead_attention_layer + self._normlayer2 + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label \
                                    + self._classficationD + self._linear + self._gate_lstm:
                        self._all_layers.append(one_layer)
                else:
                    for one_layer in [self._input_layer] + self._lstm_layer + self._mambaformer_layer + self._normlayer1 + self._second_lstm_layer\
                                    + self._kan_layer + self._multihead_attention_layer + self._normlayer2 + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label \
                                    + self._gate_lstm:
                        self._all_layers.append(one_layer)    
                                
            else:
                for one_layer in [self._input_layer, self._lstm_layer, self._mambaformer_layer, self._normlayer1, self._second_lstm_layer] \
                                + self._kan_layer + self._multihead_attention_layer + self._normlayer2 + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label:
                    self._all_layers.append(one_layer)
                if self._conf.is_adversary:
                    self._all_layers.append(self._classficationD[0])
        
        
        if self._conf.is_meta:
            self._meta_layers.append(WeightNetwork(self._source_model_list, self._pairs).cuda(self._cuda_device))
            self._meta_layers.append(LossWeightNetwork(self._source_model_list,\
                    self._pairs, self._loss_weight_type, self._loss_weight_init).cuda(self._cuda_device))

            self._feature_match_layer.append(FeatureMatching(self._source_model_list,self._target_model_list,self._pairs).cuda(self._cuda_device))
            self._all_layers.append(self._feature_match_layer[0])
        """查看结构"""    
        # print(self._all_layers) 
        with open("/home/ljj/3-biaffine-taketurn/exp-conll-matching-reviewnew-3Lfinetune-best3bilstm-chvi/src-matchingloss-review-3lxiugai-wopremamba/structure/structure.txt", "w", encoding="utf-8") as fw:
            # fw.write(str(self._all_layers))
            for layer in self._all_layers:
                fw.write(str(layer))
                fw.write("\n") 
              

    # This function is useless, and will probably never be used
    def put_models_on_cpu_if_need(self):
        if not self._use_cuda:
            return
        # If the nnModule is on GPU, then .to(torch.device('cpu')) will lead to the unnecessary use of gpu:0
        for one_layer in self._all_layers:
            one_layer.to(self._cpu_device)

    def put_models_on_gpu_if_need(self):
        if not self._use_cuda:
            return
        for one_layer in self._all_layers + self._meta_layers:
            one_layer.cuda(self._cuda_device)  # the argument can be removed

    def collect_all_params(self, all_layers, all_params, all_params_requires_grad):
        all_param_count = 0
        requires_grad_param_count = 0
        assert([] == all_params)
        for one_layer in all_layers:
            for one_param in one_layer.parameters():
                all_params.append(one_param)
        assert([] == all_params_requires_grad)
        all_params_requires_grad = [param for param in all_params if param.requires_grad]
        
        all_param_count = sum(p.numel() for p in all_params)
        requires_grad_param_count = sum(p.numel() for p in all_params_requires_grad)
        print("------------")     
        print(f"all_param_count:{all_param_count}---{all_param_count/(1024*1024):.2f}M")   
        print(f"requires_grad_param_count:{requires_grad_param_count}---{requires_grad_param_count/(1024*1024):.2f}M") 
        print("------------")  
        return all_params, all_params_requires_grad    
              
        
    def run(self, use_unlabel=False):
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.train_files, self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max)  # trainfilename,[],-1
            # self.open_and_load_datasets(self._conf.unlabel_train_files, self._unlabel_train_datasets,
            #                             inst_num_max=self._conf.inst_num_max)  # trainfilename,[],-1,unlabel_train_files
            if self._conf.is_dictionary_exist is False:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset, self._label_dict)
                # for dataset in self._unlabel_train_datasets:
                #    self.create_dictionaries(dataset, self._label_dict,True)
                self.save_dictionaries(self._conf.dict_dir)
                print("create dict done")
                return

        self.load_dictionaries(self._conf.dict_dir)

        if self._conf.is_train:
            warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
            self.open_and_load_datasets(self._conf.dev_files, self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._conf.test_files, self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print('numeralizing [and pad if use-bucket] all instances in all datasets', flush=True)
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:  # all datasets in one [].
            self.numeralize_all_instances(dataset, self._label_dict)  # 将所有的instence都转换成数字形式
            if self._use_bucket:
                self.pad_all_inst(dataset)  # 对桶中的实例进行padding  把不需要的填充

        for dataset in self._unlabel_train_datasets:  # all datasets in one [].
            self.numeralize_all_instances(dataset, self._label_dict, True)
            if self._use_bucket:
                self.pad_all_inst(dataset, True)

        print('init models', flush=True)
        self.init_models()

        if self._conf.is_train:
            self.put_models_on_gpu_if_need() 
            #self.collect_all_params()
            self._all_params, self._all_params_requires_grad = self.collect_all_params(self._all_layers,\
                    self._all_params, self._all_params_requires_grad)
            self._meta_params, self._meta_params_requires_grad = self.collect_all_params(self._meta_layers, self._meta_params, self._meta_params_requires_grad)
            assert self._optimizer is None
            self._optimizer = Optimizer(self._all_params_requires_grad, self._conf)
            assert self._meta_optimizer is None
            self._meta_optimizer = MetaOptimizer(self._meta_params_requires_grad, self._conf)
            self.train()
            return

        assert self._conf.is_test
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        self.load_model(self._conf.model_dir, self._conf.model_eval_num)
        self.put_models_on_gpu_if_need()
        """设置Test数据集的选择"""
        for dataset in self._test_datasets:
            warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
            print(dataset.file_name_short)
            # 测试之后并写入一个.out文件中
            self.evaluate(dataset, use_unlabel, output_file_name='./' + dataset.file_name_short + '.out')
            self._eval_metrics.compute_and_output(self._test_datasets[0], self._conf.model_eval_num)
            self._eval_metrics.clear()

    """主要作用是将输入数据传递给模型的不同组件，包括分词、词嵌入、序列建模（LSTM）以及输出计算。最终，模型的输出被返回以供后续的训练和测试使用。"""
    """ 1 分词和子词处理：使用 bertvocab 的 subword_tokenize_to_ids 函数将其分词成子词，并获取子词的ID、掩码和标记起始位置。
        这些分词后的子词信息被存储在 subword_idxs、subword_masks 和 token_starts_masks 中。
        2 转化tensor向量处理：以上的三个信息被转换成PyTorch的Tensor格式，并移动到GPU上（cuda(self._cuda_device)）。
        3 输入层处理：使用 _input_layer 对输入数据进行处理，包括词嵌入(embadding)、（经过MLP）特征提取等操作。
        4 LSTM层处理：input_out 经过输入层--1 2 3都是，处理后，被传递给LSTM层（_lstm_layer[0]）进行序列建模。lstm_masks 用于指示哪些位置需要进行序列建模---需要进行编码。
        如果模型正在进行训练（is_training=True），则应用了一个dropout操作。
        6 多任务或单任务输出：根据模型配置（self._conf.is_multi），模型可能支持多任务训练。如果是多任务训练，模型会计算多个任务的输出分数，包括弧-标签分数（arc_scores 和 label_scores）。
        如果不是多任务训练，模型会假定一个默认的任务（通常是1）进行输出计算。
        7 返回输出：arc_scores 和 label_scores 是模型的输出，通常用于后续的损失计算和预测。这些输出的具体含义和用途取决于模型的任务（例如，句法分析、语义分析等）。
    """

    def apply_masking(self, token_ids_tensor, mask_probability, mask_token_id):
        """
        Applies random masking to a tensor of token ids, ignoring specified token ranges.

        Parameters:
        - token_ids_tensor (torch.Tensor): The tensor of token ids to mask, shape (seq_len,).
        - mask_probability (float): The probability of masking a token.
        - mask_token_id (int): The token id to use for masking.

        Returns:
        - masked_token_ids_tensor (torch.Tensor): The tensor of token ids after masking.
        """
                
        seq_len = token_ids_tensor.size(0)
        masked_token_ids_tensor = token_ids_tensor.clone()

        # 忽略第 0 个和最后一个 token，以及第 1 到第 5 个 token
        valid_indices = list(range(6, seq_len - 1))
        # print("总长度：", seq_len)
        # print("有效长度：",len(valid_indices))

        # 确定需要掩码的子词数量
        num_to_mask = int(len(valid_indices) * mask_probability)
        if num_to_mask > 0:
            mask_indices = np.random.choice(valid_indices, num_to_mask, replace=False)
            for idx in mask_indices:
                masked_token_ids_tensor[idx] = mask_token_id

        return masked_token_ids_tensor


    def forward(self, words, ext_words, tags, masks, domains, domain_id, word_lens, chars_i, wordbert, unlabel=False):

        # forward是神经网络模型的整体前向传播过程中的最后一步，也是计算模型的输出的关键步骤。在深度学习中，前向传播是模型用于处理输入数据并生成预测或输出的过程
        # 总之，forward 方法是神经网络模型整体前向传播过程的关键步骤，它将输入数据转化为模型的输出，并通常涉及到计算损失以用于训练。
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        is_training = self._input_layer.training
        i = domain_id


        subword_idxs, subword_masks, token_starts_masks, subword_maskidxs = [], [], [], []
        for e in wordbert:
            subword_ids, mask, token_starts = self.bertvocab.subword_tokenize_to_ids(e)
            token_starts[[0, -1]] = 0
            subword_idxs.append(subword_ids)
            subword_masks.append(mask)
            token_starts_masks.append(token_starts)
            if (sum(token_starts) != len(e)):
                print("mis match")
        
        mask_token_id = self.bertvocab.tokenizer.mask_token_id     
        #训练的时候对原词中文进行随机mask---防止过拟合
        if is_training and domain_id == 1:  
            for e_id in subword_idxs:  
                emask_idxs = self.apply_masking(e_id, self._conf.subword_dropout, mask_token_id) 
                subword_maskidxs.append(emask_idxs)  
                
            subword_idxs = pad_sequence(subword_maskidxs, batch_first=True).cuda(self._cuda_device)
            
        else:
            subword_idxs = pad_sequence(subword_idxs, batch_first=True).cuda(self._cuda_device) # 44,31
            
        subword_masks = pad_sequence(subword_masks, batch_first=True).cuda(self._cuda_device) #44,31
        token_starts_masks = pad_sequence(token_starts_masks, batch_first=True).cuda(self._cuda_device) #44,31
        
        

        input_out, language_id = self._input_layer(words, ext_words, tags, domains,domain_id, word_lens, chars_i, subword_idxs, subword_masks,
                                      token_starts_masks, wordbert, self._cuda_device) #[1,9,200]

        input_out = input_out.transpose(0, 1)  #[9,1,第三维度=200] [BLD] -> [LBD]
        # input_out_domain = input_out_domain.transpose(0, 1)
        
        lstm_masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)  # [9,1,1]
        
        
        if self._conf.is_use_lstm and self._conf.is_use_mamba is False:
            lstm_out = self._lstm_layer[0](input_out, lstm_masks, initial=None, is_training=is_training)
            encode_out = lstm_out   # [9,1,800]
            
        elif self._conf.is_use_mamba and self._conf.is_use_lstm is False:
            if is_training:
                input_out = drop_sequence_shared_mask(input_out, self._conf.mlp_input_dropout_ratio)          
            
            """ 
            private两个src+tgt  lstm 200->800, 一个share lstm -----3个bilstm
            share: Bilstm[0]是share; 
            private：Bilstm[1]是src-ch; Bilstm[2]是tgt-vi
            """  
            pri_out, pri_out_all = self._lstm_layer[i](input_out, lstm_masks, initial=None, is_training=is_training)
            #encode_out = lstm_out   # [11,44,800]
            if (i == 2 and self.use_meta): # 越南语
                src_out, src_out_all = self._lstm_layer[1](input_out, lstm_masks, initial=None, is_training=is_training) # vi也走一遍ch的 Bilstm[1]
                weights = self._meta_layers[0](src_out_all)
                if self._conf.is_meta_weight:
                    loss_weights = self._meta_layers[1](src_out_all)
                else:
                    loss_weights = None
                meta_loss_weigt = [self._conf.meta_loss] * len(self._pairs) # 乘以系数meta_loss 0.5
                matching_loss = self._feature_match_layer[0](src_out_all,pri_out_all, weights, meta_loss_weigt, loss_weights)
            
            if self._conf.is_shared_lstm: #share Bilstm #中文和越南语
                mamba_in = input_out.transpose(0,1) # BLD 1 9 ,200  
                mamba_out = self._mambaformer_layer[0](mamba_in, is_training=is_training) #B L D  44, 11, 200
                mamba_out = mamba_out.transpose(0,1) # L,B,D 11,44,200  
                # 也是单独一层 共0,1,2,3
                sha_out, sha_out_all =self._lstm_layer[0](mamba_out, lstm_masks, initial=None, is_training=is_training)
                
                #kan
                kan_in = mamba_out.contiguous().view(-1, mamba_out.size()[2]) #L,B,d 9,1 ,800 -> L*B,D     
                kan_out = self._proj_layer[0](kan_in) # L*B,D 484 ,800
                encode_mamba = kan_out.view(mamba_out.size()[0], mamba_out.size()[1], -1) ## L,B,D 11,44,800
                
                #残差归一化
                sha_encode_out = self._normlayer1[0](encode_mamba + sha_out) #L B D[9,1,800]
                lstm_out = sha_encode_out + pri_out
            else:
                lstm_out = pri_out
                       
                       
            encode_out = self._normlayer1[0](lstm_out) #L B D[9,1,800]
            #加一个归一化
            
        elif self._conf.is_use_lstm and self._conf.is_use_mamba:
            #串联 lstm*3 ->mamba*1 -> mambaformer*L
            lstm_out = self._lstm_layer[0](input_out, lstm_masks, initial=None, is_training=is_training) #L B D[9,1,800]
            
            mamba_in = lstm_out.transpose(0,1) # BLD 1 9 ,800
            mamba_out = self._mambaformer_layer[0](mamba_in, is_training=is_training) #B L D [1,9,200]  800 800  L层的mambaformer
            finalmamba_out = mamba_out.transpose(0, 1) # [9,1,800] 变成 L B D
            
            # residual add
            encode_out = self._normlayer1[0](finalmamba_out + lstm_out) #L B D[9,1,800]
           
                      
        if is_training:
            encode_out = drop_sequence_shared_mask(encode_out, self._conf.mlp_input_dropout_ratio) 
            sha_out = drop_sequence_shared_mask(sha_out, self._conf.mlp_input_dropout_ratio)
            pri_out = drop_sequence_shared_mask(pri_out, self._conf.mlp_input_dropout_ratio)   
        # else:
        #     if self.use_meta:
        #         print("loss_weights", loss_weights)
        #         # print("weights", weights)  
                
        diff = self.diff_module(lstm_masks, sha_out, pri_out) 
        # if unlabel:
        #     return diff, matching_loss             
                
        # 控制对抗学习进行
        if self._conf.is_adversary:
            classfication_module = self.classfication_module(encode_out)
        else:
            classfication_module = None    

        """控制是多任务还是单任务"""
        if self._conf.is_multi:
            arc_scores, label_scores = self.mlp_biaffine_module(domain_id, encode_out, is_training)
        else:
            arc_scores, label_scores = self.mlp_biaffine_module(1, encode_out, is_training)
            
            
        if (self._conf.is_shared_lstm and self.use_meta): #training vi
            return arc_scores, label_scores, diff, matching_loss, classfication_module
        elif self._conf.is_shared_lstm: # training ch
            return arc_scores, label_scores, diff, classfication_module
        elif self.use_meta:
            return arc_scores, label_scores, matching_loss, classfication_module
        else:
            return arc_scores, label_scores, classfication_module    
        #return arc_scores, label_scores, classfication_module
    


    def save_source_out(self, source_out):
        self._source_out = source_out

    def del_source_out(self):
        self._source_out = Tensor
    
    """将两个张量的前两维度统一成最大的"""
    def uniform_tensors(self, tensor1, tensor2):
        # 获取每个张量的形状 确保张量在CPU上
        tensor1 = tensor1.data.cpu().numpy()
        tensor2 = tensor2.data.cpu().numpy()
        
        shape1 = tensor1.shape
        shape2 = tensor2.shape
        
        # 确定填充后的大小
        max_dim1 = max(shape1[0], shape2[0])
        max_dim2 = max(shape1[1], shape2[1])
        
        # 创建填充张量的模板，使用0填充
        padded_tensor1 = np.zeros((max_dim1, max_dim2, shape1[2]), dtype=tensor1.dtype)
        padded_tensor2 = np.zeros((max_dim1, max_dim2, shape2[2]), dtype=tensor2.dtype)
        
        # 将原始张量放入填充张量中
        padded_tensor1[:shape1[0], :shape1[1], :] = tensor1
        padded_tensor2[:shape2[0], :shape2[1], :] = tensor2
        
         # 将填充后的NumPy数组转换回PyTorch张量
        padded_tensor1_torch = torch.from_numpy(padded_tensor1)
        padded_tensor2_torch = torch.from_numpy(padded_tensor2)
        
        return padded_tensor1_torch, padded_tensor2_torch
    
    """将两个张量的前两维度统一成目标的，源语言大的截断，小的用0填充"""
    def clip_uniform_tensors(self, tensor1, tensor2):
        # 确保张量在CPU上，并转换为NumPy数组
        tensor1 = tensor1.data.cpu().numpy()
        tensor2 = tensor2.data.cpu().numpy()
        
        # 获取每个张量的形状
        shape1 = tensor1.shape
        shape2 = tensor2.shape
        
        # 根据tensor2的形状确定目标大小（前两个维度）
        target_shape = (shape2[0], shape2[1], shape1[2])
        
        # 创建目标形状的张量，使用0填充
        padded_tensor1 = np.zeros(target_shape, dtype=tensor1.dtype)
        padded_tensor2 = np.zeros(target_shape, dtype=tensor2.dtype)
        
        # 计算截断或填充的尺寸
        truncate_dim1 = min(shape1[0], shape2[0])
        truncate_dim2 = min(shape1[1], shape2[1])
        
        # 将原始张量放入填充张量中，或进行截断
        padded_tensor1[:truncate_dim1, :truncate_dim2, :] = tensor1[:truncate_dim1, :truncate_dim2, :]
        padded_tensor2[:shape2[0], :shape2[1], :] = tensor2
        
        # 将填充或截断后的NumPy数组转换回PyTorch张量
        padded_tensor1_torch = torch.from_numpy(padded_tensor1)
        padded_tensor2_torch = torch.from_numpy(padded_tensor2)
        
        return padded_tensor1_torch, padded_tensor2_torch

    def classfication_module(self, shared_lstm_out):
        """3.
         - “shared_lstm_out”：来自共享 LSTM 层或分支的输出。 -共享
         - “private_lstm_out”：来自私有 LSTM 层或分支的输出。-不共享

        """
        # print(len(self._classficationD))
        classficationd = self._classficationD[0](shared_lstm_out)  # 主任务
        # nadv_class = self._classficationD[1](private_lstm_out)    # 对抗
        classficationd = classficationd.transpose(0, 1)
        # transpose(0, 1)转置Original Tensor: 会交换第0 1两个维度
        # tensor([[1, 2, 3],
        #         [4, 5, 6]])
        #
        # Transposed Tensor:
        # tensor([[1, 4],
        #         [2, 5],
        #         [3, 6]])
        # nadv_class = nadv_class.transpose(0, 1)
        return classficationd  # , nadv_class

    """多/单任务"""

    def mlp_biaffine_module(self, domain_id, lstm_out, is_training):
        """
        :param domain_id:领域标识符，用于区分不同的领域或任务。
        :param lstm_out:LSTM或者mamba层的输出，通常是模型的中间表示
        :param is_training:一个布尔值，指示当前是否在模型的训练阶段
        :return:
        """
        # 使用kan
        if self._conf.is_use_kan:
            kan_in = lstm_out.view(-1, lstm_out.size()[2])  #[9,1,800] -> [9,800]
            kan_out = self._kan_layer[domain_id - 1](kan_in) #[9,800] -> [9,1200]
            mlp_out = kan_out.view(lstm_out.size()[0], lstm_out.size()[1], -1) #[9,1,1200] 
            assert(kan_out.size()[1] == mlp_out.size()[2])
        # 使用mlp
        if self._conf.is_use_mlp:    
            mlp_out = self._mlp_layer[domain_id - 1](lstm_out)
            
        if is_training:
            mlp_out = drop_sequence_shared_mask(mlp_out, self._conf.mlp_output_dropout_ratio)
        mlp_out = mlp_out.transpose(0, 1)  # 进行转置满足后续的维度要求
        """mlp_out 被分割成四个部分：mlp_arc_dep、mlp_arc_head、mlp_label_dep 和 
        mlp_label_head。这些部分分别代表弧的依存词、弧的头部词、标签的依存词和标签的头部词的特征表示。 """
        # 其中mlp_arc_dep, mlp_arc_head是每个词wi作为修饰词和中心词的---依存弧特征表示
        # 其中mlp_label_dep, mlp_label_head是每个词wi作为修饰词和中心词的---依存标签特征表示
        mlp_arc_dep, mlp_arc_head, mlp_label_dep, mlp_label_head = \
            torch.split(mlp_out, [self._conf.mlp_output_dim_arc, self._conf.mlp_output_dim_arc, \
                                  self._conf.mlp_output_dim_rel, self._conf.mlp_output_dim_rel], dim=2)
        """算出依存弧的得分"""
        arc_scores = self._bi_affine_layer_arc[domain_id - 1](mlp_arc_dep, mlp_arc_head)
        arc_scores = torch.squeeze(arc_scores, dim=3)  # 弧的维度被减少
        """算出依存标签的得分"""
        label_scores = self._bi_affine_layer_label[domain_id - 1](mlp_label_dep, mlp_label_head)
        return arc_scores, label_scores

    # def diff_module(self, lstm_masks, shared_lstm_out, private_lstm_out):
    #     length, batch, dim = shared_lstm_out.size()
    #     lstm_mask1 = lstm_masks.expand(length, batch, dim)
    #     b = torch.bmm(torch.mul(shared_lstm_out, lstm_mask1).transpose(1, 2), torch.mul(private_lstm_out, lstm_mask1))
    #     diff = torch.mul(b, b)
    #     diff1 = torch.sum(diff, dim=2)
    #     # diff1 = torch.sum(b,dim=2)
    #     diff2 = torch.sum(diff1)
    #     return diff2
    
    def diff_module(self,lstm_masks,shared_lstm_out,private_lstm_out):
        diff = (shared_lstm_out - private_lstm_out).pow(2).mean(2).mean(1).mean(0)
        return diff

    @staticmethod
    def compute_loss(arc_scores, label_scores, gold_arcs, gold_labels, total_word_num, one_batch):
        batch_size, len1, len2 = arc_scores.size()
        assert (len1 == len2)

        # gold_arcs, gold_labels: batch_size max-len
        penalty_on_ignored = []  # so that certain scores are ignored in computing cross-entropy loss
        for inst in one_batch:
            length = inst.size()
            penalty = arc_scores.new_tensor([0.] * length + [-1e10] * (len1 - length))
            penalty_on_ignored.append(penalty.unsqueeze(dim=0))
        penalty_on_ignored = torch.stack(penalty_on_ignored, 0)
        arc_scores = arc_scores + penalty_on_ignored

        # print(arc_scores.shape, arc_scores.dtype)
        # print(gold_arcs.shape, gold_arcs.dtype)
        # print(label_scores.shape, label_scores.dtype)
        # print(gold_labels.shape, gold_labels.dtype)
        
        arc_loss = F.cross_entropy(
            arc_scores.view(batch_size * len1, len2), gold_arcs.view(batch_size * len1),
            ignore_index=ignore_id_head_or_label, size_average=False)

        batch_size2, len12, len22, label_num = label_scores.size()
        assert batch_size2 == batch_size and len12 == len2 and len22 == len2

        # Discard len2 dim: batch len1 L
        label_scores_of_concern = arc_scores.new_full((batch_size, len1, label_num), 0)  # discard len2 dim

        scores_one_sent = [label_scores[0][0][0]] * len1
        for i_batch, (scores, arcs) in enumerate(zip(label_scores, gold_arcs)):
            for i in range(one_batch[i_batch].size()):
                scores_one_sent[i] = scores[i, arcs[i]]  # [mod][gold-head]: L * float
            label_scores_of_concern[i_batch] = torch.stack(scores_one_sent, dim=0)

        rel_loss = F.cross_entropy(label_scores_of_concern.view(batch_size * len1, label_num),
                                   gold_labels.view(batch_size * len1),
                                   ignore_index=ignore_id_head_or_label, size_average=False)

        loss = (arc_loss + rel_loss) / total_word_num
        return loss

    @staticmethod
    def adversary_loss(classficationd, domains):
        """4.该函数计算对抗损失，用于培训对抗学习的模型。函数首先将输入的大小调整为合适的形状，然后计算分类器和对抗分类器的交叉熵损失。
        :param classficationd:分类器的输出
        :param nadv_class:对抗分类器的输出
        :param domains:实际领域标签
        :param domains_nadv:对抗学习的领域标签
        :param total_word_num:总词数
        :return:adv_loss 和 nadv_loss 两个损失值。
        """
        batch_size, len1, len2 = classficationd.size()
        # classficationd = F.softmax(classficationd)
        adv_loss = F.cross_entropy(classficationd.contiguous().view(batch_size * len1, len2), \
                                   domains.view(batch_size * len1), ignore_index=0)
        # adv_loss = adv_loss / total_word_num

        # nadv_loss = nadv_loss / total_word_num

        return adv_loss

    def train_set(self, bc, pc, pb, zx, domain, domain_src, domain_tgt, dataset, unlabel):
        """这个函数似乎用于训练数据集的一个迭代。"""
        inst_num, loss = self.train_or_eval_one_batch(dataset[domain], is_training=True, unlabel=unlabel)
        domain_tgt += 1
        domain = domain_tgt % 2  # 修改
        return domain_src, domain_tgt, domain, inst_num, loss

    def train_set_label(self, bc, pb, zx, domain, domain_src, domain_tgt, dataset, unlabel, eval_iter=-1):
        """这个函数似乎用于训练标签数据集的一个迭代，区分了不同的领域。"""
        if (domain_tgt < bc):
            domain = 0
        elif (bc <= domain_tgt < bc + pb):
            domain = 1
        else:
            domain = 2
        inst_num, loss = self.train_or_eval_one_batch(dataset[domain], is_training=True, unlabel=unlabel,
                                                      eval_iter=eval_iter)
        domain_tgt += 1
        return domain_src, domain_tgt, domain, inst_num, loss

    def train(self):
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        print("begin train")
        update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = 0, 0, 0, 0.
        self._eval_metrics.clear()
        current_las = 0  # 以上都是初始化
        self.set_training_mode(is_training=True)  # 设置训练模型开始-以确保在训练过程中使用了 dropout 和 batch normalization。
        #label_batch_num = self._train_datasets[0].batch_num  # 单个数据集26
        ch, vi = self._train_datasets[0].batch_num, self._train_datasets[1].batch_num
        domain, domain_src, domain_tgt, train_iter, udomain, udomain_src, udomain_tgt, u_cnt, l_cnt = 0, 0, 0, 1, 0, 0, 0, 0, 0
        datasets_names = ["ch", "vi"]

        while True:
            # train_iter += 1
            # inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[0], is_training=True, unlabel=False,
            #                                               eval_iter=-1)           
            # print(train_iter)  # 单个数据集


            """【方法二 按domain值】设置在两个训练数据集的选择-两个数据集交叉在一起训练"""
            if domain == 0:
                dataset_name = "ch"
                self.use_meta, self.match_only, self.is_diff_loss =False,False,True # 只有一个share
                print(f"parser src — {train_iter}")
                inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[0], is_training=True, unlabel=False, eval_iter=-1)
                           
            elif domain == 1:
                dataset_name = "vi"
                self.use_meta, self.match_only, self.is_diff_loss =True,True,True
                print(f"parser tgt — {train_iter}")
                inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[1], is_training=True, unlabel=False, eval_iter=-1)
                    
            else:
                print("the filename is wrong, we cann't distinguish its domain")
            # inst_num, loss = self.train_or_eval_one_batch(dataset, is_training=True, unlabel=False, eval_iter=-1)
            # print(f"parser is training on dataset {dataset_name}:{train_iter}")
            #切换到下一个数据集
            domain = (domain + 1) % 2  # domain+1迭代，对2取余
            train_iter += 1

            assert inst_num > 0
            assert loss is not None
            """优化---三部曲"""
            loss.backward()  # loss反向传播
            nn.utils.clip_grad_norm_(self._all_params_requires_grad, max_norm=self._conf.clip)  # 进行梯度裁剪，以防止梯度爆炸
            self._optimizer.step()  # 优化器
            self.zero_grad(self._all_layers)  # 梯度清0
            if self.use_meta:
                nn.utils.clip_grad_norm_(self._meta_params_requires_grad, max_norm=self._conf.clip)
                self._meta_optimizer.step()
                self.zero_grad(self._meta_layers)

            update_step_cnt += 1
            # print("update_step_cnt ",update_step_cnt)
            use_unlabel = False
            #eval_every_update_step_num = label_batch_num  # 单个数据集
            eval_every_update_step_num = ch + vi  # self._conf.eval_every_update_step_num(需要config.txt手动改)  # 如果是两个数据集，则需要计算两个数据集的总和并且在config.txt中修改

            """选择在第几个训练集上进行评估"""
            if 0 == update_step_cnt % eval_every_update_step_num:
                eval_cnt += 1
                domain, domain_src, domain_tgt, train_iter, udomain, udomain_src, udomain_tgt, u_cnt, l_cnt = 0, 0, 0, 0, 0, 0, 0, 0, 0
                self._eval_metrics.compute_and_output(self._train_datasets[1], eval_cnt, use_unlabel)
                # 调用 self._eval_metrics.compute_and_output 方法计算和输出评估指标，这通常用于在训练过程中监控模型性能。
                self._eval_metrics.clear()

                print("begin evaluate")
                # 执行模型在验证集上的评估
                self.evaluate(self._dev_datasets[0], use_unlabel)
                self._eval_metrics.compute_and_output(self._dev_datasets[0], eval_cnt, use_unlabel)
                if use_unlabel == False:
                    # 并计算评估指标。如果不使用无标签数据（use_unlabel == False）_有标签，则计算当前的 LAS（Label Attachment Score）。
                    current_las = self._eval_metrics.las
                    current_uas = self._eval_metrics.uas
                self._eval_metrics.clear()

                if best_accuracy < current_las - 1e-3:
                    """1. **模型保存逻辑：**
                        - 代码跟踪评估步骤（“eval_cnt”）和最佳评估准确性（“best_accuracy”）。
                        - 它检查当前评估精度（“current_las”）是否比最佳精度（“best_accuracy”）好（更高），幅度为 1e-3 （0.001）。如果更好，它将更新最佳精度并保存模型。
                        - 代码还会检查评估步骤数 （“eval_cnt”） 是否大于配置中指定的“save_model_after_eval_num”。如果是，它将保存当前模型并删除以前保存的最佳模型（如果有）。
                      
                      2. **评估和停止标准：**
                        - 训练循环继续，直到满足以下两个条件之一：
                        - 评估步骤数（“eval_cnt”）超过“train_max_eval_num”，表示已达到最大评估数。
                        - 或者，如果“train_stop_after_eval_num_no_improve”评估步骤通过，但未达到最佳评估范围，则训练将停止。config.txt中设置为100（节省计算资源）
                    """
                    if eval_cnt > self._conf.save_model_after_eval_num:
                        # 检查当前的评估次数 eval_cnt 是否超过了一个阈值
                        # self._conf.save_model_after_eval_num。这个阈值表示在开始保存模型之前必须执行多少次评估。
                        if best_eval_cnt > self._conf.save_model_after_eval_num:
                            # 如果满足上述条件，并且之前保存的最佳模型的评估次数 best_eval_cnt 也超过了 self._conf.save_model_after_eval_num，则执行以下操作：
                            self.del_model(self._conf.model_dir, best_eval_cnt)
                            # 删除之前保存的最佳模型。这是为了确保只保留最新的和性能更好的模型。
                        self.save_model(self._conf.model_dir, eval_cnt)
                        # 无论上述条件如何，都会保存当前模型。这是为了确保在训练过程中随时可以恢复到最新的模型。
                        self.evaluate(self._test_datasets[0], use_unlabel, output_file_name=None)

                        self._eval_metrics.compute_and_output(self._test_datasets[0], eval_cnt, use_unlabel)

                        self._eval_metrics.clear()

                    # 更新记录最佳评估次数和最佳准确性
                    best_eval_cnt = eval_cnt
                    best_accuracy = current_las
                # 确保模型在继续训练之前处于训练模式，以便使用 dropout 和 batch normalization。
                self.set_training_mode(is_training=True)

            if (best_eval_cnt + self._conf.train_stop_after_eval_num_no_improve < eval_cnt) or \
                    (eval_cnt > self._conf.train_max_eval_num):
                break

    """每一个batch中计算parser loss"""

    def train_or_eval_one_batch(self, dataset, is_training, unlabel=False, eval_iter=-1):
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        # print(dataset.get_one_batch)
        one_batch, total_word_num, max_len = dataset.get_one_batch(rewind=is_training)
        # NOTICE: total_word_num does not include w_0
        if len(one_batch) == 0:
            print("one_batch is none " + dataset.file_name_short)
            return 0, None
        
        if unlabel == False:
            words, ext_words, tags, gold_heads, gold_labels, lstm_masks, domains, domains_nadv, word_lens, chars_i, wordbert = \
                self.compose_batch_data_variable(one_batch, max_len)
                
            # forward      
            if (self._conf.is_shared_lstm and self.use_meta): # 越南语
                arc_scores, label_scores, diff, matching_loss, classficationd  = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id, word_lens, chars_i, wordbert)
            elif self._conf.is_shared_lstm: # 中文
                arc_scores, label_scores, diff, classficationd = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id, word_lens, chars_i, wordbert)
            elif self.use_meta:
                arc_scores, label_scores, matching_loss, classficationd = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id, word_lens, chars_i, wordbert)
            else:
                arc_scores, label_scores, classficationd = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id, word_lens, chars_i, wordbert)

            self.decode(arc_scores, label_scores, one_batch, self._label_dict)
            loss = Parser.compute_loss(arc_scores, label_scores, gold_heads, gold_labels, total_word_num, one_batch)
            self.compute_accuracy(one_batch, self._eval_metrics)
            self._eval_metrics.loss_accumulated += loss.item()
            print("parser loss:", loss)
            final_loss = loss
            
            if self.use_meta:
                self._eval_metrics.loss_accumulated += matching_loss.item()   #之前没有添加
                final_loss += matching_loss # 在forward里面已经 乘了系数meta_loss 0.5
                print("matching_loss: ",matching_loss)
            
            if self._conf.is_adversary:
                # adv_loss = Parser.adversary_loss(classficationd, domains,total_word_num)
                adv_loss = Parser.adversary_loss(classficationd, domains)
                adv_loss = self._conf.adversary_lambda_loss * adv_loss  # 乘上系数lambda
                print("adversary loss:", adv_loss)
                self._eval_metrics.loss_accumulated += adv_loss.item()
                self.compute_accuracy(one_batch, self._eval_metrics)  # Parser.compute_accuray(classficationd, domains)
                final_loss += adv_loss
                
            if self.is_diff_loss:
                diff_loss = self._conf.diff_bate_loss * diff#/total_word_num  # 乘上系数beta
                #diff_loss = diff/total_word_num
                self._eval_metrics.loss_accumulated += diff_loss.item()
                final_loss += diff_loss
                print("diff_loss", diff_loss)
                    
        return len(one_batch), final_loss

    def evaluate(self, dataset, use_unlabel, output_file_name=None):
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略警告
        self.set_training_mode(is_training=False)
        while True:
            self.use_meta,self.match_only,self.match_only = True,True,True #因为是对目标语言测试验证，所有全是True
            inst_num, loss = self.train_or_eval_one_batch(dataset, is_training=False, unlabel=use_unlabel)
            if 0 == inst_num:
                break
            assert loss is not None
        """生成预测的test文件只有第1 3 6 7列的数据"""
        if output_file_name is not None:
            with open(output_file_name, 'w', encoding='utf-8') as out_file:
                all_inst = dataset.all_inst
                for inst in all_inst:
                    inst.write(out_file)  # 每一个填充后的实例写入  方法在Instance.py的write()方法中

    @staticmethod
    def decode(arc_scores, label_scores, one_batch, label_dict):
        # detach(): Returns a new Tensor, detached from the current graph.
        arc_scores = arc_scores.detach().cpu().numpy()
        label_scores = label_scores.detach().cpu().numpy()

        for (arc_score, label_score, inst) in zip(arc_scores, label_scores, one_batch):
            arc_pred = np.argmax(arc_score, axis=1)  # mod-head order issue. BE CAREFUL
            label_score_of_concern = label_score[np.arange(inst.size()), arc_pred[:inst.size()]]
            label_pred = np.argmax(label_score_of_concern, axis=1)
            Parser.set_predict_result(inst, arc_pred, label_pred, label_dict)

    def create_dictionaries(self, dataset, label_dict, unlabel=False):
        all_inst = dataset.all_inst
        max_char = 0
        for inst in all_inst:
            for i in range(1, inst.size()):
                self._word_dict.add_key_into_counter(inst.words_s[i])
                if self._conf.is_charlstm:
                    c = 0
                    for char in inst.words_s[i]:
                        self._char_dict.add_key_into_counter(char)
                        c += 1
                    if max_char < c:
                        max_char = c
                self._tag_dict.add_key_into_counter(inst.tags_s[i])
                if unlabel == False:
                    if inst.heads_i[i] != ignore_id_head_or_label:
                        label_dict.add_key_into_counter(inst.labels_s[i])
        print("max_char:", max_char)

    def numeralize_all_instances(self, dataset, label_dict, unlabel=False):
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(0, inst.size()):
                inst.words_i[i] = self._word_dict.get_id(inst.words_s[i])
                if self._conf.is_charlstm:
                    c = 0
                    for char in inst.words_s[i]:
                        # print(inst.words_s[i])
                        inst.chars_i[i, c] = self._char_dict.get_id(char)
                        c += 1
                    inst.word_lens[i] = c
                inst.ext_words_i[i] = self._ext_word_dict.get_id(inst.words_s[i])
                inst.tags_i[i] = self._tag_dict.get_id(inst.tags_s[i])
                if unlabel == False:
                    if inst.heads_i[i] != ignore_id_head_or_label:
                        inst.labels_i[i] = label_dict.get_id(inst.labels_s[i])

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._word_dict.load(path + self._word_dict.name, cutoff_freq=self._conf.word_freq_cutoff,
                             default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._char_dict.load(path + self._char_dict.name, cutoff_freq=self._conf.word_freq_cutoff,
                             default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._tag_dict.load(path + self._tag_dict.name,
                            default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._label_dict.load(path + self._label_dict.name, default_keys_ids=())

        self._ext_word_dict.load(self._conf.ext_word_dict_full_path,
                                 default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self.load_ext_word_emb(self._conf.ext_word_emb_full_path,
                               default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path) is False
        if not os.path.exists(path):
            os.mkdir(path)
        self._word_dict.save(path + self._word_dict.name)
        self._char_dict.save(path + self._char_dict.name)
        self._tag_dict.save(path + self._tag_dict.name)
        self._label_dict.save(path + self._label_dict.name)

    def load_ext_word_emb(self, full_file_name, default_keys_ids=()):
        assert os.path.exists(full_file_name)
        with open(full_file_name, 'rb') as f:
            self._ext_word_emb_np = pickle.load(f)
        dim = self._ext_word_emb_np.shape[1]
        assert dim == self._conf.word_emb_dim
        for i, (k, v) in enumerate(default_keys_ids):
            assert (i == v)
        pad_and_unk_embedding = np.zeros((len(default_keys_ids), dim), dtype=data_type)
        self._ext_word_emb_np = np.concatenate([pad_and_unk_embedding, self._ext_word_emb_np])
        self._ext_word_emb_np = self._ext_word_emb_np / np.std(self._ext_word_emb_np)

    @staticmethod
    def del_model(path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        assert os.path.exists(path)
        # os.rmdir(path)
        shutil.rmtree(path)
        print('Delete model %s done.' % path)

    def load_model(self, path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        assert os.path.exists(path)
        for layer in self._all_layers:
            # Without 'map_location='cpu', you may find the unnecessary use of gpu:0, unless CUDA_VISIBLE_DEVICES=6 python $exe ...
            layer.load_state_dict(torch.load(path + layer.name, map_location='cpu'))
        # layer.load_state_dict(torch.load(path + layer.name))
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        # assert os.path.exists(path) is False
        if os.path.exists(path) is False:
            os.mkdir(path)
        for layer in self._all_layers:
            torch.save(layer.state_dict(), path + layer.name)
        print('Save model %s done.' % path)

    def open_and_load_datasets(self, file_names, datasets, inst_num_max):
        assert len(datasets) == 0
        names = file_names.strip().split(':')
        assert len(names) > 0
        for name in names:
            datasets.append(Dataset(name, max_bucket_num=self._conf.max_bucket_num,
                                    word_num_one_batch=self._conf.word_num_one_batch,
                                    sent_num_one_batch=self._conf.sent_num_one_batch,
                                    inst_num_max=inst_num_max))  # 80,1500,60,-1

    @staticmethod
    def set_predict_result(inst, arc_pred, label_pred, label_dict):
        # assert arc_pred.size(0) == inst.size()
        for i in np.arange(1, inst.size()):
            inst.heads_i_predict[i] = arc_pred[i]
            inst.labels_i_predict[i] = label_pred[i]
            inst.labels_s_predict[i] = label_dict.get_str(inst.labels_i_predict[i])

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics):
        (a, b, c) = inst.eval()
        eval_metrics.word_num += inst.word_num()
        eval_metrics.word_num_to_eval += a
        eval_metrics.word_num_correct_arc += b
        eval_metrics.word_num_correct_label += c

    @staticmethod
    def compute_accuracy(one_batch, eval_metrics):
        eval_metrics.sent_num += len(one_batch)
        eval_metrics.batch_num += 1
        for inst in one_batch:
            Parser.compute_accuracy_one_inst(inst, eval_metrics)

    @staticmethod
    def compute_unlabel(one_batch, eval_metrics):
        eval_metrics.sent_num += len(one_batch)
        eval_metrics.batch_num += 1
        for inst in one_batch:
            eval_metrics.word_num += inst.word_num()

    def set_training_mode(self, is_training=True):
        for one_layer in self._all_layers:
            one_layer.train(mode=is_training)

    def zero_grad(self, all_layers):
        for one_layer in all_layers:
            one_layer.zero_grad()

    def pad_all_inst(self, dataset, unlabel=False):
        for (max_len, inst_num_one_batch, this_bucket) in dataset.all_buckets:
            for inst in this_bucket:
                assert inst.lstm_mask is None
                if unlabel == False:
                    inst.words_i, inst.ext_words_i, inst.tags_i, inst.heads_i, inst.labels_i, inst.lstm_mask, inst.domains_i, \
                    inst.domains_nadv_i, inst.word_lens, inst.chars_i, inst.words_s = self.pad_one_inst(inst, max_len)
                else:
                    inst.words_i, inst.ext_words_i, inst.tags_i, inst.lstm_mask, inst.domains_i, inst.domains_nadv_i, inst.word_lens, \
                    inst.chars_i, inst.words_s = self.pad_one_inst(inst, max_len, unlabel)

    def pad_one_inst(self, inst, max_sz, unlabel=False):
        sz = inst.size()
        assert len(inst.words_i) == sz
        assert max_sz >= sz
        pad_sz = (0, max_sz - sz)
        if max_sz > sz:
            chars_i_pad = np.zeros((max_sz - sz, 39), dtype=data_type_int)
            inst.chars_i = np.concatenate((inst.chars_i, chars_i_pad), axis=0)
        if unlabel == False:  # 把不需要的标记填充掉
            return np.pad(inst.words_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.tags_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
                   np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
                   np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.domains_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.domains_nadv_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.word_lens, pad_sz, 'constant', constant_values=1), \
                   inst.chars_i, inst.words_s
        else:
            return np.pad(inst.words_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.tags_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.domains_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.domains_nadv_i, pad_sz, 'constant', constant_values=0), \
                   np.pad(inst.word_lens, pad_sz, 'constant', constant_values=1), \
                   inst.chars_i, inst.words_s

    def compose_batch_data_variable(self, one_batch, max_len, unlabel=False):
        words, ext_words, tags, heads, labels, lstm_masks, domains, domains_nadv, wordbert = [], [], [], [], [], [], [], [], []
        chars_i = None
        i = 0
        for inst in one_batch:
            if i == 0:
                chars_i = inst.chars_i
                word_lens = inst.word_lens
            else:
                chars_i = np.concatenate((chars_i, inst.chars_i), axis=0)
                word_lens = np.concatenate((word_lens, inst.word_lens), axis=0)
            i += 1
            if self._use_bucket:
                words.append(inst.words_i)
                ext_words.append(inst.ext_words_i)
                tags.append(inst.tags_i)
                if unlabel == False:
                    heads.append(inst.heads_i)
                    labels.append(inst.labels_i)
                lstm_masks.append(inst.lstm_mask)
                domains.append(inst.domains_i)
                domains_nadv.append(inst.domains_nadv_i)
                wordbert.append(inst.words_s)
            '''
			else:
				ret = self.pad_one_inst(inst, max_len)
				words.append(ret[0])
				ext_words.append(ret[1])
				tags.append(ret[2])
				if unlabel==False:
					heads.append(ret[3])
					labels.append(ret[4])
				lstm_masks.append(ret[5])
				domains.append(ret[6])
				domains_nadv.append(ret[7])
				wordbert.append(ret[8])
			'''
        # print("i",i)#
        # print("word_lens:",word_lens)
        # print("chars_i:",chars_i)
        # print("chars_i shape:",chars_i.shape)
        # print("word_lens shape:",word_lens.shape)
        if unlabel == False:
            words, ext_words, tags, heads, labels, lstm_masks, domains, domains_nadv, word_lens, chars_i = \
                torch.from_numpy(np.stack(words, axis=0)), torch.from_numpy(np.stack(ext_words, axis=0)), \
                torch.from_numpy(np.stack(tags, axis=0)), torch.from_numpy(np.stack(heads, axis=0)), \
                torch.from_numpy(np.stack(labels, axis=0)), torch.from_numpy(np.stack(lstm_masks, axis=0)), \
                torch.from_numpy(np.stack(domains, axis=0)), torch.from_numpy(np.stack(domains_nadv, axis=0)), \
                torch.from_numpy(word_lens), torch.from_numpy(chars_i)
        else:
            words, ext_words, lstm_masks, domains, domains_nadv, word_lens, chars_i, tags = \
                torch.from_numpy(np.stack(words, axis=0)), torch.from_numpy(np.stack(ext_words, axis=0)), \
                torch.from_numpy(np.stack(lstm_masks, axis=0)), torch.from_numpy(np.stack(domains, axis=0)), \
                torch.from_numpy(np.stack(domains_nadv, axis=0)), torch.from_numpy(word_lens), \
                torch.from_numpy(chars_i), torch.from_numpy(np.stack(tags, axis=0))

        # MUST assign for Tensor.cuda() unlike nn.Module
        if self._use_cuda:
            if unlabel == False:
                words, ext_words, tags, heads, labels, lstm_masks, domains, domains_nadv, word_lens, chars_i = \
                    words.cuda(self._cuda_device), ext_words.cuda(self._cuda_device), \
                    tags.cuda(self._cuda_device), heads.cuda(self._cuda_device), \
                    labels.cuda(self._cuda_device), lstm_masks.cuda(self._cuda_device), \
                    domains.cuda(self._cuda_device), domains_nadv.cuda(self._cuda_device), \
                    word_lens.cuda(self._cuda_device), chars_i.cuda(self._cuda_device)
            else:
                words, ext_words, lstm_masks, domains, domains_nadv, word_lens, chars_i, tags = \
                    words.cuda(self._cuda_device), ext_words.cuda(self._cuda_device), \
                    lstm_masks.cuda(self._cuda_device), domains.cuda(self._cuda_device), \
                    domains_nadv.cuda(self._cuda_device), word_lens.cuda(self._cuda_device), \
                    chars_i.cuda(self._cuda_device), tags.cuda(self._cuda_device)
        if unlabel == False:
            return words, ext_words, tags, heads, labels, lstm_masks, domains, domains_nadv, word_lens, chars_i, wordbert
        else:
            return words, ext_words, lstm_masks, domains, domains_nadv, word_lens, chars_i, tags, wordbert


class EvalMetrics(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sent_num = 0
        self.word_num = 0
        # self.unlabel_sent_num = 0
        self.batch_num = 0
        self.word_num_to_eval = 0
        self.word_num_correct_arc = 0
        self.word_num_correct_label = 0
        self.uas = 0.
        self.las = 0.
        self.loss_accumulated = 0.
        self.start_time = time.time()
        self.time_gap = 0.
        self.adv_acc = 0.
        self.nadv_acc = 0.
        self.fadv_acc = 0.
        self.fnadv_acc = 0.

    def compute_and_output(self, dataset, eval_cnt, use_unlabel=False):
        assert self.word_num > 0
        self.time_gap = float(time.time() - self.start_time)
        if use_unlabel == False:
            self.uas = 100. * self.word_num_correct_arc / self.word_num_to_eval
            self.las = 100. * self.word_num_correct_label / self.word_num_to_eval
            self.time_gap = float(time.time() - self.start_time)
            print("%30s(%5d): loss=%.3f las=%.3f, uas=%.3f, %d (%d) words, %d sentences, time=%.3f [%s]" % \
                  (dataset.file_name_short, eval_cnt, self.loss_accumulated, self.las, self.uas, \
                   self.word_num_to_eval, self.word_num, self.sent_num, self.time_gap, get_time_str()), flush=True)
 # type: ignore
