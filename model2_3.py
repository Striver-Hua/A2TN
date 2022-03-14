import numpy as np
import tensorflow as tf
from Dataset1 import Dataset
import base_model_A
import base_model_R
from collections import defaultdict
import time
import sys
import math
import random
from progressbar import *


class MTL(object):
    def __init__(self, config, sess):
        t1 = time.time()
        # 数据目录
        self.data_dir = config['data_dir']
        # 任务一：cloth推荐
        self.data_name_cloth = config['data_name_cloth']
        # 调用Dataset类获得数据
        dataset_cloth = Dataset(self.data_dir + self.data_name_cloth)
        # 分别是训练集数据稀疏矩阵、测试集数据列表、测试集负样本列表
        self.train_cloth, self.testRatings_cloth, self.testNegatives_cloth, self.aesthetic_cloth_all = dataset_cloth.trainMatrix, dataset_cloth.testRatings, dataset_cloth.testNegatives, dataset_cloth.aesthetic_all
        # cloth训练数据集中用户数量和物品数量
        self.nUsers_cloth, self.nItems_cloth = self.train_cloth.shape
        print("Load cloth data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d, #test_neg=%d"
          %(time.time()-t1, self.nUsers_cloth, self.nItems_cloth, self.train_cloth.nnz, len(self.testRatings_cloth), len(self.testNegatives_cloth)*99))
        # cloth测试集中根据用户的id得到物品的id字典
        self.user_gt_item_cloth = defaultdict(int)
        for user, gt_item in self.testRatings_cloth:
            self.user_gt_item_cloth[user] = gt_item
        # 存放训练实例的列表：真正的输入
        self.user_input_cloth, self.item_input_cloth, self.labels_cloth = [], [], []
        # 存放测试实例的列表：真正的输入
        self.test_user_input_cloth, self.test_item_input_cloth, self.test_labels_cloth = [], [], []

        # 任务一：Tools推荐（类似）
        self.data_name_tools = config['data_name_tools']
        dataset_tools = Dataset(self.data_dir + self.data_name_tools)
        self.train_tools, self.testRatings_tools, self.testNegatives_tools, self.aesthetic_tools_all = dataset_tools.trainMatrix, dataset_tools.testRatings, dataset_tools.testNegatives, dataset_tools.aesthetic_all
        self.nUsers_tools, self.nItems_tools = self.train_tools.shape
        print("Load tools data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d, #test_neg=%d"
          %(time.time()-t1, self.nUsers_tools, self.nItems_tools, self.train_tools.nnz, len(self.testRatings_tools), len(self.testNegatives_tools)*99))
        self.user_gt_item_tools = defaultdict(int)
        for user, gt_item in self.testRatings_tools:
            self.user_gt_item_tools[user] = gt_item
        self.user_input_tools, self.item_input_tools, self.labels_tools = [], [], []
        self.test_user_input_tools, self.test_item_input_tools, self.test_labels_tools = [], [], []

        # 用户是共享的，所以用户数量应该是相等的
        if self.nUsers_cloth != self.nUsers_tools:
            print('nUsers_cloth != nUsers_tools. However, they should be shared. exit...')
            sys.exit(0)
        self.nUsers = self.nUsers_cloth

        # 超参数
        self.init_std = config['init_std']
        self.batch_size = config['batch_size']
        self.nepoch = config['nepoch']
        # !在main函数中对layers的描述，加了个edim_w加上了美学特征
        self.layers_A = config['layers_A']
        self.layers_R = config['layers_R']
        # self.layers_Out = config['layers_Out']
        self.edim_u = config['edim_u']
        self.edim_v = config['edim_v']
        self.edim_w = config['edim_w']
        self.edim_A = self.edim_w + self.edim_u
        self.edim_R = self.edim_u + self.edim_v
        self.nhop = len(self.layers_A)
        # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
        self.max_grad_norm = config['max_grad_norm']
        self.negRatio = config['negRatio']
        self.activation = config['activation']
        self.learner = config['learner']
        # 目标函数即损失函数
        self.objective = config['objective']
        # (pos, neg)
        self.class_size = 2     # 输出是0还是1
        # (user, item)
        self.input_size = 2
        # 可调
        self.adv_weight = 0.05
        self.show = config['show']
        self.checkpoint_dir = config['checkpoint_dir']

        # （用户、cloth）或者（用户、tolls）
        # tf.placeholder()表示占位(没有实际意义)，通过feed_dict()向占位符喂入数据
        self.input_cloth = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")
        self.target_cloth = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")
        self.input_tools = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")
        self.target_tools = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")

        self.lr = None
        self.init_lr = config['init_lr']
        self.current_lr = config['init_lr']
        self.loss_joint = None
        self.loss_cloth_joint = None
        self.loss_tools_joint = None
        self.loss_cloth_only = None
        self.loss_tools_only = None
        self.optim_joint = None
        self.optim_cloth = None
        self.optim_tools = None
        self.step = None
        self.sess = sess
        self.log_loss_cloth = []
        self.log_perp_cloth = []
        self.log_loss_tools = []
        self.log_perp_tools = []
        self.isDebug = config['isDebug']
        self.isOneBatch = config['isOneBatch']
        self.weights_cloth_tools = config['weights_cloth_tools']
        self.cross_layers = config['cross_layers']

        assert self.cross_layers > 0 and self.cross_layers < self.nhop

        # 指标
        self.topK = config['topK']
        self.bestHR_cloth = 0.
        self.bestHR_epoch_cloth = -1
        self.bestNDCG_cloth = 0.
        self.bestNDCG_epoch_cloth = -1
        self.bestMRR_cloth = 0.
        self.bestMRR_epoch_cloth = -1
        self.bestAUC_cloth = 0.
        self.bestAUC_epoch_cloth = -1
        self.HR_cloth, self.NDCG_cloth, self.MRR_cloth, self.AUC_cloth = 0, 0, 0, 0
        self.bestHR_tools = 0.
        self.bestHR_epoch_tools = -1
        self.bestNDCG_tools = 0.
        self.bestNDCG_epoch_tools = -1
        self.bestMRR_tools = 0.
        self.bestMRR_epoch_tools = -1
        self.bestAUC_tools = 0.
        self.bestAUC_epoch_tools = -1
        self.HR_tools, self.NDCG_tools, self.MRR_tools, self.AUC_tools = 0, 0, 0, 0

    def build_memory_shared(self):
        # 参数共享
        # 共享的用户矩阵表示，U是将one-hot编码embedding的矩阵
        # tf.random_normal是从服从指定正态分布的序列中随机取出指定个数的值
        self.U_A = tf.Variable(tf.random_normal([self.nUsers, self.edim_u], stddev=self.init_std))
        # 共享的公共特征提取时（对抗网络下面的部分）的权重和bias
        self.weights_shared_A = defaultdict(object)
        self.bias_shared_A = defaultdict(object)
        # # 共享的交叉网络矩阵H
        # self.shared_Hs = defaultdict(object)
        # self.shared_Hs_A = tf.Variable(tf.random_normal([self.layers_A[2], self.layers_A[3]], stddev=self.init_std))

        self.shared_Hs_A = tf.Variable(tf.random_normal([self.layers_A[3], self.layers_A[3]], stddev=self.init_std))

        for h in range(1, self.cross_layers+1):
            self.weights_shared_A[h] = tf.Variable(tf.random_normal([self.layers_A[h-1], self.layers_A[h]], stddev=self.init_std))
            self.bias_shared_A[h] = tf.Variable(tf.random_normal([self.layers_A[h]], stddev=self.init_std))
            # self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h], self.layers[h+1]], stddev=self.init_std))


        # 参数共享
        # 共享的用户矩阵表示，U是将one-hot编码embedding的矩阵
        # tf.random_normal是从服从指定正态分布的序列中随机取出指定个数的值
        self.U_R = tf.Variable(tf.random_normal([self.nUsers, self.edim_u], stddev=self.init_std))
        # 共享的公共特征提取时（对抗网络下面的部分）的权重和bias
        self.weights_shared_R = defaultdict(object)
        self.bias_shared_R = defaultdict(object)
        # # 共享的交叉网络矩阵H
        # self.shared_Hs = defaultdict(object)
        # self.shared_Hs_R = tf.Variable(tf.random_normal([self.layers_R[2], self.layers_R[3]], stddev=self.init_std))

        self.shared_Hs_R = tf.Variable(tf.random_normal([self.layers_R[3], self.layers_R[3]], stddev=self.init_std))

        for h in range(1, self.cross_layers+1):     # (1, 4)
            self.weights_shared_R[h] = tf.Variable(tf.random_normal([self.layers_R[h-1], self.layers_R[h]], stddev=self.init_std))
            self.bias_shared_R[h] = tf.Variable(tf.random_normal([self.layers_R[h]], stddev=self.init_std))
            # self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h], self.layers[h+1]], stddev=self.init_std))


        # self.w_q = tf.Variable(tf.random_normal([self.edim_w, self.edim_w], stddev=self.init_std))
        # self.w_k = tf.Variable(tf.random_normal([self.edim_w, self.edim_w], stddev=self.init_std))
        # self.w_v = tf.Variable(tf.random_normal([self.edim_w, self.edim_w], stddev=self.init_std))

    def build_memory_cloth_specific(self):
        self.aesthetic_cloth = tf.constant(self.aesthetic_cloth_all, shape=[len(self.aesthetic_cloth_all), 1024], verify_shape=True)
        # cloth网络特有的参数
        # # 美学网络中的参数
        # self.A_cloth = tf.Variable(tf.random_normal([self.nItems_cloth, self.edim_w], stddev=self.init_std))
        # V_cloth是将one-hot编码embedding的矩阵
        # self.V_cloth_A = tf.Variable(tf.random_normal([self.nItems_cloth, self.edim_v], stddev=self.init_std))
        # cloth网络中间隐层的参数
        self.weights_cloth_A = defaultdict(object)
        self.biases_cloth_A = defaultdict(object)
        # 一共有nhop-1个，self.nhop=4，则一共有三层隐层参数
        for h in range(1, self.nhop):
            self.weights_cloth_A[h] = tf.Variable(tf.random_normal([self.layers_A[h-1], self.layers_A[h]], stddev=self.init_std))
            self.biases_cloth_A[h] = tf.Variable(tf.random_normal([self.layers_A[h]], stddev=self.init_std))
        # # !输出层的参数
        # self.h_cloth_A = tf.Variable(tf.random_normal([self.layers_A[-1], self.class_size], stddev=self.init_std))
        # self.b_cloth_A = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))


        # cloth网络特有的参数
        # # 美学网络中的参数
        # self.A_cloth = tf.Variable(tf.random_normal([self.nItems_cloth, self.edim_w], stddev=self.init_std))
        # V_cloth是将one-hot编码embedding的矩阵
        self.V_cloth_R = tf.Variable(tf.random_normal([self.nItems_cloth, self.edim_v], stddev=self.init_std))
        # cloth网络中间隐层的参数
        self.weights_cloth_R = defaultdict(object)
        self.biases_cloth_R = defaultdict(object)
        # 一共有nhop-1个，self.nhop=4，则一共有三层隐层参数
        for h in range(1, self.nhop):       # (1, 4)
            self.weights_cloth_R[h] = tf.Variable(tf.random_normal([self.layers_R[h-1], self.layers_R[h]], stddev=self.init_std))
            self.biases_cloth_R[h] = tf.Variable(tf.random_normal([self.layers_R[h]], stddev=self.init_std))
        # # !输出层的参数
        # self.h_cloth_R = tf.Variable(tf.random_normal([self.layers_R[-1], self.class_size], stddev=self.init_std))
        # self.b_cloth_R = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))


        # self.weights_cloth = defaultdict(object)
        # self.biases_cloth = defaultdict(object)
        # for h in range(1, len(self.layers_Out)):
        #     self.weights_cloth[h] = tf.Variable(tf.random_normal([self.layers_Out[h-1], self.layers_Out[h]], stddev=self.init_std))
        #     self.biases_cloth[h] = tf.Variable(tf.random_normal([self.layers_Out[h]], stddev=self.init_std))

        self.w_cloth = tf.Variable(tf.random_normal([16, self.class_size], stddev=self.init_std))
        self.b_cloth = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

        # self.attention_w_cloth = tf.Variable(tf.random_normal([8, 8], stddev=self.init_std))

    def build_model_cloth_training(self):
        USERin_cloth_A = tf.nn.embedding_lookup(self.U_A, self.input_cloth[:, 0])
        Ain_cloth_A = tf.gather(self.aesthetic_cloth, self.input_cloth[:, 1])
        UAin_cloth_A = tf.concat(values=[USERin_cloth_A, Ain_cloth_A], axis=1)

        USERin_cloth_R = tf.nn.embedding_lookup(self.U_R, self.input_cloth[:, 0])
        ITEMin_cloth_R = tf.nn.embedding_lookup(self.V_cloth_R, self.input_cloth[:, 1])
        UIin_cloth_R = tf.concat(values=[USERin_cloth_R, ITEMin_cloth_R], axis=1)


        self.layer_h_clothes_A = defaultdict(object)
        layer_h_cloth_A = tf.reshape(UAin_cloth_A, [-1, self.edim_A])
        self.layer_h_clothes_A[0] = layer_h_cloth_A

        self.layer_h_clothes_R = defaultdict(object)
        # self.edim = self.edim_u + self.edim_v + self.edim_w
        layer_h_cloth_R = tf.reshape(UIin_cloth_R, [-1, self.edim_R])
        # layer_h_cloth = UIAin_cloth
        self.layer_h_clothes_R[0] = layer_h_cloth_R


        for h in range(1, self.nhop):
            layer_h_cloth_A = tf.add(tf.matmul(self.layer_h_clothes_A[h-1], self.weights_cloth_A[h]), self.biases_cloth_A[h])
            if self.activation == 'relu':
                layer_h_cloth_A = tf.nn.relu(layer_h_cloth_A)
            elif self.activation == 'sigmoid':
                layer_h_cloth_A = tf.nn.sigmoid(layer_h_cloth_A)
            self.layer_h_clothes_A[h] = layer_h_cloth_A


            layer_h_cloth_R = tf.add(tf.matmul(self.layer_h_clothes_R[h-1], self.weights_cloth_R[h]), self.biases_cloth_R[h])
            if self.activation == 'relu':
                layer_h_cloth_R = tf.nn.relu(layer_h_cloth_R)
            elif self.activation == 'sigmoid':
                layer_h_cloth_R = tf.nn.sigmoid(layer_h_cloth_R)
            self.layer_h_clothes_R[h] = layer_h_cloth_R


        self.layer_h_clothes = defaultdict(object)
        layer_h_cloth = tf.reshape(tf.concat(values=[layer_h_cloth_A, layer_h_cloth_R], axis=1), [-1, 16])
        # layer_h_cloth = tf.reshape(tf.concat(values=[tf.matmul(layer_h_cloth_A, self.attention_w_cloth), tf.matmul(layer_h_cloth_R, tf.constant(1.0, shape=[8, 8]) - self.attention_w_cloth)], axis=1), [-1, 16])
        self.layer_h_clothes[0] = layer_h_cloth


        # for h in range(1, len(self.layers_Out)):
        #     layer_h_cloth = tf.add(tf.matmul(self.layer_h_clothes[h-1], self.weights_cloth[h]), self.biases_cloth[h])
        #     if self.activation == 'relu':
        #         layer_h_cloth = tf.nn.relu(layer_h_cloth)
        #     elif self.activation == 'sigmoid':
        #         layer_h_cloth = tf.nn.sigmoid(layer_h_cloth)
        #     self.layer_h_clothes[h] = layer_h_cloth


        # 输出层
        self.z_cloth_only = tf.matmul(layer_h_cloth, self.w_cloth) + self.b_cloth
        self.pred_cloth_only = tf.nn.softmax(self.z_cloth_only)


        # loss函数和optimization
        if self.objective == 'cross':
            self.loss_cloth_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_cloth_only, labels=self.target_cloth)
        elif self.objective == 'log':
            self.loss_cloth_only = tf.losses.log_loss(predictions=self.pred_cloth_only, labels=self.target_cloth)
        else:
            self.loss_cloth_only = tf.losses.hinge_loss(logits=self.z_cloth_only, labels=self.target_cloth)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_cloth = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_cloth = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_cloth = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_cloth = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # 输入层和输出层中的参数
        params = [self.U_A, self.U_R, self.V_cloth_R, self.w_cloth, self.b_cloth]
        # 隐层中的参数
        for h in range(1, self.nhop):
            params.append(self.weights_cloth_A[h])
            params.append(self.biases_cloth_A[h])

            params.append(self.weights_cloth_R[h])
            params.append(self.biases_cloth_R[h])

        # for h in range(1, len(self.layers_Out)):
        #     params.append(self.weights_cloth[h])
        #     params.append(self.biases_cloth[h])


        grads_and_vars = self.opt_cloth.compute_gradients(self.loss_cloth_only, params)
        # tf.clip_by_norm（）对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。
        # 梯度和变量的元组对
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars if not gv[0] is None]

        inc = self.global_step.assign_add(1)
        # 只有inc被执行才会执行后面的函数
        with tf.control_dependencies([inc]):
            # 更新梯度的操作
            self.optim_cloth = self.opt_cloth.apply_gradients(clipped_grads_and_vars)

    def build_memory_tools_specific(self):
        self.aesthetic_tools = tf.constant(self.aesthetic_tools_all, shape=[len(self.aesthetic_tools_all), 1024], verify_shape=True)
        # self.V_tools_A = tf.Variable(tf.random_normal([self.nItems_tools, self.edim_v], stddev=self.init_std))
        self.weights_tools_A = defaultdict(object)
        self.biases_tools_A = defaultdict(object)
        for h in range(1, self.nhop):
            self.weights_tools_A[h] = tf.Variable(tf.random_normal([self.layers_A[h-1], self.layers_A[h]], stddev=self.init_std))
            self.biases_tools_A[h] = tf.Variable(tf.random_normal([self.layers_A[h]], stddev=self.init_std))


        self.V_tools_R = tf.Variable(tf.random_normal([self.nItems_tools, self.edim_v], stddev=self.init_std))
        self.weights_tools_R = defaultdict(object)
        self.biases_tools_R = defaultdict(object)
        for h in range(1, self.nhop):
            self.weights_tools_R[h] = tf.Variable(tf.random_normal([self.layers_R[h-1], self.layers_R[h]], stddev=self.init_std))
            self.biases_tools_R[h] = tf.Variable(tf.random_normal([self.layers_R[h]], stddev=self.init_std))


        # self.weights_tools = defaultdict(object)
        # self.biases_tools = defaultdict(object)
        # for h in range(1, len(self.layers_Out)):
        #     self.weights_tools[h] = tf.Variable(tf.random_normal([self.layers_Out[h-1], self.layers_Out[h]], stddev=self.init_std))
        #     self.biases_tools[h] = tf.Variable(tf.random_normal([self.layers_Out[h]], stddev=self.init_std))

        self.w_tools = tf.Variable(tf.random_normal([16, self.class_size], stddev=self.init_std))
        self.b_tools = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

        # self.attention_w_tools = tf.Variable(tf.random_normal([8, 8], stddev=self.init_std))

    def build_model_tools_training(self):
        USERin_tools_A = tf.nn.embedding_lookup(self.U_A, self.input_tools[:, 0])
        Ain_tools_A = tf.gather(self.aesthetic_tools, self.input_tools[:, 1])
        UAin_tools_A = tf.concat(values=[USERin_tools_A, Ain_tools_A], axis=1)

        USERin_tools_R = tf.nn.embedding_lookup(self.U_R, self.input_tools[:, 0])
        ITEMin_tools_R = tf.nn.embedding_lookup(self.V_tools_R, self.input_tools[:, 1])
        UIin_tools_R = tf.concat(values=[USERin_tools_R, ITEMin_tools_R], axis=1)


        self.layer_h_toolss_A = defaultdict(object)
        layer_h_tools_A = tf.reshape(UAin_tools_A, [-1, self.edim_A])
        self.layer_h_toolss_A[0] = layer_h_tools_A

        self.layer_h_toolss_R = defaultdict(object)
        layer_h_tools_R = tf.reshape(UIin_tools_R, [-1, self.edim_R])
        self.layer_h_toolss_R[0] = layer_h_tools_R


        for h in range(1, self.nhop):
            layer_h_tools_A = tf.add(tf.matmul(self.layer_h_toolss_A[h-1], self.weights_tools_A[h]), self.biases_tools_A[h])
            if self.activation == 'relu':
                layer_h_tools_A = tf.nn.relu(layer_h_tools_A)
            elif self.activation == 'sigmoid':
                layer_h_tools_A = tf.nn.sigmoid(layer_h_tools_A)
            self.layer_h_toolss_A[h] = layer_h_tools_A

            layer_h_tools_R = tf.add(tf.matmul(self.layer_h_toolss_R[h-1], self.weights_tools_R[h]), self.biases_tools_R[h])
            if self.activation == 'relu':
                layer_h_tools_R = tf.nn.relu(layer_h_tools_R)
            elif self.activation == 'sigmoid':
                layer_h_tools_R = tf.nn.sigmoid(layer_h_tools_R)
            self.layer_h_toolss_R[h] = layer_h_tools_R


        self.layer_h_toolss = defaultdict(object)
        layer_h_tools = tf.reshape(tf.concat(values=[layer_h_tools_A, layer_h_tools_R], axis=1), [-1, 16])
        # layer_h_tools = tf.reshape(tf.concat(values=[tf.matmul(layer_h_tools_A, self.attention_w_tools), tf.matmul(layer_h_tools_R, tf.constant(1.0, shape=[8, 8]) - self.attention_w_tools)], axis=1), [-1, 16])
        self.layer_h_toolss[0] = layer_h_tools


        # for h in range(1, len(self.layers_Out)):
        #     layer_h_tools = tf.add(tf.matmul(self.layer_h_toolss[h-1], self.weights_tools[h]), self.biases_tools[h])
        #     if self.activation == 'relu
        #     ':
        #         layer_h_tools = tf.nn.relu(layer_h_tools)
        #     elif self.activation == 'sigmoid':
        #         layer_h_tools = tf.nn.sigmoid(layer_h_tools)
        #     self.layer_h_toolss[h] = layer_h_tools


        # 输出层
        self.z_tools_only = tf.matmul(layer_h_tools, self.w_tools) + self.b_tools
        self.pred_tools_only = tf.nn.softmax(self.z_tools_only)


        if self.objective == 'cross':
            self.loss_tools_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_tools_only, labels=self.target_tools)
        elif self.objective == 'log':
            self.loss_tools_only = tf.losses.log_loss(predictions=self.pred_tools_only, labels=self.target_tools)
        else:
            self.loss_tools_only = tf.losses.hinge_loss(logits=self.z_tools_only, labels=self.target_tools)
        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_tools = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_tools = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_tools = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_tools = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U_A, self.U_R, self.V_tools_R, self.w_tools, self.b_tools]
        for h in range(1, self.nhop):
            params.append(self.weights_tools_A[h])
            params.append(self.biases_tools_A[h])

            params.append(self.weights_tools_R[h])
            params.append(self.biases_tools_R[h])

        # for h in range(1, len(self.layers_Out)):
        #     params.append(self.weights_tools[h])
        #     params.append(self.biases_tools[h])

        grads_and_vars = self.opt_tools.compute_gradients(self.loss_tools_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars if not gv[0] is None]
        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim_tools = self.opt_tools.apply_gradients(clipped_grads_and_vars)

    # add
    def adversarial_loss_A(self, feature):
        # Flip the gradient when backpropagating through this operation
        flip_gradient_A = base_model_A.FlipGradientBuilder()
        feature_A = flip_gradient_A(feature)
        # !这两个参数没有定义
        # if self.is_train:
        #     feature = tf.nn.dropout(feature, self.keep_prob1)
        # 如果变量存在，函数tf.get_variable( ) 会返回现有的变量。如果变量不存在，会根据给定形状和初始值创建变量。
        W0_adv_A = tf.get_variable(name='W0_adv_A', shape=[8, 4],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=self.init_std))
        b0_adv_A = tf.get_variable(name='b0_adv_A', shape=[4], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=self.init_std))
        d_h_fc0_A = tf.nn.relu(tf.matmul(feature_A, W0_adv_A) + b0_adv_A)

        W1_adv_A = tf.get_variable(name='W1_adv_A', shape=[4, 2],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=self.init_std))
        b1_adv_A = tf.get_variable(name='b1_adv_A', shape=[2], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=self.init_std))
        d_logits_A = tf.matmul(d_h_fc0_A, W1_adv_A) + b1_adv_A
        # domain_pred = tf.nn.softmax(d_logits)
        # 用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        adv_loss_A = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_A, labels=self.domain_arr))
        return adv_loss_A

    # add
    def adversarial_loss_R(self, feature):
        # Flip the gradient when backpropagating through this operation
        flip_gradient_R = base_model_R.FlipGradientBuilder()
        feature_R = flip_gradient_R(feature)
        # !这两个参数没有定义
        # if self.is_train:
        #     feature = tf.nn.dropout(feature, self.keep_prob1)
        # 如果变量存在，函数tf.get_variable( ) 会返回现有的变量。如果变量不存在，会根据给定形状和初始值创建变量。
        W0_adv_R = tf.get_variable(name='W0_adv_R', shape=[8, 4],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=self.init_std))
        b0_adv_R = tf.get_variable(name='b0_adv_R', shape=[4], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=self.init_std))
        d_h_fc0_R = tf.nn.relu(tf.matmul(feature_R, W0_adv_R) + b0_adv_R)

        W1_adv_R = tf.get_variable(name='W1_adv_R', shape=[4, 2],
                                       dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=self.init_std))
        b1_adv_R = tf.get_variable(name='b1_adv_R', shape=[2], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=self.init_std))
        d_logits_R = tf.matmul(d_h_fc0_R, W1_adv_R) + b1_adv_R
        # domain_pred = tf.nn.softmax(d_logits)
        # 用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        adv_loss_R = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_R, labels=self.domain_arr))
        return adv_loss_R

    def build_model_joint_training(self):
        # 联合训练
        # 输入层和embedding层
        USERin_cloth_A = tf.nn.embedding_lookup(self.U_A, self.input_cloth[:, 0])
        Ain_cloth_A = tf.gather(self.aesthetic_cloth, self.input_cloth[:, 1])
        UAin_cloth_A = tf.concat(values=[USERin_cloth_A, Ain_cloth_A], axis=1)

        USERin_cloth_R = tf.nn.embedding_lookup(self.U_R, self.input_cloth[:, 0])
        ITEMin_cloth_R = tf.nn.embedding_lookup(self.V_cloth_R, self.input_cloth[:, 1])
        UIin_cloth_R = tf.concat(values=[USERin_cloth_R, ITEMin_cloth_R], axis=1)


        USERin_tools_A = tf.nn.embedding_lookup(self.U_A, self.input_tools[:, 0])
        Ain_tools_A = tf.gather(self.aesthetic_tools, self.input_tools[:, 1])
        UAin_tools_A = tf.concat(values=[USERin_tools_A, Ain_tools_A], axis=1)

        USERin_tools_R = tf.nn.embedding_lookup(self.U_R, self.input_tools[:, 0])
        ITEMin_tools_R = tf.nn.embedding_lookup(self.V_tools_R, self.input_tools[:, 1])
        UIin_tools_R = tf.concat(values=[USERin_tools_R, ITEMin_tools_R], axis=1)


        self.layer_h_clothes_A = defaultdict(object)
        layer_h_cloth_A = tf.reshape(UAin_cloth_A, [-1, self.edim_A])
        self.layer_h_clothes_A[0] = layer_h_cloth_A

        self.layer_h_clothes_R = defaultdict(object)
        # self.edim = self.edim_u + self.edim_v + self.edim_w
        layer_h_cloth_R = tf.reshape(UIin_cloth_R, [-1, self.edim_R])
        # layer_h_cloth = UIAin_cloth
        self.layer_h_clothes_R[0] = layer_h_cloth_R


        self.layer_h_toolss_A = defaultdict(object)
        layer_h_tools_A = tf.reshape(UAin_tools_A, [-1, self.edim_A])
        self.layer_h_toolss_A[0] = layer_h_tools_A

        self.layer_h_toolss_R = defaultdict(object)
        layer_h_tools_R = tf.reshape(UIin_tools_R, [-1, self.edim_R])
        self.layer_h_toolss_R[0] = layer_h_tools_R


        # 共享特征提取的部分
        self.layer_h_shared_A = defaultdict(object)
        self.layer_h_shared_A[0] = tf.concat([layer_h_cloth_A, layer_h_tools_A], 0)

        self.layer_h_shared_R = defaultdict(object)
        self.layer_h_shared_R[0] = tf.concat([layer_h_cloth_R, layer_h_tools_R], 0)

        self.domain_arr = np.vstack((np.tile([1., 0.], [self.batch_size, 1]),
                                     np.tile([0., 1.], [self.batch_size, 1])))

        for h in range(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.layer_h_shared_A[h] = tf.add(tf.matmul(self.layer_h_shared_A[h - 1], self.weights_shared_A[h]), self.bias_shared_A[h])
            self.layer_h_shared_R[h] = tf.add(tf.matmul(self.layer_h_shared_R[h - 1], self.weights_shared_R[h]), self.bias_shared_R[h])


            layer_h_cloth_A = tf.add(tf.matmul(self.layer_h_clothes_A[h - 1], self.weights_cloth_A[h]), self.biases_cloth_A[h])
            layer_h_cloth_R = tf.add(tf.matmul(self.layer_h_clothes_R[h - 1], self.weights_cloth_R[h]), self.biases_cloth_R[h])
            layer_h_tools_A = tf.add(tf.matmul(self.layer_h_toolss_A[h - 1], self.weights_tools_A[h]), self.biases_tools_A[h])
            layer_h_tools_R = tf.add(tf.matmul(self.layer_h_toolss_R[h - 1], self.weights_tools_R[h]), self.biases_tools_R[h])

            # if h == 3:
            #     layer_h_cloth_A = tf.add(layer_h_cloth_A, tf.matmul(self.layer_h_shared_A[h][:128], self.shared_Hs_A))
            #     layer_h_cloth_R = tf.add(layer_h_cloth_R, tf.matmul(self.layer_h_shared_R[h][:128], self.shared_Hs_R))
            #     layer_h_tools_A = tf.add(layer_h_tools_A, tf.matmul(self.layer_h_shared_A[h][128:], self.shared_Hs_A))
            #     layer_h_tools_R = tf.add(layer_h_tools_R, tf.matmul(self.layer_h_shared_R[h][128:], self.shared_Hs_R))

            if self.activation == 'relu':
                layer_h_cloth_A = tf.nn.relu(layer_h_cloth_A)
                layer_h_cloth_R = tf.nn.relu(layer_h_cloth_R)
                layer_h_tools_A = tf.nn.relu(layer_h_tools_A)
                layer_h_tools_R = tf.nn.relu(layer_h_tools_R)
            elif self.activation == 'sigmoid':
                layer_h_cloth_A = tf.nn.sigmoid(layer_h_cloth_A)
                layer_h_cloth_R = tf.nn.sigmoid(layer_h_cloth_R)
                layer_h_tools_A = tf.nn.sigmoid(layer_h_tools_A)
                layer_h_tools_R = tf.nn.sigmoid(layer_h_tools_R)

            self.layer_h_clothes_A[h] = layer_h_cloth_A
            self.layer_h_clothes_R[h] = layer_h_cloth_R
            self.layer_h_toolss_A[h] = layer_h_tools_A
            self.layer_h_toolss_R[h] = layer_h_tools_R


        # adv_loss
        self.adv_loss_A = self.adversarial_loss_A(self.layer_h_shared_A[3])
        self.adv_loss_R = self.adversarial_loss_R(self.layer_h_shared_R[3])

        layer_h_cloth_A = tf.add(layer_h_cloth_A, tf.matmul(self.layer_h_shared_A[3][:128], self.shared_Hs_A))
        layer_h_cloth_R = tf.add(layer_h_cloth_R, tf.matmul(self.layer_h_shared_R[3][:128], self.shared_Hs_R))
        layer_h_tools_A = tf.add(layer_h_tools_A, tf.matmul(self.layer_h_shared_A[3][128:], self.shared_Hs_A))
        layer_h_tools_R = tf.add(layer_h_tools_R, tf.matmul(self.layer_h_shared_R[3][128:], self.shared_Hs_R))

        self.layer_h_clothes = defaultdict(object)
        layer_h_cloth = tf.reshape(tf.concat(values=[layer_h_cloth_A, layer_h_cloth_R], axis=1), [-1, 16])
        # layer_h_cloth = tf.reshape(tf.concat(values=[tf.matmul(layer_h_cloth_A, self.attention_w_cloth), tf.matmul(layer_h_cloth_R, tf.constant(1.0, shape=[8, 8]) - self.attention_w_cloth)], axis=1), [-1, 16])
        self.layer_h_clothes[0] = layer_h_cloth

        self.layer_h_toolss = defaultdict(object)
        layer_h_tools = tf.reshape(tf.concat(values=[layer_h_tools_A, layer_h_tools_R], axis=1), [-1, 16])
        # layer_h_tools = tf.reshape(tf.concat(values=[tf.matmul(layer_h_tools_A, self.attention_w_tools), tf.matmul(layer_h_tools_R, tf.constant(1.0, shape=[8, 8]) - self.attention_w_tools)], axis=1), [-1, 16])
        self.layer_h_toolss[0] = layer_h_tools


        # for h in range(1, len(self.layers_Out)):
        #     layer_h_cloth = tf.add(tf.matmul(self.layer_h_clothes[h-1], self.weights_cloth[h]), self.biases_cloth[h])
        #     if self.activation == 'relu':
        #         layer_h_cloth = tf.nn.relu(layer_h_cloth)
        #     elif self.activation == 'sigmoid':
        #         layer_h_cloth = tf.nn.sigmoid(layer_h_cloth)
        #     self.layer_h_clothes[h] = layer_h_cloth
        #
        #     layer_h_tools = tf.add(tf.matmul(self.layer_h_toolss[h-1], self.weights_tools[h]), self.biases_tools[h])
        #     if self.activation == 'relu':
        #         layer_h_tools = tf.nn.relu(layer_h_tools)
        #     elif self.activation == 'sigmoid':
        #         layer_h_tools = tf.nn.sigmoid(layer_h_tools)
        #     self.layer_h_toolss[h] = layer_h_tools

        self.z_cloth_joint = tf.matmul(layer_h_cloth, self.w_cloth) + self.b_cloth
        self.pred_cloth_joint = tf.nn.softmax(self.z_cloth_only)

        self.z_tools_joint = tf.matmul(layer_h_tools, self.w_tools) + self.b_tools
        self.pred_tools_joint = tf.nn.softmax(self.z_tools_only)


        # 损失函数和优化器
        if self.objective == 'cross':
            self.loss_cloth_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_cloth_joint, labels=self.target_cloth)
        elif self.objective == 'log':
            self.loss_cloth_joint = tf.losses.log_loss(predictions=self.pred_cloth_joint, labels=self.target_cloth)
        else:
            self.loss_cloth_joint = tf.losses.hinge_loss(logits=self.z_cloth_joint, labels=self.target_cloth)
        if self.objective == 'cross':
            self.loss_tools_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_tools_joint, labels=self.target_tools)
        elif self.objective == 'log':
            self.loss_tools_joint = tf.losses.log_loss(predictions=self.pred_tools_joint, labels=self.target_tools)
        else:
            self.loss_tools_joint = tf.losses.hinge_loss(logits=self.z_tools_joint, labels=self.target_tools)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_joint = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_joint = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_joint = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_joint = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # params = [self.U_R, self.V_cloth_R, self.V_tools_R, self.w_cloth, self.b_cloth, self.w_tools, self.b_tools]

        params = [self.U_A, self.U_R, self.V_cloth_R, self.V_tools_R, self.w_cloth, self.b_cloth, self.w_tools, self.b_tools, self.shared_Hs_A, self.shared_Hs_R]

        for h in range(1, self.nhop):  # 隐层中的权重和biases
            params.append(self.weights_cloth_A[h])
            params.append(self.biases_cloth_A[h])

            params.append(self.weights_cloth_R[h])
            params.append(self.biases_cloth_R[h])


            params.append(self.weights_tools_A[h])
            params.append(self.biases_tools_A[h])

            params.append(self.weights_tools_R[h])
            params.append(self.biases_tools_R[h])


            params.append(self.weights_shared_A[h])
            params.append(self.bias_shared_A[h])

            params.append(self.weights_shared_R[h])
            params.append(self.bias_shared_R[h])

        # for h in range(1, len(self.layers_Out)):
        #     params.append(self.weights_cloth[h])
        #     params.append(self.biases_cloth[h])
        #
        #
        #     params.append(self.weights_tools[h])
        #     params.append(self.biases_tools[h])


        self.loss_joint = self.weights_cloth_tools[0] * self.loss_cloth_joint + \
                          self.weights_cloth_tools[1] * self.loss_tools_joint + \
                          self.adv_weight * self.adv_loss_A + self.adv_weight * self.adv_loss_R

        grads_and_vars = self.opt_joint.compute_gradients(self.loss_joint, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars if not gv[0] is None]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim_joint = self.opt_joint.apply_gradients(clipped_grads_and_vars)

    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.build_memory_shared()

        self.build_memory_cloth_specific()
        self.build_model_cloth_training()

        self.build_memory_tools_specific()
        self.build_model_tools_training()

        self.build_model_joint_training()

        tf.global_variables_initializer().run()

    def get_train_instances_cloth(self):
        self.user_input_cloth, self.item_input_cloth, self.labels_cloth = [], [], []
        for (u, i) in self.train_cloth.keys():
            # positive instance
            self.user_input_cloth.append(u)
            self.item_input_cloth.append(i)
            self.labels_cloth.append(1)
            # self.aesthetic_cloth.append(self.aesthetic_cloth_all[i])
            # negative negRatio instances
            # 取负样本的比例
            for _ in range(self.negRatio):
                j = np.random.randint(self.nItems_cloth)
                # 确保得到负样本
                while (u, j) in self.train_cloth:
                    j = np.random.randint(self.nItems_cloth)
                self.user_input_cloth.append(u)
                self.item_input_cloth.append(j)
                self.labels_cloth.append(0)
                # self.aesthetic_cloth.append(self.aesthetic_cloth_all[j])

    def get_train_instances_tools(self):
        self.user_input_tools, self.item_input_tools, self.labels_tools = [], [], []
        for (u, i) in self.train_tools.keys():
            # positive instance
            self.user_input_tools.append(u)
            self.item_input_tools.append(i)
            self.labels_tools.append(1)
            # self.aesthetic_tools.append(self.aesthetic_tools_all[i])
            # negative negRatio instances
            for _ in range(self.negRatio):
                j = np.random.randint(self.nItems_tools)
                while (u, j) in self.train_tools:
                    j = np.random.randint(self.nItems_tools)
                self.user_input_tools.append(u)
                self.item_input_tools.append(j)
                self.labels_tools.append(0)
                # self.aesthetic_tools.append(self.aesthetic_tools_all[j])

    def get_test_instances_cloth(self):
        self.test_user_input_cloth, self.test_item_input_cloth, self.test_labels_cloth = [], [], []
        for idx in range(len(self.testRatings_cloth)):
            # 留一法
            rating = self.testRatings_cloth[idx]
            u = rating[0]
            gtItem = rating[1]
            self.test_user_input_cloth.append(u)
            self.test_item_input_cloth.append(gtItem)
            self.test_labels_cloth.append(1)
            # self.test_aesthetic_cloth.append(self.aesthetic_cloth_all[gtItem])
            # random 99 neg_items
            items = self.testNegatives_cloth[idx]
            for i in items:
                self.test_user_input_cloth.append(u)
                self.test_item_input_cloth.append(i)
                self.test_labels_cloth.append(0)
                # self.test_aesthetic_cloth.append(self.aesthetic_cloth_all[i])

    def get_test_instances_tools(self):
        self.test_user_input_tools, self.test_item_input_tools, self.test_labels_tools = [], [], []
        for idx in range(len(self.testRatings_tools)):
            rating = self.testRatings_tools[idx]
            u = rating[0]
            gtItem = rating[1]
            self.test_user_input_tools.append(u)
            self.test_item_input_tools.append(gtItem)
            self.test_labels_tools.append(1)
            # self.test_aesthetic_tools.append(self.aesthetic_tools_all[gtItem])
            # random 99 neg_items
            items = self.testNegatives_tools[idx]
            for i in items:
                self.test_user_input_tools.append(u)
                self.test_item_input_tools.append(i)
                self.test_labels_tools.append(0)
                # self.test_aesthetic_tools.append(self.aesthetic_tools_all[i])

    # def get_user_all_items(self, sample_id):
    #     cloth_user_id = self.user_input_cloth[sample_id]
    #     count_cloth = self.user_input_cloth.count(cloth_user_id) // 100
    #     start_cloth = self.item_input_cloth.index(self.item_input_cloth[sample_id])
    #     cloth_item_ids = []
    #     for i in range(count_cloth):
    #         cloth_item_ids.append(self.item_input_cloth[start_cloth + i * 100])
    #
    #     tools_user_id = self.user_input_tools[sample_id]
    #     count_tools = self.user_input_tools.count(tools_user_id) // 100
    #     start_tools = self.item_input_tools.index(self.item_input_tools[sample_id])
    #     tools_item_ids = []
    #     for i in range(count_tools):
    #         tools_item_ids.append(self.item_input_tools[start_tools + i * 100])
    #
    #     return cloth_item_ids, tools_item_ids

    def train_model(self):
        # 分别得到cloth和tools的训练实例，每个epoch每次随机取样负样本
        self.get_train_instances_cloth()
        self.get_train_instances_tools()

        # 用于训练的实例个数
        num_examples_cloth = len(self.labels_cloth)
        # 一共有多少个batch_size(向上取整)
        num_batches_cloth = int(math.ceil(num_examples_cloth / self.batch_size))
        num_examples_tools = len(self.labels_tools)
        num_batches_tools = int(math.ceil(num_examples_tools / self.batch_size))
        # 最大的batch和最小的batch
        num_batches_max = max(num_batches_cloth, num_batches_tools)
        num_batches_min = min(num_batches_cloth, num_batches_tools)
        print('#batch: cloth={}, tools={}, max={}, min={}'.format(num_batches_cloth, num_batches_tools, num_batches_max, num_batches_min))
        x_cloth = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)
        target_cloth = np.zeros([self.batch_size, self.class_size])
        x_tools = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)
        target_tools = np.zeros([self.batch_size, self.class_size])
        # cloth_ids = np.zeros([self.batch_size, 99], dtype=np.int)
        # tools_ids = np.zeros([self.batch_size, 103], dtype=np.int)
        # 打乱的实例
        sample_ids_cloth = [sid for sid in range(num_examples_cloth)]
        random.shuffle(sample_ids_cloth)
        sample_ids_tools = [sid for sid in range(num_examples_tools)]
        random.shuffle(sample_ids_tools)
        cost_total = 0.0
        cost_joint = 0.0
        cost_cloth = 0.0
        cost_tools = 0.0
        cost_domain = 0.0
        # batch_size的编号
        batches_cloth = [b for b in range(num_batches_cloth)]
        batches_tools = [b for b in range(num_batches_tools)]
        # 先单独训练再联合训练
        if num_batches_cloth < num_batches_tools:
            # 列表：存的是共同训练的部分编号
            batches_join = batches_cloth
            # 列表：取出较长部分的编号，存的是单独训练部分的编号
            batches_single = batches_tools[num_batches_cloth:]
            bar_tools = ProgressBar()
            # 多余的部分
            for _ in bar_tools(range(len(batches_single))):
                # if self.show: bar_tools.update(10 * _ + 1)
                target_tools.fill(0)
                # cloth_ids.fill(0)
                # tools_ids.fill(0)
                for b in range(self.batch_size):
                    # 用完了cloth的id
                    if not sample_ids_tools:
                        sample_id_tools = random.randrange(0, num_examples_tools)
                        x_tools[b][0] = self.user_input_tools[sample_id_tools]
                        x_tools[b][1] = self.item_input_tools[sample_id_tools]
                        # cloth_item_ids, tools_item_ids = self.get_user_all_items(sample_id_tools)
                        # for i in range(len(tools_item_ids)):
                        #     tools_ids[b][i] = tools_item_ids[i]
                        if self.labels_tools[sample_id_tools] == 1:
                            target_tools[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_tools[b][1] = 1  # negative
                    else:
                        # 删除最后一个元素
                        sample_id_tools = sample_ids_tools.pop()
                        x_tools[b][0] = self.user_input_tools[sample_id_tools]
                        x_tools[b][1] = self.item_input_tools[sample_id_tools]
                        if self.labels_tools[sample_id_tools] == 1:
                            target_tools[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_tools[b][1] = 1  # negative
                # 以字典的方式填充占位，placeholder函数（执行过程中再具体赋值）
                keys = [self.input_tools, self.target_tools]
                values = [x_tools, target_tools]
                # 训练模型
                # run()函数中放入要fetch的对象，也就是第一个参数
                _, loss_tools, pred_tools, self.step = self.sess.run([self.optim_tools,
                                                    self.loss_tools_only,
                                                    self.pred_tools_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k: v for k, v in zip(keys, values)
                                                    })
                cost_tools += np.sum(loss_tools)
                cost_total += np.sum(loss_tools)
            # if self.show: bar_tools.finish()
        else:
            batches_join = batches_tools
            batches_single = batches_cloth[num_batches_tools:]
            bar_cloth = ProgressBar()
            for _ in bar_cloth(range(len(batches_single))):
                # if self.show: bar_cloth.update(10 * _ + 1)
                target_cloth.fill(0)
                for b in range(self.batch_size):
                    if not sample_ids_cloth:
                        sample_id_cloth = random.randrange(0, num_examples_cloth)
                        x_cloth[b][0] = self.user_input_cloth[sample_id_cloth]
                        x_cloth[b][1] = self.item_input_cloth[sample_id_cloth]
                        if self.labels_cloth[sample_id_cloth] == 1:
                            target_cloth[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_cloth[b][1] = 1  # negative
                    else:
                        sample_id_cloth = sample_ids_cloth.pop()
                        x_cloth[b][0] = self.user_input_cloth[sample_id_cloth]
                        x_cloth[b][1] = self.item_input_cloth[sample_id_cloth]
                        if self.labels_cloth[sample_id_cloth] == 1:
                            target_cloth[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_cloth[b][1] = 1  # negative
                keys = [self.input_cloth, self.target_cloth]
                values = [x_cloth, target_cloth]
                _, loss_cloth, pred_cloth, self.step = self.sess.run([self.optim_cloth,
                                                    self.loss_cloth_only,
                                                    self.pred_cloth_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k: v for k, v in zip(keys, values)
                                                    })
                cost_cloth += np.sum(loss_cloth)
                cost_total += np.sum(loss_cloth)
            # if self.show: bar_cloth.finish()
        # joint training on both datasets after single training on the bigger one
        bar_join = ProgressBar()
        for _ in bar_join(range(len(batches_join))):
            # if self.show: bar_join.update(10 * _ + 1)
            target_cloth.fill(0)
            target_tools.fill(0)
            for b in range(self.batch_size):
                if not sample_ids_cloth:
                    sample_id_cloth = random.randrange(0, num_examples_cloth)
                    x_cloth[b][0] = self.user_input_cloth[sample_id_cloth]
                    x_cloth[b][1] = self.item_input_cloth[sample_id_cloth]
                    if self.labels_cloth[sample_id_cloth] == 1:
                        target_cloth[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_cloth[b][1] = 1  # negative
                else:
                    sample_id_cloth = sample_ids_cloth.pop()
                    x_cloth[b][0] = self.user_input_cloth[sample_id_cloth]
                    x_cloth[b][1] = self.item_input_cloth[sample_id_cloth]
                    # print(x_cloth[b].shape)
                    if self.labels_cloth[sample_id_cloth] == 1:
                        target_cloth[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_cloth[b][1] = 1  # negative
                if not sample_ids_tools:
                    sample_id_tools = random.randrange(0, num_examples_tools)
                    x_tools[b][0] = self.user_input_tools[sample_id_tools]
                    x_tools[b][1] = self.item_input_tools[sample_id_tools]
                    if self.labels_tools[sample_id_tools] == 1:
                        target_tools[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_tools[b][1] = 1  # negative
                else:
                    sample_id_tools = sample_ids_tools.pop()
                    x_tools[b][0] = self.user_input_tools[sample_id_tools]
                    x_tools[b][1] = self.item_input_tools[sample_id_tools]
                    if self.labels_tools[sample_id_tools] == 1:
                        target_tools[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_tools[b][1] = 1  # negative
            keys = [self.input_cloth, self.input_tools, self.target_cloth, self.target_tools]
            values = [x_cloth, x_tools, target_cloth, target_tools]
            _, loss, loss_cloth, loss_tools, loss_domain_A, loss_domain_R, pred_cloth, pred_tools, self.step = self.sess.run([self.optim_joint,
                                                self.loss_joint,
                                                self.loss_cloth_joint,
                                                self.loss_tools_joint,
                                                self.adv_loss_A,
                                                self.adv_loss_R,
                                                self.pred_cloth_joint,
                                                self.pred_tools_joint,
                                                self.global_step],
                                                feed_dict={
                                                    k: v for k, v in zip(keys, values)
                                                })
            cost_joint += np.sum(loss)
            cost_total += np.sum(loss)
            cost_cloth += np.sum(loss_cloth)
            cost_tools += np.sum(loss_tools)
            cost_domain += np.sum(loss_domain_A)
            cost_domain += np.sum(loss_domain_R)
            # print(cost_domain)

        # if self.show: bar_join.finish()
        return [cost_total/num_batches_max/self.batch_size, cost_joint/num_batches_min/self.batch_size,
                cost_cloth/num_batches_cloth/self.batch_size, cost_tools/num_batches_tools/self.batch_size,
                cost_domain/num_batches_min/self.batch_size]

    def run(self):
        self.get_test_instances_cloth()  # only need to get once
        self.get_test_instances_tools()  # only need to get once

        self.para_str = 'time'+str(int(time.time())) + '_' + 'eu'+str(self.edim_u)+'ev'+str(self.edim_v) + 'ew' + str(self.edim_w)\
                        + str(self.layers_A) + str(self.layers_R) + 'batch'+str(self.batch_size) + 'wtask'+str(self.weights_cloth_tools) \
                        + 'shared_layers'+str(self.cross_layers) + 'lr'+str(self.init_lr) + 'std'+str(self.init_std) + 'nr'+str(self.negRatio)\
                        + str(self.activation) + str(self.learner) + 'loss'+str(self.objective)
        print(self.para_str)
        with open('results_' + self.para_str + '.log', 'w') as ofile:
            ofile.write(self.para_str + '\n')
            start_time = time.time()
            for idx in range(self.nepoch):
                start = time.time()
                # train_loss, train_loss_cloth, train_loss_tools = np.sum(self.train_model())
                train_loss_total, train_loss_joint, train_loss_cloth, train_loss_tools, train_loss_domain = self.train_model()
                train_time = time.time() - start

                # print(train_loss_domain)

                start = time.time()
                valid_loss_cloth = np.sum(self.valid_model_cloth())
                valid_loss_tools = np.sum(self.valid_model_tools())
                valid_loss = (valid_loss_cloth + valid_loss_tools) / 2
                valid_time = time.time() - start

                if self.HR_cloth > self.bestHR_cloth and self.HR_cloth < 0.99 and idx > 3:
                    self.bestHR_cloth = self.HR_cloth
                    self.bestHR_epoch_cloth = idx
                if self.NDCG_cloth > self.bestNDCG_cloth and self.NDCG_cloth < 0.99 and idx > 3:
                    self.bestNDCG_cloth = self.NDCG_cloth
                    self.bestNDCG_epoch_cloth = idx
                if self.MRR_cloth > self.bestMRR_cloth and self.MRR_cloth < 0.99 and idx > 3:
                    self.bestMRR_cloth = self.MRR_cloth
                    self.bestMRR_epoch_cloth = idx
                if self.AUC_cloth > self.bestAUC_cloth and self.AUC_cloth < 0.99 and idx > 3:
                    self.bestAUC_cloth = self.AUC_cloth
                    self.bestAUC_epoch_cloth = idx

                if self.HR_tools > self.bestHR_tools and self.HR_tools < 0.99 and idx > 3:
                    self.bestHR_tools = self.HR_tools
                    self.bestHR_epoch_tools = idx
                if self.NDCG_tools > self.bestNDCG_tools and self.NDCG_tools < 0.99 and idx > 3:
                    self.bestNDCG_tools = self.NDCG_tools
                    self.bestNDCG_epoch_tools = idx
                if self.MRR_tools > self.bestMRR_tools and self.MRR_tools < 0.99 and idx > 3:
                    self.bestMRR_tools = self.MRR_tools
                    self.bestMRR_epoch_tools = idx
                if self.AUC_tools > self.bestAUC_tools and self.AUC_tools < 0.99 and idx > 3:
                    self.bestAUC_tools = self.AUC_tools
                    self.bestAUC_epoch_tools = idx

                print('{:.1f}s. epoch={}, loss_total={:.6f}, loss_joint={:.6f},loss_domain={:.6f}, val_l={:.6f}. {:.1f}s'.format(
                        train_time, idx, train_loss_total, train_loss_joint, train_loss_domain, valid_loss, valid_time))
                print('Cloth: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}.'.format(
                        train_loss_cloth, valid_loss_cloth, self.HR_cloth, self.NDCG_cloth, self.MRR_cloth, self.AUC_cloth))
                print('Tools: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}.'.format(
                        train_loss_tools, valid_loss_tools, self.HR_tools, self.NDCG_tools, self.MRR_tools, self.AUC_tools))
                ofile.write('{:.1f}s. epoch={}, loss_total={:.6f}, loss_joint={:.6f}, val_l={:.6f}. {:.1f}s\n'.format(
                        train_time, idx, train_loss_total, train_loss_joint, valid_loss, valid_time))
                ofile.write('Cloth: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}\n'.format(
                        train_loss_cloth, valid_loss_cloth, self.HR_cloth, self.NDCG_cloth, self.MRR_cloth, self.AUC_cloth))
                ofile.write('Tools: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}\n'.format(
                        train_loss_tools, valid_loss_tools, self.HR_tools, self.NDCG_tools, self.MRR_tools, self.AUC_tools))
                ofile.flush()
            ofile.write('bestHR_cloth = {:.6f} at epoch {}\n'.format(self.bestHR_cloth, self.bestHR_epoch_cloth))
            ofile.write('bestNDCG_cloth = {:.6f} at epoch {}\n'.format(self.bestNDCG_cloth, self.bestNDCG_epoch_cloth))
            ofile.write('bestMRR_cloth = {:.6f} at epoch {}\n'.format(self.bestMRR_cloth, self.bestMRR_epoch_cloth))
            ofile.write('bestAUC_cloth = {:.6f} at epoch {}\n'.format(self.bestAUC_cloth, self.bestAUC_epoch_cloth))
            ofile.write('bestHR_tools = {:.6f} at epoch {}\n'.format(self.bestHR_tools, self.bestHR_epoch_tools))
            ofile.write('bestNDCG_tools = {:.6f} at epoch {}\n'.format(self.bestNDCG_tools, self.bestNDCG_epoch_tools))
            ofile.write('bestMRR_tools = {:.6f} at epoch {}\n'.format(self.bestMRR_tools, self.bestMRR_epoch_tools))
            ofile.write('bestAUC_tools = {:.6f} at epoch {}\n'.format(self.bestAUC_tools, self.bestAUC_epoch_tools))
            print('bestHR_cloth = {:.6f} at epoch {}'.format(self.bestHR_cloth, self.bestHR_epoch_cloth))
            print('bestNDCG_cloth = {:.6f} at epoch {}'.format(self.bestNDCG_cloth, self.bestNDCG_epoch_cloth))
            print('bestMRR_cloth = {:.6f} at epoch {}'.format(self.bestMRR_cloth, self.bestMRR_epoch_cloth))
            print('bestAUC_cloth = {:.6f} at epoch {}'.format(self.bestAUC_cloth, self.bestAUC_epoch_cloth))
            print('bestHR_tools = {:.6f} at epoch {}'.format(self.bestHR_tools, self.bestHR_epoch_tools))
            print('bestNDCG_tools = {:.6f} at epoch {}'.format(self.bestNDCG_tools, self.bestNDCG_epoch_tools))
            print('bestMRR_tools = {:.6f} at epoch {}'.format(self.bestMRR_tools, self.bestMRR_epoch_tools))
            print('bestAUC_tools = {:.6f} at epoch {}'.format(self.bestAUC_tools, self.bestAUC_epoch_tools))
            print('total time = {:.1f}m'.format((time.time() - start_time)/60))
            ofile.write('total time = {:.1f}\n'.format((time.time() - start_time)/60))
        print(self.para_str)

    def valid_model_cloth(self):
        num_test_examples = len(self.test_labels_cloth)
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        if self.show:
            bar = ProgressBar()
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # user,item
        target = np.zeros([self.batch_size, self.class_size])  # one-hot encoding: (pos, neg)
        sample_id = 0
        test_preds = []
        for current_batch in bar(range(num_test_batches)):
            # if self.show: bar.update(10 * current_batch + 1)
            target.fill(0)
            for b in range(self.batch_size):
                if sample_id >= len(self.test_labels_cloth):  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_cloth[0]
                    x[b][1] = self.test_item_input_cloth[0]
                    if self.test_labels_cloth[0] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                else:
                    x[b][0] = self.test_user_input_cloth[sample_id]
                    x[b][1] = self.test_item_input_cloth[sample_id]
                    if self.test_labels_cloth[sample_id] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                sample_id += 1

            keys = [self.input_cloth, self.target_cloth]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_cloth_only, self.pred_cloth_only],
                                        feed_dict={
                                            k: v for k, v in zip(keys, values)
                                        })
            cost += np.sum(loss)
            test_preds.extend(pred)
            if self.isOneBatch:
                break
        # if self.show: bar.finish()

        # evaluation
        user_item_preds = defaultdict(lambda: defaultdict(float))
        user_pred_gtItem = defaultdict(float)
        for sample_id in range(len(self.test_labels_cloth)):
            user = self.test_user_input_cloth[sample_id]
            item = self.test_item_input_cloth[sample_id]
            label = self.test_labels_cloth[sample_id]
            pred = test_preds[sample_id]  # [pos_prob, neg_prob]
            user_item_preds[user][item] = pred[0]
            if item == self.user_gt_item_cloth[user]:
                user_pred_gtItem[user] = pred[0]
        # print("1", user_item_preds)
        # print("2", user_pred_gtItem)
        # print("3", pred)
        HR, NDCG, MRR, AUC = 0, 0, 0, 0
        for user, item_preds in user_item_preds.items():
            # compute AUC
            gt_pred = user_pred_gtItem[user]
            hit = 0
            for i, p in item_preds.items():
                if i != self.user_gt_item_cloth[user] and p < gt_pred:
                    hit += 1
            AUC += hit / 99.0
            # compute HR, NDCG, MRR
            item_preds = sorted(item_preds.items(), key=lambda x: -x[1])
            item_preds_topK = item_preds[:self.topK]
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_cloth[user]:
                    HR += 1
                    break
            for position in range(len(item_preds_topK)):
                item, pred = item_preds_topK[position]
                if item == self.user_gt_item_cloth[user]:
                    NDCG += math.log(2) / math.log(position + 2)
                    # MRR += 1 / (position + 1)
                    break
            rank = 1
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_cloth[user]:
                    break
                rank += 1
            MRR += 1 / rank
            if self.isDebug and user == 1:
                print('gt_pred = {:.6f}, topK_pred={:.6f}'.format(gt_pred, item_preds[self.topK][1]))
        self.HR_cloth = HR / len(user_item_preds)
        self.NDCG_cloth = NDCG / len(user_item_preds)
        self.MRR_cloth = MRR / len(user_item_preds)
        self.AUC_cloth = AUC / len(user_item_preds)
        return cost/num_test_batches/self.batch_size

    def valid_model_tools(self):
        num_test_examples = len(self.test_labels_tools)
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        if self.show:
            bar = ProgressBar()
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # user,item
        target = np.zeros([self.batch_size, self.class_size])  # one-hot encoding: (pos, neg)
        sample_id = 0
        test_preds = []
        for current_batch in bar(range(num_test_batches)):
            # if self.show: bar.update(10 * current_batch + 1)
            target.fill(0)
            for b in range(self.batch_size):
                if sample_id >= len(self.test_labels_tools):  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_tools[0]
                    x[b][1] = self.test_item_input_tools[0]
                    if self.test_labels_tools[0] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                else:
                    x[b][0] = self.test_user_input_tools[sample_id]
                    x[b][1] = self.test_item_input_tools[sample_id]
                    if self.test_labels_tools[sample_id] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                sample_id += 1

            keys = [self.input_tools, self.target_tools]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_tools_only, self.pred_tools_only],
                                        feed_dict={
                                            k: v for k, v in zip(keys, values)
                                        })
            cost += np.sum(loss)
            test_preds.extend(pred)
            if current_batch == 0 and self.isDebug:
                print()
                print(target[0:3, :])
                print(pred[0:3, :])
            if self.isOneBatch:
                break
        # if self.show: bar.finish()

        # evaluation
        user_item_preds = defaultdict(lambda: defaultdict(float))
        user_pred_gtItem = defaultdict(float)
        for sample_id in range(len(self.test_labels_tools)):
            user = self.test_user_input_tools[sample_id]
            item = self.test_item_input_tools[sample_id]
            label = self.test_labels_tools[sample_id]
            pred = test_preds[sample_id]  # [pos_prob, neg_prob]
            user_item_preds[user][item] = pred[0]
            if item == self.user_gt_item_tools[user]:
                user_pred_gtItem[user] = pred[0]
        HR, NDCG, MRR, AUC = 0, 0, 0, 0
        for user, item_preds in user_item_preds.items():
            # compute AUC
            gt_pred = user_pred_gtItem[user]
            hit = 0
            for i, p in item_preds.items():
                if i != self.user_gt_item_tools[user] and p < gt_pred:
                    hit += 1
            AUC += hit / 99.0
            # compute HR, NDCG, MRR
            item_preds = sorted(item_preds.items(), key=lambda x: -x[1])
            item_preds_topK = item_preds[:self.topK]
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_tools[user]:
                    HR += 1
                    break
            for position in range(len(item_preds_topK)):
                item, pred = item_preds_topK[position]
                if item == self.user_gt_item_tools[user]:
                    NDCG += math.log(2) / math.log(position + 2)
                    # MRR += 1 / (position + 1)
                    break
            rank = 1
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_tools[user]:
                    break
                rank += 1
            MRR += 1 / rank
            if self.isDebug and user == 1:
                print('gt_pred = {:.6f}, topK_pred={:.6f}'.format(gt_pred, item_preds[self.topK][1]))
        self.HR_tools = HR / len(user_item_preds)
        self.NDCG_tools = NDCG / len(user_item_preds)
        self.MRR_tools = MRR / len(user_item_preds)
        self.AUC_tools = AUC / len(user_item_preds)
        return cost/num_test_batches/self.batch_size
