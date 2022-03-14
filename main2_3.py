import pprint
import tensorflow as tf
import time
from collections import defaultdict
from model2_3 import MTL
import math
import numpy as np


pp = pprint.PrettyPrinter()


def main():

    paras_setting = {
        'edim_u': 32,
        'edim_v': 32,
        'edim_w': 1024,
        'layers_A': [1056, 256, 64, 8],  # layers[0] must equal to edim_u + edim_v + edim_w
        'layers_R': [64, 32, 16, 8],    # layers[0] must equal to edim_u + edim_v
        # 'layers_Out': [264, 128, 32, 8],
        'batch_size': 128,  # "batch size to use during training [128,256,512,]"
        'nepoch': 50,  # "number of epoch to use during training [80]"
        'init_lr': 0.0001,  # "initial learning rate [0.01]"
        'init_std': 0.01,  # "weight initialization std [0.05]"
        'max_grad_norm': 10,  # "clip gradients to this norm [50]"
        'negRatio': 1,  # "negative sampling ratio [5]"
        'cross_layers': 3,  # cross between 1st & 2nd, and 2nd & 3rd layers
        # 'merge_ui': 0,  # "merge embeddings of user and item: 0-add, 1-mult [1], 2-concat"
        'activation': 'relu',  # "0:relu, 1:tanh, 2:softmax"
        'learner': 'adam',  # {adam, rmsprop, adagrad, sgd}
        # 论文中提到用cross_entropy loss
        'objective': 'cross',  # 0:cross, 1: hinge, 2:log
        # '+': [0.5, 0.5],  # weight of carry/copy gate
        'topK': 5,
        'data_dir': './data/Amazon1/clothing&home/',  # "data directory [../data]"
        'data_name_cloth': 'clothing',  # "user-info", "data state [user-info]"
        'data_name_tools': 'home',  # "user-info", "data state [user-info]"
        'weights_cloth_tools': [1, 1],  # weights of each task [0.8,0.2], [0.5,0.5], [1,1]
        'checkpoint_dir': 'checkpoints',  # "checkpoints", "checkpoint directory [checkpoints]"
        'show': True,  # "print progress [True]"
        # 'isDebug': True,  # "isDebug mode [True]"
        'isDebug': False,  # "isDebug mode [True]"
        'isOneBatch': False,  # "isOneBatch mode for quickly run through [True]"
    }
    # setenv CUDA_VISIBLE_DEVICES 1
    isRandomSearch = False

    if not isRandomSearch:
        start_time = time.time()
        pp.pprint(paras_setting)

        # train_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.train'
        # valid_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.valid'
        # valid_neg_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.neg.valid'

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            model = MTL(paras_setting, sess)
            model.build_model()
            model.run()
            metrics = {
                'bestHR_cloth': model.bestHR_cloth,
                'bestNDCG_cloth': model.bestNDCG_cloth,
                'bestMRR_cloth': model.bestMRR_cloth,
                'bestAUC_cloth': model.bestAUC_cloth,
                'bestHR_epoch_cloth': model.bestHR_epoch_cloth,
                'bestNDCG_epoch_cloth': model.bestNDCG_epoch_cloth,
                'bestMRR_epoch_cloth': model.bestMRR_epoch_cloth,
                'bestAUC_epoch_cloth': model.bestAUC_epoch_cloth,
                'bestHR_tools': model.bestHR_tools,
                'bestNDCG_tools': model.bestNDCG_tools,
                'bestMRR_tools': model.bestMRR_tools,
                'bestAUC_tools': model.bestAUC_tools,
                'bestHR_epoch_tools': model.bestHR_epoch_tools,
                'bestNDCG_epoch_tools': model.bestNDCG_epoch_tools,
                'bestMRR_epoch_tools': model.bestMRR_epoch_tools,
                'bestAUC_epoch_tools': model.bestAUC_epoch_tools,
            }
            pp.pprint(metrics)
            print(model.para_str)
            pp.pprint(paras_setting)
        print('total time {:.2f}m'.format((time.time() - start_time)/60))
    else:
        para_ranges_map = {
            'edim_u': [5, 10, 20, 32, 50, 64],
            'edim_v': [5, 10, 20, 32, 50, 64],
            'mem_size': [5, 10, 20, 32, 50, 64, 100],
            'nhop': [1, 2, 3, 4, 5, 6],
            'init_lr': [0.001, 0.005, 0.01, 0.05],
            'negRatio': [1, 2, 3, 4, 5],
            'batch_size': [128, 256, 512],
            'activation': [0, 1, 2],
            'learner': [0, 1, 2],
            'init_std': [0.01, 0.05, 0.1]
        }
        total_random_searches = 100
        g_idx_paras = defaultdict(lambda: defaultdict(object))
        for idx_rand_search in range(total_random_searches):
            for key, value in para_ranges_map.items():
                rint = np.random.randint(len(value))
                paras_setting[key] = value[rint]
                paras_setting['lindim'] = math.floor(0.5 * (paras_setting['edim_u'] + paras_setting['edim_v']))
            g_idx_paras[idx_rand_search] = paras_setting

        start_time = time.time()
        g_bestHR10 = -1
        g_bestHR10_paras = defaultdict(object)
        g_bestNDCG = -1
        g_bestNDCG_paras = defaultdict(object)
        g_MetricsParas = []
        for idx_rand_search, paras in g_idx_paras.items():
            paras_setting = paras
            pp.pprint(paras_setting)

            train_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.train'
            valid_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.valid'
            valid_neg_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.neg.valid'

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                # model = MemN2N(paras_setting, sess, train_file, valid_file, valid_neg_file)
                model = MTL(paras_setting, sess, train_file, valid_file, valid_neg_file)
                model.build_model()
                model.run()
                metrics = {
                    #'bestMRR': model.bestMRR,
                    'bestHR10': model.bestHR10,
                    'bestNDCG': model.bestNDCG,
                    #'bestMRR_epoch': model.bestMRR_epoch,
                    'bestHR10_epoch': model.bestHR10_epoch,
                    'bestNDCG_epoch': model.bestNDCG_epoch,
                }
                pp.pprint(metrics)
                print(model.para_str)
                pp.pprint(paras_setting)
            print('total time {:.2f}m'.format((time.time() - start_time)/60))

            if model.bestHR10 > g_bestHR10:
                g_bestHR10 = model.bestHR10
                g_bestHR10_paras = paras_setting
                print('current best HR = {}'.format(g_bestHR10))
            if model.bestNDCG > g_bestNDCG:
                g_bestNDCG = model.bestNDCG
                g_bestNDCG_paras = paras_setting
                print('current best NDCG = {}'.format(g_bestNDCG))
            g_MetricsParas.append([metrics, paras_setting, {'sport': 2}])
        print('best metric (HR, NDCG) and corresponding meta-parameters')
        print(g_bestHR10)
        print(g_bestHR10_paras)
        print(g_bestNDCG)
        print(g_bestNDCG_paras)
        print('total time {:.2f}m'.format((time.time() - start_time)/60))

        with open('g_MetricsParas.txt', 'w', encoding='utf-8') as ofile:
            ofile.write('\n'.join([str(e) for e in g_MetricsParas]))


if __name__ == '__main__':
    main()
