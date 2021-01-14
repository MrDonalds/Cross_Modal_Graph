from pyHGT.data import Graph, renamed_load
import numpy as np
import dill
import scipy.spatial
from scipy.io import loadmat, savemat
from tqdm import tqdm
import os


def ind2vec(ind, N=None):  # 将标签变成标签数组
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def get_loader(path):
    #     img_train = loadmat(path+"train_img_new.mat")['train_img']  # for Pascal new
    #     img_test = loadmat(path + "test_img_new.mat")['test_img']
    #     text_train = loadmat(path+"train_txt_new.mat")['train_img']
    #     text_test = loadmat(path + "test_txt_new.mat")['test_img']

    img_train = loadmat(path + "train_img_new.mat")['train_img']
    img_test = loadmat(path + "test_img_new.mat")['test_img']
    text_train = loadmat(path + "train_txt_new.mat")['train_txt']
    text_test = loadmat(path + "test_txt_new.mat")['test_txt']

    label_train = loadmat(path + "train_img_lab.mat")['train_img_lab']
    label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

    label_train = ind2vec(label_train).astype(int)
    label_test = ind2vec(label_test).astype(int)

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}

    return imgs, texts, labels


# 计算两两之间的cosine相似度
def cosine(image, text):
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    return dist


def write_edge(path, connection, num_train, num_val, num_test, target_type, source_type, link_type):
    with open(path, 'a', encoding='utf-8') as f:
        for i, t in enumerate(connection):
            if i < num_train:  # train 0  or test 2
                t_v_t = '0'
            elif i < num_train+num_val:
                t_v_t = '1'
            else:
                t_v_t = '2'
            # link_type = 'i2i'
            # target_type, source_type = 'img', 'img'
            t_id = str(i)
            for j in t:
                s_id = str(j)
                weight = str(1)
                f.write(target_type + '\t' + source_type + '\t' + link_type + '\t')
                f.write(t_id + '\t' + s_id + '\t' + weight + '\t')
                f.write(t_v_t + '\n')


k = 5
base_path = '../data/'
dataset = 'nus_wide'

edgelist_path = base_path +dataset+'/edgelist'
path = base_path + dataset + '/mat/'
graph_output = base_path + dataset + '/graph.pkl'

img, text, lab = get_loader(path)

iall = np.concatenate((img['train'],  img['test']), axis=0)
tall = np.concatenate((text['train'], text['test']), axis=0)
lall = np.concatenate((lab['train'],  lab['test']), axis=0)


num_iall  = len(iall)          # 数据集大小
len_train = len(img['train'])  # 训练集大小
len_test  = len(img['test'])   # 测试集大小
# nus_wide
num_train = len_train-len_test
num_val = len_test
num_test = len_test
print('train:', num_train, 'num_val:', num_val, 'num_test:', num_test)

'''
（1）
 得到模态内KNN的边连接关系，写到 edge_list 路径，格式如下：
 target_type, source_type, link_type, t_i, s_i, weigh, t_v_t
'''

print('calculating similarity...   ')
isimilarity = cosine(iall, iall)
tsimilarity = cosine(tall, tall)
print('done!\n')

iconnection = []  # 得到 imgaes 的 KNN 邻居
for i, A in enumerate(isimilarity):
    # if i>=num_train:  # 如果是测试样本，那么测试样本只会连接到训练样本。换句话说：测试样本不连接测试样本
    A = A[:num_train]   # 上面的说法不对，应该是所有样本的KNN都不会连接到测试样本。这样后面对所有样本进行相似性扩展就合理了。
    idx = np.argpartition(A, k)
    iconnection.append(idx[:k])


tconnection = []
for i, A in enumerate(tsimilarity):
    # if i>=num_train:
    A = A[:num_train]
    idx = np.argpartition(A, k)
    tconnection.append(idx[:k])


if os.path.exists(edgelist_path):
    os.remove(edgelist_path)
# with open(edgelist_path, 'w', encoding='utf-8') as f:
#     f.write('target' + '\t' + 'source' + '\t' + 'link' + '\t')  # 节点和边的类型
#     f.write('t_id' + '\t' + 's_id' + '\t' + 'weight' + '\t')  # id 和 权重
#     f.write('t/v/t' + '\n')  # 训练/验证/测试
# 追加写
write_edge(edgelist_path, iconnection, num_train, num_val, num_test, 'img', 'img', 'i2i')
write_edge(edgelist_path, tconnection, num_train, num_val, num_test, 'txt', 'txt', 't2t')


graph = Graph()
graph.node_feature['img'] = iall
graph.node_feature['txt'] = tall
node_type = {'img', 'txt'}

with open(edgelist_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in tqdm(f, total=sum(1 for line in open(edgelist_path))):
        target_type, source_type, link_type, t_id, s_id, weigh, t_v_t = line.strip().split('\t')
        graph.edge_list[target_type][source_type][link_type][(t_id)][(s_id)] = t_v_t
        # if t_v_t != '0': continue  # 只对训练集进行相似性传播
        for s_type in node_type:
            if s_type == target_type: continue
            link_type = target_type[0]+'2'+s_type[0]
            graph.edge_list[target_type][s_type][link_type][(t_id)][(s_id)] = t_v_t

graph.labels = lall

dill.dump(graph, open(graph_output, 'wb'))


# graph = renamed_load(open(os.path.join(graph_output), 'rb'))
