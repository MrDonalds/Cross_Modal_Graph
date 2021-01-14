# import sys
# sys.path.append('../pyHGT')
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
from loss import calc_loss, fx_calc_map_label

filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Training GNN on Paper-Field (L2) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='/home/pyHGT/data/nus_wide/',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='/home/pyHGT/model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PF',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=20,
                    help='Number of epoch to run')

parser.add_argument('--n_pool', type=int, default=5,
                    help='Number of process to sample subgraph')

parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')

args = parser.parse_args()

if args.cuda == -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")


def node_classification_sample(seed, graph, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size//len(graph.get_types()), replace=False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    inp = {'img': np.array(target_info), 'txt': np.array(target_info)}

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, ylabel = sample_subgraph(graph, time_range, inp=inp, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
    '''
    # 原图中 valid 和 test 节点都没有模态间连接，并且没有进行相似性传播。
    # 所以不需要再 mask 了。

    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    for t in ylabel.keys():
        ylabel[t] = torch.FloatTensor(ylabel[t])

    node_time = []
    for t in types:
        node_time += list(times[t])
    node_time = torch.LongTensor([int(t) for t in node_time])
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''

    return node_feature, node_type, edge_time, edge_index, edge_type, ylabel, node_time


def node_classification(seed, graph, pairs, time_range):

    target_ids = list(pairs.keys())
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    inp = {'img': np.array(target_info), 'txt': np.array(target_info)}

    feature, times, edge_list, _, ylabel = train_subgraph(graph, time_range, inp=inp)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
        to_torch(feature, times, edge_list, graph)

    for t in ylabel.keys():
        ylabel[t] = torch.FloatTensor(ylabel[t])

    node_time = []
    for t in types:
        # node_feature += list(feature[t])
        node_time += list(times[t])
    node_time = torch.LongTensor([int(t) for t in node_time])
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''


    return node_feature, node_type, edge_time, edge_index, edge_type, ylabel, node_time


train_range = {'0': True}
valid_range = {'1': True}
test_range = {'2': True}


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), graph, train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), graph, valid_pairs, valid_range))
    jobs.append(p)
    return jobs


graph = renamed_load(open(os.path.join(args.data_dir+'graph.pkl'), 'rb'))
types = graph.get_types()
num_label = graph.labels.shape[1]

train_pairs = {}
valid_pairs = {}
test_pairs = {}

'''
    Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
'''
for target_id in graph.edge_list['img']['img']['i2i']:
    for source_id in graph.edge_list['img']['img']['i2i'][target_id]:
        _time = graph.edge_list['img']['img']['i2i'][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _time]
            train_pairs[target_id][0] += [source_id]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _time]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id] = [[], _time]
            test_pairs[target_id][0] += [source_id]

np.random.seed(43)

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
in_dimension = []
for node_type in graph.node_feature:
    in_dimension.append(graph.node_feature[node_type].shape[1])

gnn = GNN(conv_name=args.conv_name, in_dim=in_dimension, \
          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, \
          num_types=len(graph.get_types()), num_relations=len(graph.get_meta_graph()) + 1).to(device)

classifier = Classifier(args.n_hid, num_label).to(device)

model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 1500

# # valid_data = [node_classification_sample(randint(), graph, valid_pairs, valid_range)]

# pool = mp.Pool(args.n_pool)
st = time.time()
# jobs = prepare_data(pool)

train_data = [node_classification(randint(), graph, train_pairs, [])]
valid_data = node_classification(randint(), graph, {**train_pairs, **valid_pairs}, [])
test_data  = node_classification(randint(), graph, {**train_pairs, **test_pairs}, [])

for epoch in np.arange(args.n_epoch) + 1:
    # '''
    #     Prepare Training and Validation Data
    # '''
    # train_data = [job.get() for job in jobs[:-1]]
    # valid_data = jobs[-1].get()
    # pool.close()
    # pool.join()
    # '''
    #     After the data is collected, close the pool and then reopen it.
    # '''
    # pool = mp.Pool(args.n_pool)
    # jobs = prepare_data(pool)

    # train_data = [node_classification_sample(randint(), graph, train_pairs, train_range) for batch_id in np.arange(args.n_batch)]
    # valid_data = node_classification_sample(randint(), graph, valid_pairs, valid_range)

    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))

    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, ylabel, node_time in train_data:
            node_rep = gnn.forward(node_feature, node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier.forward(node_rep)

            y = ylabel['img'].to(device)
            f1 = node_rep[node_type==0]
            f2 = node_rep[node_type==1]
            loss = calc_loss(f1, f2, res[node_type==0], res[node_type==1], y, y, 1, 1)

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss


    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, ylabel, node_time = test_data
        node_rep = gnn.forward(node_feature, node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep)

        node_time = node_time[node_type == 0]

        t_v_t = 2  # valid

        y = ylabel['img'].to(device)
        f1 = node_rep[node_type == 0][node_time == t_v_t]
        f2 = node_rep[node_type == 1][node_time == t_v_t]
        r1 = res[node_type == 0][node_time == t_v_t]
        r2 = res[node_type == 1][node_time == t_v_t]
        loss = calc_loss(f1, f2, r1, r2, y[node_time == t_v_t], y[node_time == t_v_t], 1, 1)

        '''
            Calculate Valid score. Update the best model based on highest score.
        '''
        img = f1.cpu().numpy()
        txt = f2.cpu().numpy()
        ylab = y[node_time == t_v_t].cpu().numpy().argmax(1)
        valid_score = fx_calc_map_label(img, txt, ylab)  # i2t
        valid_score2 = fx_calc_map_label(txt, img, ylab)

        if valid_score > best_val:
            best_val = valid_score
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Score: %.4f, %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_score, valid_score2))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    # del train_data, valid_data


'''
    Evaluate the trained model via test set (time > 2016)
'''

# with torch.no_grad():
#     test_res = []
#     for _ in range(10):
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
#             node_classification_sample(randint(), test_pairs, test_range)
#         paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                 edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
#         res = classifier.forward(paper_rep)
#         for ai, bi in zip(ylabel, res.argsort(descending=True)):
#             test_res += [ai[bi.cpu().numpy()]]
#     test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
#     print('Last Test NDCG: %.4f' % np.average(test_ndcg))
#     test_mrr = mean_reciprocal_rank(test_res)
#     print('Last Test MRR:  %.4f' % np.average(test_mrr))

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    node_feature, node_type, edge_time, edge_index, edge_type, ylabel, node_time = valid_data

    paper_rep = gnn.forward(node_feature, node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
    res = classifier.forward(paper_rep)

    node_time = node_time[node_type==0]

    t_v_t = 1  # test

    y = ylabel['img'].to(device)
    f1 = paper_rep[node_type == 0][node_time==t_v_t]
    f2 = paper_rep[node_type == 1][node_time==t_v_t]
    r1 = res[node_type == 0][node_time==t_v_t]
    r2 = res[node_type == 1][node_time==t_v_t]
    loss = calc_loss(f1, f2, r1, r2, y[node_time==t_v_t], y[node_time==t_v_t], 1, 1)

    '''
        Calculate Valid score. Update the best model based on highest score.
    '''
    img = f1.cpu().numpy()
    txt = f2.cpu().numpy()
    ylab = y[node_time==t_v_t].cpu().numpy().argmax(1)
    valid_score = fx_calc_map_label(img, txt, ylab)  # i2t
    valid_score2 = fx_calc_map_label(txt, img, ylab)

    print('Best Test i2t: %.4f' % valid_score)
    print('Best Test t2i:  %.4f' % valid_score2)
