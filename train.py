from __future__ import division
from __future__ import print_function

import time
import os
from sklearn import metrics
from metrics import softmax_cross_entropy,accuracy
from argparse import ArgumentParser
from utils import *
from models_pytorch import GNN
import numpy as np
import torch
import torch.nn as nn
# Set random seed
seed = 123
set_seed(seed)
# parameters
parser = ArgumentParser()
parser.add_argument('--logger_name', help='Logger string.', default='TextING-pytorch')
parser.add_argument('--checkpoint_dir', help='Checkpoint string', default='checkpoints')
parser.add_argument('--dataset', help='Dataset string.', default='R8')
parser.add_argument('--model', help='Model string.', default='gnn')
parser.add_argument('--learning_rate', help='Initial learning rate.', default=0.005)
parser.add_argument('--epochs', help='Number of epochs to train.', default=200)
parser.add_argument('--batch_size', help='Size of batches per epoch.', default=4096) # 4096
parser.add_argument('--input_dim', help='Dimension of input.', default=300)
parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', default=96)
parser.add_argument('--steps', help='Number of graph layers.', default=2)
parser.add_argument('--dropout', help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', default=0)
parser.add_argument('--early_stopping', help='Tolerance for early stopping (# of epochs).', default=-1)
parser.add_argument('--max_degree', help='Maximum Chebyshev polynomial degree.', default=3) # not used
args = parser.parse_args()
# Gpu
device = torch.device('cpu')
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
args.device = device
# logger
os.makedirs(args.checkpoint_dir, exist_ok=True)
logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))
# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(args.dataset)
# Some preprocessing
logger.info('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
logger.info('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
logger.info('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)
array_lst = [train_adj,train_mask,train_feature,train_y,val_adj,val_mask,val_feature,val_y,test_adj,test_mask,test_feature,test_y]
device_lst = [args.device]*len(array_lst)
train_adj,train_mask,train_feature,train_y,val_adj,val_mask,val_feature,val_y,test_adj,test_mask,test_feature,test_y = map(to_tensor_func, array_lst, device_lst)
train_y, val_y, val_y = train_y.long(), val_y.long(), val_y.long()
if args.model == 'gnn':
    model_func = GNN
elif args.model == 'gcn_cheby': # not used
    num_supports = 1 + args.max_degree
    model_func = GNN
elif args.model == 'dense': # not used
    num_supports = 1
    model_func = MLP  # not used
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))
import torch
model = model_func(args=args,
                   input_dim=args.input_dim,
                   output_dim=train_y.shape[1],
                   hidden_dim=args.hidden_dim,
                   gru_step = args.steps,
                   dropout_p=args.dropout)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss()
# Define model evaluation function
def evaluate(model, features, support, mask, labels):
    # test_cost, test_acc, test_duration, embeddings, pred, labels
    t_test = time.time()
    outputs,embeddings = model(features, support, mask)
    cost = softmax_cross_entropy(loss_fn, outputs, labels)
    acc = accuracy(outputs, labels)
    duration = (time.time() - t_test)
    pred = torch.argmax(outputs,1)
    return cost, acc, duration, embeddings,pred, labels


cost_val = []
best_val = 0
best_epoch = 0
best_acc = 0
best_cost = 0
test_doc_embeddings = None
preds = None
labels = None

logger.info('train start...')
# Train model
model.train()
for epoch in range(args.epochs):
    t = time.time()
    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)
    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), args.batch_size):
        end = start + args.batch_size
        idx = indices[start:end]
        outputs,_= model(train_feature[idx], train_adj[idx], train_mask[idx]) # embeddings not used
        loss = softmax_cross_entropy(loss_fn, outputs, train_y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = accuracy(outputs, train_y[idx])
        train_loss += loss.item()*len(idx)
        train_acc += acc.item()*len(idx)
    train_loss /= len(train_y)
    train_acc /= len(train_y)
    # Validation
    val_cost, val_acc, val_duration, _, _, _ = evaluate(model, val_feature, val_adj, val_mask, val_y)
    cost_val.append(val_cost.item())
    # Test
    test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(model, test_feature, test_adj, test_mask, test_y)

    if val_acc >= best_val:
        best_val = val_acc
        best_epoch = epoch
        best_acc = test_acc
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred

    # Print results
    logger.info("Epoch: {:04d} train_loss= {:.5f} train_acc= {:.5f} val_loss= {:.5f} val_acc= {:.5f} test_acc= {:.5f} time= {:.5f}".format(
        epoch+1,train_loss,train_acc,val_cost,val_acc,test_acc,time.time() - t))

    if args.early_stopping > 0 and epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        logger.info("Early stopping...")
        break

logger.info("Optimization Finished!")
# Best results
logger.info(f'Best epoch: {best_epoch}')
logger.info("Test set results: cost= {:.5f} accuracy= {:.5f}".format(best_cost,best_acc))
"""
logger.info("Test set results:", "cost=", "{:.5f}".format(best_cost),
      "accuracy=", "{:.5f}".format(best_acc))
"""


logger.info("Test Precision, Recall and F1-Score...")
logger.info(metrics.classification_report(labels, preds, digits=4))
logger.info("Macro average Test Precision, Recall and F1-Score...")
logger.info(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
logger.info("Micro average Test Precision, Recall and F1-Score...")
logger.info(metrics.precision_recall_fscore_support(labels, preds, average='micro'))

'''
# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)
'''
