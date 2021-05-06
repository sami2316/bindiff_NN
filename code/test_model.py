"""
BinDiff_NN: Learning Distributed Representation of Assembly for Robust
            Binary Diffing against Semantic Differences
Copyright (c) 2020-2021, Sami Ullah

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import logging
import numpy as np
from deeper_attention_SM.model import *
from deeper_attention_SM.dataset_builder import *
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
# Set the timer
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1, help='random_seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--mnemonic_embed_size', type=int, default=100, help="mnemonic_embed_size")
parser.add_argument('--operand_embed_size', type=int, default=100, help="operand_embed_size")
parser.add_argument('--encode_size', type=int, default=300, help="encode_size")
parser.add_argument('--max_inst_length', type=int, default=500, help="max_inst_length")
parser.add_argument('--max_inst_token', type=int, default=4, help="max_inst_token")
parser.add_argument("--max_epoch", type=int, default=40, help="max_epoch")
parser.add_argument('--lr', type=float, default=0.0001, help="lr")
parser.add_argument('--beta_min', type=float, default=0.0, help="beta_min")
parser.add_argument('--beta_max', type=float, default=0.5, help="beta_max")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay")

parser.add_argument('--dropout_prob', type=float, default=0.0, help="dropout_prob")
parser.add_argument("--no_cuda", action="store_true", default=False, help="no_cuda")
parser.add_argument("--gpu", type=str, default="cuda:0", help="gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

parser.add_argument("--find_hyperparams", action="store_true", default=False, help="find optimal hyperparameters")
parser.add_argument("--num_trials", type=int, default=100, help="num_trials")

parser.add_argument("--angular_margin_loss", action="store_true", default=False, help="use angular margin loss")
parser.add_argument("--angular_margin", type=float, default=0.5, help="angular margin")
parser.add_argument("--inverse_temp", type=float, default=30.0, help="inverse temperature")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
logger.info("device: {0}".format(device))

eps = 1e-9
margin = 1
# configuration of model
class Option(object):
  def __init__(self, reader):
    self.max_inst_length = args.max_inst_length
    self.max_inst_token = args.max_inst_token

    self.mnemonic_count = reader.mnemonic_vocab.len()
    self.op_count = reader.op_vocab.len()
    self.label_count = reader.label_vocab.len()
    logger.info('mnemonic size: {0}'.format(self.mnemonic_count))

    self.mnemonic_embed_size = args.mnemonic_embed_size
    self.operand_embed_size = args.mnemonic_embed_size
    self.encode_size = args.encode_size

    self.dropout_prob = args.dropout_prob
    self.batch_size = args.batch_size

    self.angular_margin_loss = args.angular_margin_loss
    self.angular_margin = args.angular_margin
    self.inverse_temp = args.inverse_temp

    self.device = device

def exact_match(expected_labels, actual_labels):
  expected_labels = np.array(expected_labels, dtype=np.uint64)
  actual_labels = np.array(actual_labels, dtype=np.uint64)
  precision, recall, f1, _ = precision_recall_fscore_support(expected_labels, actual_labels, average='weighted')
  accuracy = accuracy_score(expected_labels, actual_labels)
  return accuracy, precision, recall, f1

reader = DatasetReader('test_data.txt',args.max_inst_token)
reader.load_data()
options = Option(reader)
builder = DatasetBuilder(reader, options, 0)

model = Asm2Vec(options).to(device)
state_dict = torch.load('code2vec.model')
model.load_state_dict(state_dict)
model.eval()

builder.refresh_train_dataset()
train_data_loader = DataLoader(builder.train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=args.num_workers)

expected_labels = []
actual_labels = []

with torch.no_grad():
  for i_batch, sample_batched in enumerate(train_data_loader):
    mnemonics_a = sample_batched['mnemonic_a'].to(options.device)
    op1s_a = sample_batched['op1_a'].to(options.device)
    op2s_a = sample_batched['op2_a'].to(options.device)
    mnemonics_b = sample_batched['mnemonic_b'].to(options.device)
    op1s_b = sample_batched['op1_b'].to(options.device)
    op2s_b = sample_batched['op2_b'].to(options.device)
    label = sample_batched['label'].to(options.device)
    _, given_label = torch.max(label, dim=1)
    expected_labels.extend(given_label)

    preds, _ = model.siamese_forward(mnemonics_a, op1s_a, op2s_a, mnemonics_b, op1s_b, op2s_b)
    _, preds_label = torch.max(preds, dim=1)
    actual_labels.extend(preds_label)

expected_labels = np.array(expected_labels)
actual_labels = np.array(actual_labels)
accuracy, precision, recall, f1 = None, None, None, None
accuracy, precision, recall, f1 = exact_match(expected_labels, actual_labels)
cm = confusion_matrix(np.array(actual_labels, dtype=np.uint64), np.array(expected_labels, dtype=np.uint64))
FNR = cm[0][1] / (cm[0][1]+cm[1][1]) 
FPR = cm[1][0] / (cm[0][0]+cm[1][0])
print(FPR, FNR)
print(cm)

logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))
logger.info("--- %s seconds ---" % (time.time() - start_time))
