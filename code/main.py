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

import argparse
import numpy as np
import torch
import os
import pickle
import numpy as np
import torch.nn.functional as F
from distutils.util import strtobool
from torch.utils.data import DataLoader
from deeper_attention_SM.model import *
from deeper_attention_SM.dataset_builder import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
eloss = torch.nn.CosineEmbeddingLoss(margin=0.1, reduction='mean')


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1, help='random_seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--mnemonic_embed_size', type=int, default=100, help="mnemonic_embed_size")
parser.add_argument('--operand_embed_size', type=int, default=100, help="operand_embed_size")
parser.add_argument('--encode_size', type=int, default=300, help="encode_size")
parser.add_argument('--max_inst_length', type=int, default=500, help="max_inst_length")
parser.add_argument('--max_inst_token', type=int, default=4, help="max_inst_token")
parser.add_argument("--max_epoch", type=int, default=80, help="max_epoch")
parser.add_argument('--lr', type=float, default=0.0025, help="lr")
parser.add_argument('--beta_min', type=float, default=0.0, help="beta_min")
parser.add_argument('--beta_max', type=float, default=0.5, help="beta_max")
parser.add_argument('--weight_decay', type=float, default=1.5581392558680177e-06, help="weight_decay")

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
if args.find_hyperparams:
  import optuna
use_cuda = False #torch.cuda.is_available()
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

# Train the model
def train():
  torch.manual_seed(args.random_seed)

  reader = DatasetReader('assembly_data.txt', args.max_inst_token)
  reader.load_data()
  options = Option(reader)
  with open('options.pkl', 'wb') as fp:
    pickle.dump(options, fp, 2)
    logger.info('Saving Model options')

  builder = DatasetBuilder(reader, options)

  label_freq = torch.tensor(reader.label_vocab.get_freq_list(), dtype=torch.float32).to(device)
  #criterion = nn.NLLLoss(weight=1 / options.batch_size/label_freq).to(device)
  criterion = nn.BCEWithLogitsLoss().to(device)

  model = Asm2Vec(options).to(device)
  print(model)
  learning_rate = args.lr
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(args.beta_min, args.beta_max), weight_decay=args.weight_decay)

  _train(model, optimizer, criterion, options, reader, builder, None)


def _train(model, optimizer, criterion, options, reader, builder, trial):
  f1 = 0.0
  best_f1 = None
  last_loss = None
  last_accuracy = None
  bad_count = 0
  output = open('output.csv', 'w')
  output.write('train_loss,test_loss, accuracy, precision, recall, f1\n')

  try:
    for epoch in range(args.max_epoch):
      train_loss = 0.0
      builder.refresh_train_dataset()
      train_data_loader = DataLoader(builder.train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=args.num_workers)

      model.train()
      for i_batch, sample_batched in enumerate(train_data_loader):
        mnemonics_a = sample_batched['mnemonic_a'].to(options.device)
        op1s_a = sample_batched['op1_a'].to(options.device)
        op2s_a = sample_batched['op2_a'].to(options.device)
        mnemonics_b = sample_batched['mnemonic_b'].to(options.device)
        op1s_b = sample_batched['op1_b'].to(options.device)
        op2s_b = sample_batched['op2_b'].to(options.device)
        label = sample_batched['label'].to(options.device)
        #label = torch.tensor([1 if x == 0 else -1 for x in label.tolist()]).to(options.device)

        optimizer.zero_grad()
        preds, _ = model.siamese_forward(mnemonics_a, op1s_a, op2s_a, mnemonics_b, op1s_b, op2s_b)

        loss = calculate_loss(preds, _, label, options, criterion)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

      builder.refresh_test_dataset()
      test_data_loader = DataLoader(builder.test_dataset, batch_size=options.batch_size, shuffle=True, num_workers=args.num_workers)
      test_loss, accuracy, precision, recall, f1 = test(model, test_data_loader, criterion, options, reader.label_vocab)

      output.write(str(train_loss)+','+str(test_loss) +','+ str(accuracy) +','+ str(precision) +','+ str(recall) +','+ str(f1) + '\n')
      logger.info("epoch {0}".format(epoch))
      logger.info('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
      logger.info('{{"metric": "test_loss", "value": {0}}}'.format(test_loss))
      logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
      logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
      logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
      logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))

      if trial is not None:
        intermediate_value = 1.0 - f1
        trial.report(intermediate_value, epoch)
        if trial.should_prune(epoch):
          raise optuna.exceptions.TrialPruned()
  finally:
    output.close()
    logger.info("Training Done")
    logger.info('Saving the model...')
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "code2vec.model"))

  return 1.0 - f1

# Contrastive loss function
def calculate_loss(preds, distance, label, options, criterion):
  '''
  # Binary entropy loss
  preds = distance
  tmp1 = label * torch.log(preds)
  tmp2 = (1-label)* torch.log(1-preds)
  return -(tmp1 + tmp2).sum()/options.batch_size/2
  '''

  # NN loss
  #preds = F.sigmoid(distance)
  #print(preds, label)
  loss = criterion(preds, label)
  return loss

  '''
  # contractive loss function
  tmp = label * distance.pow(2)
  tmp2 = (1-label) * (torch.max((1-distance), 0)[0].pow(2))
  return torch.sum(tmp + tmp2)/options.batch_size/2
  '''

def test(model, data_loader, criterion, options, label_vocab):
  """test the model"""
  model.eval()
  with torch.no_grad():
    test_loss = 0.0
    expected_labels = []
    actual_labels = []

    for i_batch, sample_batched in enumerate(data_loader):
      mnemonic_a = sample_batched['mnemonic_a'].to(options.device)
      op1_a = sample_batched['op1_a'].to(options.device)
      op2_a = sample_batched['op2_a'].to(options.device)
      mnemonic_b = sample_batched['mnemonic_b'].to(options.device)
      op1_b = sample_batched['op1_b'].to(options.device)
      op2_b = sample_batched['op2_b'].to(options.device)
      label = sample_batched['label'].to(options.device)
      _, given_label = torch.max(label, dim=1)
      #label = torch.tensor([1 if x == 0 else -1 for x in label.tolist()]).to(options.device)
      expected_labels.extend(given_label)

      preds, _ = model.siamese_forward(mnemonic_a, op1_a, op2_a, mnemonic_b, op1_b, op2_b)
      loss = calculate_loss(preds, _, label, options, criterion)
      test_loss += loss.item()
      #preds_label = torch.eq(preds, label).int()
      _, preds_label = torch.max(preds, dim=1)
      #print(preds_label)
      #preds = torch.tensor([1 if x > 0.5 else 0 for x in preds_values.tolist()])
      #print(preds)
      actual_labels.extend(preds_label)
      #actual_labels.extend(preds)

    expected_labels = np.array(expected_labels)
    actual_labels = np.array(actual_labels)
    accuracy, precision, recall, f1 = None, None, None, None
    accuracy, precision, recall, f1 = exact_match(expected_labels, actual_labels)

    return test_loss, accuracy, precision, recall, f1

def exact_match(expected_labels, actual_labels):
  expected_labels = np.array(expected_labels, dtype=np.uint64)
  actual_labels = np.array(actual_labels, dtype=np.uint64)
  precision, recall, f1, _ = precision_recall_fscore_support(expected_labels, actual_labels, average='weighted', labels=np.unique(actual_labels))
  accuracy = accuracy_score(expected_labels, actual_labels)
  return accuracy, precision, recall, f1

#
# for optuna
#
def find_optimal_hyperparams():
  """find optimal hyperparameters"""
  torch.manual_seed(args.random_seed)

  reader = DatasetReader('assembly_data.txt',args.max_inst_token)
  reader.load_data()
  option = Option(reader)

  builder = DatasetBuilder(reader, option)

  label_freq = torch.tensor(reader.label_vocab.get_freq_list(), dtype=torch.float32).to(device)
  criterion = nn.BCEWithLogitsLoss().to(device)

  def objective(trial):
    option.lr = trial.suggest_loguniform('lr', 0.0001, 10)
    #option.dropout_prob = trial.suggest_loguniform('dropout_prob', 0.1, 0.9)
    #option.max_inst_length = int(trial.suggest_loguniform('max_inst_length', 300, 1000))
    #option.encode_size = int(trial.suggest_loguniform('encode_size', 100, 300))
    #option.batch_size = int(trial.suggest_loguniform('batch_size', 32, 256))

    model = Asm2Vec(option).to(device)
    optimizer = get_optimizer(trial, model)

    return _train(model, optimizer, criterion, option, reader, builder, trial)

  study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
  study.optimize(objective, n_trials=args.num_trials)

  best_params = study.best_params
  best_value = study.best_value
  logger.info("optimal hyperparams: {0}".format(best_params))
  logger.info('best value: {0}'.format(best_value))


def get_optimizer(trial, model):
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  return adam(model, trial, weight_decay)


def adam(model, trial, weight_decay):
  lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
  return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def momentum(model, trial, weight_decay):
  lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
  return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

model_path = os.path.join(os.getcwd(), 'code2vec.model')
if __name__ == '__main__':
  if args.find_hyperparams:
    find_optimal_hyperparams()
  else:
    train()
