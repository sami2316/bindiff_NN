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


import torch
import random
import logging
import numpy as np
import gc
from .dataset_reader import *

logger = logging.getLogger()
#logger.setLevel(logging.INFO)

# transform dataset for training and test
class DatasetBuilder(object):
  def __init__(self, reader, option, split_ratio=0.2):
    self.reader = reader
    self.option = option

    test_count = int(len(reader.items) * split_ratio)
    random.shuffle(reader.items)
    self.train_items = reader.items[test_count:]
    self.test_items = reader.items[0:test_count]
    logger.info('train item size: {0}'.format(len(self.train_items)))
    logger.info('test item size: {0}'.format(len(self.test_items)))
    self.train_dataset = None
    self.test_dataset = None

  # shuffle instructions and picking up training items
  def refresh_train_dataset(self):
    mn_as, op1_as, op2_as, mn_bs, op1_bs, op2_bs, inputs_label = self.build_data(self.reader, self.train_items)
    self.train_dataset = AsmDataset(mn_as, op1_as, op2_as, mn_bs, op1_bs, op2_bs, inputs_label)

  # shuffle instructions and picking up test items
  def refresh_test_dataset(self):
    mn_as, op1_as, op2_as, mn_bs, op1_bs, op2_bs, inputs_label = self.build_data(self.reader, self.test_items)
    self.test_dataset = AsmDataset(mn_as, op1_as, op2_as, mn_bs, op1_bs, op2_bs, inputs_label)

  def build_data(self, reader, items):
    inputs_mnemonic_as = []
    inputs_mnemonic_bs = []
    inputs_op1_as = []
    inputs_op1_bs = []
    inputs_op2_as = []
    inputs_op2_bs = []
    inputs_label = []

    for item in items:
      label = [0., 0.] #
      label_index = reader.label_vocab.word2index[item.label]
      label[label_index] = 1.0 #
      inputs_label.append(label) # append(label_index)
      mnemonic_as = []
      op1_as = []
      op2_as = []
      len_a = []
      mnemonic_bs = []
      op1_bs = []
      op2_bs = []
      len_b = []


      for mnemonic, op1, op2 in item.instructions_a[:self.option.max_inst_length]:
        mnemonic_as.append(mnemonic)
        op1_as.append(self.pad_inputs(op1[:self.option.max_inst_token], self.option.max_inst_token))
        op2_as.append(self.pad_inputs(op2[:self.option.max_inst_token], self.option.max_inst_token))
      for mnemonic, op1, op2 in item.instructions_b[:self.option.max_inst_length]:
        mnemonic_bs.append(mnemonic)
        op1_bs.append(self.pad_inputs(op1[:self.option.max_inst_token], self.option.max_inst_token))
        op2_bs.append(self.pad_inputs(op2[:self.option.max_inst_token], self.option.max_inst_token))

      mnemonic_as = self.pad_inputs(mnemonic_as, self.option.max_inst_length)
      op1_as = self.pad_operands(op1_as, self.option.max_inst_length)
      op2_as = self.pad_operands(op2_as, self.option.max_inst_length)
      mnemonic_bs = self.pad_inputs(mnemonic_bs, self.option.max_inst_length)
      op1_bs = self.pad_operands(op1_bs, self.option.max_inst_length)
      op2_bs = self.pad_operands(op2_bs, self.option.max_inst_length)
      inputs_mnemonic_as.append(mnemonic_as)
      inputs_op1_as.append(op1_as)
      inputs_op2_as.append(op2_as)
      inputs_mnemonic_bs.append(mnemonic_bs)
      inputs_op1_bs.append(op1_bs)
      inputs_op2_bs.append(op2_bs)

    inputs_mnemonic_as = torch.tensor(inputs_mnemonic_as, dtype=torch.long)
    inputs_op1_as = torch.tensor(inputs_op1_as, dtype=torch.long)
    inputs_op2_as = torch.tensor(inputs_op2_as, dtype=torch.long)
    inputs_mnemonic_bs = torch.tensor(inputs_mnemonic_bs, dtype=torch.long)
    inputs_op1_bs = torch.tensor(inputs_op1_bs, dtype=torch.long)
    inputs_op2_bs = torch.tensor(inputs_op2_bs, dtype=torch.long)
    inputs_label = torch.tensor(inputs_label, dtype=torch.float)
    return inputs_mnemonic_as, inputs_op1_as, inputs_op2_as, inputs_mnemonic_bs, inputs_op1_bs, inputs_op2_bs, inputs_label

  def pad_inputs(self, data, length, pad_value=0):
    assert len(data) <= length

    count = length - len(data)
    data.extend([pad_value] * count)
    return data

  def pad_operands(self, data, length, pad_value=0):
    assert len(data) <= length

    count = length - len(data)
    data.extend([[pad_value]*self.option.max_inst_token] * count)
    return data
