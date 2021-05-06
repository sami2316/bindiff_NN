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

from torch.utils.data import Dataset

# Dataset for training/test --> torch based training
class AsmDataset(Dataset):
  def __init__(self, mnemonic_as, op1as, op2as, mnemonic_bs, op1bs, op2bs, labels, transform=None):
    # left function data
    self.mnemonic_as = mnemonic_as
    self.op1_as = op1as
    self.op2_as = op2as
    # right function data
    self.mnemonic_bs = mnemonic_bs
    self.op1_bs = op1bs
    self.op2_bs = op2bs
    # Label of function
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.mnemonic_as)

  def __getitem__(self, index):
    item = {
      'mnemonic_a': self.mnemonic_as[index],
      'op1_a': self.op1_as[index],
      'op2_a': self.op2_as[index],
      'mnemonic_b': self.mnemonic_bs[index],
      'op1_b': self.op1_bs[index],
      'op2_b': self.op2_bs[index],
      'label': self.labels[index]
    }
    if self.transform:
      item = self.transform(item)

    return item

# Data corresponding to a single function
class AsmData(object):
  def __init__(self):
    self.label = None
    self.instructions_a = []
    self.instructions_b = []


# Vocabulary of mnemonic, operands, and labels
class Vocab(object):
  def __init__(self, skip=False):
    self.word2index = {}
    self.index2word = {}
    self.freq = {}
    if not skip:
      # 0 index is reserved for pad
      self.word2index['<PAD/>'] = 0
      self.index2word[0] = '<PAD/>'
      self.freq[0] = 1

      self.word2index['EMPTY'] = 1
      self.index2word[1] = 'EMPTY'
      self.freq[1] = 1

      self.word2index['ret'] = 2
      self.index2word[2] = 'ret'
      self.freq[2] = 1

      self.word2index['je'] = 3
      self.index2word[3] = 'je'
      self.freq[3] = 1

  def append(self, word):
    if word not in self.word2index:
      index = len(self.word2index)
      if self.freq.get(index) is None:
        self.freq[index] = 0

      self.word2index[word] = index
      self.index2word[index] = word
      self.freq[index] += 1
    else:
      index = self.word2index[word]
      self.freq[index] += 1

  def get_freq_list(self):
    freq = self.freq
    freq_list = [0] * self.len()
    # copy the content into new list
    for i in range(self.len()):
      freq_list[i] = freq[i]

    return freq_list

  def len(self):
    return len(self.word2index)

# Read preprocessed Vocabulary
class PVocab(object):
  def __init__(self, skip=False):
    self.word2index = {}
    self.index2word = {}
    self.freq = {}

  def append(self, word, index=None):
    if word not in self.word2index:
      if index is None:
        index = len(self.word2index)
      if self.freq.get(index) is None:
        self.freq[index] = 0

      self.word2index[word] = index
      self.index2word[index] = word
      self.freq[index] += 1

  def get_freq_list(self):
    freq = self.freq
    freq_list = [0] * self.len()
    # copy the content into new list
    for i in range(self.len()):
      freq_list[i] = freq[i]

    return freq_list

  def len(self):
    return len(self.word2index)
