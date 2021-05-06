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

import os
import re
import logging
from .dataset import *

logger = logging.getLogger()

# extract tokens from an operand
def operand_tokens(operand):
  # remove address brackets
  tokens = re.split(r"[\[\]+\-*]", operand)
  tokens = [w.strip() for w in tokens if w != '']
  return tokens

# Read assembly and prepare Vocabulary
class VocabReader(object):
  def __init__(self, filename):
    self.filepath = os.path.join(os.getcwd(), 'asmdata', filename)
    self.mnemonic_vocab = Vocab()
    self.op_vocab = Vocab()

  def read(self):
    with open(self.filepath, 'r') as fp:
      # read the dataset file line by line
      for line in fp:
        # first filter id and name lines
        if line.startswith('label:') or line.startswith('assembly_b:') or line.startswith('assembly_a:'):
          continue
        elif line is '' or line is '\n':
          continue

        self.preprocess(line)
    #return self.vocab

  def get_vocab(self, type):
    if type == 'mnemonic':
      return self.mnemonic_vocab
    elif type == 'op':
      return self.op_vocab
    else:
      return None

  # Split the lines to get words
  def preprocess(self, line):
    line = line.strip('\n\r\t')
    if line == 'XXXX:':
      return

    delim = line.find(' ')
    if delim == -1:
      self.mnemonic_vocab.append(line.strip('\n\t\r'))
    else:
      self.mnemonic_vocab.append(line[:delim])
      operands = list(map(lambda arg: arg.strip(), line[delim+1:].split(',')))
      for op in operands:
        if op is '':
          continue

        tokens = operand_tokens(op)
        for w in tokens:
          self.op_vocab.append(w)


# Prepare Trainable dataset
class DatasetReader(object):
  def __init__(self, filename, token_size):
    self.filepath = os.path.join(os.getcwd(), 'asmdata', filename)
    self.token_size = token_size
    if os.path.exists('mnemonic_vocab.txt'):
      self.mnemonic_vocab = self.read_vocab_files('mnemonic_vocab.txt')
      self.op_vocab = self.read_vocab_files('op_vocab.txt')
    else:
      vocab_reader = VocabReader(filename)
      vocab_reader.read()
      self.mnemonic_vocab = vocab_reader.get_vocab('mnemonic')
      self.op_vocab = vocab_reader.get_vocab('op')
      self.write_vocab(self.mnemonic_vocab, 'mnemonic_vocab.txt')
      self.write_vocab(self.op_vocab, 'op_vocab.txt')
    logger.info('Mnemonic vocab size: {0}'.format(self.mnemonic_vocab.len()))
    logger.info('Operands vocab size: {0}'.format(self.op_vocab.len()))

  def load_data(self):
    self.label_vocab = Vocab(skip=True)
    self.items=[]
    if os.path.exists('corpus.txt'):
      self.load_corpus()
    else:
      self.load()
      self.write_corpus()

    count = 0
    for item in self.items:
      if item.label.strip(' \r\t\n') == "partial":
        count += 1
    logger.info('corpus -: {0}'.format(count))
    logger.info('corpus +: {0}'.format(len(self.items) - count))
    logger.info('label vocab size: {0}'.format(self.label_vocab.len()))
    logger.info('corpus: {0}'.format(len(self.items)))

  # write Vocabulary
  def write_vocab(self, vocab, filename):
    with open(filename, mode='w', encoding='utf-8') as fp:
      for i,w in vocab.index2word.items():
        fp.write(str(i) + '\t' + str(w) + '\n')

  def write_corpus(self):
    with open('corpus.txt', mode='w', encoding='utf-8') as fp:
      for item in self.items:
        fp.write('label:{0}\n'.format(item.label))
        fp.write('assembly_a:\n')
        for m,op1,op2 in item.instructions_a:
          fp.write(str(m) + '\t' + str(op1) + '\t' + str(op2)+'\n')
        fp.write('assembly_b:\n')
        for m,op1,op2 in item.instructions_b:
          fp.write(str(m) + '\t' + str(op1) + '\t' + str(op2)+'\n')
        fp.write('\n')

  def read_vocab_files(self, filename):
    index = 0
    with open(filename, mode='r', encoding='utf-8') as fp:
      vocab = PVocab()
      for line in fp:
        data = line.strip(' \r\n\t').split('\t')
        index = int(data[0])
        if len(data) > 1:
          word = data[1]
        else:
          word = ''
        vocab.append(word, index)
    return vocab

  def load_corpus(self):
    with open('corpus.txt', mode='r', encoding='utf-8') as fp:
      asm_data = None
      parse_mode = 0
      instA_contexts_append = None
      instB_contexts_append = None
      label_vocab = self.label_vocab
      size = {}
      # read the dataset file line by line
      for line in fp:
        if line == '\n':
          if asm_data is not None:
            self.items.append(asm_data)
            asm_data = None
          continue

        if asm_data is None:
          asm_data = AsmData()
          instA_contexts_append = asm_data.instructions_a.append
          instB_contexts_append = asm_data.instructions_b.append

        # first filter id and name lines
        if line.startswith('label:'):
          asm_data.label = line[6:].strip(' \r\t\n')
          label_vocab.append(asm_data.label)
        elif line.startswith('assembly_a'):
          parse_mode = 1
        elif line.startswith('assembly_b'):
          parse_mode = 2
        else:
          data = line.strip(' \r\n\t').split('\t')
          mnemonic = int(data[0])
          operands = [[0],[0]] # initialize with zero
          for i in range(1, len(data)):
            if i == self.token_size:
              break
            tmp = re.split(r"[\[\],]", data[i])
            tmp = [int(w) for w in tmp if w != '']
            operands[i-1] = tmp

          if parse_mode == 1:
            instA_contexts_append((mnemonic, operands[0], operands[1]))
          elif parse_mode == 2:
            instB_contexts_append((mnemonic, operands[0], operands[1]))

      if asm_data is not None:
        self.items.append(asm_data)

  def load(self):
    with open(self.filepath, mode='r', encoding='utf-8') as fp:
      asm_data = None
      parse_mode = 0
      instA_contexts_append = None
      instB_contexts_append = None
      label_vocab = self.label_vocab

      # read the dataset file line by line
      for line in fp:
        if line.strip('\n\r\t') == 'XXXX:':
          continue
        if line == '\n':
          if asm_data is not None:
            self.items.append(asm_data)
            asm_data = None
          continue

        if asm_data is None:
          asm_data = AsmData()
          instA_contexts_append = asm_data.instructions_a.append
          instB_contexts_append = asm_data.instructions_b.append

        # first filter id and name lines
        if line.startswith('label:'):
          asm_data.label = line[6:].strip(' \r\t\n')
          label_vocab.append(asm_data.label)
        elif line.startswith('assembly_a'):
          parse_mode = 1
        elif line.startswith('assembly_b'):
          parse_mode = 2
        else:
          delim = line.find(' ')
          # for instructions like 'ret', 'leave' with no operands.
          if delim == -1:
            try:
              mnemonic = self.mnemonic_vocab.word2index[line.strip('\n\t\r')]
            except:
              mnemonic = 0
            operands = [[0],[0]] # initialize with zero
            continue

          # Get the embedding vector
          try:
            mnemonic = self.mnemonic_vocab.word2index[line[:delim]]
          except:
            mnemonic = 0
          operands = [[0],[0]] # initialize with zero
          op_words = list(map(lambda arg: arg.strip(), line[delim+1:].split(',')))
          for i,op in enumerate(op_words):
            if op is '':
              continue

            if i < 2:
              tokens = operand_tokens(op)
              try:
                tokens_index = [self.op_vocab.word2index[w] for w in tokens]
              except:
                tokens_index = [0]
              operands[i] = tokens_index
          if parse_mode == 1:
            instA_contexts_append((mnemonic, operands[0], operands[1]))
          elif parse_mode == 2:
            instB_contexts_append((mnemonic, operands[0], operands[1]))

      if asm_data is not None:
        self.items.append(asm_data)
