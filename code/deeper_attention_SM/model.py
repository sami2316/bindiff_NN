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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

NINF = - 3.4 * math.pow(10,38) # -Inf
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
eloss = torch.nn.CosineEmbeddingLoss(margin=0.1, reduction='mean')
pdist = torch.nn.PairwiseDistance(p=1, eps=0, keepdim=True)

class Asm2Vec(nn.Module):
  def __init__(self, options):
    super(Asm2Vec, self).__init__()
    self.options = options
    self.encode_size = options.encode_size
    self.embeddings_mnemonic = nn.Embedding(options.mnemonic_count, options.mnemonic_embed_size)
    self.embeddings_operand = nn.Embedding(options.op_count, options.operand_embed_size)

    # operand NN layers
    self.operand_linear = nn.Linear(options.operand_embed_size, options.operand_embed_size, bias=False)
    self.operand_layer_norm = nn.LayerNorm(options.operand_embed_size)

    self.operand_attention_parameter = Parameter(torch.nn.init.xavier_normal_(torch.zeros(options.operand_embed_size, 1, dtype=torch.float32, requires_grad=True)).view(-1), requires_grad=True)

    # Instructions NN layers
    self.input_linear = nn.Linear(options.mnemonic_embed_size + (options.operand_embed_size*2), options.encode_size, bias=False)
    self.input_layer__norm = nn.LayerNorm(options.encode_size)

    if 0.0 < options.dropout_prob < 1.0:
      self.input_dropout = nn.Dropout(p=options.dropout_prob)
    else:
      self.input_dropout = None

    self.instruction_attention_parameter = Parameter(torch.nn.init.xavier_normal_(torch.zeros(options.encode_size, 1, dtype=torch.float32, requires_grad=True)).view(-1), requires_grad=True)

    self.output_linear = nn.Linear(options.encode_size*3, options.label_count, bias=True)
    self.output_linear.bias.data.fill_(0.0)


  def compute_operand(self, operand):

    if len(list(operand.size())) == 2:
      operand = operand.unsqueeze(0)

    emb_list = []
    # compute embeddings for each token in operand
    for i in range(0,self.options.max_inst_token):
      emb_list.append(self.embeddings_operand(operand[:,:,i]))
    # FNN, Layer Normalization, tanh
    op_vectors = torch.sum(torch.stack(emb_list, dim=2), dim=2)


    op_vectors = self.operand_linear(torch.stack(emb_list, dim=2))
    ccv_size = op_vectors.size()
    op_vectors = self.operand_layer_norm(op_vectors.view(-1, self.options.operand_embed_size)).view(ccv_size)
    op_vectors = torch.tanh(op_vectors)
    '''
    return op_vectors
    '''
    # attention
    attn_mask = (operand > 0).float()
    expand_attn_param = self.operand_attention_parameter.unsqueeze(0).expand_as(op_vectors)
    attn_ca = torch.mul(torch.sum(op_vectors * expand_attn_param, dim=-1), attn_mask) + (1-attn_mask)*NINF
    attention = F.softmax(attn_ca, dim=2)

    # operand vector
    expand_attn = attention.unsqueeze(-1).expand_as(op_vectors)
    operand_vector = torch.sum(torch.mul(op_vectors, expand_attn), dim=2)

    return operand_vector
    
  def forward(self, mnemonic, op1, op2):
    # embeddings

    if len(list(mnemonic.size())) == 1:
      mnemonic = mnemonic.unsqueeze(0)

    emb_mnemonic = self.embeddings_mnemonic(mnemonic)
    emb_operand1 = self.compute_operand(op1)
    emb_operand2 = self.compute_operand(op2)

    combined_context_vectors = torch.cat((emb_mnemonic, emb_operand1, emb_operand2), dim=2)

    # FNN, Layer Normalization, tanh
    combined_context_vectors = self.input_linear(combined_context_vectors)
    ccv_size = combined_context_vectors.size()
    combined_context_vectors = self.input_layer__norm(combined_context_vectors.view(-1, self.encode_size)).view(ccv_size)
    combined_context_vectors = torch.tanh(combined_context_vectors)

    # dropout
    if self.input_dropout is not None:
      combined_context_vectors = self.input_dropout(combined_context_vectors)

    # attention
    attn_mask = (mnemonic > 0).float()
    attention = self.get_attention(combined_context_vectors, attn_mask)

    # assembly function vector
    expand_attn = attention.unsqueeze(-1).expand_as(combined_context_vectors)
    func_vector = torch.sum(torch.mul(combined_context_vectors, expand_attn), dim=1)

    return func_vector, attention

  def siamese_forward(self, mnemonic_a, op1_a, op2_a, mnemonic_b, op1_b, op2_b):
    output_a, _ = self.forward(mnemonic_a, op1_a, op2_a)
    output_b, _ = self.forward(mnemonic_b, op1_b, op2_b)


    distance = 1 - torch.exp(-torch.abs(output_a - output_b))
    preds = self.output_linear(torch.cat((distance, output_a, output_b), dim=1))

    #loss = eloss(output_a, output_b, label)
    #preds = F.sigmoid(distance)
    #preds_values, _ = torch.max(preds, dim=1)
    #preds = torch.tensor([1 if x > 0.5 else 0 for x in preds_values.tolist()])
    #print(preds, label)
    #preds = torch.sub(torch.ones_like(non_linear), non_linear.round())
    return preds, distance

  def get_attention(self, vectors, mask):
    expand_attn_param = self.instruction_attention_parameter.unsqueeze(0).expand_as(vectors)
    attn_ca = torch.mul(torch.sum(vectors * expand_attn_param, dim=2), mask) + (1-mask)*NINF
    attention = F.softmax(attn_ca, dim=1)

    return attention
