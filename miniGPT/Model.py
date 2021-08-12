import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class GPTConfig:
    """ Configuration for GPT Model """
    dropout = 0.1

    def __init__(self,
                 vocab_size, embed_dim, max_seq_len, pad_idx,
                 depth, num_attention_heads,
                 ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.depth = depth  # number of attention block layers
        self.num_attention_heads = num_attention_heads


class GPT(nn.Module):
    """ The full GPT model """

    def __init__(self, config):
        super(GPT, self).__init__()
        self.embedding_layer = GPTEmbedding(config)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.depth)])
        self.ln = nn.Layernorm(config.embed_dim)
        self.token_pred_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.embed_dim = config.embed_dim

    def forward(self, idx_sequences, masks):
        """
        :param idx_sequences:  batch of <s> one two three four five . </s>  <pad> <pad>
        :param masks:          batch of  1   1   1   1     1    1   1  0     0     0
        :return:
        """
        inputs = idx_sequences[:-1]
        targets = idx_sequences[1:]

        bsz, seq_len = inputs.shape

        embedding = self.embedding_layer.forward(inputs, masks)
        last_hidden_state = self.ln(self.blocks(embedding))
        logits = self.token_pred_head(last_hidden_state)

        loss = F.cross_entropy(logits.view(-1, self.embed_dim), targets.flatten(), reduction='none')
        loss = loss.view(bsz, seq_len).masked_fill(masks == 0, 0).mean()

        return logits, loss 


class GPTEmbedding(nn.Module):
    """ Embedding Layer """

    def __init__(self, config):
        super(GPTEmbedding, self).__init__()
        self.embed_dim = config.embed_dim
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_idx)
        self.pos_embedding = nn.Parameters(torch.randn(1, config.max_seq_len, config.embed_dim))
        self.dropout = nn.Dropout(p=config.dropout)

        self.register_parameter('pos_embedding', self.pos_embedding)

    def forward(self, inputs, masks):
        """
        :param: inputs:     batch of <s> one two three four five six seven  .
                                     <s> app ban crawl doll fish .   </s> <pad>
                                     <s> can you figure out the rule ?    </s>
        :param masks:       batch of  1   1   1    1     1   1   1    1     1
                                      1   1   1    1     1   1   1    0     0
                                      1   1   1    1     1   1   1    1     0
        :return: embedding
        """
        # inputs = idx_sequence[:-1]
        # targets = idx_sequence[1:]
        embedding = self.token_embedding(inputs)  # bsz, max_seq_len, embed_dim
        embedding = embedding + self.pos_embedding
        embedding = embedding.masked_fill(masks.unsqueeze(-1).repeat(1, 1, self.embed_dim), 0)
        return self.dropout(embedding)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attn_block = MultiHeadAttention(config)
        self.attention_ln = nn.Layernorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),  # conv 1d ?
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),  # conv 1d >
            nn.Dropout(config.dropout)
        )
        self.mlp_ln = nn.Layernorm(config.embed_dim)

    def forward(self, hidden_states):
        """
        :param hidden_states:  bsz, seq_len, embed_dim
        :return: next hidden states: bsz, seq_lem, embed_dim
        """
        hidden_states = self.attn_block.forward(self.attention_ln(hidden_states)) + hidden_states
        hidden_states = self.mlp(self.mlp_ln(hidden_states)) + hidden_states
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_attention_heads
        self.scale = math.sqrt(config.embed_dim)
        self.query_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.key_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.value_proj = nn.Linear(config.embed_dim, config.embed_dim)

        msl = config.max_seq_len
        self.attn_mask = torch.tril(torch.ones(msl, msl).view(1, 1, msl, msl))

        self.attn_drop = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer("attn_mask", self.attn_mask)

    def forward(self, hidden_states):
        bsz, seq_len, embed_dim = hidden_states.shape
        queries = self.query_proj(hidden_states).view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        keys = self.key_proj(hidden_states).view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        values = self.value_proj(hidden_states).view(bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        # batch_size x num_heads, sequence_length, attention_dim

        attention_score = (queries @ keys.transpose(-2, -1)) / self.scale  # (bsz, num_heads, seq_len, seq_len)
        attention_score = attention_score.masked_fill(self.attn_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attention = F.softmax(attention_score, dim=-1)
        attention = self.attn_drop(attention)

        nxt_hidden_states = attention @ values  # bsz, num_heads, seq_len, embed_dim
        nxt_hidden_states = nxt_hidden_states.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        return self.resid_drop(self.out_proj(nxt_hidden_states))
