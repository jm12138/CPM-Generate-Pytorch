import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embedding_size):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(embedding_size, embedding_size*4)
        self.dense_4h_to_h = nn.Linear(embedding_size*4, embedding_size)
        self.act = nn.functional.gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2

class Attention(nn.Module):
    def __init__(self, 
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Attention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size

        self.query_key_value = nn.Linear(embedding_size, embedding_size * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)
        self.dense = nn.Linear(embedding_size, embedding_size)

    def split_heads(self, x):
        x = x.reshape([-1, self.seq_len, self.num_attention_heads, self.size_per_head])
        return x.permute(0, 2, 1, 3)

    def forward(self, x, kv_cache=None):
        self.seq_len = x.shape[1]
        x = self.query_key_value(x)
        q, k, v = torch.split(x, split_size_or_sections=self.embedding_size, dim=2)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        if kv_cache is not None:
            pk, pv = kv_cache[0], kv_cache[1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        cached_kv = torch.stack([k, v])

        attn = torch.matmul(q, k.transpose(-1, -2))  # [B, N, L, S]
        attn = attn / math.sqrt(self.size_per_head)

        # [L, S]
        attention_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.float32))
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])

        # adding to softmax -> its like removing them entirely
        attn = attn * attention_mask - 10000.0 * (1.0 - attention_mask)
        attn = nn.Softmax(dim=-1)(attn)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)
        # [B, N, L, S] -> [B, L, N, S]
        y = y.permute(0, 2, 1, 3)
        y = torch.reshape(y, [-1, self.seq_len, self.embedding_size])
        y = self.resid_drop(self.dense(y))

        return y, cached_kv

class Block(nn.Module):
    def __init__(self, 
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Block, self).__init__()
        self.input_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)
        self.attention = Attention(embedding_size, num_attention_heads, attention_dropout, residual_dropout)
        self.post_attention_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)
        self.mlp = MLP(embedding_size)

    def forward(self, x, kv_cache=None):
        attn, cached_kv = self.attention(self.input_layernorm(x), kv_cache=kv_cache)
        x = x + attn
        z = self.post_attention_layernorm(x)
        z = self.mlp(z)
        x = x + z
        return x, cached_kv

class Transformer(nn.Module):
    def __init__(self, 
                layer_size,
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([Block(
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout) 
            for _ in range(layer_size)])

        self.final_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)
    
    def forward(self, x, kv_cache=None):
        cached_kvs = []
        for i, layer in enumerate(self.layers):
            x, cached_kv = layer(
                x, 
                kv_cache=kv_cache[i] if kv_cache is not None else None)
            cached_kvs.append(cached_kv)
        x = self.final_layernorm(x)
        return x, torch.stack(cached_kvs)



class GPT2Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 layer_size,
                 block_size,
                 embedding_dropout,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout):
        super(GPT2Model, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(block_size, embedding_size)
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.transformer = Transformer(
            layer_size,
            embedding_size, 
            num_attention_heads,
            attention_dropout,
            residual_dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        if kv_cache is None:
            past_length = 0
        else:
            past_length = kv_cache[0][0].shape[-2]
        position_ids = torch.arange(past_length, x.shape[-1] + past_length, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        # print(position_ids)
        x = self.word_embeddings(x)
        x = self.emb_drop(x + self.position_embeddings(position_ids))
        # print(x)
        x, cached_kvs = self.transformer(x, kv_cache)
        x = torch.matmul(x, self.word_embeddings.weight.transpose(-1, -2))
        if use_cache:
            return x, cached_kvs
        return x


if __name__ == '__main__':
    gpt = GPT2Model(
    vocab_size=30000,
    layer_size=32,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=2560,
    num_attention_heads=32,
    attention_dropout=0.0,
    residual_dropout=0.0).half()
    gpt.eval()
    for x, y in gpt.state_dict().items():
        print(x, y.shape)
    # out, cached_kvs = gpt(torch.ones(1,1,dtype=torch.int64), torch.randn(2, 2, 1, 32, 1, 80, dtype=torch.float32), use_cache=True)
    # print(out.shape, cached_kvs.shape)
