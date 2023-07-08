import jax
import flax
from flax import linen as nn
from jax import numpy as jnp
from flax.linen import Embed, Sequential

class FeedForward(nn.Module):
    n_embd: int
    dropout: float
    
    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(4 * self.n_embd),
            nn.relu,
            nn.Dense(self.n_embd)
        ])
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x, deterministic: bool):
        x = self.net(x)
        return self.drop(x, deterministic=deterministic)


class CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    dropout: float

    def setup(self):
        assert self.n_embd % self.n_head == 0
        head_size = self.n_embd // self.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(self.n_embd * 3)
        # output projection
        self.c_proj = nn.Dense(self.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)


    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=deterministic)
        return y

class TransformerBlock(nn.Module):
    n_embd: int
    n_head: int
    dropout: float

    def setup(self):
        self.head_size = self.n_embd // self.n_head
        #self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        #self.sa = SelfAttention(n_head, dropout_rate=dropout) # head size is calculated internally
        self.sa = CausalSelfAttention(self.n_head, self.n_embd, self.dropout)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.ffwd = FeedForward(self.n_embd, self.dropout)

    def __call__(self, x, deterministic):
        x = x + self.sa(self.ln1(x), deterministic=deterministic)
        x = x + self.ffwd(self.ln2(x), deterministic=deterministic)
        return x

class GPTLanguageModel(nn.Module):
    """ This block performs the embedding of the tokens and the positional embedding, it feeds the result
    into the transformer blocks, normalizes the result and feeds it into the final linear layer."""
    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int
    dropout: float
    n_layer: int

    def setup(self):
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = Embed(self.vocab_size, self.n_embd)
        self.position_embedding_table = Embed(self.block_size, self.n_embd)
        self.blocks = [TransformerBlock(self.n_embd, self.n_head, self.dropout) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5) # final layer norm
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, xb, deterministic=True):
        B, T = xb.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(xb) # (B,T,C)
        pos_emb = self.position_embedding_table(jnp.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, deterministic) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits