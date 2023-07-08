from flax import linen as nn
import optax
from jax import numpy as jnp
from jax import random
import jax
from flax.training import train_state
from networks import GPTLanguageModel
import os
import requests

# hyperparameters

# network setup
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
n_embd = 384 # dimension to which a token and its position are embedded
n_head = 6 # number of parallel attention heads per block
n_layer = 6 # number of transformer blocks
dropout = 0.2

# training
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 100

# ------------
key = random.PRNGKey(1337)

# Download tinyshakespeare if not present
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.isfile('tinyshakespeare.txt'):
    response = requests.get(url)
    with open('tinyshakespeare.txt', 'wb') as file:
        file.write(response.content)

# Open shakespeare
with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers (tokens)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Split data in train and test
data = jnp.array(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    key, cur_key = random.split(key)
    ix = random.randint(cur_key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def estimate_loss(key, params):
    out = {}
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            key, batch_key, loss_fn_key = random.split(key, 3)
            xb, yb = get_batch(batch_key, split)
            loss_fn = make_loss_fn(loss_fn_key, xb, yb)
            loss = loss_fn(params)
            losses = losses.at[k].set(loss)
        out[split] = losses.mean()
    return out

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, dropout, n_layer)
key, batch_key, params_key, dropout_rng = random.split(key, 4)
xb, yb = get_batch(batch_key, 'train')

model_key = {'params': params_key, 'dropout': dropout_rng}

params = model.init(model_key, xb, deterministic=True)

# print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create an optimizer
optimizer = optax.adamw(learning_rate=learning_rate)
#opt_state = optimizer.init(params)
trn_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

key, dropout_key = random.split(key)

def make_loss_fn(key, xb, yb):
    def loss_fn(params):
        logits = model.apply(params, xb, deterministic=False, rngs={'dropout':key})
        #B, T, C = logits.shape
        #logits = logits.reshape(B*T, C)
        #yb_inner = yb_inner.reshape(B*T)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        return loss
    return loss_fn

for iter in range(max_iters):

    key, batch_key, loss_fn_key, est_loss_key = random.split(key, 4)

    # sample a batch of data
    xb, yb = get_batch(batch_key, 'train')

    loss_fn = make_loss_fn(loss_fn_key, xb, yb)

    loss, grads = jax.value_and_grad(loss_fn)(trn_state.params)
    trn_state = trn_state.apply_gradients(grads=grads)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(est_loss_key, trn_state.params)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")



def generate(key, params, idx, max_new_tokens):
    loc_key = key
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        loc_key, sample_key, dropout_key = random.split(loc_key, 3)
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits = model.apply(params, idx_cond, deterministic=True, rngs={'dropout' : dropout_key})
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # sample from the distribution
        idx_next = random.categorical(sample_key, logits) # (B, 1)

        idx_next = jnp.reshape(idx_next, (1,-1))
        # append sampled index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1) # (B, T+1)
    return idx


key, generate_key = random.split(key)

# generate from the model
context = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(generate(generate_key, trn_state.params, context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(generate(generate_key, trn_state.params, context, max_new_tokens=10000)[0].tolist()))

