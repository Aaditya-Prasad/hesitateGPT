"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O



always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'enwik8'
wandb_run_name = 'eval_baseline' # 'run' + str(time.time())
out_dir = 'out/' + wandb_project + '/' + wandb_run_name[5:] #gets rid of 'eval_' in the name

print("WRITING TO: ", out_dir)

# data
dataset = 'enwik8'
mem_length = 0
block_size = 1048 #- mem_length # we want to feed the model sequences of this size, it will prepend memory internally
# model
n_layer = 8
n_head = 8
n_embd = 624
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
test_data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')

class enwikIterator():
    def __init__(self, split, block_size, device_type):
        self.data = val_data if split == 'val' else test_data
        self.idx = 0
        self.block_size = block_size
        self.device_type = device_type

    def __iter__(self):
        return self

    def __next__(self):
        index = self.idx * self.block_size
        if index > self.data_len():
            raise StopIteration
        
        x = torch.from_numpy((self.data[index:index+self.block_size]).astype(np.int64)).unsqueeze(0) # B T
        y = torch.from_numpy((self.data[index+1:index+self.block_size+1]).astype(np.int64)).unsqueeze(0)

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        self.idx += 1
        return x, y

    def len(self):
        return (len(self.data) // self.block_size) - self.idx
    
    def data_len(self):
        return len(self.data) - self.block_size
    
@torch.no_grad()    
def eval_non_persistent(batch, model):
    x, y = batch

    logits, _, _ = model(x)
    target = y[:, [-1]]
    bpc = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1) / math.log(2)

    return bpc


@torch.no_grad()
def evaluate_bpc():
    bpc_dict = {}

    model.eval()
    
    for split in ['val', 'test']:
        dataset = enwikIterator(split, block_size, device_type)
        ds = iter(dataset)

        bpcs = torch.zeros(dataset.len())

        for seq in ds:
            bpc = eval_non_persistent(seq, model)
            bpcs[ds.idx-1] = bpc

        bpc_dict[split] = bpcs.mean()
        print(f"{split} bpc: {bpc_dict[split].item()}")

    return bpc_dict

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, mem_length=mem_length) # start with model_args from command line

print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, 'best_ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_test_bpc = checkpoint['best_test_bpc']
print(f"resuming from iteration {iter_num} and best test bpc {best_test_bpc}")
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Eval
model.eval()
model = model.to(device_type)
n_params = sum([p.nelement() for p in model.parameters()])
print(f"number of parameters: {n_params}")
bpc_dict = evaluate_bpc()
if wandb_log and master_process:
    wandb.log({
        'val_bpc': bpc_dict['val'].item(),
        'test_bpc': bpc_dict['test'].item(),
        'n_params': n_params,
    })
print(f"val bpc: {bpc_dict['val'].item()}, test bpc: {bpc_dict['test'].item()}")

if ddp:
    destroy_process_group()
