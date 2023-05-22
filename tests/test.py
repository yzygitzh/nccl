import ctypes
import os
import torch
import torch.distributed as dist
import pickle

rank = int(os.getenv('RANK'))
local_rank = int(os.getenv('LOCAL_RANK'))
world_size = int(os.getenv('WORLD_SIZE'))
print(f'Rank/World Size: {rank}/{world_size}')
torch.distributed.init_process_group(
    backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()
torch.cuda.set_device(local_rank)

assert world_size == 2

libmsamp = ctypes.cdll.LoadLibrary('libmsamp.so')

# test case 1
# Check uint8 all reduce
xs = [torch.tensor([100, 200, 300], dtype=torch.uint8, device='cuda'),
      torch.tensor([130, 220, 307], dtype=torch.uint8, device='cuda')]
target = xs[0] + xs[1]

x = xs[rank].clone()

libmsamp.DisableFP8()
torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)

if torch.all(x == target):
    print('[passed] uint8 all reduce passed')
else:
    print(f'[failed] uint8 all reduce failed. Output: {x}, target: {target}')

# test case 2
# Check fp8e4m3 all reduce
# E4M3: 4 exponent bits, 3 mantissa bits
xs = [torch.tensor([0b01001010, 0b01011000], dtype=torch.uint8, device='cuda'),
      torch.tensor([0b01010000, 0b01011100], dtype=torch.uint8, device='cuda')]
target = torch.tensor([0b01010101, 0b01100010], dtype=torch.uint8, device='cuda')

x = xs[rank].clone()

libmsamp.EnableFP8E4M3()
torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)

if torch.all(x == target):
    print('[passed] fp8e4m3 all reduce passed')
else:
    print(f'[failed] fp8e4m3 all reduce failed. Output: {x}, target: {target}')
