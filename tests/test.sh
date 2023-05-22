NCCL_DIR=../build
g++ -shared -I/usr/local/cuda/include -I${NCCL_DIR}/include -L/usr/local/cuda/lib64 -ldl -fPIC msamp.cc -o libmsamp.so
LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH LD_PRELOAD=./libmsamp.so:../build/lib/libnccl.so:$LD_PRELOAD python -m torch.distributed.launch --nnodes=1 --nproc_per_node 2 test.py