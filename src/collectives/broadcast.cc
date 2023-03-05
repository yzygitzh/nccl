/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_PARAM(TreeBroadcastEnabled, "TREE_BROADCAST_ENABLED", 0);

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  if (ncclParamTreeBroadcastEnabled()) {
    if (comm->rank != root) {
      CUDACHECK(cudaMemsetAsync((void*)sendbuff, 0, count * ncclTypeSize(datatype), stream));
    }
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
      sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
      BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  }
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

