/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_PARAM(TreeAllGatherEnabled, "TREE_ALLGATHER_ENABLED", 0);

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  if (ncclParamTreeAllGatherEnabled()) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    uint64_t totalcount = sendcount * nRanks;
    uint64_t sendsize = sendcount * ncclTypeSize(datatype);
    uint64_t totalsize = sendsize * nRanks;
    uint64_t offset = comm->rank * sendsize;
    if (offset > 0) {
      CUDACHECK(cudaMemsetAsync((void*)recvbuff, 0, offset, stream));
    }
    if ((uint64_t)sendbuff != (uint64_t)recvbuff + offset) {
      CUDACHECK(cudaMemcpyAsync((void*)((uint64_t)recvbuff + offset), sendbuff, sendsize, cudaMemcpyDeviceToDevice, stream));
    }
    if (offset + sendsize < totalsize) {
      CUDACHECK(cudaMemsetAsync((void*)((uint64_t)recvbuff + offset + sendsize), 0, totalsize - offset - sendsize, stream));
    }
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
      recvbuff, recvbuff, totalcount, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    struct ncclInfo info = { ncclFuncAllGather, "AllGather",
      sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  }
}
