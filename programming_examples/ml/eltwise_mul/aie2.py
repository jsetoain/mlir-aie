#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils


def my_eltwise_mul(trace_size):

    word_size_in = 2
    N = 65536
    N_in_bytes = N * word_size_in

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores
    buffer_depth = 2

    @device(AIEDevice.npu1_1col)
    def device_body():
        memRef_ty = T.memref(n, T.bf16())

        # Type used in the tile memory
        memRef_A_ty = T.memref(n, T.bf16())
        memRef_B_ty = T.memref(n, T.bf16())
        memRef_C_ty = T.memref(n, T.bf16())

        # Type used in the memory tile which aggregates across the 4 cores
        memRef_A_MT_ty = T.memref(n * n_cores, T.bf16())
        memRef_B_MT_ty = T.memref(n * n_cores, T.bf16())
        memRef_C_MT_ty = T.memref(n * n_cores, T.bf16())

        # AIE Core Function declarations

        eltwise_mul_bf16_scalar = external_func(
            "eltwise_mul_bf16_scalar", inputs=[memRef_ty, memRef_ty, memRef_ty]
        )
        eltwise_mul_bf16_vector = external_func(
            "eltwise_mul_bf16_vector", inputs=[memRef_ty, memRef_ty, memRef_ty]
        )
        # elwise_int32 = external_func("scale_int32", inputs=[memRef_ty, memRef_ty])

        # Tile declarations
        ShimTile = tile(0, 0)

        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(cores[0], WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        inA_fifo_names = [f"memA{i}" for i in range(n_cores)]
        inB_fifo_names = [f"memB{i}" for i in range(n_cores)]
        outC_fifo_names = [f"memC{i}" for i in range(n_cores)]

        inA_fifos = {}
        inB_fifos = {}
        outC_fifos = {}

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, memRef_A_MT_ty)
        for i in range(n_cores):
            inA_fifos[inA_fifo_names[i]] = object_fifo(
                inA_fifo_names[i], MemTile, cores[i], buffer_depth, memRef_A_ty
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(memRef_A_MT_ty.shape) // n_cores) * i for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inA, inA_fifo_names, [], of_offsets)

        # Input B
        inB = object_fifo("inB", ShimTile, MemTile, buffer_depth, memRef_B_MT_ty)
        for i in range(n_cores):
            inB_fifos[inB_fifo_names[i]] = object_fifo(
                inB_fifo_names[i], MemTile, cores[i], buffer_depth, memRef_B_ty
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(memRef_B_MT_ty.shape) // n_cores) * i for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inB, inB_fifo_names[0:n_cores], [], of_offsets)

        # Output C
        for i in range(n_cores):
            outC_fifos[outC_fifo_names[i]] = object_fifo(
                outC_fifo_names[i], cores[i], MemTile, buffer_depth, memRef_C_ty
            )
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, memRef_C_MT_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(memRef_C_MT_ty.shape) // n_cores) * i for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifo_names[0:n_cores], outC, of_offsets, [])

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "mul.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles):
                        elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        elem_in_b = inB_fifos[inB_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 1
                        )

                        call(
                            eltwise_mul_bf16_vector,
                            [elem_in_a, elem_in_b, elem_out],
                        )
                        inA_fifos[inA_fifo_names[i]].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[inB_fifo_names[i]].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[outC_fifo_names[i]].release(
                            ObjectFifoPort.Produce, 1
                        )
                        yield_([])
                    yield_([])

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.bf16())

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):

            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    cores[0],
                    ShimTile,
                    ddr_id=2,
                    size=trace_size,
                    offset=N_in_bytes,
                )

            npu_dma_memcpy_nd(metadata="outC", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="inA", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="inB", bd_id=2, mem=B, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)


try:
    trace_size = 0 if (len(sys.argv) < 2) else int(sys.argv[1])
except ValueError:
    print("Argument is not an integer")

with mlir_mod_ctx() as ctx:
    my_eltwise_mul(trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
