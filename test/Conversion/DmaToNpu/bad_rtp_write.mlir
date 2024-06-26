//===- bad_rtp_write.mlir ---------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu -verify-diagnostics %s

aie.device(npu1_4col) {
  func.func @sequence() {
    // expected-error@+2 {{'aiex.npu.rtp_write' op RTP buffer address cannot be found. Has an RTP buffer been allocated?}}
    // expected-error@+1 {{failed to legalize operation 'aiex.npu.rtp_write' that was explicitly marked illegal}}
    aiex.npu.rtp_write(0, 2, 4, 99) { buffer_sym_name = "RTP" }
    return
  }
}
