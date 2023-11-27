// RUN: aie-opt --aie-lower-multicast %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(7, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(7, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(7, 4)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(6, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(6, 4)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         }

module @test_multicast {
 AIE.device(xcvc1902) {
  %70 = AIE.tile(7, 0)
  %73 = AIE.tile(7, 3)
  %74 = AIE.tile(7, 4)
  %63 = AIE.tile(6, 3)
  %64 = AIE.tile(6, 4)
  AIEX.multicast(%70, "DMA" : 0){
    AIEX.multi_dest<%73, "DMA" : 0>
    AIEX.multi_dest<%74, "DMA" : 0>
    AIEX.multi_dest<%63, "DMA" : 0>
    AIEX.multi_dest<%64, "DMA" : 0>
  }
 }
}