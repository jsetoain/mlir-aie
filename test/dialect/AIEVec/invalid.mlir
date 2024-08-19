// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @invalidElementType(%A : vector<4x8xf16>, %B : vector<8x4xf16>,
                              %C : vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+1 {{op operand #0 must be a vector compatible with a lhs operand of matrix-multiply and accumulate, but got 'vector<4x8xf16>'}}
  %0 = aievec.matmul %A, %B, %C : vector<4x8xf16>, vector<8x4xf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

func.func @invalidShape(%A : vector<4x4xbf16>, %B : vector<4x4xbf16>,
                        %C : vector<4x4xf32>) -> vector<4x4xf32> {
  // expected-error @+1 {{op operand #0 must be a vector compatible with a lhs operand of matrix-multiply and accumulate, but got 'vector<4x4xbf16>'}}
  %0 = aievec.matmul %A, %B, %C : vector<4x4xbf16>, vector<4x4xbf16>
                                  into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

func.func @invalidContraction(%A : vector<2x4xi16>, %B : vector<2x8xi16>,
                              %C : vector<4x8xi32>) -> vector<4x8xi32> {
  // expected-error @+1 {{op failed to verify that [lhs x rhs = acc] is a valid contraction}}
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<2x8xi16>
                                  into vector<4x8xi32>
  return %0 : vector<4x8xi32>
}

// -----

func.func @invalidAccumulatorType(%A : vector<2x4xi16>, %B : vector<4x8xi16>,
                                  %C : vector<2x8xi32>) -> vector<2x8xi32> {
  // expected-error @+1 {{op operand #2 must be a vector compatible with an accumulator of matrix-multiply and accumulate, but got 'vector<2x8xi32>'}}
  %0 = aievec.matmul %A, %B, %C : vector<2x4xi16>, vector<4x8xi16>
                                  into vector<2x8xi32>
  return %0 : vector<2x8xi32>
}

// -----

func.func @invalidShuffleModeElementType(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't32_4x4' requires vectors of 32-bit elements}}
  %r = aievec.shuffle %v [t32_4x4] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidShuffleModeExtraOperand(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't16_4x8' does not admit a second operand}}
  %r = aievec.shuffle %v, %v [t16_4x8] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidShuffleModeMissingOperand(%v : vector<32xi16>)
            -> vector<32xi16> {
  // expected-error @+1 {{shuffle mode 't16_16x4_lo' requires a second operand}}
  %r = aievec.shuffle %v [t16_16x4_lo] : vector<32xi16>
  return %r : vector<32xi16>
}

// -----

func.func @invalidElementTypeMulElem(%arg0 : vector<32xi8>, %arg1 : vector<32xi8>) -> vector<32xi64> {
  // expected-error @+1 {{'aievec.mul_elem' op failed to verify that result type is a valid accumulator type given the type of the operands.}}
  %t11 = aievec.mul_elem %arg0, %arg1 : vector<32xi8>, vector<32xi8>, vector<32xi64>
  return %t11 : vector<32xi64>
}

// -----

func.func @invalidInsertDestShape(%src : i32, %dst : vector<4x4xi32>)
            -> vector<4x4xi32> {
  // expected-error @+1 {{'aievec.insert' op failed to verify that the destination is a 1D vector}}
  %r = aievec.insert %src, %dst[0, 0] : i32 into vector<4x4xi32>
  return %r : vector<4x4xi32>
}

// -----

func.func @invalidInsertAddressing(%src : i32, %dst : vector<16xi32>)
            -> vector<16xi32> {
  // expected-error @+1 {{'aievec.insert' op expected a single insertion index}}
  %r = aievec.insert %src, %dst[0, 0] : i32 into vector<16xi32>
  return %r : vector<16xi32>
}

// -----

func.func @invalidInsertSrcVector(%src : vector<2x2xi32>, %dst : vector<16xi32>)
            -> vector<16xi32> {
  // expected-error @+1 {{'aievec.insert' op failed to verify that the source vector is 1D}}
  %r = aievec.insert %src, %dst[0] : vector<2x2xi32> into vector<16xi32>
  return %r : vector<16xi32>
}

// -----

func.func @invalidInsertSrcInvalidSize(%src : vector<3xi32>,
                                       %dst : vector<16xi32>)
            -> vector<16xi32> {
  // expected-error @+1 {{'aievec.insert' op failed to verify that the source is 8, 16, 32, 64, 128, or 256 bits wide}}
  %r = aievec.insert %src, %dst[0] : vector<3xi32> into vector<16xi32>
  return %r : vector<16xi32>
}

// -----

func.func @invalidInsertMisaligned(%src : vector<2xi32>, %dst : vector<16xi32>)
            -> vector<16xi32> {
  // expected-error @+1 {{'aievec.insert' op failed to verify that insertion index is aligned to source size}}
  %r = aievec.insert %src, %dst[1] : vector<2xi32> into vector<16xi32>
  return %r : vector<16xi32>
}

// -----

func.func @invalidInsertSrcWontFit(%src : vector<4xi32>, %dst : vector<16xi32>)
            -> vector<16xi32> {
  // expected-error @+1 {{'aievec.insert' op failed to verify that source fits in destination at index 16}}
  %r = aievec.insert %src, %dst[16] : vector<4xi32> into vector<16xi32>
  return %r : vector<16xi32>
}

