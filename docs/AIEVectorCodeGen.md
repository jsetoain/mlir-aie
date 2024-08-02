# Targeting AIE Vector Units

In order to maximize performance on AIE, it's paramount to maximize the usage of
its complex vector units (see [LINK](https://whereever)) in our kernels.
While _mlir-aie_ itself does not provide mechanisms to vectorize the kernels,
it provides passes to convert standard [MLIR Vector
Dialect](https://mlir.llvm.org/docs/Dialects/Vector/) code into AIE Vector code,
so we can leverage existing vectorization tools within MLIR instead, or even
build our own vectorization strategies within MLIR's framework.

Currently, although there's an ongoing sustained effort to support as much of
Vector Dialect as possible, there are gaps that might prevent your vector code
from compiling to AIE vector units unmodified. If you find one such cases,
please let us know by [opening an issue in the
project](https://github.com/Xilinx/mlir-aie/issues/new/choose), and we will do
our best to add support for it as soon as possible.

In the meantime, this document exists to provide guidance on what you can
expect to work out of the box, and what might require some additional work, as
well as some suggestions as to how to generate vector code that will run
efficiently on AIEngine.

# Canonicalization of Vector for conversion to AIEVec

Some times, some vectorization tools will generate Vector code that can't be
directly lowered to AIEVec code. The purpose of this pass is to get rid of
"invalid" Vector dialect by either transforming it to valid code, or unrolling
it down to scalar ops.

Examples:

- Unaligned memory accesses
- Complex vector transfer ops (like transfers with broadcasting dimensions)
- Unsupported element types
- Unsupported vector sizes

# Conversion from Vector to AIEVec dialect

Some complex AIE vector instructions are better encoded by intrinsics, instead
of general, back-end agnostic, Vector operations. These intrinsics are
represented within _mlir-aie_ in the AIEVec dialect. _mlir-aie_ also provides
a pass to convert Vector dialect into a combination of Vector + AIEVec
dialects, allowing for efficient vector code generation for AIE.

The conversion pass from Vector to AIEVec is:
```bash
-convert-vector-to-aievec
```

And it takes two main options: _aie-target_ and _target-backend_.

The first option, _aie-target_, indicates which version of the AIEngine vector
unit we are targeting. The second option, _target-backend_, indicates whether
the backend compiler will be an AIE _C++_ compiler or will take _LLVM IR_
directly.

# Optimization on AIEVec dialect

Merge and simplify AIEVec ops

Examples:
- Merge linked multi-channel ops that only use one channel
- Merge sequences of mac ops that come from broadcasted and shifted operands
  into conv1d AIEVec ops

# Supported Vector Dialect

## Directly supported

## Conversion supported

# Examples of Vector Dialect targeting AIE vector units

## Using the supervectorizer

# Contributing to AIE Vector Code Generation

# How AIE vector units work