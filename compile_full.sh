#!/bin/bash

# Complete compilation pipeline for Matrix language
# Usage: ./compile_full.sh <input.mx> [output_name]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mx> [output_name]"
    echo "Example: $0 examples/main_input.mx my_program"
    exit 1
fi

INPUT=$1
OUTPUT_NAME=${2:-"output"}

# Compile Matrix source to MLIR
source venv/bin/activate >/dev/null 2>&1
matrixc $INPUT --emit-standard -o ${OUTPUT_NAME}.mlir 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Failed to compile Matrix source" >&2
    exit 1
fi

# Convert MLIR to LLVM IR
# First lower to LLVM dialect, then translate to LLVM IR
/opt/homebrew/opt/llvm@20/bin/mlir-opt ${OUTPUT_NAME}.mlir \
    -convert-scf-to-cf \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o ${OUTPUT_NAME}_llvm.mlir 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Error: Failed to lower to LLVM dialect" >&2
    exit 1
fi

/opt/homebrew/opt/llvm@20/bin/mlir-translate --mlir-to-llvmir ${OUTPUT_NAME}_llvm.mlir -o ${OUTPUT_NAME}.ll 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Failed to translate to LLVM IR" >&2
    exit 1
fi

# Clean up intermediate file
rm -f ${OUTPUT_NAME}_llvm.mlir

# Compile LLVM IR to assembly
/opt/homebrew/opt/llvm@20/bin/llc ${OUTPUT_NAME}.ll -o ${OUTPUT_NAME}.s 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate assembly" >&2
    exit 1
fi

# Assemble and link to executable
/opt/homebrew/opt/llvm@20/bin/clang ${OUTPUT_NAME}.s -o ${OUTPUT_NAME} 2>/dev/null
if [ $? -ne 0 ]; then
    # Silent failure for executable (runtime support may be missing)
    rm -f ${OUTPUT_NAME}
fi

# Report success
echo "Generated: ${OUTPUT_NAME}.mlir ${OUTPUT_NAME}.ll ${OUTPUT_NAME}.s"
if [ -f ${OUTPUT_NAME} ]; then
    echo "Executable: ${OUTPUT_NAME}"
fi