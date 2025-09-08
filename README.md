# Matrix Language Compiler

A complete compiler for a matrix-oriented programming language built using MLIR and xDSL. This compiler demonstrates the full compilation pipeline from high-level matrix operations to native machine code.

## Overview

The Matrix Language Compiler implements a Python-like syntax for matrix operations and compiles it through multiple intermediate representations (IRs) down to executable machine code. It showcases:

- Custom MLIR dialect for matrix operations
- Optimization passes (e.g., double transpose elimination)
- Progressive lowering from high-level operations to LLVM IR
- Full compilation to native assembly and executables

## Architecture

```
Matrix Source Code (.mx)
        ↓
    [Parser]
        ↓
    Matrix IR (Custom Dialect)
        ↓
    [Optimizer]
        ↓
    Optimized Matrix IR
        ↓
    [Lowering Pass]
        ↓
    Standard MLIR (memref/scf/arith)
        ↓
    [LLVM Lowering]
        ↓
    LLVM IR
        ↓
    [LLVM Backend]
        ↓
    Native Assembly/Executable
```

## Installation

### Prerequisites
- Python 3.10+
- LLVM 20 (for mlir-opt and mlir-translate)
- xDSL framework

### Setup

1. Clone the repository:
```bash
git clone https://github.com/djtodoro/matrix-toy-lang.git
cd matrix-toy-lang
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Basic Compilation

Compile a Matrix source file to MLIR:
```bash
matrixc examples/main_input.mx -o output.mlir
```

Compile with lowering to standard dialects:
```bash
matrixc examples/main_input.mx --emit-standard -o output.mlir
```

### Full Compilation Pipeline

Use the provided script to compile from source to executable:
```bash
./compile_full.sh examples/main_input.mx my_program
```

This generates:
- `my_program.mlir` - Standard MLIR representation
- `my_program.ll` - LLVM IR
- `my_program.s` - Assembly code
- `my_program` - Native executable

### Manual LLVM Conversion

Convert MLIR to LLVM IR:
```bash
# Lower to LLVM dialect
mlir-opt output.mlir \
    -convert-scf-to-cf \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o output_llvm.mlir

# Translate to LLVM IR
mlir-translate --mlir-to-llvmir output_llvm.mlir -o output.ll
```

## Example Program

```python
# examples/main_input.mx
def matrix_computation(A, B):
    """Matrix operations with optimizations."""
    C = A.T.T  # Double transpose (optimized away)
    D = B.T
    E = D.T    # E equals B after optimization
    F = C + E  # Matrix addition
    result = F.T.T  # Another double transpose
    return result

def main():
    """Main function."""
    A = matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    
    B = matrix([[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]])
    
    result = matrix_computation(A, B)
```

## Example Output

### Without Optimization
```mlir
func.func @matrix_computation(...) {
  %2 = "matrix.transpose"(%0) : ... // A.T
  %3 = "matrix.transpose"(%2) : ... // A.T.T
  %4 = "matrix.transpose"(%1) : ... // B.T
  %5 = "matrix.transpose"(%4) : ... // B.T.T
  %6 = "matrix.add"(%3, %5) : ...   // Add
  %7 = "matrix.transpose"(%6) : ... // Result.T
  %8 = "matrix.transpose"(%7) : ... // Result.T.T
  func.return %8
}
```

### With Optimization (Double Transpose Elimination)
```mlir
func.func @matrix_computation(...) {
  %4 = "matrix.add"(%0, %1) : ...   // Direct add A + B
  func.return %4
}
```

### After Lowering to Standard Dialects
```mlir
func.func @matrix_computation(%0 : memref<3x3xf32>, %1 : memref<3x3xf32>) -> memref<3x3xf32> {
  %result = memref.alloc() : memref<3x3xf32>
  scf.for %i = %c0 to %c3 step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      %a = memref.load %0[%i, %j] : memref<3x3xf32>
      %b = memref.load %1[%i, %j] : memref<3x3xf32>
      %sum = arith.addf %a, %b : f32
      memref.store %sum, %result[%i, %j] : memref<3x3xf32>
    }
  }
  func.return %result : memref<3x3xf32>
}
```

## Project Structure

```
matrix-toy-lang/
├── src/
│   ├── dialect.py                   # Matrix dialect definition
│   ├── parser.py                    # Python AST to IR parser
│   ├── ir_generator.py              # IR generation from parsed code
│   ├── middle_end.py                # Optimization passes
│   ├── standard_lowering.py         # Lowering to standard MLIR dialects
├── examples/
│   └── main_input.mx                # Example Matrix programs
├── compiler.py                      # Main compiler driver
├── compile_full.sh                  # Full compilation pipeline script
└── setup.py                         # Package setup
```

## Implementation Details

### Custom Dialect Operations

The Matrix dialect defines:
- `matrix.alloc` - Allocate matrix memory
- `matrix.add` - Matrix addition
- `matrix.transpose` - Matrix transpose
- `matrix.matmul` - Matrix multiplication  
- `matrix.scalar_mul` - Scalar multiplication

### Lowering Strategy

1. **Matrix to MemRef**: Convert matrix types to memref (memory reference) types
2. **Operations to Loops**: Lower matrix operations to nested loops
3. **Standard Dialects**: Use scf (structured control flow) for loops, arith for arithmetic
4. **LLVM Conversion**: Final conversion to LLVM dialect for code generation

### Optimization Passes

- **Double Transpose Elimination**: Detects patterns like `A.T.T` and replaces with `A`
- **Dead Code Elimination**: Removes unused operations
- **Constant Folding**: Evaluates compile-time constants

## Command-Line Options

```bash
matrixc [options] input.mx

Options:
  -o OUTPUT        Output file path
  --emit-standard  Generate standard MLIR dialects (prepared for LLVM conversion)
  --no-optimize    Disable optimization passes
  --verbose        Verbose output
```

## Testing

Run the test suite:
```bash
# Compile and verify example programs
./compile_full.sh examples/main_input.mx test_output

# Verify MLIR output
/path/to/llvm/bin/mlir-opt output.mlir -verify-diagnostics
```

## Troubleshooting

### Common Issues

1. **LLVM Tools Not Found**: Ensure LLVM 20 is installed and in PATH
   ```bash
   export PATH=/path/to/llvm/bin:$PATH
   ```

2. **Module Import Errors**: Ensure the package is installed in development mode
   ```bash
   pip install -e .
   ```

3. **Type Mismatch Errors**: Check that matrix dimensions match in operations

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

- Built with [xDSL](https://github.com/xdslproject/xdsl) framework
- Uses [MLIR](https://mlir.llvm.org/) infrastructure
- Inspired by various compiler design courses and tutorials
