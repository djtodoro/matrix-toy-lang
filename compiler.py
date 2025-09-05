#!/usr/bin/env python3
"""
Matrix Operations Compiler
A complete compiler for matrix operations with optimization.
"""

import sys
import argparse
from pathlib import Path

from matrix_toy_lang.src.parser import parse_matrix_program
from matrix_toy_lang.src.dialect import MatrixDialect
from matrix_toy_lang.src.ir_generator import MatrixIRGenerator
from matrix_toy_lang.src.middle_end import MatrixOptimizationPipeline

from xdsl.context import Context
from xdsl.printer import Printer
from xdsl.dialects import builtin, func

class MatrixCompiler:
    """Main compiler class."""
    
    def __init__(self):
        # Initialize xDSL context
        self.ctx = Context()
        self.ctx.load_dialect(builtin.Builtin)
        self.ctx.load_dialect(func.Func)
        self.ctx.load_dialect(MatrixDialect())
        
        self.printer = Printer()
    
    def compile_file(self, input_file: Path, 
                    output_file: Path = None,
                    optimize: bool = True) -> builtin.ModuleOp:
        """Compile a Python file with matrix operations."""
        
        print(f"Compiling {input_file}...")
        
        # Step 1: Parse Python source
        with open(input_file, 'r') as f:
            source_code = f.read()
        
        functions = parse_matrix_program(source_code)
        total_ops = sum(len(f['operations']) for f in functions.values())
        print(f"Parsed {len(functions)} functions with {total_ops} operations")
        
        # Step 2: Generate IR
        generator = MatrixIRGenerator()
        module = generator.generate(functions)
        print("Generated initial IR")
        
        if not optimize:
            print("Skipping optimizations")
        else:
            # Step 3: Optimize
            print("Running optimization passes...")
            optimizer = MatrixOptimizationPipeline()
            optimizer.apply(self.ctx, module)
        
        # Step 4: Output IR
        if output_file:
            with open(output_file, 'w') as f:
                self.printer.print_op(module, f)
            print(f"Wrote optimized IR to {output_file}")
        else:
            print("\nGenerated IR:")
            self.printer.print_op(module)
        
        return module
    
    def verify_optimization(self, original: str, optimized: str):
        """Verify that optimization preserves semantics."""
        # TODO: Implement semantic equivalence checking
        # Could involve:
        # - Symbolic execution
        # - Testing with concrete values
        # - Formal verification
        pass

def main():
    parser = argparse.ArgumentParser(
        description="Compile Python matrix operations to optimized IR"
    )
    parser.add_argument(
        'input',
        help='Input Python file with matrix operations'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for generated IR'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable optimizations'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create compiler
    compiler = MatrixCompiler()
    
    # Compile the file
    module = compiler.compile_file(
        Path(args.input),
        Path(args.output) if args.output else None,
        optimize=not args.no_optimize
    )
    
    print("\nCompilation successful!")

if __name__ == "__main__":
    main()
