"""
Standard Lowering Pass for Matrix Dialect
Converts Matrix dialect operations to standard MLIR dialects (memref, scf, arith, func).
This prepares the IR for subsequent LLVM conversion using mlir-opt.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from xdsl.context import Context
from xdsl.ir import Operation, SSAValue, Block, Region, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier
)
from xdsl.dialects import builtin, func, memref, arith, scf
from xdsl.dialects.builtin import (
    ModuleOp, 
    Float32Type, 
    IndexType,
    IntegerAttr,
    FloatAttr,
    UnrealizedConversionCastOp,
)

from .dialect import (
    MatrixType,
    MatMulOp,
    TransposeOp,
    ScalarMulOp,
    AddOp,
    AllocOp
)


def convert_matrix_type(matrix_type: MatrixType) -> memref.MemRefType:
    """Convert MatrixType to MemRefType."""
    rows = matrix_type.parameters[0].data
    cols = matrix_type.parameters[1].data
    element_type = matrix_type.parameters[2]
    return memref.MemRefType(element_type, [rows, cols])


def get_operand_as_memref(operand: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Get an operand as a memref, inserting a cast if necessary."""
    if isinstance(operand.type, memref.MemRefType):
        return operand
    elif isinstance(operand.type, MatrixType):
        # Insert a cast to convert matrix type to memref
        memref_type = convert_matrix_type(operand.type)
        cast = UnrealizedConversionCastOp.get([operand], [memref_type])
        rewriter.insert_op_before_matched_op(cast)
        return cast.results[0]
    else:
        return operand


@dataclass
class AllocOpLowering(RewritePattern):
    """Lower matrix.alloc to memref.alloc."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AllocOp, rewriter: PatternRewriter):
        # Get shape from attributes
        shape_attr = op.attributes['shape']
        rows = shape_attr.data[0].data
        cols = shape_attr.data[1].data
        
        # Get element type (default to f32)
        element_type = Float32Type()
        
        # Create memref type
        memref_type = memref.MemRefType(element_type, [rows, cols])
        
        # Create memref.alloc operation
        alloc_op = memref.AllocOp([], [], memref_type)
        
        # Cast back to matrix type to maintain interface
        matrix_type = op.result.type
        cast = UnrealizedConversionCastOp.get([alloc_op.results[0]], [matrix_type])
        
        # Replace the operation
        rewriter.replace_matched_op([alloc_op, cast], cast.results)


@dataclass
class AddOpLowering(RewritePattern):
    """Lower matrix.add to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter):
        # Get matrix dimensions from the result type
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return  # Already lowered
            
        rows = result_type.parameters[0].data
        cols = result_type.parameters[1].data
        element_type = result_type.parameters[2]
        
        # Create result memref
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        # Get operands as memrefs
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        # Create constants for loop bounds
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Create outer loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        # Create inner loop
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Inside inner loop: load, add, store
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, j])
        rhs_elem = memref.LoadOp.get(rhs_memref, [i, j])
        sum_elem = arith.AddfOp(lhs_elem, rhs_elem)
        store_op = memref.StoreOp.get(sum_elem, result, [i, j])
        
        # Build inner loop body
        inner_block.add_ops([lhs_elem, rhs_elem, sum_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        # Add inner loop to outer loop body
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        # Cast result back to matrix type
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        # Replace the operation
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class TransposeOpLowering(RewritePattern):
    """Lower matrix.transpose to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        # Get result dimensions from the result type
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return  # Already lowered
            
        # For transpose, result has swapped dimensions
        cols = result_type.parameters[0].data  # Result rows = input cols
        rows = result_type.parameters[1].data  # Result cols = input rows
        element_type = result_type.parameters[2]
        
        # Create result memref (transposed dimensions)
        memref_type = memref.MemRefType(element_type, [cols, rows])
        result = memref.AllocOp([], [], memref_type)
        
        # Get input as memref
        input_memref = get_operand_as_memref(op.input, rewriter)
        
        # Create constants for loop bounds (iterate over input dimensions)
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Create outer loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        # Create inner loop
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Inside inner loop: load from [i,j], store to [j,i]
        elem = memref.LoadOp.get(input_memref, [i, j])
        store_op = memref.StoreOp.get(elem, result, [j, i])
        
        # Build inner loop body
        inner_block.add_ops([elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        # Add inner loop to outer loop body
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        # Cast result back to matrix type
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        # Replace the operation
        rewriter.replace_matched_op(
            [result, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class ScalarMulOpLowering(RewritePattern):
    """Lower matrix.scalar_mul to nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ScalarMulOp, rewriter: PatternRewriter):
        # Get result dimensions
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return  # Already lowered
            
        rows = result_type.parameters[0].data
        cols = result_type.parameters[1].data
        element_type = result_type.parameters[2]
        
        # Get scalar value
        scalar_attr = op.attributes['scalar']
        scalar_value = scalar_attr.value.data
        
        # Create result memref
        memref_type = memref.MemRefType(element_type, [rows, cols])
        result = memref.AllocOp([], [], memref_type)
        
        # Get matrix as memref
        matrix_memref = get_operand_as_memref(op.matrix, rewriter)
        
        # Create scalar constant
        scalar_const = arith.ConstantOp(FloatAttr(scalar_value, element_type))
        
        # Create constants for loop bounds
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        rows_bound = arith.ConstantOp(IntegerAttr(rows, IndexType()))
        cols_bound = arith.ConstantOp(IntegerAttr(cols, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Create outer loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, rows_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        # Create inner loop
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, cols_bound, one, [], inner_block)
        j = inner_block.args[0]
        
        # Inside inner loop: load, multiply, store
        elem = memref.LoadOp.get(matrix_memref, [i, j])
        mul_elem = arith.MulfOp(elem, scalar_const)
        store_op = memref.StoreOp.get(mul_elem, result, [i, j])
        
        # Build inner loop body
        inner_block.add_ops([elem, mul_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        # Add inner loop to outer loop body
        outer_block.add_op(inner_loop)
        outer_block.add_op(scf.YieldOp())
        
        # Cast result back to matrix type
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        # Replace the operation
        rewriter.replace_matched_op(
            [result, scalar_const, zero, rows_bound, cols_bound, one, outer_loop, result_cast],
            result_cast.results
        )


@dataclass
class MatMulOpLowering(RewritePattern):
    """Lower matrix.matmul to triple nested loops with memref operations."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatMulOp, rewriter: PatternRewriter):
        # Get result dimensions
        result_type = op.result.type
        if not isinstance(result_type, MatrixType):
            return  # Already lowered
            
        m = result_type.parameters[0].data  # rows of result
        n = result_type.parameters[1].data  # cols of result
        
        # Get k from lhs type (cols of lhs)
        lhs_type = op.lhs.type
        if isinstance(lhs_type, MatrixType):
            k = lhs_type.parameters[1].data
        else:
            # Already converted to memref
            k = lhs_type.shape.data[1].data
            
        element_type = result_type.parameters[2]
        
        # Create result memref
        memref_type = memref.MemRefType(element_type, [m, n])
        result = memref.AllocOp([], [], memref_type)
        
        # Get operands as memrefs
        lhs_memref = get_operand_as_memref(op.lhs, rewriter)
        rhs_memref = get_operand_as_memref(op.rhs, rewriter)
        
        # Create zero constant for initialization
        zero_float = arith.ConstantOp(FloatAttr(0.0, element_type))
        
        # Create constants for loop bounds
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        m_bound = arith.ConstantOp(IntegerAttr(m, IndexType()))
        n_bound = arith.ConstantOp(IntegerAttr(n, IndexType()))
        k_bound = arith.ConstantOp(IntegerAttr(k, IndexType()))
        one = arith.ConstantOp(IntegerAttr(1, IndexType()))
        
        # Initialize result matrix to zero
        init_outer_block = Block(arg_types=[IndexType()])
        init_outer = scf.ForOp(zero, m_bound, one, [], init_outer_block)
        i_init = init_outer_block.args[0]
        
        init_inner_block = Block(arg_types=[IndexType()])
        init_inner = scf.ForOp(zero, n_bound, one, [], init_inner_block)
        j_init = init_inner_block.args[0]
        
        init_store = memref.StoreOp.get(zero_float, result, [i_init, j_init])
        init_inner_block.add_ops([init_store])
        init_inner_block.add_op(scf.YieldOp())
        
        init_outer_block.add_op(init_inner)
        init_outer_block.add_op(scf.YieldOp())
        
        # Matrix multiplication triple loop
        outer_block = Block(arg_types=[IndexType()])
        outer_loop = scf.ForOp(zero, m_bound, one, [], outer_block)
        i = outer_block.args[0]
        
        middle_block = Block(arg_types=[IndexType()])
        middle_loop = scf.ForOp(zero, n_bound, one, [], middle_block)
        j = middle_block.args[0]
        
        inner_block = Block(arg_types=[IndexType()])
        inner_loop = scf.ForOp(zero, k_bound, one, [], inner_block)
        k_idx = inner_block.args[0]
        
        # Load elements
        lhs_elem = memref.LoadOp.get(lhs_memref, [i, k_idx])
        rhs_elem = memref.LoadOp.get(rhs_memref, [k_idx, j])
        result_elem = memref.LoadOp.get(result, [i, j])
        
        # Multiply and accumulate
        mul_elem = arith.MulfOp(lhs_elem, rhs_elem)
        acc_elem = arith.AddfOp(result_elem, mul_elem)
        
        # Store back
        store_op = memref.StoreOp.get(acc_elem, result, [i, j])
        
        # Build inner loop body
        inner_block.add_ops([lhs_elem, rhs_elem, result_elem, mul_elem, acc_elem, store_op])
        inner_block.add_op(scf.YieldOp())
        
        # Add inner loop to middle loop body
        middle_block.add_op(inner_loop)
        middle_block.add_op(scf.YieldOp())
        
        # Add middle loop to outer loop body
        outer_block.add_op(middle_loop)
        outer_block.add_op(scf.YieldOp())
        
        # Cast result back to matrix type
        result_cast = UnrealizedConversionCastOp.get([result.results[0]], [result_type])
        
        # Replace the operation
        rewriter.replace_matched_op(
            [result, zero_float, zero, m_bound, n_bound, k_bound, one,
             init_outer, outer_loop, result_cast],
            result_cast.results
        )


class CleanupCasts(RewritePattern):
    """Clean up UnrealizedConversionCastOps by removing redundant ones."""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: UnrealizedConversionCastOp, rewriter: PatternRewriter):
        # If cast from and to the same type, remove it
        if op.inputs and op.results:
            if len(op.inputs) == 1 and len(op.results) == 1:
                # Check for back-to-back casts that cancel out
                input_val = op.inputs[0]
                if isinstance(input_val.owner, UnrealizedConversionCastOp):
                    prev_cast = input_val.owner
                    if prev_cast.inputs and len(prev_cast.inputs) == 1:
                        # If we're casting A->B->A, replace with A
                        if prev_cast.inputs[0].type == op.results[0].type:
                            rewriter.replace_matched_op([], [prev_cast.inputs[0]])
                            return


class MatrixToStandardLoweringPass(ModulePass):
    """Pass to lower Matrix dialect operations to standard MLIR dialects."""
    
    name = "matrix-to-standard"
    
    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the lowering pass to the module."""
        
        # First pass: Lower matrix operations to memref/scf/arith
        patterns = [
            AllocOpLowering(),
            AddOpLowering(),
            TransposeOpLowering(),
            ScalarMulOpLowering(),
            MatMulOpLowering(),
        ]
        
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(patterns),
            apply_recursively=True
        )
        walker.rewrite_module(op)
        
        # Second pass: Update function signatures and clean up casts
        self._update_functions(op)
        
        # Third pass: Clean up redundant casts
        cleanup_patterns = [CleanupCasts()]
        cleanup_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(cleanup_patterns),
            apply_recursively=True
        )
        cleanup_walker.rewrite_module(op)
    
    def _update_functions(self, module: ModuleOp):
        """Update function signatures and handle casts."""
        for op in module.body.ops:
            if isinstance(op, func.FuncOp):
                # Update function type
                func_type = op.function_type
                new_inputs = []
                new_outputs = []
                
                for input_type in func_type.inputs.data:
                    if isinstance(input_type, MatrixType):
                        new_inputs.append(convert_matrix_type(input_type))
                    else:
                        new_inputs.append(input_type)
                
                for output_type in func_type.outputs.data:
                    if isinstance(output_type, MatrixType):
                        new_outputs.append(convert_matrix_type(output_type))
                    else:
                        new_outputs.append(output_type)
                
                # Update the function type
                new_func_type = func.FunctionType.from_lists(new_inputs, new_outputs)
                op.attributes['function_type'] = new_func_type
                
                # Update block arguments
                if op.body.blocks:
                    entry_block = op.body.blocks[0]
                    for i, arg in enumerate(entry_block.args):
                        if i < len(new_inputs):
                            arg._type = new_inputs[i]
                
                # Handle returns - insert casts if needed
                for block in op.body.blocks:
                    for i, block_op in enumerate(list(block.ops)):
                        if isinstance(block_op, func.ReturnOp):
                            new_operands = []
                            ops_to_insert = []
                            
                            for j, operand in enumerate(block_op.operands):
                                if j < len(new_outputs):
                                    expected_type = new_outputs[j]
                                    if operand.type != expected_type:
                                        # Need to insert a cast
                                        if isinstance(operand.type, MatrixType):
                                            # Cast from matrix to memref
                                            cast = UnrealizedConversionCastOp.get(
                                                [operand], [expected_type]
                                            )
                                            ops_to_insert.append(cast)
                                            new_operands.append(cast.results[0])
                                        elif isinstance(operand.owner, UnrealizedConversionCastOp):
                                            # Use the input of the cast
                                            cast_op = operand.owner
                                            if cast_op.inputs:
                                                new_operands.append(cast_op.inputs[0])
                                            else:
                                                new_operands.append(operand)
                                        else:
                                            new_operands.append(operand)
                                    else:
                                        new_operands.append(operand)
                                else:
                                    new_operands.append(operand)
                            
                            # Insert casts before return
                            for cast_op in ops_to_insert:
                                block.insert_op_before(cast_op, block_op)
                            
                            # Update return operands
                            if new_operands != list(block_op.operands):
                                # Create new return with updated operands
                                new_return = func.ReturnOp(*new_operands)
                                block.insert_op_before(new_return, block_op)
                                block_op.detach()


def cleanup_final_types(ctx: Context, module: ModuleOp) -> ModuleOp:
    """Final cleanup to ensure all matrix types are eliminated."""
    
    # First, remove all UnrealizedConversionCastOps that involve matrix types
    def remove_matrix_casts(op: Operation):
        """Recursively remove casts involving matrix types."""
        ops_to_remove = []
        
        # Process all operations in the current operation's regions
        for region in op.regions:
            for block in region.blocks:
                for block_op in list(block.ops):
                    if isinstance(block_op, UnrealizedConversionCastOp):
                        # Check if this cast involves matrix types or is redundant
                        has_matrix = False
                        is_redundant = False
                        
                        for result in block_op.results:
                            if isinstance(result.type, MatrixType):
                                has_matrix = True
                                break
                        
                        # Check if it's a redundant cast (memref to same memref)
                        if block_op.inputs and block_op.results:
                            if (len(block_op.inputs) == 1 and len(block_op.results) == 1 and
                                isinstance(block_op.inputs[0].type, memref.MemRefType) and
                                isinstance(block_op.results[0].type, memref.MemRefType) and
                                block_op.inputs[0].type == block_op.results[0].type):
                                is_redundant = True
                        
                        if (has_matrix or is_redundant) and block_op.inputs:
                            # Replace uses of the cast result with the input
                            for i, result in enumerate(block_op.results):
                                if i < len(block_op.inputs):
                                    for use in list(result.uses):
                                        use.operation.operands[use.index] = block_op.inputs[i]
                            ops_to_remove.append(block_op)
                    else:
                        # Recursively process nested operations
                        remove_matrix_casts(block_op)
        
        # Remove collected operations
        for op_to_remove in ops_to_remove:
            op_to_remove.detach()
    
    # Remove matrix casts from the entire module
    remove_matrix_casts(module)
    
    # Update function signatures
    for op in module.body.ops:
        if isinstance(op, func.FuncOp):
            # Update function type
            func_type = op.function_type
            new_inputs = []
            new_outputs = []
            
            for input_type in func_type.inputs.data:
                if isinstance(input_type, MatrixType):
                    new_inputs.append(convert_matrix_type(input_type))
                else:
                    new_inputs.append(input_type)
            
            for output_type in func_type.outputs.data:
                if isinstance(output_type, MatrixType):
                    new_outputs.append(convert_matrix_type(output_type))
                else:
                    new_outputs.append(output_type)
            
            # Update the function type
            new_func_type = func.FunctionType.from_lists(new_inputs, new_outputs)
            op.attributes['function_type'] = new_func_type
            op.function_type = new_func_type
            
            # Update block arguments
            if op.body.blocks:
                entry_block = op.body.blocks[0]
                for i, arg in enumerate(entry_block.args):
                    if i < len(new_inputs):
                        arg._type = new_inputs[i]
            
            # Update function calls
            for block in op.body.blocks:
                for block_op in list(block.ops):
                    if isinstance(block_op, func.CallOp):
                        # Check if any result type needs updating
                        needs_update = False
                        new_result_types = []
                        for result_type in block_op.result_types:
                            if isinstance(result_type, MatrixType):
                                new_result_types.append(convert_matrix_type(result_type))
                                needs_update = True
                            else:
                                new_result_types.append(result_type)
                        
                        if needs_update:
                            # Create a new CallOp with updated types
                            new_call = func.CallOp(
                                block_op.callee,
                                block_op.operands,
                                new_result_types
                            )
                            # Replace the old op
                            block.insert_op_before(new_call, block_op)
                            # Update uses of the old result
                            for i, (old_res, new_res) in enumerate(zip(block_op.results, new_call.results)):
                                for use in list(old_res.uses):
                                    use.operation.operands[use.index] = new_res
                            block_op.detach()
    
    return module


def lower_to_standard_dialects(ctx: Context, module: ModuleOp, verbose: bool = False) -> ModuleOp:
    """Main entry point for lowering Matrix dialect to standard MLIR dialects."""
    lowering_pass = MatrixToStandardLoweringPass()
    lowering_pass.apply(ctx, module)
    
    # Final cleanup to ensure all matrix types are gone
    module = cleanup_final_types(ctx, module)
    
    if verbose:
        print("Successfully lowered Matrix dialect to standard MLIR dialects")
        print("All Matrix types have been eliminated - IR is ready for LLVM conversion")
    
    return module
