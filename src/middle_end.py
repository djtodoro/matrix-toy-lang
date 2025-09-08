from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern
)
from xdsl.ir import Operation
from xdsl.dialects import builtin, func
from typing import List

# Import TransposeOp from dialect
from matrix_toy_lang.src.dialect import TransposeOp

class DoubleTransposeElimination(RewritePattern):
    """Eliminate double transpose patterns: (A^T)^T = A"""
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, 
                         rewriter: PatternRewriter) -> None:
        """Match and eliminate double transpose."""
        
        # Check if input to this transpose is also a transpose
        input_val = op.operands[0]
        if not input_val.owner:
            return
            
        input_op = input_val.owner
        
        if isinstance(input_op, TransposeOp):
            # Found pattern: transpose(transpose(X)) = X
            original_matrix = input_op.operands[0]
            
            # Replace all uses of the double transpose result with the original matrix
            op.results[0].replace_by(original_matrix)
            
            # Check if the inner transpose has no other uses
            # Convert uses to list to check if empty
            if not list(input_val.uses):
                # Erase the inner transpose first if it's dead
                rewriter.erase_op(input_op)
            
            # Erase the outer transpose operation
            rewriter.erase_op(op)

class TransposeChainOptimization(ModulePass):
    """Optimize chains of transpose operations."""
    
    name = "optimize-transpose"
    
    def apply(self, ctx, module: builtin.ModuleOp) -> None:
        """Apply transpose optimizations to module."""
        
        # Track statistics
        self.transposes_eliminated = 0
        
        # Apply pattern rewriting
        walker = PatternRewriteWalker(DoubleTransposeElimination())
        walker.rewrite_module(module)
        
        # Second pass: find hidden transpose chains
        self.find_transpose_chains(module)
        
        # Report results
        if self.transposes_eliminated > 0:
            print(f"Eliminated {self.transposes_eliminated} transpose operations")
    
    def find_transpose_chains(self, module: builtin.ModuleOp):
        """Find and optimize transpose chains across assignments."""
        
        # Build use-def chains
        transpose_ops = []
        for op in module.walk():
            if isinstance(op, TransposeOp):
                transpose_ops.append(op)
        
        # Analyze patterns
        for t_op in transpose_ops:
            if self.can_eliminate_transpose(t_op):
                self.eliminate_transpose(t_op)
                self.transposes_eliminated += 1
    
    def can_eliminate_transpose(self, op: TransposeOp) -> bool:
        """Check if transpose can be eliminated."""
        # Look for patterns where transpose cancels out
        # TODO: This is simplified - we should implement full analysis
        return False  # TODO: Implement
    
    def eliminate_transpose(self, op: TransposeOp):
        """Eliminate unnecessary transpose."""
        # TODO: Implement elimination logic
        pass

class DeadCodeElimination(ModulePass):
    """Remove operations whose results are never used."""
    
    name = "dead-code-elimination"
    
    def apply(self, ctx, module: builtin.ModuleOp) -> None:
        """Remove dead operations from the module."""
        eliminated = 0
        changed = True
        
        # Repeat until no more dead code can be eliminated
        while changed:
            changed = False
            ops_to_remove = []
            
            # Walk through all operations
            for op in module.walk():
                # Skip certain operations that shouldn't be removed
                if isinstance(op, (builtin.ModuleOp, func.FuncOp)):
                    continue
                    
                # Skip operations with side effects (like return statements)
                if op.name == "func.return" or op.name == "func.call":
                    continue
                
                # Check if any result of this operation is used
                has_uses = False
                for result in op.results:
                    if list(result.uses):  # Convert to list and check if non-empty
                        has_uses = True
                        break
                
                # If no results are used, mark for removal
                if not has_uses and len(op.results) > 0:
                    ops_to_remove.append(op)
            
            # Remove dead operations
            for op in ops_to_remove:
                op.detach()  # Detach from parent first
                op.erase()
                eliminated += 1
                changed = True
        
        if eliminated > 0:
            print(f"Eliminated {eliminated} dead operations")

class MatrixOptimizationPipeline(ModulePass):
    """Complete optimization pipeline for matrix operations."""
    
    name = "matrix-opt-pipeline"
    
    def apply(self, ctx, module: builtin.ModuleOp) -> None:
        """Apply all matrix optimizations."""
        
        # Apply transpose optimization
        walker = PatternRewriteWalker(DoubleTransposeElimination())
        walker.rewrite_module(module)
        
        # Apply dead code elimination
        dce = DeadCodeElimination()
        dce.apply(ctx, module)
        
        # Verify IR is still valid
        module.verify()
