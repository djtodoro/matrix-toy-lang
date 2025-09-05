from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.builtin import Float32Type, FloatAttr, ModuleOp
from typing import Dict, List, Any
from matrix_toy_lang.src.dialect import MatrixType, MatMulOp, TransposeOp, AddOp, ScalarMulOp, AllocOp

class MatrixIRGenerator:
    """Generate matrix dialect IR from parsed operations."""
    
    def __init__(self):
        self.var_maps = {}  # Variable maps per function
    
    def generate(self, functions: Dict[str, Any]) -> builtin.ModuleOp:
        """Generate IR module from parsed functions."""
        
        func_ops = []
        
        # Generate each function
        for func_name, func_info in functions.items():
            func_op = self.generate_function(func_name, func_info)
            if func_op:
                func_ops.append(func_op)
        
        # Create and return module
        module = builtin.ModuleOp(func_ops)
        return module
    
    def generate_function(self, func_name: str, func_info: Dict[str, Any]) -> func.FuncOp:
        """Generate a single function."""
        
        # Create a new variable map for this function
        self.var_maps[func_name] = {}
        var_map = self.var_maps[func_name]
        
        # Determine function signature
        args = func_info.get('args', [])
        operations = func_info.get('operations', [])
        
        # Create block with appropriate arguments
        if func_name == 'matrix_computation':
            # Two matrix arguments
            matrix_type = MatrixType(4, 4, Float32Type())
            block = Block(arg_types=[matrix_type, matrix_type])
            
            # Map arguments to variables
            if len(args) >= 2:
                var_map[args[0]] = block.args[0]
                var_map[args[1]] = block.args[1]
                
            input_types = [matrix_type, matrix_type]
            output_types = [matrix_type]
            
        elif func_name == 'main':
            # No arguments for main
            block = Block(arg_types=[])
            input_types = []
            output_types = []
        else:
            # Generic function
            block = Block(arg_types=[])
            input_types = []
            output_types = []
        
        # Generate operations
        last_result = None
        return_value = None
        
        for op_info in operations:
            if op_info['type'] == 'assign':
                result = self.generate_operation(op_info, block, var_map)
                if result:
                    last_result = result
            elif op_info['type'] == 'return':
                # Handle return statement
                return_op = op_info['operation']
                if return_op['op'] == 'var':
                    return_value = var_map.get(return_op['name'])
                else:
                    return_value = last_result
        
        # Add return operation
        if return_value:
            ret_op = func.ReturnOp(return_value)
        else:
            ret_op = func.ReturnOp()
        block.add_op(ret_op)
        
        # Create function region with the block
        region = Region([block])
        
        # Create function type
        func_type = func.FunctionType.from_lists(
            input_types,
            output_types if return_value else []
        )
        
        # Create function operation
        func_op = func.FuncOp(func_name, func_type, region)
        
        return func_op
    
    def generate_operation(self, op_info: Dict[str, Any], block: Block, var_map: Dict[str, SSAValue]) -> SSAValue:
        """Generate IR for a single operation."""
        op_type = op_info['operation']['op']
        target = op_info['target']
        
        result = None
        
        if op_type == 'matmul':
            left = var_map.get(op_info['operation']['left'])
            right = var_map.get(op_info['operation']['right'])
            if left and right:
                matmul_op = MatMulOp(left, right)
                block.add_op(matmul_op)
                result = matmul_op.result
                
        elif op_type == 'transpose':
            matrix = var_map.get(op_info['operation']['matrix'])
            if matrix:
                transpose_op = TransposeOp(matrix)
                block.add_op(transpose_op)
                result = transpose_op.result
                
        elif op_type == 'double_transpose':
            # Handle double transpose - create two transpose ops
            matrix = var_map.get(op_info['operation']['matrix'])
            if matrix:
                transpose1_op = TransposeOp(matrix)
                block.add_op(transpose1_op)
                transpose2_op = TransposeOp(transpose1_op.result)
                block.add_op(transpose2_op)
                result = transpose2_op.result
                
        elif op_type == 'scalar_mul':
            matrix = var_map.get(op_info['operation']['matrix'])
            scalar = op_info['operation'].get('scalar', 1.0)
            if matrix:
                scalar_mul_op = ScalarMulOp(matrix, scalar)
                block.add_op(scalar_mul_op)
                result = scalar_mul_op.result
                
        elif op_type == 'add':
            left = var_map.get(op_info['operation']['left'])
            right_name = op_info['operation']['right']
            
            # Handle B.T (B transpose)
            if right_name == 'B.T':
                right = var_map.get('B')
                if right:
                    # Create transpose of B
                    transpose_b = TransposeOp(right)
                    block.add_op(transpose_b)
                    right = transpose_b.result
            else:
                right = var_map.get(right_name)
            
            if left and right:
                add_op = AddOp(left, right)
                block.add_op(add_op)
                result = add_op.result
                
        elif op_type == 'var':
            # Simple variable assignment
            source = var_map.get(op_info['operation']['name'])
            if source:
                result = source
                
        elif op_type == 'alloc':
            # Matrix allocation with literal values
            shape = op_info['operation']['shape']
            if shape and len(shape) >= 2:
                rows, cols = shape[0], shape[1]
                # Create an allocation operation
                alloc_op = AllocOp(rows, cols, Float32Type())
                block.add_op(alloc_op)
                result = alloc_op.result
                # TODO: Initialize with values if needed
                
        elif op_type == 'call':
            # Function call - for now, handle matrix_computation specially
            func_name = op_info['operation']['function']
            args = op_info['operation']['args']
            
            if func_name == 'matrix_computation' and len(args) >= 2:
                # Get the arguments
                arg_vals = [var_map.get(arg) for arg in args]
                if all(arg_vals):
                    # Create a function call operation
                    # For simplicity, we'll inline the result
                    # In a real compiler, we'd generate a call instruction
                    # For now, just use the first argument as placeholder
                    result = arg_vals[0]
        
        # Store result in variable map
        if result:
            var_map[target] = result
            
        return result