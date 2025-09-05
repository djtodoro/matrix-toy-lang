from xdsl.builder import Builder
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from typing import Dict, List, Any

class MatrixIRGenerator:
    """Generate matrix dialect IR from parsed operations."""
    
    def __init__(self):
        self.builder = Builder()
        self.var_map: Dict[str, SSAValue] = {}
    
    def generate(self, operations: List[Dict[str, Any]]) -> builtin.ModuleOp:
        """Generate IR module from operations list."""
        
        # Create function type (simplified: 2 matrices in, 1 out)
        func_type = func.FunctionType.from_lists(
            [MatrixType(None, None, Float32Type()),  # A
             MatrixType(None, None, Float32Type())], # B
            [MatrixType(None, None, Float32Type())]  # Result
        )
        
        # Create function
        with func.FuncOp("matrix_computation", func_type).body as body:
            # Map parameters to variables
            self.var_map['A'] = body.block.args[0]
            self.var_map['B'] = body.block.args[1]
            
            # Generate operations
            for op_info in operations:
                self.generate_operation(op_info)
            
            # Return final result
            result = self.var_map.get('result', self.var_map['A'])
            func.Return(result)
        
        # Create module
        return builtin.ModuleOp([body.owner])
    
    def generate_operation(self, op_info: Dict[str, Any]):
        """Generate IR for a single operation."""
        op_type = op_info['operation']['op']
        target = op_info['target']
        
        if op_type == 'matmul':
            left = self.var_map[op_info['operation']['left']]
            right = self.var_map[op_info['operation']['right']]
            result = MatMulOp(left, right).result
            self.var_map[target] = result
        
        elif op_type == 'transpose':
            matrix = self.var_map[op_info['operation']['matrix']]
            result = TransposeOp(matrix).result
            self.var_map[target] = result
        
        elif op_type == 'double_transpose':
            # This is where optimization opportunity exists!
            # For now, generate two transposes (will be optimized later)
            matrix = self.var_map[op_info['operation']['matrix']]
            temp = TransposeOp(matrix).result
            result = TransposeOp(temp).result
            self.var_map[target] = result
        
        elif op_type == 'scalar_mul':
            matrix = self.var_map[op_info['operation']['matrix']]
            scalar = op_info['operation']['scalar']
            result = ScalarMulOp(matrix, scalar).result
            self.var_map[target] = result
        
        elif op_type == 'add':
            left = self.var_map[op_info['operation']['left']]
            right = self.var_map[op_info['operation']['right']]
            result = AddOp(left, right).result
            self.var_map[target] = result
