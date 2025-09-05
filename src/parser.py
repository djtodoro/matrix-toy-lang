import ast
from typing import List, Dict, Any

class MatrixOperationExtractor(ast.NodeVisitor):
    """Extract matrix operations from Python AST."""
    
    def __init__(self):
        self.functions = {}  # Store operations by function name
        self.current_function = None
        self.variables = {}
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process function definition."""
        self.current_function = node.name
        self.functions[node.name] = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'operations': []
        }
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
        
        self.current_function = None
    
    def visit_Assign(self, node: ast.Assign):
        """Process assignment statements."""
        if not self.current_function:
            return
            
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            
            # Analyze the right-hand side
            op_info = self.analyze_expression(node.value)
            
            operation = {
                'type': 'assign',
                'target': var_name,
                'operation': op_info
            }
            
            self.functions[self.current_function]['operations'].append(operation)
            
            self.variables[var_name] = {
                'type': 'matrix',
                'defined_by': op_info
            }
    
    def visit_Return(self, node: ast.Return):
        """Process return statement."""
        if not self.current_function or not node.value:
            return
            
        op_info = self.analyze_expression(node.value)
        operation = {
            'type': 'return',
            'operation': op_info
        }
        self.functions[self.current_function]['operations'].append(operation)
    
    def visit_Expr(self, node: ast.Expr):
        """Process expression statements (like print calls)."""
        if not self.current_function:
            return
            
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
                # For now, skip print statements
                pass
    
    def analyze_expression(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze expression to identify matrix operations."""
        
        # Matrix multiplication (A @ B)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return {
                'op': 'matmul',
                'left': self.get_var_name(node.left),
                'right': self.get_var_name(node.right)
            }
        
        # Element-wise multiplication (A * scalar)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return {
                'op': 'scalar_mul',
                'matrix': self.get_var_name(node.left),
                'scalar': self.get_value(node.right)
            }
        
        # Transpose (A.T)
        elif isinstance(node, ast.Attribute) and node.attr == 'T':
            base = self.analyze_expression(node.value)
            
            # Check for double transpose pattern
            if base.get('op') == 'transpose':
                return {
                    'op': 'double_transpose',
                    'matrix': base['matrix']
                }
            else:
                return {
                    'op': 'transpose',
                    'matrix': self.get_var_name(node.value)
                }
        
        # Matrix addition (A + B)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {
                'op': 'add',
                'left': self.get_var_name(node.left),
                'right': self.get_var_name(node.right)
            }
        
        # Variable reference
        elif isinstance(node, ast.Name):
            return {
                'op': 'var',
                'name': node.id
            }
        
        # Matrix allocation (matrix([[1, 2], [3, 4]]))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'matrix':
                # Extract matrix literal values
                if node.args and isinstance(node.args[0], ast.List):
                    rows = []
                    for row in node.args[0].elts:
                        if isinstance(row, ast.List):
                            row_values = [self.get_value(elem) for elem in row.elts]
                            rows.append(row_values)
                    
                    return {
                        'op': 'alloc',
                        'shape': [len(rows), len(rows[0]) if rows else 0],
                        'values': rows
                    }
            # Function call (like matrix_computation)
            elif isinstance(node.func, ast.Name):
                args = [self.get_var_name(arg) for arg in node.args]
                return {
                    'op': 'call',
                    'function': node.func.id,
                    'args': args
                }
        
        return {'op': 'unknown', 'node': ast.dump(node)}
    
    def get_var_name(self, node: ast.AST) -> str:
        """Get variable name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_var_name(node.value) + '.' + node.attr
        return 'unknown'
    
    def get_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        return None

def parse_matrix_program(source_code: str) -> Dict[str, Any]:
    """Parse Python source code and extract matrix operations by function."""
    tree = ast.parse(source_code)
    extractor = MatrixOperationExtractor()
    extractor.visit(tree)
    return extractor.functions