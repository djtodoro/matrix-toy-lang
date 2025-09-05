import ast
from typing import List, Dict, Any

class MatrixOperationExtractor(ast.NodeVisitor):
    """Extract matrix operations from Python AST."""
    
    def __init__(self):
        self.operations = []
        self.variables = {}
        self.current_function = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process function definition."""
        self.current_function = node.name
        self.extract_parameters(node.args)
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
    
    def extract_parameters(self, args: ast.arguments):
        """Extract function parameters as matrix variables."""
        for arg in args.args:
            self.variables[arg.arg] = {
                'type': 'matrix',
                'shape': 'unknown'  # Will be inferred or annotated
            }
    
    def visit_Assign(self, node: ast.Assign):
        """Process assignment statements."""
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            
            # Analyze the right-hand side
            op_info = self.analyze_expression(node.value)
            
            self.operations.append({
                'type': 'assign',
                'target': var_name,
                'operation': op_info
            })
            
            self.variables[var_name] = {
                'type': 'matrix',
                'defined_by': op_info
            }
    
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

def parse_matrix_program(source_code: str) -> List[Dict[str, Any]]:
    """Parse Python source code and extract matrix operations."""
    tree = ast.parse(source_code)
    extractor = MatrixOperationExtractor()
    extractor.visit(tree)
    return extractor.operations
