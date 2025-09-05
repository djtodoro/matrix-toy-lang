from xdsl.irdl import (
    IRDLOperation,
    OpResult,
    Operand,
    irdl_op_definition,
    irdl_attr_definition,
    operand_def,
    result_def,
    attr_def,
    ParametrizedAttribute
)
from xdsl.dialects.builtin import (
    TensorType,
    Float32Type,
    IntAttr,
    ArrayAttr,
    TypeAttribute,
    FloatAttr
)
from xdsl.ir import Dialect, Operation, SSAValue
from typing import List, Optional

# Define the Matrix dialect
@irdl_attr_definition
class MatrixType(ParametrizedAttribute, TypeAttribute):
    """Type for matrix with known dimensions."""
    name = "matrix.type"
    
    rows: IntAttr
    cols: IntAttr
    dtype: TypeAttribute  # f32, f64, i32, etc.
    
    def __init__(self, rows: int, cols: int, dtype: TypeAttribute):
        super().__init__(
            IntAttr(rows),
            IntAttr(cols),
            dtype
        )

@irdl_op_definition
class MatMulOp(IRDLOperation):
    """Matrix multiplication operation."""
    name = "matrix.matmul"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        # Infer result type from operand shapes
        lhs_type = lhs.type
        rhs_type = rhs.type
        result_type = MatrixType(
            lhs_type.parameters[0].data,
            rhs_type.parameters[1].data,
            lhs_type.parameters[2]
        )
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type]
        )
    
    def verify(self):
        """Verify that matrix dimensions are compatible."""
        if self.lhs.type.parameters[1].data != self.rhs.type.parameters[0].data:
            raise ValueError(
                f"Incompatible matrix dimensions for multiplication: "
                f"({self.lhs.type.parameters[0].data}x{self.lhs.type.parameters[1].data}) @ "
                f"({self.rhs.type.parameters[0].data}x{self.rhs.type.parameters[1].data})"
            )

@irdl_op_definition
class TransposeOp(IRDLOperation):
    """Matrix transpose operation."""
    name = "matrix.transpose"
    
    input = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, input: SSAValue):
        input_type = input.type
        # Transpose swaps rows and columns
        result_type = MatrixType(
            input_type.parameters[1].data,
            input_type.parameters[0].data,
            input_type.parameters[2]
        )
        super().__init__(
            operands=[input],
            result_types=[result_type]
        )

@irdl_op_definition
class ScalarMulOp(IRDLOperation):
    """Element-wise scalar multiplication."""
    name = "matrix.scalar_mul"
    
    matrix = operand_def(MatrixType)
    scalar = attr_def(FloatAttr)
    result = result_def(MatrixType)
    
    def __init__(self, matrix: SSAValue, scalar: float):
        super().__init__(
            operands=[matrix],
            attributes={'scalar': FloatAttr(scalar, Float32Type())},
            result_types=[matrix.type]  # Same shape as input
        )

@irdl_op_definition
class AddOp(IRDLOperation):
    """Element-wise matrix addition."""
    name = "matrix.add"
    
    lhs = operand_def(MatrixType)
    rhs = operand_def(MatrixType)
    result = result_def(MatrixType)
    
    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type]  # Same shape as inputs
        )
    
    def verify(self):
        """Verify that matrices have the same shape."""
        if (self.lhs.type.parameters[0].data != self.rhs.type.parameters[0].data or
            self.lhs.type.parameters[1].data != self.rhs.type.parameters[1].data):
            raise ValueError("Matrix addition requires same dimensions")

@irdl_op_definition
class AllocOp(IRDLOperation):
    """Allocate matrix in memory."""
    name = "matrix.alloc"
    
    shape = attr_def(ArrayAttr)
    result = result_def(MatrixType)
    
    def __init__(self, rows: int, cols: int, dtype: TypeAttribute):
        matrix_type = MatrixType(rows, cols, dtype)
        super().__init__(
            operands=[],
            attributes={'shape': ArrayAttr([IntAttr(rows), IntAttr(cols)])},
            result_types=[matrix_type]
        )

class MatrixDialect(Dialect):
    """Dialect for matrix operations."""
    name = "matrix"
    
    def __init__(self):
        super().__init__("matrix")
        
    @property
    def operations(self):
        return [
            MatMulOp,
            TransposeOp,
            ScalarMulOp,
            AddOp,
            AllocOp
        ]
