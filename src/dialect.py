from xdsl.irdl import (
    IRDLOperation,
    OpResult,
    Operand,
    irdl_op_definition,
    irdl_attr_definition,
    operand_def,
    result_def,
    attr_def
)
from xdsl.dialects.builtin import (
    TensorType,
    Float32Type,
    IntAttr,
    ArrayAttr
)
from xdsl.ir import Dialect, Operation, SSAValue
from typing import List, Optional

# Define the Matrix dialect
@irdl_attr_definition
class MatrixType(TypeAttribute):
    """Type for matrix with known dimensions."""
    name = "matrix.type"
    
    rows: IntAttr
    cols: IntAttr
    dtype: TypeAttribute  # f32, f64, i32, etc.
    
    def __init__(self, rows: int, cols: int, dtype: TypeAttribute):
        self.rows = IntAttr(rows)
        self.cols = IntAttr(cols)
        self.dtype = dtype

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
            lhs_type.rows.value,
            rhs_type.cols.value,
            lhs_type.dtype
        )
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type]
        )
    
    def verify(self):
        """Verify that matrix dimensions are compatible."""
        if self.lhs.type.cols.value != self.rhs.type.rows.value:
            raise ValueError(
                f"Incompatible matrix dimensions for multiplication: "
                f"({self.lhs.type.rows.value}x{self.lhs.type.cols.value}) @ "
                f"({self.rhs.type.rows.value}x{self.rhs.type.cols.value})"
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
            input_type.cols.value,
            input_type.rows.value,
            input_type.dtype
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
    scalar = attr_def(Float32Type)
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
        if (self.lhs.type.rows.value != self.rhs.type.rows.value or
            self.lhs.type.cols.value != self.rhs.type.cols.value):
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
    
    operations = [
        MatMulOp,
        TransposeOp,
        ScalarMulOp,
        AddOp,
        AllocOp
    ]
