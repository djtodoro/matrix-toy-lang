"""
Matrix operations program to be compiled.
This program demonstrates various matrix operations including
the inefficient double transpose pattern that should be optimized.
"""

import numpy as np

def matrix_computation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix computations with optimization opportunities.
    
    Args:
        A: First input matrix (m x n)
        B: Second input matrix (n x p)
    
    Returns:
        Result matrix after operations
    """
    # Matrix multiplication
    C = A @ B
    
    # Inefficient double transpose (should be optimized away)
    D = C.T.T  # This is equivalent to just C
    
    # Element-wise operations
    E = D * 2.0
    
    # Another double transpose pattern (hidden in sequence)
    F = E.T
    G = F.T  # G should be equal to E
    
    # Matrix addition
    if G.shape == B.T.shape:
        H = G + B.T
    else:
        H = G
    
    # Final double transpose (for demonstration)
    result = H.T.T
    
    return result

# Example usage
if __name__ == "__main__":
    # Create sample matrices
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3 matrix
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])  # 3x2 matrix
    
    # Run computation
    result = matrix_computation(A, B)
    print("Result shape:", result.shape)
    print("Result:\n", result)
