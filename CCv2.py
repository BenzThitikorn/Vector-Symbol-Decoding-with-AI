import numpy as np
import Channel_Coding as cc
def Compute_Gauss_Jordan_Reduction_Optimized(Original_matrix):
    Binary_matrix = np.array(Original_matrix, dtype=int)
    index_operation_col = 0
    rows = 0
    Index_Rows = []
    num_rows, num_cols = Binary_matrix.shape

    while rows < num_rows and index_operation_col < num_cols:
        columns_one = np.where(Binary_matrix[rows, :] == 1)[0]
        if len(columns_one) > 0:
            if index_operation_col in columns_one:
                other_indices = columns_one[columns_one != index_operation_col]
                if len(other_indices) > 0:
                    Binary_matrix[:, other_indices] ^= Binary_matrix[:, [index_operation_col]]
                Index_Rows.append(rows)
                index_operation_col += 1
                rows += 1
            else:
                change_index_col = columns_one[columns_one > index_operation_col]
                if len(change_index_col) > 0:
                    Binary_matrix[:, [index_operation_col, change_index_col[0]]] = Binary_matrix[:, [change_index_col[0], index_operation_col]]
                else:
                    rows += 1
        else:
            rows += 1
    return Binary_matrix, index_operation_col, Index_Rows

def VSD_normal_get0fast(H,Y):
    Y_Binary = np.array(Y, dtype=int)

    # Compute Syndrome Matrix
    S_Binary = (np.dot(H, Y)%2).astype(int)
    S_Gauss, S_rank, Index_Rows = Compute_Gauss_Jordan_Reduction_Optimized(S_Binary)

    # Compute Error Locating Vector
    Error_Locating_Vector = cc.Compute_Error_Locating_Vector(S_Gauss, Index_Rows, H)

    # Find Number of Erroneous Symbols
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)

    if Number_Error == 0:
        return Y, 1  # No errors detected

    # Check if Rank of S matches Number of Erroneous Symbols
    if S_rank == Number_Error:
        S_Sub = S_Binary[Index_Rows, :]
        Position_Error = np.where(Error_Locating_Vector == 0)[0]
        H_Sub = H[np.ix_(Index_Rows, Position_Error)]

        if np.linalg.det(H_Sub) == 0:
            return Y_Binary, 0  # Cannot correct errors


        return 0,   1  # Successful correction
    else:
        return 0,   0  # Unable to correct errors