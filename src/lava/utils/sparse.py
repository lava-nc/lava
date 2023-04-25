from scipy.sparse import csr_matrix, find


def find_with_explicit_zeros(mat: csr_matrix):
    """Behaves as find but also returns explict zeros."""
    idx = mat.data == 0

    mat.data[idx] = 1
    dst, src, _ = find(mat)
    mat.data[idx] = 0

    vals = mat[dst, src].A1

    return dst, src, vals
