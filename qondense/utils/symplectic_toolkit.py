import numpy as np
from typing import List, Dict, Tuple


def number_of_qubits(operator:Dict[str, float]) -> int:
    """ Extract number of qubits from operator in dictionary representation
    Enforces that each term has same length
    Will also accept a list of Pauli operators!
    """
    qubits_numbers = set([len(pauli) for pauli in operator])
    assert(len(qubits_numbers)==1) # each pauli must be same length
    num_qubits = list(qubits_numbers)[0]

    return num_qubits


def pauli_to_symplectic(p_str: str) -> np.array:
    """Convert Pauli string to symplectic representation
    e.g. "XXYY" -> np.array([1. 1. 1. 1. 0. 0. 1. 1.])
    """
    num_qubits = len(p_str)
    p_sym = np.zeros(2*num_qubits)
    for index,p in enumerate(p_str):
        if p=='X':
            p_sym[index]=1
        elif p=='Z':
            p_sym[index+num_qubits]=1
        elif p=='Y':
            p_sym[index]=1
            p_sym[index+num_qubits]=1
    
    return np.array(p_sym, dtype=int).reshape(1, 2*num_qubits)


def pauli_from_symplectic(p_sym: np.array) -> str:
    """Convert symplectic representation of Pauli operator to string
    e.g. np.array([0. 1. 0. 1. 1. 0. 0. 1.]) -> ZXIY
    """
    num_qubits = len(p_sym)//2
    p_str = ['I' for i in range(num_qubits)]
    for i in range(num_qubits):
        if p_sym[i]==1:
            if p_sym[i+num_qubits]==0:
                p_str[i]='X'
            else:
                p_str[i]='Y'
        else:
            if p_sym[i+num_qubits]==1:
                p_str[i]='Z'
    
    return ''.join(p_str)


def build_symplectic_matrix(p_list: List[str]) -> np.array:
    """Stack of paulis in symplectic form
    One matrix row per pauli term, number of columns in 2*num_qubits
    """
    num_qubits = number_of_qubits({P:1 for P in p_list})
    p_sym_list = [pauli_to_symplectic(p) for p in p_list]
    sym_mat = np.array(np.stack(p_sym_list), dtype=int).reshape(len(p_list), 2*num_qubits)

    return sym_mat


def symplectic_operator(operator: Dict[str, float])->Tuple[np.array, np.array]:
    """ Converts operator stored as dict to syplectic representation
    """
    num_qubits = number_of_qubits(operator)
    pauli_terms, coeffs = zip(*operator.items())
    Pterms = build_symplectic_matrix(pauli_terms)
    coeffs = np.array(coeffs).reshape(len(operator), 1)

    return Pterms, coeffs


def dictionary_operator(Pterms: np.array, coeffvec: np.array):
    """ Converts operator stored in symplectic representation to dictionary form
    """
    op_dict = {}
    for P, coeff in zip(Pterms, coeffvec):
        P_str = pauli_from_symplectic(P)
        op_dict[P_str] = coeff[0]

    return op_dict


def symplectic_inner_product(P:str,Q:str) -> int:
    """If 0 is returned then P and Q commute, else they anticommute
    """
    assert(len(P)==len(Q))
    num_qubits = len(P)

    P_sym = pauli_to_symplectic(P)
    Q_sym = pauli_to_symplectic(Q)

    half_sym_form = np.eye(2*num_qubits,2*num_qubits,num_qubits)
    sym_form = np.array(half_sym_form - half_sym_form.T, dtype=int)

    return P_sym@sym_form@Q_sym.T % 2
    

def adjacency_matrix(p_list: List[str], 
                    num_qubits: int
                    ) -> np.matrix:
    """Adjacency matrix of pauli list w.r.t. commutation
    if entry i,j == 0 then pauli i and j commute, elif == 1 they anticommute
    """
    sym_mat  = build_symplectic_matrix(p_list)
    half_sym_form = np.eye(2*num_qubits,2*num_qubits,num_qubits)
    sym_form = np.array(half_sym_form + half_sym_form.T, dtype=int)
    
    return sym_mat@sym_form@sym_mat.T % 2


def cleanup_symplectic(terms, coeff):    
    """ Remove duplicated rows of symplectic matrix terms, whilst summing the corresponding
    coefficients of the deleted rows in coeff
    """ 
    # order the pauli terms
    sort_order = np.lexsort(terms.T)
    sorted_terms = terms[sort_order,:]
    sorted_coeff = coeff[sort_order,:]
    
    # take difference between adjacent terms and drop 0 rows (duplicates)
    row_mask = np.append([True],np.any(np.diff(sorted_terms,axis=0),1))
    out_terms = sorted_terms[row_mask]
    
    # sum coefficients of like-terms
    #if row_mask[-1] == False:
    row_mask = np.append(row_mask, True)
    mask_indices = np.where(row_mask==True)[0]
    mask_diff = np.diff(mask_indices)
    out_coeff = []
    for i, d in zip(mask_indices, mask_diff):
        dup_term_sum = np.sum([sorted_coeff[i+j] for j in range(d)])
        out_coeff.append([dup_term_sum])
        
    out_coeff = np.stack(out_coeff)
    
    return out_terms,  out_coeff


def gf2_gaus_elim(gf2_matrix: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    pseudocode: http://dde.binghamton.edu/filler/mct/hw/1/assignment.pdf

    Args:
        gf2_matrix (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        gf2_matrix_rref (np.array): reduced row echelon form of M
    """
    gf2_matrix_rref = gf2_matrix.copy()
    m_rows, n_cols = gf2_matrix_rref.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        if sum(gf2_matrix_rref[row_i:, col_j]) == 0:
            # case when col_j all zeros
            # No pivot in this column, pass to next column
            col_j += 1
            continue

        # find index of row with first "1" in the vector defined by column j (note previous if statement removes all zero column)
        k = np.argmax(gf2_matrix_rref[row_i:, col_j]) + row_i
        # + row_i gives correct index (as we start search from row_i!)

        # swap row k and row_i (row_i now has 1 at top of column j... aka: gf2_matrix_rref[row_i, col_j]==1)
        gf2_matrix_rref[[k, row_i]] = gf2_matrix_rref[[row_i, k]]
        # next need to zero out all other ones present in column j (apart from on the i_row!)
        # to do this use row_i and use modulo addition to zero other columns!

        # make a copy of j_th column of gf2_matrix_rref, this includes all rows (0 -> M)
        Om_j = np.copy(gf2_matrix_rref[:, col_j])

        # zero out the i^th position of vector Om_j (this is why copy needed... to stop it affecting gf2_matrix_rref)
        Om_j[row_i] = 0
        # note this was orginally 1 by definition...
        # This vector now defines the indices of the rows we need to zero out
        # by setting ith position to zero - it stops the next steps zeroing out the i^th row (which we need as our pivot)


        # next from row_i of rref matrix take all columns from j->n (j to last column)
        # this is vector of zero and ones from row_i of gf2_matrix_rref
        i_jn = gf2_matrix_rref[row_i, col_j:]
        # we use i_jn to zero out the rows in gf2_matrix_rref[:, col_j:] that have leading one (apart from row_i!)
        # which rows are these? They are defined by that Om_j vector!

        # the matrix to zero out these rows is simply defined by the outer product of Om_j and i_jn
        # this creates a matrix of rows of i_jn terms where Om_j=1 otherwise rows of zeros (where Om_j=0)
        Om_j_dependent_rows_flip = np.einsum('i,j->ij', Om_j, i_jn, optimize=True)
        # note flip matrix is contains all m rows ,but only j->n columns!

        # perfrom bitwise xor of flip matrix to zero out rows in col_j that that contain a leading '1' (apart from row i)
        gf2_matrix_rref[:, col_j:] = np.bitwise_xor(gf2_matrix_rref[:, col_j:], Om_j_dependent_rows_flip)

        row_i += 1
        col_j += 1

    return gf2_matrix_rref


def gf2_basis_for_gf2_rref(gf2_matrix_in_rreform: np.array) -> np.array:
    """
    Function that gets the kernel over GF2(2) of ow reduced  gf2 matrix!

    uses method in: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Basis

    Args:
        gf2_matrix_in_rreform (np.array): GF(2) matrix in row reduced form
    Returns:
        basis (np.array): basis for gf2 input matrix that was in row reduced form
    """
    rows_to_columns = gf2_matrix_in_rreform.T
    eye = np.eye(gf2_matrix_in_rreform.shape[1], dtype=int)

    # do column reduced form as row reduced form
    rrf = gf2_gaus_elim(np.hstack((rows_to_columns, eye.T)))

    zero_rrf = np.where(~rrf[:, :gf2_matrix_in_rreform.shape[0]].any(axis=1))[0]
    basis = rrf[zero_rrf, gf2_matrix_in_rreform.shape[0]:]

    return basis