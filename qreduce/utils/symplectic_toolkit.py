import numpy as np
from typing import List, Dict, Tuple


def number_of_qubits(operator:Dict[str, float]) -> int:
    """ Extract number of qubits from operator in dictionary representation
    Enforces that each term has same length
    """
    qubits_numbers = set([len(pauli) for pauli in operator.keys()])
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