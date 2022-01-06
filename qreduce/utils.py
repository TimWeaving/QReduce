import numpy as np
from typing import List, Dict

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
    
    return p_sym


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


def build_symplectic_matrix(p_list: List[str], num_qubits: int) -> np.matrix:
    """Stack of paulis in symplectic form
    One matrix row per pauli term, number of columns in 2*num_qubits
    """
    p_sym_list = [pauli_to_symplectic(p) for p in p_list]
    sym_mat = np.matrix(np.stack(p_sym_list))

    return sym_mat


def adjacency_matrix(p_list: List[str], num_qubits: int) -> np.matrix:
    """Adjacency matrix of pauli list w.r.t. commutation
    if entry i,j == 0 then pauli i and j commute, elif == 1 they anticommute
    """
    sym_mat  = build_symplectic_matrix(p_list, num_qubits)
    half_sym_form = np.eye(2*num_qubits,2*num_qubits,num_qubits)
    sym_form = np.matrix(half_sym_form + half_sym_form.T)
    
    return sym_mat@sym_form@sym_mat.T % 2