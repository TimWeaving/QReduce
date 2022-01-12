import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Union

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


def build_symplectic_matrix(p_list: List[str]) -> np.matrix:
    """Stack of paulis in symplectic form
    One matrix row per pauli term, number of columns in 2*num_qubits
    """
    p_sym_list = [pauli_to_symplectic(p) for p in p_list]
    sym_mat = np.array(np.stack(p_sym_list), dtype=int)

    return sym_mat


def symplectic_inner_product(P,Q):
    """If 0 is returned then P and Q commute, else they anticommute
    """
    assert(len(P)==len(Q))
    num_qubits = len(P)

    P_sym = pauli_to_symplectic(P)
    Q_sym = pauli_to_symplectic(Q)

    half_sym_form = np.eye(2*num_qubits,2*num_qubits,num_qubits)
    sym_form = np.array(half_sym_form - half_sym_form.T, dtype=int)

    return P_sym@sym_form@Q_sym.T % 2
    

def adjacency_matrix(p_list: List[str], num_qubits: int) -> np.matrix:
    """Adjacency matrix of pauli list w.r.t. commutation
    if entry i,j == 0 then pauli i and j commute, elif == 1 they anticommute
    """
    sym_mat  = build_symplectic_matrix(p_list)
    half_sym_form = np.eye(2*num_qubits,2*num_qubits,num_qubits)
    sym_form = np.array(half_sym_form + half_sym_form.T, dtype=int)
    
    return sym_mat@sym_form@sym_mat.T % 2


def multiply_paulis(P: str, Q: str) -> str:
    """Multiply two Pauli strings via their sympletic representation
    """
    P_sym = pauli_to_symplectic(P)
    Q_sym = pauli_to_symplectic(Q)
    PQ = pauli_from_symplectic((P_sym+Q_sym)%2)

    return PQ


def multiply_pauli_list(pauli_list: List[str]) -> str:
    """Multiply a list of Pauli strings via their sympletic representation
    """
    pauli_list_sym = [pauli_to_symplectic(P) for P in pauli_list]
    Prod = pauli_from_symplectic(sum(pauli_list_sym)%2)

    return Prod


def multiply_paulis_with_coeff(P,Q):
    i_factors=[]
    for p,q in zip(P,Q):
        if p=='X':
            if q=='Y':
                i_factors.append(1j)
            elif q=='Z':
                i_factors.append(-1j)
        elif p=='Y':
            if q=='Z':
                i_factors.append(1j)
            elif q=='X':
                i_factors.append(-1j)
        elif p=='Z':
            if q=='X':
                i_factors.append(1j)
            elif q=='Y':
                i_factors.append(-1j)
    
    coeff = np.prod(i_factors)
    PQ = multiply_paulis(P,Q)
    
    return PQ, coeff 


def pauli_matrix(pauli):
    num_qubits = len(pauli)
    single_paulis ={'I': np.matrix(np.identity(2)),
                    'X': np.matrix([[0, 1],
                                    [1, 0]]),
                    'Y': np.matrix([[0,-1.j],
                                    [1.j, 0]]),
                    'Z': np.matrix([[1, 0],
                                    [0,-1]])}
    
    pauli_matrix = 1
    for p in pauli:
        pauli_matrix = np.kron(pauli_matrix, single_paulis[p])

    return pauli_matrix


def apply_rotation( P:str,
                    Q:str,
                    coeff:float=1,
                    t:float=None, 
                    gen_rot:bool=False
                    ) -> Dict[str,float]:
    """ Let R(t) = e^{i t/2 Q}, then one of the following can occur:
    R(t) P R^\dag(t) = P when [P,Q] = 0
    R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0
    For the generator rotations we have t=pi/2 so cos(pi/2) P - sin(pi/2) iPQ = -iPQ (Hence Clifford!)
    In unitary partitioning t will not be so nice and will result in an increase in the number of terms (non-Clifford)
    """
    commute = symplectic_inner_product(P, Q) == 0  # bool

    if commute:
        return {P: coeff}
    else:
        PQ, i_factor = multiply_paulis_with_coeff(P, Q)
        sign = int((-1j * i_factor).real)
        if gen_rot:
            return {PQ: sign*coeff}
        else:
            return {P: np.cos(t)*coeff,
                    PQ:np.sin(t)*sign*coeff}


def sum_operators(operators:List[Dict[str,float]])->Dict[str,float]:
    """Take in a list of operators stored as dictionaries and combine
    like-terms to obtain the summed operator
    """
    op_out={}
    for op in operators:
        for pauli,coeff in op.items():
            if pauli not in op_out:
                op_out[pauli] = coeff
            else:
                op_out[pauli]+= coeff
    
    return op_out


def cleanup_operator(operator:Dict[str,float], threshold:int=15):
    """Drop Pauli terms with negligible coefficients
    """
    op_out={pauli:round(coeff, threshold) for pauli,coeff in operator.items() if abs(coeff)>0.1**threshold}
    
    return op_out


def rotate_operator(
                    operator:Dict[str,float], 
                    rotations:List[Tuple[str,float,bool]],
                    cleanup:bool=True
                    )->Dict[str,float]:
    
    rotated_operator=deepcopy(operator)
    for rot,angle,gen_flag in rotations:
        rotated_paulis = []
        for pauli,coeff in rotated_operator.items():
            rotated_paulis.append(apply_rotation(   
                                                    P=pauli,
                                                    Q=rot,
                                                    coeff=coeff,
                                                    t=angle,
                                                    gen_rot=gen_flag
                                                )
                                )
        rotated_operator = sum_operators(rotated_paulis)
    
    if cleanup:
        rotated_operator=cleanup_operator(rotated_operator)
    
    return rotated_operator


def amend_string_index(string:Union[str,list], index:int, character:str)->str:
    """Update a string at a given index with some character 
    """
    listed = list(deepcopy(string))
    listed[index] = character
    return ''.join(listed)


def exact_gs_energy(ham:Dict[str, float]):
    ham_mat = sum(coeff * pauli_matrix(op) for op, coeff in ham.items())
    gs_energy = sorted(np.linalg.eigh(ham_mat)[0])[0]
    return gs_energy




