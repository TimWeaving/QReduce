import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from matplotlib import pyplot as plt
from itertools import permutations, product, combinations
from qreduce.utils.symplectic_toolkit import *
plt.style.use('ggplot')


def multiply_paulis(P: str, Q: str) -> str:
    """Multiply two Pauli strings via their sympletic representation
    <!> Disregards the coefficient, refer to multiply_paulis_with_coeff if needed
    """
    P_sym = pauli_to_symplectic(P)
    Q_sym = pauli_to_symplectic(Q)
    PQ_sym = (P_sym+Q_sym)%2
    PQ = pauli_from_symplectic(PQ_sym[0])
    
    return PQ


def multiply_pauli_list(pauli_list: List[str]) -> str:
    """ Multiply a list of Pauli strings via their sympletic representation
    <!> Disregards the coefficient, refer to multiply_paulis_with_coeff if needed
    """
    pauli_list_sym = [pauli_to_symplectic(P) for P in pauli_list]
    Prod = pauli_from_symplectic((sum(pauli_list_sym)%2)[0])

    return Prod


def multiply_paulis_with_coeff( P: str,
                                Q:str
                                )->Tuple[str, float]:
    """ Keep track of the coefficient when multiplying pauli operators
    """
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


def pauli_matrix(pauli:str) -> np.array:
    """ Convert a tensor product of paulis to numpy array
    """
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


def sum_operators(operators:List[Dict[str,float]]
                    )->Dict[str,float]:
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


def cleanup_operator(operator:Dict[str,float], 
                    threshold:int=15
                    ) -> Dict[str, float]:
    """Drop Pauli terms with negligible coefficients
    """
    op_out={pauli:round(coeff, threshold) for pauli,coeff in operator.items() if abs(coeff)>0.1**threshold}
    
    return op_out


def rotate_operator(
                    operator:Dict[str,float], 
                    rotations:List[Tuple[str,float,bool]],
                    cleanup:bool=True
                    )->Dict[str,float]:
    """ Applies a list of rotations of the form (pauli, angle, pi/2_flag)
    to the input operator. If cleanup is True then terms with negligible
    coefficients are dropped - this behaviour is not always desirable,
    for example when rotating an Ansatz operator prior to projection since
    the coefficients correspond with angles in this case, not Hamiltonian weights.
    """
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


def amend_string_index( string:Union[str,list], 
                        index:int, character:str
                        )->str:
    """Update a string at a given index with some character 
    """
    listed = list(deepcopy(string))
    listed[index] = character
    return ''.join(listed)


def exact_gs_energy(operator:Dict[str, float]
                    ) -> Tuple[float, np.array]:
    """ Return the ground state energy and corresponding ground state
    vector for the input operator
    """
    ham_mat = sum(coeff * pauli_matrix(op) for op, coeff in operator.items())
    eigvals, eigvecs = np.linalg.eigh(ham_mat)
    ground_energy, ground_state = sorted(zip(eigvals,eigvecs), key=lambda x:x[0])[0]

    return ground_energy, np.array(ground_state)


def plot_ground_state_amplitudes(operator: Dict[str, float], 
                                num_qubits: int, 
                                reverse_bitstrings: bool=False
                                )-> None:
    """ Prints a barplot of the probability amplitudes for each 
    basis state in the ground eigenstate of the input operator
    """
    cs_energy, cs_vector = exact_gs_energy(operator)
    bitstrings = [format(index, f'0{num_qubits}b') for index in range(2**(num_qubits))]
    if reverse_bitstrings:
        bitstrings.reverse()
    amps = [(b_str, amp) for b_str,amp 
            in zip(bitstrings, np.square(abs(cs_vector)[0])) if amp>1e-10]
    amps = sorted(amps, key=lambda x:-x[1])
    X, Y = zip(*amps)
    
    # plot and show the amplitudes
    plt.bar(X, Y)
    plt.xlabel('Basis state')
    plt.ylabel('Amplitude in ground state')
    plt.title(f'Energy = {cs_energy: .10f}')
    plt.xticks(rotation=90)
    plt.show()
    

def number_of_qubits(operator:Dict[str, float]) -> int:
    """ Extract number of qubits from operator in dictionary representation
    Enforces that each term has same length
    """
    qubits_numbers = set([len(pauli) for pauli in operator.keys()])
    assert(len(qubits_numbers)==1) # each pauli must be same length
    num_qubits = list(qubits_numbers)[0]

    return num_qubits


def simultaneous_eigenstates(stabilizers: Dict[str,float])->List[str]:
    """
    """
    all_eigenstates = {}

    for op,eigval in stabilizers.items():

        parity = (1-eigval)//2
        num_Z = op.count('Z')
        possible_states = []

        for i in range((num_Z)//2+1):
            num_1 = 2*i + parity
            if num_1<=num_Z:
                I_indices = [i for i,P in enumerate(op) if P=='I']
                Z_indices = [i for i,P in enumerate(op) if P=='Z']

                init_state = ['1' for i in range(num_1)]+['0' for i in range(num_Z-num_1)]
                Z_bit_vals = list(set(permutations(init_state)))
                I_bit_vals = list(product(['0', '1'], repeat=op.count('I')))

                for Z_bits in Z_bit_vals:
                    Z_bits_indexed = list(zip(Z_bits, Z_indices))
                    for I_bits in I_bit_vals:
                        I_bits_indexed = list(zip(I_bits, I_indices))
                        ordered_eigenstring = sorted(Z_bits_indexed+I_bits_indexed, key=lambda x:x[1])
                        eigenstring = ''.join([bit[0] for bit in ordered_eigenstring])
                        possible_states.append(eigenstring)

                all_eigenstates[op] = possible_states

    list_eigenstates = list(all_eigenstates.values())
    intersection = set(list_eigenstates[0]).intersection(*list_eigenstates)
    
    return list(intersection)