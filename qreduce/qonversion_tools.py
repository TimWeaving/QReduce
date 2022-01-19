from openfermion.ops import QubitOperator
from qiskit.aqua.operators.legacy import WeightedPauliOperator
try:
    on_QLM = False
    from qiskit.quantum_info.operators.symplectic.pauli import Pauli
    from qiskit.opflow.primitive_ops import PauliOp
    print('Will store matrices in sparse form')
except:
    on_QLM = True
    from qiskit.quantum_info.operators import Pauli
    from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
    print('Will store matrices in dense form')
    
from openfermion.ops import FermionOperator
try:
    from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
except:
    print('Qiskit Nature not installed')

PAULI_STRINGS_LOOKUP = {'I':0,'X':1,'Y':2,'Z':3}
PAULI_STRINGS_LOOKUP_REVERSE = {0:'I', 1:'X', 2:'Y', 3:'Z'}

# comment out due to incompatible versions of Cirq and OpenFermion in Orquestra
def QubitOperator_to_dict(op, num_qubits):
    assert(type(op) == QubitOperator)
    op_dict = {}
    term_dict = op.terms
    terms = list(term_dict.keys())

    for t in terms:    
        letters = ['I' for i in range(num_qubits)]
        for i in t:
            letters[i[0]] = i[1]
        p_string = ''.join(letters)        
        op_dict[p_string] = term_dict[t]
         
    return op_dict


def dict_to_QubitOperator(op):
    assert(type(op) == dict)
    p_strings = list(op.keys())
    out = QubitOperator()
    
    for p in p_strings:
        p_str = ''
        for index, p_single in enumerate(p):
            if p_single != 'I':
                p_str += (p_single + str(index) + ' ')
        out += QubitOperator(p_str, op[p])
    
    return out


def WeightedPauliOperator_to_dict(op):
    assert(type(op) == WeightedPauliOperator)
    op_dict = {(p[1]).to_label():p[0] for p in op.paulis}
    
    return op_dict


def dict_to_WeightedPauliOperator(op):
    assert(type(op) == dict)
    if on_QLM:
        return sum([PauliOp(Pauli(label=p), op[p]) for p in op.keys()])
    else:
        return sum([PauliOp(Pauli(p), op[p]) for p in op.keys()])


def dict_to_list_index(ham_dict):
    """ Convert Hamiltonian from Pauli string dictionary format to list indices
    """
    ham_list = []
    for op in ham_dict.keys():
        new_op=[]
        for i in op:
            new_op.append(PAULI_STRINGS_LOOKUP[i])
        ham_list.append([ham_dict[op], new_op])
    return ham_list

def index_list_to_dict(ham_list):
    """ Convert Hamiltonian from list indices to Pauli string dictionary
    """
    ham_dict = {}
    for coeff, op in ham_list:
        new_op = []
        for i in op:
            new_op.append(PAULI_STRINGS_LOOKUP_REVERSE[i])
        ham_dict[''.join(new_op)] = coeff
    return ham_dict


def fermionic_openfermion_to_qiskit(f_operator, num_qubits):
    assert(type(f_operator)==FermionOperator)
    f_map = {1:'+', 0:'-'}
    f_ops = list(f_operator.terms.keys())
    mapped_f_ops=[]

    for op in f_ops:
        f_string=''
        for t in op:
            q_pos = t[0]
            dag = t[1]
            f_string += f_map[dag]+'_'+str(q_pos)+' '
        mapped_f_ops.append(f_operator.terms[op]*FermionicOp(f_string, display_format="sparse", register_length=num_qubits))

    return sum(mapped_f_ops)


def PauliOp_to_dict(operator):
    op_dict = {}
    for term in operator:
        name, coeff = term.primitive.to_list()[0]
        op_dict[name] = coeff
    return op_dict