from typing import Dict
from qreduce.utils.symplectic_toolkit import *

class DictOp:
    pauli_map = {   
        'XY':[+1j, 'Z'],
        'ZX':[+1j, 'Y'],
        'YZ':[+1j, 'X'],
        'YX':[-1j, 'Z'],
        'XZ':[-1j, 'Y'],
        'ZY':[-1j, 'X']
        }

    def __init__(self, operator: Dict[str, float]) -> None:
        
        self.n_qubits = number_of_qubits(operator)
        self.operator = operator