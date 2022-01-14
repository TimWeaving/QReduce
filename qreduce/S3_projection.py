from qreduce.utils import *
from typing import Dict, List, Tuple
from copy import deepcopy

class S3_projection:
    def __init__(self, 
                hamiltonian: Dict[str, float],
                stabilizers: List[str], 
                eigenvalues: List[int],
                single_pauli: str
                ) -> None:
        """
        """
        self.hamiltonian = hamiltonian
        self.num_qubits  = number_of_qubits(hamiltonian)
        self.stabilizers = stabilizers
        self.eigenvalues = eigenvalues
        self.single_pauli = single_pauli


    def stabilizer_rotations(self):
        """ Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        Returns dictionary of stabilizers with the rotations mapping each to a single Pauli in the form
        List[Tuple[rotation, angle, gen_rot]], the corresponding qubit position and eigenvalue post-rotation
        """
        stabilizer_map = {S:{   
                            'rotations':[], 
                            'single_pauli_index':None, 
                            'single_pauli':None, 
                            'eigenvalue':None
                            } 
                            for S in self.stabilizers}
        single_pauli_map = {'X':{1:'Z', 2:'Y'},
                            'Y':{1:'X', 2:'Z'},
                            'Z':{1:'Y', 2:'X'}}
        used_qubits = []
        all_rotations=[]

        for S in self.stabilizers:
            S_rot = list(rotate_operator({S:1}, all_rotations).keys())[0]

            if set(S_rot) == {'I',self.single_pauli} and S_rot.count(self.single_pauli)==1:
                used_qubits.append(S_rot.index(self.single_pauli))
                single_pauli_index = S_rot.index(self.single_pauli)
                stabilizer_map[S]['single_pauli_index'] = single_pauli_index

            else:
                if set(S_rot) in [{self.single_pauli},{'I',self.single_pauli}]:
                    potential_single_pauli_indices = [i for i in range(self.num_qubits) if 
                                            S_rot[i]==self.single_pauli and i not in used_qubits]
                    single_pauli_index = potential_single_pauli_indices[0]
                        
                    rot_op = amend_string_index(['I' for i in range(self.num_qubits)],
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][1]
                                                )
                    stabilizer_map[S]['rotations'].append((rot_op, np.pi/2, True))
                    all_rotations.append((rot_op, np.pi/2, True))
                    S_offdiag = amend_string_index(S_rot,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][2]
                                                )
                else:
                    S_offdiag = deepcopy(S_rot)
                    single_pauli_index = [i for i in range(self.num_qubits) if 
                                S_offdiag[i] not in ['I',self.single_pauli] and i not in used_qubits][0]
                
                stabilizer_map[S]['single_pauli_index'] = single_pauli_index
                if S_offdiag[single_pauli_index] == single_pauli_map[self.single_pauli][2]:
                    rot_op = amend_string_index(S_offdiag,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][1]
                                                )
                else:
                    rot_op = amend_string_index(S_offdiag,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][2]
                                                )
                stabilizer_map[S]['rotations'].append((rot_op, np.pi/2, True))
                all_rotations.append((rot_op, np.pi/2, True))
                    
            used_qubits.append(single_pauli_index)
        
        stabilizer_assignments = {S:eigval for S,eigval in zip(self.stabilizers, self.eigenvalues)}
        rotated_stabilizers = rotate_operator(stabilizer_assignments, all_rotations)
        for S, (S_rot, eigval_rot) in zip(self.stabilizers, rotated_stabilizers.items()):
            stabilizer_map[S]['single_pauli'] = S_rot
            stabilizer_map[S]['eigenvalue'] = eigval_rot

        return all_rotations, stabilizer_map


    def _perform_projection(self, 
                        operator: Dict[str, float], 
                        stab_q: List[int], 
                        sector: List[int],
                        ) -> Dict[str, float]:
        """ method for projecting an operator over fixed qubit positions
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        # qubits for projection must be ordered
        stab_q = sorted(stab_q)
        
        operator_proj = {}
        for pauli in operator:
            pauli_stab = "".join([pauli[i] for i in stab_q])
            pauli_free = "".join([pauli[i] for i in range(self.num_qubits) if i not in stab_q])
            if set(pauli_stab) in [{"I"}, {self.single_pauli}, {"I", self.single_pauli}]:
                sign = np.prod(
                    [eigval for eigval,stab in zip(sector, pauli_stab) if stab == self.single_pauli]
                )
                if pauli_free not in operator_proj:
                    operator_proj[pauli_free] = sign * operator[pauli]
                else:
                    operator_proj[pauli_free] += sign * operator[pauli]

        return operator_proj
    

    def perform_projection(self,
                    insert_rotation:Tuple[str,float,bool]=None
                    )->Dict[float, str]:
        """
        """
        all_rotations, stabilizer_map = self.stabilizer_rotations()
        if insert_rotation is not None:
            all_rotations.insert(0, insert_rotation)

        stab_q = [S_data['single_pauli_index'] for S_data in stabilizer_map.values()]
        free_q = list(set(range(self.num_qubits))-set(stab_q))
        sector = [S_data['eigenvalue'] for S_data in stabilizer_map.values()]

        ham_rotated = rotate_operator(self.hamiltonian, all_rotations, cleanup=False)
        ham_project = self._perform_projection(
            operator=ham_rotated, stab_q=stab_q, sector=sector)

        return cleanup_operator(ham_project)