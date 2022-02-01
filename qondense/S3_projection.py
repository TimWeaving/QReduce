from qondense.utils.QubitOp import QubitOp
from qondense.utils.symplectic_toolkit import *
from itertools import product
from typing import Dict, List, Tuple, Union
from copy import deepcopy


class S3_projection:
    def __init__(self,
                stabilizers: List[str], 
                eigenvalues: List[int],
                single_pauli: str,
                ) -> None:
        """
        """
        self.n_qubits = number_of_qubits(stabilizers)

        # check the stabilizers are independent:
        check_independent = gf2_gaus_elim(build_symplectic_matrix(stabilizers))
        for row in check_independent:
            if np.all(row==0):
                raise ValueError('The supplied stabilizers are not independent')
        self.stabilizers = stabilizers
        self.eigenvalues = eigenvalues
        # store stabilizers and their assignments as QubitOp object
        # this facilitates various manipulations such as Pauli rotations
        self.stab_eigval = QubitOp({S:eigval for S,eigval 
                                            in zip(stabilizers, eigenvalues)})
        self.single_pauli= single_pauli
        
        # identify a valid mapping onto qubits
        non_identity_pos = [[i for i,Gi in enumerate(G) if Gi!='I'] for G in stabilizers]
        valid_permutations = [prod for prod in product(*non_identity_pos) if len(set(prod))==len(stabilizers)]
        self.stab_to_qubit = {S:q for S,q in zip(stabilizers, valid_permutations[0])}


    def stabilizer_rotations(self):
        """ 
        Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        
        Returns 
        - a dictionary of stabilizers with the rotations mapping each to a 
          single Pauli in the formList[Tuple[rotation, angle, gen_rot]], 
        
        - a dictionary of qubit positions that we have rotated onto and 
          the eigenvalues post-rotation
        
        TODO - this method could definitely be more elegant!
        """

        single_pauli_map = {'X':{1:'Z', 2:'Y'},
                            'Y':{1:'X', 2:'Z'},
                            'Z':{1:'Y', 2:'X'}}

        def amend_string_index( string:Union[str,list], 
                        index:int, character:str
                        )->str:
            """Update a string at a given index with some character 
            """
            listed = list(deepcopy(string))
            listed[index] = character
            return ''.join(listed)
        
        
        used_qubits = []
        rotations = []
        
        for S in self.stabilizers:
            # perform the current list of rotations 
            # (since each rotation is dependent on those preceeding it)
            S_rot = list(QubitOp({S:1}).perform_rotations(rotations)._dict.keys())[0]

            # check if the stabilizer is already a single Pauli operator
            if S.count(self.single_pauli)!=1:
                # Check if diagonal with respect to single_pauli
                if set(S_rot) in [{self.single_pauli},{'I',self.single_pauli}]:
                    # identify a qubit position that is available for rotation onto
                    potential_single_pauli_indices = [i for i,Si in enumerate(S_rot) if
                                            Si==self.single_pauli and i not in used_qubits]
                    single_pauli_index = potential_single_pauli_indices[0]
                    # this defines corresponding rotations (see link to paper above)
                    rot_op = amend_string_index(['I' for i in range(self.n_qubits)],
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][1])
                    S_offdiag = amend_string_index(S_rot,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][2])
                    rotations.append((rot_op, np.pi/2, True))
                    # now the stabilizer is non-diagonal!
                else:
                    # for non-diagonal stabilizer:
                    S_offdiag = deepcopy(S_rot)
                    single_pauli_index = [i for i,Si in enumerate(S_offdiag) if 
                                Si not in ['I',self.single_pauli] and i not in used_qubits][0]

                # rotation for non-diagonal stabilizer (wrt to chosen single_pauli) onto single Pauli operator
                if S_offdiag[single_pauli_index] == single_pauli_map[self.single_pauli][2]:
                    rot_op = amend_string_index(S_offdiag,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][1])
                else:
                    rot_op = amend_string_index(S_offdiag,
                                                single_pauli_index,
                                                single_pauli_map[self.single_pauli][2])
                rotations.append((rot_op, np.pi/2, True))
            else:
                single_pauli_index = S_rot.index(self.single_pauli)
            
            # append to used_qubits the qubit index onto which we rotated 
            # so that it cannot be used for subsequent stabilizers
            used_qubits.append(single_pauli_index)

        # perform the full list of rotations to obtain the new eigenvalues
        rotated_stabilizers = self.stab_eigval.perform_rotations(rotations)._dict
        stab_index_eigenval = {S_rot.index(self.single_pauli):eigval 
                            for S_rot,eigval in rotated_stabilizers.items()}

        return rotations, stab_index_eigenval


    def _perform_projection(self, 
                        operator: Dict[str, float],  
                        q_sector: Dict[int, int],
                        ) -> Dict[str, float]:
        """ 
        method for projecting an operator over fixed qubit positions stabilized 
        by single Pauli operators (obtained via Clifford operations)

        TODO update method to work with symplectic representation
        """
        # qubits for projection must be ordered
        stab_q, sector = zip(*sorted(q_sector.items(), key=lambda x:x[0]))
        operator_proj = {}
        for pauli in operator:
            # split the Pauli string in accordance with the 
            # projected qubits and those we wish to simulate
            pauli_stab = "".join([pauli[i] for i in stab_q])
            pauli_free = "".join([pauli[i] for i in range(self.n_qubits) if i not in stab_q])
            
            # keep only the terms for which the stabilized qubit positions are diagonal (wrt single_pauli)
            if set(pauli_stab) in [{"I"}, {self.single_pauli}, {"I", self.single_pauli}]:
                # update the sign according to the stabilizer eigenvalues
                sign = np.prod(
                    [eigval for eigval,stab in zip(sector, pauli_stab) if stab == self.single_pauli]
                )
                # append simulated part to the projected operator
                if pauli_free not in operator_proj:
                    operator_proj[pauli_free] = sign * operator[pauli]
                else:
                    operator_proj[pauli_free] += sign * operator[pauli]

        return operator_proj
    

    def perform_projection(self,
                    operator: QubitOp,
                    insert_rotation:Tuple[str,float,bool]=None
                    )->Dict[float, str]:
        """ 
        Input a QubitOp and returns the reduced operator corresponding 
        with the specified stabilizers and eigenvalues.
        
        insert_rotation allows one to include supplementary Pauli rotations
        to be performed prior to the stabilizer rotations, for example 
        unitary partitioning in CS-VQE
        """
        # obtain the full list of stabilizer rotations...
        all_rotations, stab_index_eigenval = self.stabilizer_rotations()
        # ...and insert any supplementary ones coming from the child class
        if insert_rotation is not None:
            all_rotations.insert(0, insert_rotation)
        self.all_rotations = all_rotations

        # perform the full list of rotations on the input operator...
        op_rotated = operator.perform_rotations(all_rotations)._dict
        # ...and finally perform the stabilizer subspace projection
        ham_project = self._perform_projection( operator=op_rotated, 
                                                q_sector=stab_index_eigenval)

        return QubitOp(ham_project)