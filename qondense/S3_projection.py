from qondense.utils.QubitOp import QubitOp
from qondense.utils.symplectic_toolkit import *
from typing import Dict, List, Tuple

class S3_projection:
    def __init__(self,
                stabilizers:  List[str], 
                eigenvalues:  List[int],
                single_pauli: str,
                sqp_override: List[int] = None
                ) -> None:
        """
        """
        self.n_qubits = number_of_qubits(stabilizers)

        # check the stabilizers are independent:
        check_independent = gf2_gaus_elim(QubitOp(stabilizers)._symp())
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
        if sqp_override is None:
            sqp_override = [None for S in stabilizers]
        self.sqp_override = sqp_override
        

    def stabilizer_rotations(self):
        """ 
        Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        
        Returns 
        - a dictionary of stabilizers with the rotations mapping each to a 
          single Pauli in the formList[Tuple[rotation, angle, gen_rot]], 
        
        - a dictionary of qubit positions that we have rotated onto and 
          the eigenvalues post-rotation
        """
        stabilizer_ref = self.stab_eigval.copy()
        sqpZ_check = stabilizer_ref._symp()[:,self.n_qubits:]
        rotations  = []
        while ~np.all(np.count_nonzero(stabilizer_ref._symp().T, axis=0)==1):
            unique_position = np.where(np.count_nonzero(stabilizer_ref._symp(), axis=0)==1)[0]
            reduced = stabilizer_ref._symp()[:,unique_position]
            unique_stabilizer = np.where(np.any(reduced, axis=1))
            for row,sqp_check_row in zip(stabilizer_ref._symp()[unique_stabilizer,:][0], 
                                        sqpZ_check[unique_stabilizer, :][0]):
                # find the free indices and pick one (there is some freedom over this)
                available_positions = np.intersect1d(unique_position, np.where(row))
                sqp_index = available_positions[0]
                # check if already single Pauli X or Z
                if np.count_nonzero(sqp_check_row) != 1:
                    # check if diagonal
                    if np.all(row[:self.n_qubits]==0) and self.single_pauli == 'Z':
                        # define pauli that rotates off-diagonal
                        pauli_rotation = np.zeros(2*self.n_qubits)
                        pauli_rotation[sqp_index]=1
                        pauli_rotation[(sqp_index+self.n_qubits)%self.n_qubits]=1
                        pauli_rotation = pauli_from_symplectic(pauli_rotation)
                        rotations.append(pauli_rotation)
                        stabilizer_ref = stabilizer_ref.rotate_by_pauli(pauli_rotation)
                    # pauli rotation mapping to a single-qubit Pauli operator
                    pauli_rotation=row.copy()
                    pauli_rotation[(sqp_index+self.n_qubits)%self.n_qubits]=1
                    pauli_rotation = pauli_from_symplectic(pauli_rotation)
                    rotations.append(pauli_rotation)
                    stabilizer_ref = stabilizer_ref.rotate_by_pauli(pauli_rotation)
            sqpZ_check = stabilizer_ref._symp()[:,self.n_qubits:]
        # there might be one left over at this point
        if self.single_pauli=='X' and ~np.all(sqpZ_check==0):
            for row in stabilizer_ref._symp()[np.where(np.any(sqpZ_check==1, axis=1))]:
                pauli_rotation = row.copy()
                sqp_index = np.where(pauli_rotation)
                pauli_rotation[(sqp_index[0]+self.n_qubits)%self.n_qubits]=1
                rotations.append(pauli_from_symplectic(pauli_rotation))
                stabilizer_ref = stabilizer_ref.rotate_by_pauli(pauli_rotation)
        rotations = [(rot_op, np.pi/2, True) for rot_op in rotations]

        # perform the full list of rotations to obtain the new eigenvalues
        rotated_stabilizers = self.stab_eigval.perform_rotations(rotations)._dict()
        stab_index_eigenval = {S_rot.index(self.single_pauli):eigval 
                            for S_rot,eigval in rotated_stabilizers.items()}

        return rotations, stab_index_eigenval


    def _perform_projection(self, 
                        operator: QubitOp,  
                        q_sector: Dict[int, int],
                        ) -> Dict[str, float]:
        """ 
        method for projecting an operator over fixed qubit positions stabilized 
        by single Pauli operators (obtained via Clifford operations)
        """
        stab_qubits,eigval = zip(*self.stab_index_eigval.items())
        stab_qubits,eigval = np.array(stab_qubits),np.array(eigval)
        
        # pick out relevant element of symplectic for single Pauli X or Z
        stab_qubits_ref = stab_qubits.copy()
        if self.single_pauli=='Z':
            stab_qubits_ref += self.n_qubits

        # build the single-qubits Paulis the stabilizers are rotated onto
        single_qubit_paulis = np.vstack([np.eye(1, 2*self.n_qubits, q_pos) 
                                        for q_pos in stab_qubits_ref])

        # remove terms that do not commute with the rotated stabilizers
        commute_with_stabilizers = operator.commutes_with(single_qubit_paulis).toarray()==0
        op_anticommuting_removed = operator._symp()[np.all(commute_with_stabilizers, axis=1)]
        cf_anticommuting_removed = operator.cfvec[np.all(commute_with_stabilizers, axis=1)]

        # determine sign flipping from eigenvalue assignment
        eigval_assignment = op_anticommuting_removed[:,stab_qubits_ref]*eigval
        eigval_assignment[eigval_assignment==0]=1 #so the product is not 0
        coeff_sign_flip = cf_anticommuting_removed*np.array([np.prod(eigval_assignment, axis=1)]).T

        # the projected Pauli terms:
        all_qubits = np.arange(self.n_qubits)
        free_qubits = all_qubits[~np.isin(all_qubits,stab_qubits)]
        free_qubits = np.concatenate([free_qubits, free_qubits+self.n_qubits])
        op_projected = op_anticommuting_removed[:,free_qubits]

        # there may be duplicate rows in op_projected - these are identified and
        # the corresponding coefficients collected in cleanup_symplectic function
        op_out = dictionary_operator(
            *cleanup_symplectic(
                terms=op_projected, 
                coeff=coeff_sign_flip
            )
        )
        return QubitOp(op_out) 
        
    
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
        # obtain the full list of stabilizer rotations and a dictionary of the 
        # resulting single qubit Pauli indices with the eigenvalue post-rotation
        stab_rotations, stab_index_eigval = self.stabilizer_rotations()
        # ...and insert any supplementary ones coming from the child class
        if insert_rotation is not None:
            stab_rotations.insert(0, insert_rotation)
        self.stab_rotations    = stab_rotations
        self.stab_index_eigval = stab_index_eigval

        # perform the full list of rotations on the input operator...
        op_rotated = operator.perform_rotations(stab_rotations)
        # ...and finally perform the stabilizer subspace projection
        op_project = self._perform_projection(
            operator=op_rotated,
            q_sector=stab_index_eigval
        )  
        return op_project