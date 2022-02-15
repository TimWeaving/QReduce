from shutil import ExecError
from qondense.utils.QubitOp import QubitOp
from qondense.utils.symplectic_toolkit import *
from typing import Dict, List, Tuple

class S3_projection:
    """ Base class for enabling qubit reduction techniques derived from
    the Stabilizer SubSpace (S3) projection framework, such as tapering
    and Contextual-Subspace VQE. The methods defined herein serve the 
    following purposes:

    - stabilizer_rotations
        This method determines a sequence of Clifford rotations mapping the
        provided stabilizers onto single-qubit Paulis (sqp), either X or Z
    - _perform_projection
        Assuming the input operator has been rotated via the Clifford operations 
        found in the above stabilizer_rotations method, this will effect the 
        projection onto the corresponding stabilizer subspace. This involves
        droping any operator terms that do not commute with the rotated generators
        and fixing the eigenvalues of those that do consistently.
    - perform_projection
        This method wraps _perform_projection but provides the facility to insert
        auxiliary rotations (that need not be Clifford). This is used in CS-VQE
        to implement unitary partitioning where necessary. 
    """
    
    rotated_flag = False

    def __init__(self,
                stabilizers:  List[str], 
                eigenvalues:  List[int],
                target_sqp: str,
                fix_qubits: List[int] = None
                ) -> None:
        """
        - stabilizers
            a list of stabilizers that should be enforced, given as Pauli strings
        - eigenvalues
            the list of eigenvalue assignments to complement the stabilizers
        - target_sqp
            the target single-qubit Pauli (X or Z) that we wish to rotate onto
        - fix_qubits
            Manually overrides the qubit positions selected in stabilizer_rotations, 
            although the rotation procedure can be a bit unpredictable so take care!
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
        self.target_sqp= target_sqp
        if fix_qubits is None:
            fix_qubits = [None for S in stabilizers]
        self.fix_qubits = fix_qubits
        

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
        rotations=[]

        def append_rotation(base_pauli: np.array, index: int) -> str:
            """ force the indexed qubit to a Pauli Y in the base Pauli
            """
            X_index = index % self.n_qubits # index in the X block
            base_pauli[np.array([X_index, X_index+self.n_qubits])]=1
            base_pauli = pauli_from_symplectic(base_pauli)
            rotations.append((base_pauli, np.pi/2, True))
            # return the pauli rotation to update stabilizer_ref as we go
            return base_pauli

        # This part produces rotations onto single-qubit Paulis (sqp) - might be a combination of X and Z
        # while loop active until each row of symplectic matrix contains a single non-zero element
        while np.any(~(np.count_nonzero(stabilizer_ref._symp(), axis=1)==1)):
            unique_position = np.where(np.count_nonzero(stabilizer_ref._symp(), axis=0)==1)[0]
            reduced = stabilizer_ref._symp()[:,unique_position]
            unique_stabilizer = np.where(np.any(reduced, axis=1))
            for row in stabilizer_ref._symp()[unique_stabilizer]:
                if np.count_nonzero(row) != 1:
                    # find the free indices and pick one (there is some freedom over this)
                    available_positions = np.intersect1d(unique_position, np.where(row))
                    pauli_rotation = append_rotation(row.copy(), available_positions[0])
                    # update the stabilizers by performing the rotation
                    stabilizer_ref = stabilizer_ref.rotate_by_pauli(pauli_rotation)

        # This part produces rotations onto the target sqp
        for row in stabilizer_ref._symp():
            sqp_index = np.where(row)[0]
            if ((self.target_sqp == 'Z' and sqp_index< self.n_qubits) or 
                (self.target_sqp == 'X' and sqp_index>=self.n_qubits)):
                pauli_rotation = append_rotation(np.zeros(2*self.n_qubits), sqp_index)

        # perform the full list of rotations to obtain the new eigenvalues
        rotated_stabilizers = self.stab_eigval.perform_rotations(rotations)._dict()
        stab_index_eigenval = {S_rot.index(self.target_sqp):eigval 
                            for S_rot,eigval in rotated_stabilizers.items()}

        return rotations, stab_index_eigenval


    def _perform_projection(self, 
                        operator: QubitOp
                        ) -> Dict[str, float]:
        """ method for projecting an operator over fixed qubit positions 
        stabilized by single Pauli operators (obtained via Clifford operations)
        """
        if not self.rotated_flag:
            raise ExecError('The operator has not been rotated - intended for use with perform_projection method')

        stab_qubits,eigval = zip(*self.stab_index_eigval.items())
        stab_qubits,eigval = np.array(stab_qubits),np.array(eigval)
        
        # pick out relevant element of symplectic for single Pauli X or Z
        stab_qubits_ref = stab_qubits.copy()
        if self.target_sqp=='Z':
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
        """ Input a QubitOp and returns the reduced operator corresponding 
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
        self.rotated_flag = True
        # ...and finally perform the stabilizer subspace projection
        op_project = self._perform_projection(operator=op_rotated)

        return op_project