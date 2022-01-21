from cmath import phase
from qreduce.utils.symplectic_toolkit import *

class SymOp:
    phase_matrix = np.array(
        [
            [1,  1,  1,   1 ],
            [1,  1,  1j, -1j],
            [1, -1j, 1,   1j],
            [1,  1j,-1j,  1 ]
        ]
    )

    """ Represent a Pauli operator in the symplectic form
    """
    def __init__(self, operator: Dict[str, float]):
        """
        """
        # Extract number of qubits and corresponding symplectic form
        self.n_qbits = number_of_qubits(operator)
        half_symform = np.eye(2*self.n_qbits,2*self.n_qbits,self.n_qbits)
        self.symform = np.array(half_symform + half_symform.T, dtype=int)

        # Convert operator dictionary to symplectic represention
        self._dict = operator
        self._symp, self.cfvec = symplectic_operator(operator)


    def adjacency_matrix(self):
        """
        """
        return (self._symp @ self.symform @ self._symp.T) % 2


    def commuting(self, P):
        """
        """
        P_symp = pauli_to_symplectic(P)
        return (self._symp @ self.symform @ P_symp.T) % 2


    def phases_by_term(self, P:str, apply_on='left'):
        """
        """
        if apply_on not in ['left', 'right']:
            raise ValueError('Accepted values for apply_on are left or right')

        P_symp = pauli_to_symplectic(P)
        
        # permute columns so that of form [X0 Z(0+n_qbits) X1 Z(1+n_qits) ...]
        permutation = []
        for i in range(self.n_qbits):
            permutation+=[i, i+self.n_qbits]
        permutation = np.array(permutation)
        op_permuted = self._symp[:, permutation]  # return a rearranged copy
        P_permuted = P_symp[:, permutation]

        phases_by_term = 1

        # iterate over pairs XiZi
        for op_qpos, Pi in zip(np.hsplit(op_permuted, self.n_qbits),
                            np.hsplit(P_permuted, self.n_qbits)):
            k, l = Pi[0]
            kl_index = int(f'{k}{l}', 2)
            kl_state = np.eye(1,4,kl_index)
            C4_expansion = []
            for i,j in op_qpos:
                ij_index = int(f'{i}{j}', 2)
                ij_state = np.eye(1,4,ij_index)
                C4_expansion.append(ij_state[0])
            # stores the phase arising from single Pauli multiplication on qubit position i
            C4_expand_matrix = np.array(np.stack(C4_expansion), dtype=int)
            #if apply_on == 'right':
            #phase_vec = C4_expand_matrix @ self.phase_matrix @ kl_state.T
            #elif apply_on == 'left':
            phase_vec = (kl_state @ self.phase_matrix @ C4_expand_matrix.T).T
            
            if apply_on == 'right': #comment this out for faster execution
                phase_vec = phase_vec.conjugate()
            
            # keep track of phase product over tensor factors
            phases_by_term *= phase_vec

        return phases_by_term


    def multiply(self, P:str, apply_on='left'):
        """
        """
        P_symp = pauli_to_symplectic(P)
        phases = self.phases_by_term(P, apply_on=apply_on)
        opmult = (self._symp + P_symp) % 2
        coeffs = self.cfvec * phases

        op_out = dictionary_operator(opmult, coeffs)

        return SymOp(op_out)


    def rotate_by(self, 
                P:str, 
                clifford:bool=True, 
                angle:float=None
                ):
        """ Let R = e^(i t/2 P)... this method returns R H R^\dag
        angle = pi/2 for R to be a Clifford operations
        """
        assert(len(P) == self.n_qbits)
        #if on not in ['left', 'right']:
        #    raise ValueError('Accepted values for argument on are left or right')

        P_symp = pauli_to_symplectic(P)
        commuting = (self._symp @ self.symform @ P_symp.T) % 2

        I = np.eye(2*self.n_qbits, 2*self.n_qbits)
        OmegaPtxP = np.outer(self.symform @ P_symp.T, P_symp)
        P_mult_mat = (I+OmegaPtxP) % 2
        
        # determine Pauli terms
        RQRt = (self._symp @ P_mult_mat) % 2

        # determine corresponding phase flips
        phases = self.phases_by_term(P)
        phases *= 1j # commuting terms now purely imaginary, anticommuting real
        phase_flip = phases.real + (commuting ^ 1) # +1 whenever term commutes with P, +/-1 rest of time
        coeff_flip = self.cfvec * phase_flip

        if clifford == True:
            op_out = dictionary_operator(RQRt, coeff_flip)
        else:
            assert(angle is not None)
            sin_cf_part = (commuting*np.sin(angle)+(commuting^1))*coeff_flip
            sin_op_part = RQRt
            cos_cf_part = commuting*np.cos(angle)*self.cfvec
            cos_cf_part = cos_cf_part[~np.all(cos_cf_part == 0, axis=1)]
            cos_op_part = self._symp * commuting
            cos_op_part = cos_op_part[~np.all(cos_op_part == 0, axis=1)]

            non_cliff_op = np.concatenate((sin_op_part, cos_op_part))
            non_cliff_cf = np.concatenate((sin_cf_part, cos_cf_part))
            non_cliff_op, non_cliff_cf = cleanup_symplectic(non_cliff_op, non_cliff_cf)

            op_out = dictionary_operator(non_cliff_op, non_cliff_cf)

        return SymOp(op_out)