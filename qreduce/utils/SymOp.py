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
    def __init__(self, op_dict:Dict[str, float]):
        """
        """
        # Extract number of qubits and corresponding symplectic form
        self.n_qbits = number_of_qubits(op_dict)
        half_symform = np.eye(2*self.n_qbits,2*self.n_qbits,self.n_qbits)
        self.symform = np.array(half_symform + half_symform.T, dtype=int)

        # Convert operator dictionary to symplectic represention
        self.op_dict = op_dict
        self.op_symp, self.coefvec = symplectic_operator(op_dict)


    def phases_by_term(self, P:str, apply_on='right'):
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
        op_permuted = self.op_symp[:, permutation]  # return a rearranged copy
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
            phase_vec = C4_expand_matrix @ self.phase_matrix @ kl_state.T
            # keep track of phase product over tensor factors
            phases_by_term *= phase_vec

        return phases_by_term


    def multiply(self, P:str, apply_on='right'):
        """
        """
        P_symp = pauli_to_symplectic(P)
        phases = self.phases_by_term(P)
        opmult = (self.op_symp + P_symp) % 2
        coeffs = self.coefvec * phases

        op_out = dictionary_operator(opmult, coeffs)

        return SymOp(op_out)


    def rotate_by(self, 
                P:str, 
                clifford:bool=True, 
                angle:float=None
                ):#, on:str='left'):
        """ Let R = e^(i t/2 P)... this method returns R Q R^\dag
        angle = pi/2 for R to be a Clifford operations
        """
        assert(len(P) == self.n_qbits)
        #if on not in ['left', 'right']:
        #    raise ValueError('Accepted values for argument on are left or right')

        P_symp = pauli_to_symplectic(P)
        commuting = (self.op_symp @ self.symform @ P_symp.T) % 2

        I = np.eye(2*self.n_qbits, 2*self.n_qbits)
        OmegaPtxP = np.outer(self.symform @ P_symp.T, P_symp)
        P_mult_mat = (I+OmegaPtxP) % 2
        
        # determine Pauli terms
        RQRt = (self.op_symp @ P_mult_mat) % 2

        # determine corresponding phase flips
        phases = self.phases_by_term(P)
        phases *= -1j # commuting terms now purely imaginary, anticommuting real
        phase_flip = phases.real + (commuting ^ 1) # +1 whenever term commutes with P, +/-1 rest of time
        coeff_flip = self.coefvec * phase_flip

        if clifford == True:
            op_out = dictionary_operator(RQRt, coeff_flip)
        else:
            assert(angle is not None)
            # DOESN'T WORK, DO NOT ADD IN EXTRA P WHEN COMMUTE!
            new_num_terms = len(self.op_symp) + len(RQRt)
            non_cliff_op = np.stack([   
                                    self.op_symp, 
                                    RQRt
                                    ]
                                    ).reshape(new_num_terms, 2*self.n_qbits)
            non_cliff_coeff = np.stack([
                                    np.cos(angle) * self.coefvec, 
                                    np.sin(angle) * coeff_flip
                                    ]
                                    ).reshape(new_num_terms, 1)
            print(non_cliff_op)
            print(non_cliff_coeff)
            op_out = dictionary_operator(non_cliff_op, non_cliff_coeff)


        return SymOp(op_out) #RQRt, coeffs_out