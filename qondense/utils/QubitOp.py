from qondense.utils.symplectic_toolkit import *
from copy import deepcopy

class QubitOp:

    pauli_map = {   
        'II':[1,   'I'],
        'XX':[1,   'I'],'YY':[1,   'I'],'ZZ':[1,   'I'], 
        'IX':[1,   'X'],'IY':[1,   'Y'],'IZ':[1,   'Z'],
        'XI':[1,   'X'],'YI':[1,   'Y'],'ZI':[1,   'Z'],
        'XY':[+1j, 'Z'],'ZX':[+1j, 'Y'],'YZ':[+1j, 'X'],
        'YX':[-1j, 'Z'],'XZ':[-1j, 'Y'],'ZY':[-1j, 'X']
        }

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
        if type(operator)==list:
            operator = {op:0 for op in operator}
        self._dict = operator
        self._symp, self.cfvec = symplectic_operator(operator)


    def swap_XZ_blocks(self):
            """ Reverse order of symplectic matrix so that 
            Z operators are on the left and X on the right
            """
            X = self._symp[:,:self.n_qbits]
            Z = self._symp[:,self.n_qbits:]
            ZX = np.hstack((Z,X))
            #swap_op = dictionary_operator(ZX, self.cfvec)

            return ZX # QubitOp(swap_op)


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

        return QubitOp(op_out)


    def _symplectic_rotation(self, 
                            pauli_rot:str,
                            angle:float=None,
                            clifford_flag:bool=True
                            )->Dict[str, float]:

        pauli_rot_symp = pauli_to_symplectic(pauli_rot)
        commuting = self.commuting(pauli_rot) #(self._symp @ self.symform @ pauli_rot_symp.T) % 2

        I = np.eye(2*self.n_qbits, 2*self.n_qbits)
        OmegaPtxP = np.outer(self.symform @ pauli_rot_symp.T, pauli_rot_symp)
        pauli_rot_mult_mat = (I+OmegaPtxP) % 2
        
        # determine Pauli terms
        RQRt = (self._symp @ pauli_rot_mult_mat) % 2

        # determine corresponding phase flips
        phases = self.phases_by_term(pauli_rot)
        phases *= 1j # commuting terms now purely imaginary, anticommuting real
        phase_flip = phases.real + (commuting ^ 1) # +1 whenever term commutes with pauli_rot, +/-1 rest of time
        coeff_flip = self.cfvec * phase_flip

        if clifford_flag == True:
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

        return op_out


    def _dictionary_rotation(self, 
                            pauli_rot:str, 
                            angle:float=None,
                            clifford_flag:bool=True
                            )->Dict[str, float]:

        ## determine possible Paulis in image to avoid if statements
        #pauli_rot_symp = pauli_to_symplectic(pauli_rot)
        #I = np.eye(2*self.n_qbits, 2*self.n_qbits)
        #OmegaPtxP = np.outer(self.symform @ pauli_rot_symp.T, pauli_rot_symp)
        #P_mult_mat = (I+OmegaPtxP) % 2
        ## determine Pauli terms
        #RQRt = (self._symp @ P_mult_mat) % 2
        #poss_ops = np.concatenate((self._symp, RQRt))
        #poss_ops = dictionary_operator(
        #                                poss_ops, 
        #                                np.array([[0] for i in range(len(poss_ops))])
        #                                )
        def update_op(op, P, c):
            if P not in op:
                op[P] = c.real
            else:
                op[P] += c.real

        def commutes(P, Q):
            num_diff=0
            for Pi,Qi in zip(P,Q):
                if Pi=='I' or Qi=='I':
                    pass
                else:
                    if Pi!=Qi:
                        num_diff+=1
            return not bool(num_diff%2)

        op_out = {}
        #commuting = list(self.commuting(pauli_rot).T[0]==0)
        for (pauli,coeff) in self._dict.items():#,commutes in zip(self._dict.items(), commuting):
            if commutes(pauli, pauli_rot):#commutes:
                update_op(op=op_out, P=pauli, c=coeff)
            else:
                phases, paulis = zip(*[self.pauli_map[P+Q] for P,Q in zip(pauli_rot, pauli)])
                coeff_update = coeff*1j*np.prod(phases)
                if clifford_flag:
                    update_op(op=op_out, P=''.join(paulis), c=coeff_update)
                else:
                    update_op(op=op_out, P=pauli, c=np.cos(angle)*coeff)
                    update_op(op=op_out, P=''.join(paulis), c=np.sin(angle)*coeff_update)
                
        return op_out


    def rotate_by(self, 
                pauli_rot:str,  
                angle:float=None,
                clifford_flag:bool=True,
                rot_type:str='dict'
                ):
        """ Let R = e^(i t/2 P)... this method returns R H R^\dag
        angle = pi/2 for R to be a Clifford operations
        """

        if rot_type not in ['dict', 'symp']:
            raise ValueError('Accepted values for rot_type are dict and symp')
        assert(len(pauli_rot) == self.n_qbits)

        if rot_type == 'symp':
            rotated_op = self._symplectic_rotation(pauli_rot,angle,clifford_flag,angle)
        elif rot_type == 'dict':
            rotated_op = self._dictionary_rotation(pauli_rot,angle,clifford_flag)

        return QubitOp(rotated_op)


    def perform_rotations(self, rotation_list:List[Tuple[str,float,bool]]):
        """ Allows one to perform a list of rotations sequentially
        """
        op_copy = QubitOp(deepcopy(self._dict))

        for pauli_rot, angle, clifford_flag in rotation_list:
            op_copy = op_copy.rotate_by(pauli_rot,angle,clifford_flag)

        return op_copy


        