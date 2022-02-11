import scipy
from qondense.utils.symplectic_toolkit import *
from copy import deepcopy
from typing import Union
from scipy import sparse

class QubitOp:
    """ Class to represent an operator defined over 
    the Pauli group in the symplectic form.
    """

    pauli_map = {   
        'II':[1,   'I'],
        'XX':[1,   'I'],'YY':[1,   'I'],'ZZ':[1,   'I'], 
        'IX':[1,   'X'],'IY':[1,   'Y'],'IZ':[1,   'Z'],
        'XI':[1,   'X'],'YI':[1,   'Y'],'ZI':[1,   'Z'],
        'XY':[+1j, 'Z'],'ZX':[+1j, 'Y'],'YZ':[+1j, 'X'],
        'YX':[-1j, 'Z'],'XZ':[-1j, 'Y'],'ZY':[-1j, 'X']
        }

    def __init__(self, 
            operator: Union[str, List[str], Dict[str, float]] = None,
            symp_rep: np.array = None,
            coeffvec: np.array = None
        ):
        """
        """
        if operator is None:
            # Here the operator has been specified by its 
            # symplectic representation and vector of coefficients
            assert(symp_rep is not None and coeffvec is not None)
            r1,c1=symp_rep.shape
            r2,c2=coeffvec.shape
            assert(r1==r2 and c1%2==0 and c2==1)
            
            self._symp = symp_rep
            self.cfvec = coeffvec
            self.n_qubits = c1//2
            self.n_terms  = r1
        else:
            # This handles over operator 
            # Convert operator to dictionary
            if type(operator)==str:
                operator = [operator]
            if type(operator)==list:
                operator = {op:1 for op in operator}
            if type(operator)==np.array:
                operator = dictionary_operator(operator, np.ones((operator.shape[0],1)))

            # Extract number of qubits and corresponding symplectic form
            self.n_qubits = number_of_qubits(operator)
            self.n_terms = len(operator)

            # build each of the X,Z symplectic blocks separately for accessibility
            terms, coeff = zip(*operator.items())
            XZ_search = {'X':['X', 'Y'], 'Z':['Z', 'Y']}
            def build_block(which):
                zero_block = np.zeros((self.n_terms, self.n_qubits))
                for row,P in enumerate(terms):
                    non_zero = [i for i,Pi in enumerate(P) if Pi in XZ_search[which]]
                    zero_block[row, non_zero] = 1
                return zero_block

            self.X_block = build_block('X')
            self.Z_block = build_block('Z')
            # by default we store the X and Z blocks on the left and right, respectively
            # ... and finally, the vector of coefficients:
            self.cfvec = np.array(coeff).reshape(self.n_terms,1)
            self._symp = np.hstack((self.X_block, self.Z_block))

        self._csc = sparse.csc_matrix(self._symp, dtype=int) # sparsify
        self.X_block = self._csc[:,:self.n_qubits]
        self.Z_block = self._csc[:,self.n_qubits:]
        # symplectic forms for computing inner products
        self.half_symform     = np.eye(2*self.n_qubits,2*self.n_qubits,self.n_qubits)
        self.half_symform_csc = sparse.csc_matrix(self.half_symform, dtype=int)
        self.symform     = np.array(self.half_symform + self.half_symform.T, dtype=int)
        self.symform_csc = sparse.csc_matrix(self.symform, dtype=int)

        
    def _dict(self):
        """
        """
        return dictionary_operator(self._symp, self.cfvec)


    def copy(self):
        """ Create a carbon copy of the class instance
        """
        return deepcopy(self)

    
    def reform(self, operator):
        """ funnel the input operator regardless of type into QubitOp
        """
        if not isinstance(operator, QubitOp):
            operator = QubitOp(operator)
        return operator.copy()


    def swap_XZ_blocks(self):
            """ Reverse order of symplectic matrix so that 
            Z operators are on the left and X on the right
            """
            return sparse.hstack((self.Z_block, self.X_block))


    def count_pauli_Y(self):
        """ Count the qubit positions of each term set to Pauli Y
        """
        Y_coords = self.X_block + self.Z_block == 2
        return np.array(Y_coords.sum(axis=1))


    def basis_reconstruction(self, 
            operator_basis: List[str]
        ) -> np.array:
        """ simultaneously reconstruct every operator term in the supplied basis.
        Performs Gaussian elimination on [op_basis.T | self_symp.T] and restricts 
        so that the row-reduced identity block is removed. Each row of the
        resulting matrix will index the basis elements required to reconstruct
        the corresponding term in the operator.
        """
        dim = len(operator_basis)
        basis_symp = self.reform(operator_basis)._symp
        basis_op_stack = np.vstack([basis_symp, self._symp])
        ham_reconstruction = gf2_gaus_elim(basis_op_stack.T)[:dim,dim:].T

        return ham_reconstruction


    def symplectic_inner_product(self, 
            aux_paulis: Union[str, List[str], np.array], 
            sip_type = 'full'
        ) -> np.array:
        """ Method to calculate the symplectic inner product of the represented
        operator with one (or more) specified pauli operators, .

        sip_type allows one to choose whether the inner product is: 
        - full, meaning it computes commutation properties, or... 
        - half, which computes sign flips for Pauli multiplication
        """
        if sip_type == 'full':
            U = self.symform_csc
        elif sip_type == 'half':
            U = self.half_symform_csc
        else:
            raise ValueError('Accepted values for sip_type are half or full')

        aux_paulis = self.reform(aux_paulis)
        sparse_inner_product = self._csc @ U @ aux_paulis._csc.transpose()
        sparse_inner_product.data %= 2 # effects modulo 2 in csc form

        return sparse_inner_product


    def commutes_with(self, 
            check_paulis: Union[str, List[str], np.array]
        ) -> np.array:
        """ Returns an array in which:
        - 0 entries denote commutation and
        - 1 entries denote anticommutation
        """
        return self.symplectic_inner_product(aux_paulis=check_paulis)

    
    def adjacency_matrix(self) -> np.array:
        """ Checks commutation of the represented operator with itself
        """
        return self.commutes_with(check_paulis=self._symp)


    def sign_difference(self, 
            check_paulis:Union[str, List[str], np.array]
        ) -> np.array:
        """ symplectic inner product but with a modified syplectic form.
        This keeps track of sign flips resulting from Pauli multiplication
        but disregards complex phases (we account for this elsewhere).
        """
        return self.symplectic_inner_product(aux_paulis=check_paulis, sip_type='half')


    def _multiply_by_pauli(self, pauli):
        """ performs *phaseless* Pauli multiplication 
        via binary summation of the symplectic matrix
        """
        pauli = self.reform(pauli)
        pauli_mult = self._csc + pauli._csc
        pauli_mult.data %= 2 # effects modulo 2 in csc form
        return QubitOp(symp_rep=pauli_mult.toarray(), coeffvec=self.cfvec)


    def phase_modification(self, source_pauli, target_pauli) -> np.array:
        """ compensates for the phases incurred through Pauli multiplication
        implemented as per https://doi.org/10.1103/PhysRevA.68.042318

        outputs a vector of phases to multiply termwise with the coefficient vector
        """
        
        sign_exp = self.sign_difference(source_pauli).toarray()
        sign = (-1)**sign_exp
        Y_count = self.count_pauli_Y() + source_pauli.count_pauli_Y()
        sigma_tau_compensation = (-1j)**Y_count # mapping from sigma to tau representation
        tau_sigma_compensation = (1j)**target_pauli.count_pauli_Y() # back from tau to sigma
        phase_mod = sign*sigma_tau_compensation*tau_sigma_compensation # the full phase modification
        return phase_mod


    def multiply_pauli(self, pauli):
        """ computes right-multiplication of the internal operator 
        by some single pauli operator... *with* phases
        """
        pauli = self.reform(pauli)
        assert(pauli.n_terms==1) # must be a single pauli operator

        # perform the pauli multiplication disregarding phase
        signless_product = self._multiply_by_pauli(pauli)
        phase_mod = self.phase_modification(pauli, signless_product)
        new_cfvec = self.cfvec*pauli.cfvec*phase_mod
        
        return QubitOp(symp_rep=signless_product._symp, coeffvec=new_cfvec)


    def multiply_by_operator(self, operator):
        """ computes right-multiplication of the internal operator 
        by some other operator that may contain arbitrarily many terms
        """
        operator = self.reform(operator)
        pauli_products = []
        cfvec_products = []
        for pauli, coeff in operator._dict().items():
            product = self.multiply_pauli({pauli:coeff})
            pauli_products.append(product._symp)
            cfvec_products.append(product.cfvec)

        clean_terms, clean_coeff = cleanup_symplectic(
                terms=np.vstack(pauli_products), 
                coeff=np.vstack(cfvec_products)
            )
        return QubitOp(symp_rep=clean_terms, coeffvec=clean_coeff) 
            

    def _rotate_by_pauli(self, pauli_rotation):
        """ Performs (H Omega vT \otimes v) \plus H where H is the internal operator, 
        Omega the symplectic form and v the symplectic representation of pauli_rotation
        """
        pauli_rotation = self.reform(pauli_rotation)
        commutes = self.commutes_with(pauli_rotation)
        rot_where_commutes = sparse.kron(commutes, pauli_rotation._csc)
        signless_rotation = self._csc + rot_where_commutes
        signless_rotation.data %= 2 # modulo 2
        signless_rotation = QubitOp(symp_rep = signless_rotation.toarray(),
                                    coeffvec=self.cfvec)
        return signless_rotation, commutes.toarray()

    
    def rotate_by_pauli(self, pauli_rotation):
        """ Let R(t) = e^{i t/2 Q}, then one of the following can occur:
        R(t) P R^\dag(t) = P when [P,Q] = 0
        R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0
        This operation is Clifford when t=pi/2, since cos(pi/2) P - sin(pi/2) iPQ = -iPQ
        In unitary partitioning t will not be so nice and will result in an increase in the number of terms (non-Clifford)
        
        <!> Need to add rotation by angles other than pi/2!
        """
        signless_rotation, commutes = self._rotate_by_pauli(pauli_rotation)
        phase_mod = self.phase_modification(pauli_rotation, signless_rotation)
        # zero out phase mod where term commutes
        phase_mod = phase_mod*commutes
        commutes_inv = np.array(commutes==0, dtype=int) #inverted commutes vector
        phase_mod = commutes_inv - phase_mod*1j # -1j comes from rotation derivation above
        new_cfvec = self.cfvec*pauli_rotation.cfvec*phase_mod
        
        return QubitOp(symp_rep=signless_rotation._symp, coeffvec=new_cfvec)


    ###### BELOW THIS POINT SOON TO BE DEPRECATED #######

    def _dictionary_rotation(self, 
                            pauli_rot:str, 
                            angle:float=None,
                            clifford_flag:bool=True
                            )->Dict[str, float]:

        ## determine possible Paulis in image to avoid if statements
        #pauli_rot_symp = pauli_to_symplectic(pauli_rot)
        #I = np.eye(2*self.n_qubits, 2*self.n_qubits)
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
        for (pauli,coeff) in self._dict().items():#,commutes in zip(self._dict.items(), commuting):
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
        assert(len(pauli_rot) == self.n_qubits)

        if rot_type == 'symp':
            rotated_op = self._symplectic_rotation(pauli_rot,angle,clifford_flag,angle)
        elif rot_type == 'dict':
            rotated_op = self._dictionary_rotation(pauli_rot,angle,clifford_flag)

        return QubitOp(rotated_op)


    def perform_rotations(self, rotation_list:List[Tuple[str,float,bool]]):
        """ Allows one to perform a list of rotations sequentially
        """
        op_copy = QubitOp(deepcopy(self._dict()))

        for pauli_rot, angle, clifford_flag in rotation_list:
            op_copy = op_copy.rotate_by(pauli_rot,angle,clifford_flag)

        return op_copy


        