from qondense.S3_projection import S3_projection
from qondense.utils.QubitOp import QubitOp
from qondense.utils.hypermapper_tools import hypermapper_specs
from qondense.utils.operator_toolkit import *
from qondense.utils.symplectic_toolkit import *
from qondense.utils.cs_vqe_tools_legacy import (greedy_dfs,to_indep_set,quasi_model)
from hypermapper import optimizer
from scipy.optimize import minimize_scalar

# general imports
from itertools import combinations
import numpy as np
from typing import Dict, List
import json
import sys
import csv

class cs_vqe(S3_projection):
    """ Class for performing Contextual-Subspace VQE as per https://doi.org/10.22331/q-2021-05-14-456.
    Allows one to scale a quantum problem to the available quantum resource. This is an approximate
    method but can achieve high levels of precision at a reduction in qubit count.

    1. identify a noncontextual subset of terms in the full Hamiltonian,
    2. Extract the noncontextual symmetry
    3. find an independent basis of symmetry generators,
    4. rotate each basis operator onto a single Pauli Z, 
        whilst applying the same rotations to the full Hamiltonian 
    5. drop the corresponding qubits from the Hamiltonian and
    6. fix the +/-1 eigenvalues

    Steps 1-3 are handled in this class whereas we defer to the parent S3_projection for 4-6.
    """
    def __init__(self,
                hamiltonian: Dict[str, float],
                noncontextual_set: List[str] = None,
                ref_state: List[int] = None,
                single_pauli: str = 'Z'
                ) -> None:
        """ Input a Hamiltonian in the dictionary representation and, optionally, a noncontextual 
        subset of Pauli terms (if not supplied then find_noncontextual_set() will be called.)
        There is freedom over the choice of single Pauli operator we wish to rotate onto, however 
        this is set to Z by default (in line with the original CS-VQE paper).
        """
        # Hamiltonian and noncontextual model
        self.hamiltonian  = QubitOp(hamiltonian)
        self.n_qubits     = self.hamiltonian.n_qbits
        self.single_pauli = single_pauli
        self.ref_state    = ref_state
        if noncontextual_set is not None:
            self.noncontextual_set = noncontextual_set
        else:
            self.noncontextual_set = self.find_noncontextual_set()
        self.ham_noncontextual = QubitOp({op:coeff for op,coeff in self.hamiltonian._dict.items() 
                                            if op in self.noncontextual_set})
        self.generators, self.cliquereps = self.independent_generators()
        self.generators, self.cliquereps, construction = quasi_model(self.ham_noncontextual._dict)
        # noncontextual ground state
        self.objfncprms = self.classical_obj_fnc_params()
        self.ngs_energy, self.nu, self.r = self.find_ngs()
        self.anti_clique_operator = {C: val for C,
                                     val in zip(self.cliquereps, self.r)}

        # stabilizer rotations
        Q, t = self.unitary_partitioning_rotation()
        self.unitary_partitioning = (Q, t, False)
        clique_rot = rotate_operator(self.anti_clique_operator,[self.unitary_partitioning])
        assert(len(clique_rot)==1)
        C,C_eigval = tuple(*clique_rot.items())
        # include anticommuting clique operator in set of generators
        self.generators.insert(0,C)
        self.nu = np.insert(self.nu, 0, C_eigval)
        

    def find_noncontextual_set(self, search_time=10) -> List[str]:
        """ Method for extracting a noncontextual subset of the hamiltonian terms
        """
        # for now uses the legacy greedy DFS approach
        # to be updated once more efficient/effective methods are identified
        noncontextual_set = greedy_dfs(self.hamiltonian._dict, cutoff=search_time,
                            criterion="weight")[1]
        return noncontextual_set


    def independent_generators(self) -> List[str]:
        """ Find independent generating set for noncontextual symmetry and clique representatives 
        for the anticommuting part. Does so in the symplectic representation:

        1. obtains row-reduced form of flipped symplectic matrix [Z|X],
        2. determines a basis for the kernel of the above (these are the symmetry generators),
        3. the remaining rows of the row-reduced matrix form a basis for the anticommuting cliques
        """

        # find symmetry generators
        # swap order of XZ blocks in symplectic matrix to ZX
        ZX_symp = self.ham_noncontextual.swap_XZ_blocks()
        ZX_reduced = gf2_gaus_elim(ZX_symp)
        ZX_reduced = ZX_reduced[~np.all(ZX_reduced == 0, axis=1)]
        kernel  = gf2_basis_for_gf2_rref(ZX_reduced)

        # swap XZ order back
        Z = ZX_reduced[:,:self.n_qubits]
        X = ZX_reduced[:,self.n_qubits:]
        XZ_reduced = np.hstack((X,Z))
        # Remove symmetry generators from reduced symplectic matrix
        # these should be the anticommuting generators...
        unique_rows = []
        for sympauli in XZ_reduced:
            diff = kernel-sympauli
            if list(np.where(~diff.any(axis=1))[0]) == []:
                unique_rows.append(sympauli)
        anti_kernel = np.stack(unique_rows)

        generators = [pauli_from_symplectic(row) for row in kernel]
        cliquereps = [pauli_from_symplectic(row) for row in anti_kernel]

        # some of the terms in cliquereps can actually belong 
        # to the same clique at this point... pick one!
        commutation_matrix = QubitOp(cliquereps).adjacency_matrix()
        sort_order = np.lexsort(commutation_matrix.T)
        sorted_comm_mat = commutation_matrix[sort_order,:]
        # take difference between adjacent terms to identify duplicates (i.e. commuting operators)
        row_mask = np.append([True],np.any(np.diff(sorted_comm_mat,axis=0),1))
        cliquereps = [cliquereps[i] for i,include in zip(sort_order, row_mask) if include]
        
        # check whether the generators are contained in the symmetry
        # choose another if not...
        
        #if not all([G in self.symmetry for G in generators[1:]]):
        #    raise Exception('Not all reduced generators reside within the noncontextual symmetry')

        return generators, cliquereps


    def classical_obj_fnc_params(self) -> List[Tuple[float, List[int], List[float]]]:
        """Sums over the completion (under Pauli multiplication) of G and
        extracts the non-zero Hamiltonian contributions. For each term we 
        also list the indices of the generators used in its construction. 
        For each iterate also multiplies by the clique representatives and 
        again checks whether the resulting terms exist in the Hamiltonian. 
        Any terms that do not appear have their coefficient set to zero in 
        the objective function. Returns a list of everthing required to 
        construct the classical objective function for the noncontextual 
        ground state energy (defined in classical_obj_fnc method).
        """
        # construct the closure of the commuting generating set
        G_combs = []
        for i in range(1, len(self.generators) + 1):
            G_combs += list(combinations(self.generators, i))
        G_closure = [
            (multiply_pauli_list(comb), [self.generators.index(op) for op in comb])
            for comb in G_combs
        ]
        G_closure.append(("".join(["I" for i in range(self.n_qubits)]), []))

        # extract the relevant coefficients for the sum over G_closure 
        # as in eq 13/14 of https://arxiv.org/pdf/2002.05693.pdf
        obj_fnc_params = []
        for G_op, q_indices in G_closure:
            try:
                h_G = self.hamiltonian._dict[G_op]
            except:
                h_G = 0
            C_vec = []
            for C_op in self.cliquereps:
                GC_op = multiply_paulis(G_op, C_op)
                try:
                    h_GC = self.hamiltonian._dict[GC_op]
                except:
                    h_GC = 0
                C_vec.append(h_GC)
            if h_G!=0 or not np.all(np.array(C_vec)==0):
                obj_fnc_params.append((h_G, q_indices, C_vec))

        return obj_fnc_params

    
    def classical_obj_fnc(self, input_params: Dict) -> float:
        """ Noncontextual ground state energy objective function:
        built from the data generated in classical_obj_fnc_params

        input_params is a dictionary of cofspecs required by 
        HyperMapper (defined in find_ngs).
        """
        t = input_params['theta'] #parametrizes the r unit vector

        objfnc_sum = 0
        for h_G, q_indices, h_GCs in self.objfncprms:
            q_prod = np.prod([input_params[f'q{i}'] for i in q_indices])
            objfnc_sum += (h_G+np.sin(t)*h_GCs[0]+np.cos(t)*h_GCs[1])*q_prod
            
        return objfnc_sum


    def find_ngs(self,
                hypermapper: bool = False,
                ref_energy:  float = None
                ) -> Tuple[float, List[int], List[float]]:
        """ Uses HyperMapper to perform discrete optimization over the
        generator eigenvalue assingments q_i and the continuous r unit
        vector specifying weights of anticommuting clique contributions.

        HyperMapper writes results to a .csv file that we read back in
        for use elsewhere in the cs_vqe class.

        ref_energy allows one to specify a known ground state energy 
        approximation (e.g. Hartree-Fock) as a benchmark for HyperMapper.

        If hypermapper is False then this method will require a reference
        state to be provided at the point of initialization. Operates in
        the same way as calculating a sector in tapering, in addition to
        the r vector that is optimized using SciPy's minimize_scalar.
        This is very fast but requires the noncontextual ground state to
        have a non-zero overlap with the reference to yield the correct
        result; this depends on how the noncontextual sub-Hamiltonian is 
        constructed.
        """
        q_vars = [f'q{i}' for i in range(len(self.generators))]
        if hypermapper:
            # write HyperMapper specs to file data/ngs_calculator.json
            hypermapper_specs(q_vars)
            # Jupyter uses a special stdout and HyperMapper logging overwrites it.
            stdout = sys.stdout
            # Call HyperMapper to optimize the noncontextual energy objective function
            optimizer.optimize("data/hypermapper/ngs_calculator.json", self.classical_obj_fnc)
            # restore stdout for use in Jupyter
            sys.stdout = stdout

            optimizer_output=[]
            with open("data/hypermapper/ngs_optimization_output_samples.csv", newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                # read in the optimizer output as (energy, G assignment, r vector)
                for opt_guess in reader:
                    t = float(opt_guess['theta'])
                    optimizer_output.append(
                        (
                            float(opt_guess['objfnc_sum']),
                            [int(opt_guess[qi]) for qi in q_vars], 
                            [np.sin(t), np.cos(t)] 
                        )
                    )
            energy, nu, r = sorted(optimizer_output, key=lambda x:x[0])[0]

        else:
            assert(self.ref_state is not None)
            nu = [measure_operator(pauli, self.ref_state) 
                        for pauli in self.generators]
            input_params={qi:eigval for qi,eigval in zip(q_vars, nu)}
            def f(x):
                input_params['theta'] = x
                return self.classical_obj_fnc(input_params)
            result = minimize_scalar(f)
            energy = result['fun']
            t = result['x']
            r = [np.sin(t), np.cos(t)]

        return energy, np.array(nu), np.array(r)     


    def unitary_partitioning_rotation(self):
        """ Implementation of procedure described in https://doi.org/10.1103/PhysRevA.101.062322 (Section A)
        Currently works only when number of cliques M=2
        """
        order_terms = sorted(
            self.anti_clique_operator.items(), key=lambda x: (x[0].count('X')+x[0].count('Y')))
        Aa, Bb = order_terms
        A, a = Aa
        B, b = Bb

        if a == 0 or b == 0:
            raise ValueError('Clique operator already contains one term')

        Q, coeff = multiply_paulis_with_coeff(A, B)
        sign = int((1j*coeff).real)
        t = np.arctan(-b/a)*sign
        if abs(a+np.cos(t))<1e-15:
            t+=np.pi

        return Q, t

          
    def _contextual_subspace_projection(self,   
                                        operator:QubitOp,
                                        stabilizer_indices:List[int] = None
                                        ) -> QubitOp:
        """ Returns the restriction of an operator to the contextual subspace 
        defined by a projection over stabilizers corresponing with stabilizer_indices
        """
        stabilizers = [self.generators[i] for i in stabilizer_indices]
        eigenvalues = [self.nu[i] for i in stabilizer_indices]

        # Now invoke the stabilizer subspace projection class methods given the chosen
        # stabilizers we wish to project (fixing the eigenvalues of corresponding qubits) 
        super().__init__(
                        stabilizers = stabilizers, 
                        eigenvalues = eigenvalues, 
                        single_pauli= self.single_pauli
                        )

        if 0 in stabilizer_indices:
            # Note element 0 is always the anticommuting clique operator, hence in this case
            # we need to insert the unitary partitioning rotations before applying the
            # remaining stabilizer rotations determined by S3_projection
            operator_cs = self.perform_projection(
                operator=operator,
                insert_rotation = self.unitary_partitioning
            )
        else:
            operator_cs = self.perform_projection(operator=operator)

        return operator_cs


    def contextual_subspace_hamiltonian(self,
                                        stabilizer_indices:List[int]
                                        ) -> Dict[str,float]:
        """ Construct and return the CS-VQE Hamiltonian for the stabilizers
        corresponding with stabilizer_indices
        """
        ham_cs = self._contextual_subspace_projection(operator=self.hamiltonian,
                                                    stabilizer_indices=stabilizer_indices)

        return cleanup_operator(ham_cs._dict, threshold=8)


    def noncontextual_ground_state(self,
                                stabilizer_indices:List[int],
                                projection_qubits: List[int] = None
                                ) -> str:
        if projection_qubits is not None:
            sim_qubits = list(set(range(self.n_qubits))-set(projection_qubits))
        ham_cs, free_q = self.contextual_subspace_hamiltonian(stabilizer_indices=stabilizer_indices, projection_qubits=projection_qubits)
        all_generators = {G:eigval for G, eigval in zip(self.generators, self.nu)}
        rotate_gen = rotate_operator(operator=all_generators, rotations=self.all_rotations, cleanup=True)
        stabilizers = {G:int(eig) for G,eig in cleanup_operator(rotate_gen, threshold=5).items()}
        poss_eigenstates = simultaneous_eigenstates(stabilizers)
        reduced_eigenstates = [''.join([state[i] for i in free_q]) for state in poss_eigenstates]
        
        if len(reduced_eigenstates)>1:
            print('Multiple eigenstates found:', reduced_eigenstates)
        
        return reduced_eigenstates[0]