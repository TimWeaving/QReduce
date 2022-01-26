from qreduce.S3_projection import S3_projection
from qreduce.utils.QubitOp import QubitOp
from qreduce.utils.operator_toolkit import *
from qreduce.utils.symplectic_toolkit import *
from qreduce.utils.cs_vqe_tools_legacy import (greedy_dfs,to_indep_set,quasi_model)
import qreduce.utils.qonversion_tools as qonvert
from itertools import combinations, product
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List

class cs_vqe(S3_projection):
    """
    """
    def __init__(self,
                hamiltonian: Dict[str, float],
                noncontextual_set: List[str] = None,
                single_pauli: str = 'Z'
                ) -> None:
        """
        """
        self.hamiltonian = QubitOp(hamiltonian)
        self.n_qubits = self.hamiltonian.n_qbits
        self.single_pauli = single_pauli
        if noncontextual_set is not None:
            self.noncontextual_set = noncontextual_set
        else:
            self.noncontextual_set = self.find_noncontextual_set()
        self.ham_noncontextual = QubitOp({op:coeff for op,coeff in self.hamiltonian._dict.items() 
                                            if op in self.noncontextual_set})

        self.symmetry, self.cliques = self.decompose_noncontextual_set()
        #self.cliquereps = self.choose_clique_representatives()
        #self.generators = self.find_symmetry_generators()

        model = quasi_model(self.ham_noncontextual._dict)
        self.generators= model[0]
        self.cliquereps= model[1]

        self.objfncprms = self.classical_obj_fnc_params(
            G=self.generators, C=self.cliquereps
        )
        self.ngs_energy, self.nu, self.r = self.find_noncontextual_ground_state()
        self.anti_clique_operator = {C: val for C,
                                     val in zip(self.cliquereps, self.r)}
        Q, t = self.unitary_partitioning_rotation()
        self.unitary_partitioning = (Q, t, False)
        clique_rot = rotate_operator(self.anti_clique_operator,[self.unitary_partitioning])
        assert(len(clique_rot)==1)
        C,C_eigval = tuple(*clique_rot.items())
        # include anticommuting clique operator in set of generators
        self.generators.insert(0,C)
        self.nu = np.insert(self.nu, 0, C_eigval)
        

    def find_noncontextual_set(self, search_time=10):
        """Method for extracting a noncontextual subset of the hamiltonian terms
        """
        # for now uses the legacy greedy DFS approach
        # to be updated once more efficient/effective methods are identified
        noncontextual_set = greedy_dfs(self.hamiltonian._dict, cutoff=search_time,
                            criterion="weight")[1]
        return noncontextual_set


    def decompose_noncontextual_set(self):
        """Decompose a noncontextual set into its symmetry and
        remaining pairwise anticommuting cliques
        """
        commutation_matrix = adjacency_matrix(
            self.noncontextual_set, self.n_qubits) == 0
        symmetry_indices = []
        for index, commutes_with in enumerate(commutation_matrix):
            if np.all(commutes_with):
                symmetry_indices.append(index)

        cliques = []
        for i, commute_list in enumerate(commutation_matrix):
            if i not in symmetry_indices:
                C = frozenset(
                    [
                        self.noncontextual_set[j]
                        for j, commutes in enumerate(commute_list)
                        if commutes and j not in symmetry_indices
                    ]
                )
                cliques.append(C)

        symmetry = [self.noncontextual_set[i] for i in symmetry_indices]
        cliques = [list(C) for C in set(cliques)]

        return symmetry, cliques


    def choose_clique_representatives(self):
        """Choose an operator from each of the cliques determined in decompose_noncontextual_set
        to complete the generating set and forms the observable C(r)
        """
        clique_reps = []
        # choose the operator in each clique which is identity on the most qubit positions
        # to choose the single pauli Z where possible
        # ACTUALLY...
        # perhaps best to choose representatives with minimal identity qubits...
        # results in more rotations but each qubit position effectively encodes 'more information'
        # about the collective Hamiltonian... results in chemical accuracy being achieved faster?
        
        offset=0
        index=offset
        for clique in self.cliques:   
            # it seems the choice of clique representative
            # has some bearing on the success of CS-VQE...
            
            #if all([op.find('X')==-1 for op in clique]):
            #    index = offset
            #else:
            #    index = -offset
            op_weights = [(op, op.count("I")) for op in clique]
            op_weights = sorted(op_weights, key=lambda x: -x[1])
            clique_reps.append(op_weights[index][0])

        #clique_reps = [self.cliques[0][1], self.cliques[1][7]]
        
        return clique_reps


    def find_symmetry_generators(self):
        """Find independent generating set for noncontextual symmetry
        """

        # swap order of XZ blocks in symplectic matrix to ZX
        ZX_symp = self.ham_noncontextual.swap_XZ_blocks()
        reduced = gf2_gaus_elim(ZX_symp)
        kernel  = gf2_basis_for_gf2_rref(reduced)

        generators = [pauli_from_symplectic(row) for row in kernel]
        # check whether the generators are contained in the symmetry
        # choose another if not...
        
        if not all([G in self.symmetry for G in generators[1:]]):
            raise Exception('Not all reduced generators reside within the noncontextual symmetry')

        return generators


    def classical_obj_fnc_params(self, G: List[str], C: List[str]):
        """
        """
        # construct the closure of the commuting generating set
        G_combs = []
        for i in range(1, len(G) + 1):
            G_combs += list(combinations(G, i))
        G_closure = [
            (multiply_pauli_list(comb), [G.index(op) for op in comb])
            for comb in G_combs
        ]
        G_closure.append(("".join(["I" for i in range(self.n_qubits)]), []))

        # extract the relevant coefficients for the sum over G_closure as in eq 13/14 of https://arxiv.org/pdf/2002.05693.pdf
        obj_fnc_params = []
        for G_op, q_indices in G_closure:
            try:
                h_G = self.hamiltonian._dict[G_op]
            except:
                h_G = 0
            C_vec = []
            for C_op in C:
                GC_op = multiply_paulis(G_op, C_op)
                try:
                    h_GC = self.hamiltonian._dict[GC_op]
                except:
                    h_GC = 0
                C_vec.append(h_GC)
            obj_fnc_params.append((h_G, q_indices, C_vec))

        return obj_fnc_params


    def evaluate_classical_obj_fnc(
        self, nu: np.array, r: np.array, G: List[str] = None, C: List[str] = None,
        ) -> float:
        """Evaluates the classical objective function yielding
        possible energies of the noncontextual Hamiltonian
        """
        if G is None and C is None:
            G = self.generators
            C = self.cliquereps
            objfncprms = self.objfncprms
        elif (G is None and C is not None) or (G is not None and C is None):
            raise ValueError("G and C must both be None or not None")
        else:
            objfncprms = self.classical_obj_fnc_params(G=G, C=C)

        assert len(nu) == len(G)
        assert len(r) == len(C)

        outsum = 0
        for h_G, q_indices, h_GC_vec in objfncprms:
            sign = np.prod([nu[i] for i in q_indices])
            outsum += sign * (h_G + r.dot(np.array(h_GC_vec)))

        return outsum


    def find_noncontextual_ground_state(self, G=None, C=None):
        """Minimize the function defined in evaluate_classical_obj_fnc
        """
        if G is None and C is None:
            G = self.generators
            C = self.cliquereps
        G_assignments = product([1, -1], repeat=len(G))
        energies = []
        # the value assignent q to generators G is brute force
        for nu in G_assignments:
            nu = np.array(nu, dtype=int)
            if C==[]:
                nrg = self.evaluate_classical_obj_fnc(nu, np.array([]), G=G, C=C)
                energies.append([nrg, nu, []])
            else:
                # now optimize over the parameter r
                sol = minimize_scalar(lambda x: self.evaluate_classical_obj_fnc(
                    nu, np.array([np.cos(x), np.sin(x)]),G=G,C=C))
                energies.append(
                    [sol['fun'], nu, (np.cos(sol['x']), np.sin(sol['x']))])

        energy, nu, r = sorted(energies, key=lambda x: x[0])[0]

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
        


