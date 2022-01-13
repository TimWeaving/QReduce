import qreduce.utils as utils
from qreduce.cs_vqe_tools_legacy import apply_rotation, greedy_dfs
from qreduce.tapering import gf2_gaus_elim
from itertools import combinations, product, permutations
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple
from copy import deepcopy
import json


class cs_vqe_model:
    """Class for building the Contextual-Subspace VQE model of an input Hamiltonian
    """

    def __init__(
        self, ham: Dict[str, float], num_qubits: int, noncontextual_set: List[str] = None
    ) -> None:
        self.ham = ham
        self.num_qubits = num_qubits
        if noncontextual_set is not None:
            self.noncontextual_set = noncontextual_set
        else:
            self.noncontextual_set = self.find_noncontextual_set()
        self.ham_noncontextual = {op:coeff for op,coeff in self.ham.items() if op in self.noncontextual_set}
        self.ham_contextual = {op:coeff for op,coeff in self.ham.items() if op not in self.noncontextual_set}
        self.symmetry, self.cliques = self.decompose_noncontextual_set()
        self.cliquereps = self.choose_clique_representatives()
        
        for op in self.cliquereps:
            X_index = op.find('X')
            if X_index != -1:
                self.anti_clique_qubit=X_index
        
        self.generators, self.Z_indices = self.find_symmetry_generators()
        self.objfncprms = self.classical_obj_fnc_params(
            G=self.generators, C=self.cliquereps
        )
        self.ngs_energy, self.nu, self.r = self.find_noncontextual_ground_state()
        self.generator_assignment = {G: eig for G,
                                     eig in zip(self.generators, self.nu)}
        self.anti_clique_operator = {C: val for C,
                                     val in zip(self.cliquereps, self.r)}
        Q, t = self.unitary_partitioning_rotation()
        self.unitary_partitioning = (Q, t, False)
        self.stabilizer_map = self.generator_rotations()

        for i, rot_data in self.stabilizer_map.items():
            if rot_data['symmetry']==False:
                self.anti_clique_qubit = i
        (
            self.stabilizer_eigenvals,
            self.stabilizers,
            self.stab_qubits,
            self.free_qubits
        ) = self.single_Z_stabilizers()

    def find_noncontextual_set(self, search_time=3):
        """Method for extracting a noncontextual subset of the hamiltonian terms
        """
        # for now uses the legacy greedy DFS approach
        # to be updated once more efficient/effective methods are identified
        noncontextual_set = greedy_dfs(self.ham, cutoff=search_time,
                            criterion="weight")[1]
        return noncontextual_set

    def decompose_noncontextual_set(self):
        """Decompose a noncontextual set into its symmetry and
        remaining pairwise anticommuting cliques
        """
        commutation_matrix = utils.adjacency_matrix(
            self.noncontextual_set, self.num_qubits) == 0
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
        # about the collective Hamiltonian... results in chemical accuracy being achieved faster
        for clique in self.cliques:
            op_weights = [(op, op.count("I")) for op in clique]
            op_weights = sorted(op_weights, key=lambda x: x[1])
            clique_reps.append(op_weights[0][0])

        return clique_reps

    def find_symmetry_generators(self):
        """Find independent generating set for noncontextual symmetry via Gaussian elimination
        """
        A_ij_ops=[]
        for rep,clique in zip(self.cliquereps,self.cliques):
            for C in clique:
                if C!=rep:
                    A_ij = utils.multiply_paulis(C,rep)
                    A_ij_ops.append(A_ij)
        
        universally_commuting = set(self.symmetry+A_ij_ops)
        symmetry_matrix = utils.build_symplectic_matrix(universally_commuting)

        sym_row_reduced = gf2_gaus_elim(symmetry_matrix)
        sym_row_reduced = sym_row_reduced[
            ~np.all(sym_row_reduced == 0, axis=1)
        ]  # remove zero rows
        generators = [utils.pauli_from_symplectic(p) for p in sym_row_reduced]
        
        heavy_generators=[]
        for P in generators:
            for Q in generators:
                if P!=Q:
                    P=utils.multiply_paulis(P,Q)
            heavy_generators.append(P)

        non_X_qubits = [i for i in range(self.num_qubits) if i!=self.anti_clique_qubit]
        for Z_indices in permutations(non_X_qubits):
            if all([G[i]=='Z' for G,i in zip(heavy_generators,Z_indices)]):
                break
                    
        return heavy_generators, Z_indices

    def classical_obj_fnc_params(self, G: List[str], C: List[str]):
        """
        """
        # construct the closure of the commuting generating set
        G_combs = []
        for i in range(1, len(G) + 1):
            G_combs += list(combinations(G, i))
        G_closure = [
            (utils.multiply_pauli_list(comb), [G.index(op) for op in comb])
            for comb in G_combs
        ]
        G_closure.append(("".join(["I" for i in range(self.num_qubits)]), []))

        # extract the relevant coefficients for the sum over G_closure as in eq 13/14 of https://arxiv.org/pdf/2002.05693.pdf
        obj_fnc_params = []
        for G_op, q_indices in G_closure:
            try:
                h_G = self.ham[G_op]
            except:
                h_G = 0
            C_vec = []
            for C_op in C:
                GC_op = utils.multiply_paulis(G_op, C_op)
                try:
                    h_GC = self.ham[GC_op]
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
            self.anti_clique_operator.items(), key=lambda x: -abs(x[1]))
        Aa, Bb = order_terms
        A, a = Aa
        B, b = Bb

        if a == 0 or b == 0:
            raise ValueError('Clique operator already contains one term')

        Q, coeff = utils.multiply_paulis_with_coeff(A, B)
        sign = int((1j*coeff).real)
        t = np.arctan(-b/a)*sign
        if abs(a+np.cos(t))<1e-15:
            t+=np.pi

        return Q, t

    def generator_rotations(self) -> Dict[int, List[Tuple[str, float, bool]]]:
        """ Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        Returns dictionary where keys represent qubit positions and values are lists of rotations of the form
        (rotation, angle, gen_rot) to rotate stabilizers to single Pauli Z supported by the corresponding qubit
        """
        used_qubits = []
        clique_rot = utils.rotate_operator(self.anti_clique_operator, [
                                                self.unitary_partitioning])
        stabilizer_map = {i: {'generator': None, 'symmetry': True,
                                'rotation': []} for i in range(self.num_qubits)}
        ops_to_rotate = list(clique_rot.keys())+self.generators
        
        rotations=[]
        for G in ops_to_rotate:
            G_rot = list(utils.rotate_operator({G:1}, rotations).keys())[0]

            if G_rot.count('Z')==1:
                used_qubits.append(G_rot.index('Z'))
                if G_rot in clique_rot:
                    rotations[Z_index]['symmetry'] = False
                    rotations[Z_index]['rotation'].append(self.unitary_partitioning)
            else:
                if set(G_rot) in [{'Z'},{'I','Z'}]:
                    Z_indices = [i for i in range(self.num_qubits) if G_rot[i]=='Z' and i not in used_qubits]
                    if G in clique_rot:
                        Z_index = self.anti_clique_qubit
                    else:
                        Z_index = Z_indices[0]
                    rot_op = utils.amend_string_index(['I' for i in range(self.num_qubits)],Z_index,'Y')
                    stabilizer_map[Z_index]['rotation'].append((rot_op, np.pi/2, True))
                    rotations.append((rot_op, np.pi/2, True))
                    G_offdiag = utils.amend_string_index(G_rot,Z_index,'X')
                else:
                    G_offdiag = deepcopy(G_rot)
                    Z_index = [i for i in range(self.num_qubits) if G_offdiag[i] in [
                                'X', 'Y'] and i not in used_qubits][0]
                
                stabilizer_map[Z_index]['generator'] = G
                if G_offdiag[Z_index] == 'X':
                    rot_op = utils.amend_string_index(G_offdiag,Z_index,'Y')
                else:
                    rot_op = utils.amend_string_index(G_offdiag,Z_index,'X')
                stabilizer_map[Z_index]['rotation'].append((rot_op, np.pi/2, True))
                rotations.append((rot_op, np.pi/2, True))
                
                if G in clique_rot:
                    stabilizer_map[Z_index]['rotation'].insert(0, self.unitary_partitioning)
                    stabilizer_map[Z_index]['symmetry'] = False
                    rotations.insert(0,self.unitary_partitioning)
                    
            used_qubits.append(Z_index)

        return stabilizer_map

    def single_Z_stabilizers(self):
        """Dictionary of rotated stabilizers and corresponding eigenvalue assignments
        """
        unrotated_generators = dict(list(
            self.generator_assignment.items()) + list(self.anti_clique_operator.items()))
        # order rotations so that unitary partitioning performed first
        rot_order=list(range(self.num_qubits))
        rot_order.pop(self.anti_clique_qubit)
        rot_order.insert(0,self.anti_clique_qubit)
        all_rotations = []
        for i in rot_order:
            all_rotations += self.stabilizer_map[i]['rotation']

        # perform rotation
        stabilizers = utils.rotate_operator(
            unrotated_generators, all_rotations)
        for stab,eigval in stabilizers.items():
            self.stabilizer_map[stab.find('Z')]['stabilizer'] = (stab,eigval)
        
        stabilizer_eigenvals = {stab.find('Z'): int(
            eigval) for stab, eigval in stabilizers.items()}
        
        # determine qubits that are stabilized and those that are free (always empty?)
        stab_qubits = [S.find("Z") for S in stabilizers]
        free_qubits = list(set(range(self.num_qubits)) - set(stab_qubits))

        return stabilizer_eigenvals, stabilizers, stab_qubits, free_qubits

    def _stabilizer_subspace_projection(
        self, operator: Dict[str, float], project_qubits: List[int], eigvals: List[int]
    ) -> Dict[str, float]:
        """ method for projecting an operator over fixed qubit positions
        stabilized by single Pauli Z operators (obtained via Clifford operations)
        """
        # qubits for projection must be ordered
        project_qubits = sorted(project_qubits)
        #num_proj_q = len(project_qubits)
        #single_Z_ops = [utils.amend_string_index(['I' for j in range(num_proj_q)], i, 'Z') for i in range(num_proj_q)]
        operator_proj = {}
        for pauli in operator:
            pauli_proj = "".join([pauli[i] for i in project_qubits])
            pauli_sim = "".join(
                [pauli[i]
                    for i in range(self.num_qubits) if i not in project_qubits]
            )
            #commuting = [utils.symplectic_inner_product(pauli_proj,Z)==0 for Z in single_Z_ops]
            if set(pauli_proj) in [{"I"}, {"Z"}, {"I", "Z"}]: #np.all(commuting):
                sign = np.prod(
                    [e for e, stab in zip(eigvals, pauli_proj) if stab == "Z"]
                )
                if pauli_sim not in operator_proj:
                    operator_proj[pauli_sim] = sign * operator[pauli]
                else:
                    operator_proj[pauli_sim] += sign * operator[pauli]

        return operator_proj

    def contextual_subspace_hamiltonian(self, sim_qubits: List[int]):
        cs_qubits = sim_qubits + self.free_qubits
        nc_qubits = list(set(range(self.num_qubits)) - set(cs_qubits))
        num_sim_q = len(cs_qubits)

        if self.anti_clique_qubit in nc_qubits:
            # if not simulating anticommuting clique qubit the perform unitary partitioning first
            nc_qubits.pop(nc_qubits.index(self.anti_clique_qubit))
            nc_qubits.insert(0, self.anti_clique_qubit)
            C_set = self.cliquereps
            C_vec = self.r
        else:
            C_set = []
            C_vec = np.array([])

        symmetry_stabs_to_project = []
        symmetry_stabs_nu_project=[]
        stab_rotations = []
        for i in nc_qubits:
            rot_data = self.stabilizer_map[i]
            stab_rotations += rot_data['rotation']
            if rot_data['symmetry']:
                gen,eigval = rot_data['generator']
                symmetry_stabs_to_project.append(gen)
                symmetry_stabs_nu_project.append(eigval)

        # evaluate updated classical objective function for reduced noncontextual Hamiltonian energy
        updated_ngs_energy = self.evaluate_classical_obj_fnc(
            nu=np.array(symmetry_stabs_nu_project), r=C_vec, G=symmetry_stabs_to_project, C=C_set
        )

        ham_rotated = utils.rotate_operator(self.ham, stab_rotations, cleanup=False)
        ham_contextual_rotated = utils.rotate_operator(self.ham_contextual, stab_rotations, cleanup=False)
        #ham_noncontextual_rotated = utils.rotate_operator(self.ham_noncontextual, stab_rotations, cleanup=True)

        operator_to_project = {}
        for op, coeff in ham_rotated.items():
            op_sim = "".join([op[i] for i in cs_qubits])
            if set(op_sim) in [{"Z"},{"I", "Z"}] or op in ham_contextual_rotated:
                operator_to_project[op] = coeff
        #operator_to_project = utils.sum_operators([operator_to_project, ham_contextual_rotated])
        nc_qubits = sorted(nc_qubits)
        nu = np.array([self.stabilizer_map[i]['stabilizer'][1] for i in nc_qubits])
        ham_cs = self._stabilizer_subspace_projection(
            operator=operator_to_project, project_qubits=nc_qubits, eigvals=nu
        )

        identity = "".join(["I" for i in range(num_sim_q)])
        if identity in ham_cs:
            print('Term was projected to identity')
            ham_cs[identity] += updated_ngs_energy
        else:
            ham_cs[identity] = updated_ngs_energy

        return utils.cleanup_operator(ham_cs)

    def contextual_subspace_hamiltonian_2(self, sim_qubits: List[int]):
        cs_qubits = sim_qubits + self.free_qubits
        nc_qubits = list(set(range(self.num_qubits)) - set(cs_qubits))
        nu_proj = np.array([self.stabilizer_map[i]['stabilizer'][1] for i in nc_qubits])

        if self.anti_clique_qubit in nc_qubits:
            # if not simulating anticommuting clique qubit then perform unitary partitioning first
            nc_qubits.pop(nc_qubits.index(self.anti_clique_qubit))
            nc_qubits.insert(0, self.anti_clique_qubit)

        stab_rotations=[]
        for i in nc_qubits:
            rot_data = self.stabilizer_map[i]
            stab_rotations += rot_data['rotation']

        rot_order=list(range(self.num_qubits))
        rot_order.pop(self.anti_clique_qubit)
        rot_order.insert(0,self.anti_clique_qubit)
        all_rotations = []
        for i in rot_order:
            all_rotations += self.stabilizer_map[i]['rotation']
        stab_rotations = all_rotations
        

        ham_rotated = utils.rotate_operator(self.ham, stab_rotations, cleanup=False)
        ham_cs = self._stabilizer_subspace_projection(
            operator=ham_rotated, project_qubits=nc_qubits, eigvals=nu_proj
        )

        return utils.cleanup_operator(ham_cs)