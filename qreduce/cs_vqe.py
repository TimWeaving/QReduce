import utils
from cs_vqe_tools_legacy import apply_rotation, greedy_dfs
from tapering import gf2_gaus_elim
from itertools import combinations, product
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple
from copy import deepcopy
import json


class cs_vqe_model:
    """Class for building the Contextual-Subspace VQE model of an input Hamiltonian
    """
    def __init__(
        self, ham: Dict[str, float], num_qubits: int, set_nc: List[str] = None
    ) -> None:
        self.ham = ham
        self.num_qubits = num_qubits
        if set_nc is not None:
            self.set_nc = set_nc
        else:
            self.set_nc = self.find_noncontextual_set()
        self.symmetry, self.cliques = self.decompose_noncontextual_set()
        self.generators = self.find_symmetry_generators()
        self.cliquereps = self.choose_clique_representatives()
        self.objfncprms = self.classical_obj_fnc_params(
            G=self.generators, C=self.cliquereps
        )
        self.ngs_energy, self.nu, self.r = self.find_noncontextual_ground_state()
        # self.stabilizer_rotations = self.determine_stabilizer_rotations()
        self.generator_assignment = {G:eig for G,eig in zip(self.generators, self.nu)}
        self.anti_clique_operator = {C:val for C,val in zip(self.cliquereps, self.r)}
        Q,t=self.unitary_partitioning_rotation()
        self.unitary_partitioning = (Q,t,False)
        self.stabilizer_rotations = self.generator_rotations()
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
        set_nc = greedy_dfs(self.ham, cutoff=search_time, criterion="weight")[1]
        return set_nc


    def decompose_noncontextual_set(self):
        """Decompose a noncontextual set into its symmetry and
        remaining pairwise anticommuting cliques
        """
        commutation_matrix = utils.adjacency_matrix(self.set_nc, self.num_qubits) == 0
        symmetry_indices = []
        for index, commutes_with in enumerate(commutation_matrix):
            if np.all(commutes_with):
                symmetry_indices.append(index)

        cliques = []
        for i, commute_list in enumerate(commutation_matrix):
            if i not in symmetry_indices:
                C = frozenset(
                    [
                        self.set_nc[j]
                        for j, commutes in enumerate(commute_list)
                        if commutes and j not in symmetry_indices
                    ]
                )
                cliques.append(C)

        symmetry = [self.set_nc[i] for i in symmetry_indices]
        cliques = [list(C) for C in set(cliques)]

        return symmetry, cliques


    def find_symmetry_generators(self):
        """Find independent generating set for noncontextual symmetry via Gaussian elimination
        """
        symmetry_matrix = utils.build_symplectic_matrix(self.symmetry)
        sym_row_reduced = gf2_gaus_elim(symmetry_matrix)
        sym_row_reduced = sym_row_reduced[
            ~np.all(sym_row_reduced == 0, axis=1)
        ]  # remove zero rows
        generators = [utils.pauli_from_symplectic(p) for p in sym_row_reduced]

        return generators


    def choose_clique_representatives(self):
        """Choose an operator from each of the cliques determined in decompose_noncontextual_set
        to complete the generating set and forms the observable C(r)
        """
        clique_reps = []
        # choose the operator in each clique which is identity on the most qubit positions
        # to choose the single pauli Z where possible
        for clique in self.cliques:
            op_weights = [(op, op.count("I")) for op in clique]
            op_weights = sorted(op_weights, key=lambda x: -x[1])
            clique_reps.append(op_weights[0][0])

        return clique_reps


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


    def find_noncontextual_ground_state(self):
        """Minimize the function defined in evaluate_classical_obj_fnc
        """
        G = self.generators
        C = self.cliquereps
        G_assignments = product([1,-1],repeat=len(self.generators))
        energies = []
        # the value assignent q to generators G is brute force
        for nu in G_assignments:
            nu = np.array(nu, dtype=int)
            # now optimize over the parameter r
            sol = minimize_scalar(lambda x: self.evaluate_classical_obj_fnc(nu,np.array([np.cos(x),np.sin(x)])))
            energies.append([sol['fun'],nu,(np.cos(sol['x']),np.sin(sol['x']))])
    
        energy, nu, r  = sorted(energies, key=lambda x: x[0])[0]

        return energy, nu, r


    def unitary_partitioning_rotation(self):
        """ Implementation of procedure described in https://doi.org/10.1103/PhysRevA.101.062322 (Section A)
        Currently works only when number of cliques M=2
        """
        order_terms = sorted(self.anti_clique_operator.items(), key=lambda x:-abs(x[1]))
        Aa,Bb = order_terms; A,a=Aa; B,b=Bb

        if a==0 or b==0:
            raise ValueError('Clique operator already contains one term')
        
        Q,coeff = utils.multiply_paulis_with_coeff(A,B)
        sign=int((1j*coeff).real)
        t=np.arctan(-b/a)*sign

        return Q, t


    def generator_rotations(self) -> Dict[int,List[Tuple[str,float,bool]]]:
        """ Implementation of procedure described in https://doi.org/10.22331/q-2021-05-14-456 (Lemma A.2)
        Returns dictionary where keys represent qubit positions and values are lists of rotations of the form
        (rotation, angle, gen_rot) to rotate stabilizers to single Pauli Z supported by the corresponding qubit
        """
        clique_rot=utils.rotate_operator(self.anti_clique_operator, [self.unitary_partitioning])
        ops_to_rotate=dict(list(self.generator_assignment.items()) + list(clique_rot.items()))
        single_Z = [G for G in ops_to_rotate.keys() if set(G)=={'I','Z'} and G.count('Z')==1]
        used_indices = [G.index('Z') for G in single_Z]
        rotations = {i:{'generator':None, 'symmetry':True, 'rotation':[]} for i in range(self.num_qubits)}
        for G in single_Z:
            Z_index = G.find('Z')
            rotations[Z_index]['generator'] = G
            if G in clique_rot:
                rotations[Z_index]['rotation'].append(self.unitary_partitioning)
                rotations[Z_index]['symmetry'] = False

        for G,assignment in ops_to_rotate.items():
            if G not in single_Z:
                print(G)
                if set(G) in [{'I', 'Z'},{'Z'}]:
                    Z_indices = [i for i in range(self.num_qubits) if G[i]=='Z']
                    available = list(set(Z_indices)-set(used_indices))
                    Z_index = available[0]
                    rot_op = ['I' for i in range(self.num_qubits)]; rot_op[Z_index]='Y'
                    rot_op=''.join(rot_op)
                    rotations[Z_index]['rotation'].append((rot_op, np.pi/2, True))
                    G_offdiag,assignment= list(utils.apply_rotation(   
                                                                P=G, 
                                                                Q=rot_op,
                                                                coeff=assignment, 
                                                                gen_rot=True
                                                                ).items())[0]
                else:
                    G_offdiag = G
                    Z_index = [i for i in range(self.num_qubits) if G_offdiag[i] in ['X','Y'] and i not in used_indices][0]
                    
                rot_op = list(deepcopy(G_offdiag))
                if rot_op[Z_index]=='X':
                    rot_op[Z_index]='Y'
                else:
                    rot_op[Z_index]='X'
                rot_op = ''.join(rot_op)
                rotations[Z_index]['generator']=G
                rotations[Z_index]['rotation'].append((rot_op, np.pi/2, True))
                if G in clique_rot:
                    rotations[Z_index]['rotation'].insert(0,self.unitary_partitioning)
                    rotations[Z_index]['symmetry'] = False
                
                used_indices.append(Z_index)

        return rotations
                

    def single_Z_stabilizers(self):
        """Dictionary of single Pauli Z stabilizers 
        and corresponding eigenvalue assignment
        """
        unrotated_generators = dict(list(self.generator_assignment.items()) + list(self.anti_clique_operator.items()))

        # apply rotations here
        all_rotations = []
        for qubit in self.stabilizer_rotations.values():
            all_rotations += qubit['rotation']
        stabilizers = utils.rotate_operator(unrotated_generators, all_rotations)
        stabilizer_eigenvals = {stab.find('Z'):int(eigval) for stab,eigval in stabilizers.items()}
        
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
        # TODO dictionary mapping qubit index of generators before and after rotation
        # perform rotations for only the generators corresponding with stab_indices

        operator_proj = {}
        for pauli in operator:
            pauli_proj = "".join([pauli[i] for i in project_qubits])
            pauli_sim = "".join(
                [pauli[i] for i in range(self.num_qubits) if i not in project_qubits]
            )
            if set(pauli_proj) in [{"I"}, {"Z"}, {"I", "Z"}]:
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

        project_clique_flag=False
        symmetry_stabs_to_project=[]
        stab_rotations = []
        for i in nc_qubits:
            stab_data = self.stabilizer_rotations[i]
            if stab_data['symmetry']:
                symmetry_stabs_to_project.append(stab_data['generator'])
            else:
                project_clique_flag=True
            stab_rotations+=stab_data['rotation']

        if project_clique_flag:
            r=np.array(self.r)
        else:
            r=np.zeros(len(self.cliquereps))
        nu = np.array([self.stabilizer_eigenvals[i] for i in nc_qubits if self.stabilizer_rotations[i]['symmetry']])
        
        updated_ngs_energy = self.evaluate_classical_obj_fnc(
            nu, r, G=symmetry_stabs_to_project,C=self.cliquereps
        )
        # rotate just the stabilized qubits, for example via:
        # rotations = [self.stabilizer_rotations[stab] for stab in stabs_to_project]
        # rotate Hamiltonian terms accordingly

        ham_nc = {op: coeff for op, coeff in self.ham.items() if op in self.set_nc}
        ham_cx = {op: coeff for op, coeff in self.ham.items() if op not in self.set_nc}
        ham_nc_rot = utils.rotate_operator(ham_nc,stab_rotations)
        ham_cx_rot = utils.rotate_operator(ham_cx,stab_rotations)

        append_terms={}
        for op,coeff in ham_nc_rot.items():
            op_sim = "".join([op[i] for i in cs_qubits])
            if set(op_sim) in [{"Z"}, {"I", "Z"}]:
                append_terms[op] = coeff
        
        ham_contextual = utils.sum_operators([ham_cx_rot, append_terms])

        nu = np.array([self.stabilizer_eigenvals[i] for i in nc_qubits])
        ham_cs = self._stabilizer_subspace_projection(
            operator=ham_contextual, project_qubits=nc_qubits, eigvals=nu
        )

        identity = "".join(["I" for i in range(num_sim_q)])
        if identity in ham_cs:
            ham_cs[identity] += updated_ngs_energy
        else:
            ham_cs[identity] = updated_ngs_energy

        return ham_cs


if __name__ == "__main__":
    with open('model_data.json', 'r') as infile:
        mol_data = json.load(infile)
    mol=mol_data['HF_STO-3G_SINGLET']
    ham=mol['ham']
    n_q=mol['num_qubits']
    cs_vqe_mol = cs_vqe_model(ham, n_q)
    print("Symmetry:", cs_vqe_mol.symmetry)
    print("Cliques:", cs_vqe_mol.cliques)
    print("Generators:", cs_vqe_mol.generators)
    print("Clique reps:", cs_vqe_mol.cliquereps)
    print("Noncon nrg:", cs_vqe_mol.ngs_energy)
    print("G assignment:", cs_vqe_mol.nu)
    print("r vector:", cs_vqe_mol.r)
    print("Stabilizers:", cs_vqe_mol.stabilizers)
    print("Stab qubits:", cs_vqe_mol.stab_qubits)

    ham_mat = sum(
        coeff * utils.pauli_matrix(op) for op, coeff in cs_vqe_mol.ham.items()
    )
    exact = sorted(np.linalg.eigh(ham_mat)[0])[0]
    print('\n exact:',exact, '\n')
    for comb in combinations(range(n_q), 4):
        comb=list(comb)
        ham_cs = cs_vqe_mol.contextual_subspace_hamiltonian(comb)
        ham_mat = sum(coeff * utils.pauli_matrix(op) for op, coeff in ham_cs.items())
        print('error:', sorted(np.linalg.eigh(ham_mat)[0])[0]-exact)
    
    
    #ham_cs = cs_vqe_mol.contextual_subspace_hamiltonian([1,4,0])
    #ham_mat = sum(coeff * utils.pauli_matrix(op) for op, coeff in ham_cs.items())
    #print(sorted(np.linalg.eigh(ham_mat)[0])[0])