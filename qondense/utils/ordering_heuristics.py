import numpy as np
from copy import deepcopy
from qondense.cs_vqe import cs_vqe
from typing import Dict, List
from openfermion import get_fermion_operator, jordan_wigner, FermionOperator
from openfermionpyscf import PyscfMolecularData
from qondense.utils.qonversion_tools import QubitOperator_to_dict
from qondense.tapering import tapering
from qondense.utils.operator_toolkit import exact_gs_energy
from itertools import combinations

class ordering_heuristics(cs_vqe):
    """ Class for assessing various generator removal ordering heuristics
    """
    def __init__(self,
                calculated_molecule: PyscfMolecularData)-> None:
        """
        """
        dashes = "------------------------------------------------"
        # Orbital nums and HF state
        self.n_electrons  = calculated_molecule.n_electrons
        self.n_qubits     = 2*calculated_molecule.n_orbitals
        self.hf_state= [1 for i in range(self.n_electrons)]+[0 for i in range(self.n_qubits-self.n_electrons)]
        hf_string   = ''.join([str(i) for i in self.hf_state])
        print(dashes)
        print('Information concerning the full system:')
        print(dashes)
        print(f'Number of qubits in full problem: {self.n_qubits}')
        print(f'The Hartree-Fock state is |{hf_string}>')
        
        # reference energies
        self.hf_energy = calculated_molecule.hf_energy
        self.mp_energy = calculated_molecule.mp2_energy
        self.fci_energy = calculated_molecule.fci_energy
        print(f'Hartree-Fock energy   = {self.hf_energy: .8f}')
        print(f'Møller–Plesset energy = {self.mp_energy: .8f}')
        if self.fci_energy is not None:
            print(f'FCI energy            = {self.fci_energy:.8f}')
        print(dashes)
        
        # Hamiltonian
        ham_ferm_data = calculated_molecule.get_molecular_hamiltonian()
        self.ham_fermionic = get_fermion_operator(ham_ferm_data)
        ham_jw = jordan_wigner(self.ham_fermionic)
        self.ham_dict = QubitOperator_to_dict(ham_jw, self.n_qubits)
        ham_sor = self.second_order_response()

        # taper Hamiltonian
        taper_hamiltonian = tapering(hamiltonian=self.ham_dict, 
                               ref_state=self.hf_state)
        self.ham_tap = taper_hamiltonian.taper_it()
        self.sor_tap = taper_hamiltonian.taper_it(ham_sor)
        self.n_taper = taper_hamiltonian.n_taper
        self.hf_tapered = taper_hamiltonian.taper_ref_state()
        hf_tap_str = ''.join([str(i) for i in self.hf_tapered])
        self.HL_index = self.hf_tapered.index(0) #index HOMO-LUMO gap
        print("Tapering information:")
        print(dashes)
        print(f'We are able to taper {self.n_taper} qubits from the Hamiltonian')
        print(f'The symmetry sector is {taper_hamiltonian.symmetry_sec}')
        print(f'The tapered Hartree-Fock state is |{hf_tap_str}>')
        print(dashes)

        # build CS-VQE model
        terms_noncon = [op for op in self.ham_tap._dict if set(op) in [{'I'},{'Z'},{'I','Z'}]]
        super().__init__(hamiltonian=self.ham_tap._dict,
                        noncontextual_set=terms_noncon,
                        ref_state=self.hf_tapered)
        print("CS-VQE information:")
        print(dashes)
        print("Noncontextual GS energy:", self.ngs_energy)#, ' // matches original?', match_original)
        print("Symmetry generators:    ", self.generators)
        if self.cliquereps != []:
            print("Clique representatives: ", self.cliquereps)
            print("Clique operator coeffs: ", self.r)
        print("Generator eigenvalues:  ", self.nu)
        
        print(dashes)

    
    def sor_data(self):
        """ Calculate the w(i) function 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = {i:0 for i in range(self.n_qubits)}
        for f_op,coeff in self.ham_fermionic.terms.items():
            if len(f_op)==2:
                (p,p_ex),(q,q_ex) = f_op
                # self-interaction terms p==q
                if p==q:
                    w[p] += coeff
            if len(f_op)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = f_op
                #want p==r and q==s for hopping
                if p==r:
                    if q==s and self.hf_state[q]==1:
                        w[p]+=coeff
        return w


    def second_order_response(self):
        """ Calculate the I_a Hamiltonian term importance metric 
        as in https://arxiv.org/pdf/1406.4920.pdf
        """
        w = self.sor_data()
        f_out = FermionOperator()
        for H_a,coeff in self.ham_fermionic.terms.items():
            if len(H_a)==4:
                (p,p_ex),(q,q_ex),(r,r_ex),(s,s_ex) = H_a
                Delta_pqrs = abs(w[p]+w[q]-w[r]-w[s])
                if Delta_pqrs == 0:
                    I_a = 1e15
                else:
                    I_a = (abs(coeff)**2)/Delta_pqrs
                
                f_out += FermionOperator(H_a, I_a)
        f_out_jw = jordan_wigner(f_out)
        f_out_q = QubitOperator_to_dict(f_out_jw, self.n_qubits)
        return f_out_q
            

    ###############################################
    # heuristics with HOMO-LUMO as starting point #
    ###############################################

    def HOMO_LUMO_outwards(self):
        """
        """
        x = np.arange(0,self.n_qubits,1)
        a = (1-(2*self.HL_index)/self.n_qubits)*(x-self.n_qubits/2)
        b = self.HL_index*np.log(self.HL_index/self.n_qubits)*np.exp(-((x-self.HL_index)**2)/self.n_qubits)
        f = a+b*10/3
        g = f/np.min(f)
        stab_removal_order = []
        for i in np.arange(0,2*self.n_qubits+1):
            threshold = 1-i/self.n_qubits
            stab_indices = list(np.where(g>=threshold)[0])
            if stab_indices != []:
                if stab_indices not in stab_removal_order: 
                    stab_removal_order.append(stab_indices)

        stab_order = [list(set(range(self.n_qubits))-set(stabs)) for stabs in stab_removal_order 
                        if set(stabs)!=set(range(self.n_qubits))]
        return stab_order


    def unocc_first(self):
        """
        """
        order = list(range(self.HL_index, self.n_qubits))+list(range(self.HL_index))[::-1]
        stab_order = [order[i:] for i in range(1,self.n_qubits)]
        return stab_order


    ###################################
    # Hamiltonian ordering heuristics #
    ###################################

    def stab_index_from_term_weighting(self,ham_ordering):
        """
        """
        order = []
        used = []
        for pauli, coeff in ham_ordering:
            # index qubit positions of X,Y paulis
            relax_G_indices = [index for index,Pi in enumerate(pauli) if Pi not in ['I', 'Z']]
            #relax_G_indices = [index for index,G in enumerate(self.generators) 
            #                        if 'Z' in [G[i] for i in off_diag]]
            # set difference with those already included
            relax_G_indices = list(set(relax_G_indices)-set(used))
            if relax_G_indices != []:
                used += relax_G_indices
                used = list(set(used))
                order.append(deepcopy(used))
                
        stab_order = [list(set(range(self.n_qubits))-set(o)) for o in order if set(o)!=set(range(self.n_qubits))]
        
        return stab_order


    def by_Hamiltonian_magnitude(self):
        """
        """
        ham_sorted = sorted(self.ham_tap._dict.items(), key=lambda x:-abs(x[1]))
        return self.stab_index_from_term_weighting(ham_sorted)


    def by_second_order_response(self):
        """
        """
        ham_sorted = sorted(self.sor_tap._dict.items(), key=lambda x:-abs(x[1]))
        return self.stab_index_from_term_weighting(ham_sorted)

    ##########################
    # brute-force heuristics #
    ##########################

    # <!> only use if FCI energy is not None <!>

    def generator_search(self, full=False):
        stab_index_pool = list(range(len(self.generators)))
        stab_order = []
        optimal_errors = {}
        for num_sim_q in range(1,self.n_qubits):
            if num_sim_q < 5:
                m_type='dense'
            else:
                m_type='sparse'
            cs_vqe_errors = []
            if full:
                stab_index_pool = list(range(len(self.generators)))
            for order in combinations(stab_index_pool, self.n_qubits - num_sim_q):
                order = list(order)
                ham_cs = self.contextual_subspace_hamiltonian(stabilizer_indices=order)
                cs_energy, cs_vector = exact_gs_energy(ham_cs, matrix_type=m_type)
                cs_vqe_errors.append((cs_energy-self.fci_energy, order))

            cs_vqe_errors = sorted(cs_vqe_errors, key=lambda x:x[0])
            error, stab_index_pool = cs_vqe_errors[0]
            stab_order.append(list(stab_index_pool))
        return stab_order
        
        
    def generator_walk(self):
        return self.generator_search()

    def perfect_removal(self):
        return self.generator_search(full=True)


    ##########################################
    # Finally, to evaluate heuristic errors: #
    ##########################################

    def heuristic_errors(self, heuristic, print_info=False):
        assert(heuristic in ["a", "b", "c", "d", "e", "f"])
        # generator search heuristics not wise for large moleulces!
        if heuristic in ["e", "f"]:
            assert(self.n_qubits<=10)
        heuristics={
            "a":{"name":"HOMO_LUMO_outwards",       "func":self.HOMO_LUMO_outwards},
            "b":{"name":"unocc_first",              "func":self.unocc_first},
            "c":{"name":"by_Hamiltonian_magnitude", "func":self.by_Hamiltonian_magnitude},
            "d":{"name":"by_second_order_response", "func":self.by_second_order_response},
            "e":{"name":"generator_walk",           "func":self.generator_walk},
            "f":{"name":"perfect_removal",          "func":self.perfect_removal}
        }
        optimal_energy={'heuristic':heuristics[heuristic]["name"]}
        stab_order = heuristics[heuristic]["func"]()
        for o in stab_order:
            num_sim_q = self.n_qubits-len(o)
            if num_sim_q <=18:
                if num_sim_q < 5:
                    m_type='dense'
                else:
                    m_type='sparse'
                ngs = ''.join([str(self.hf_tapered[i]) for i in range(self.n_qubits) if i not in o])
                ngs = np.eye(1,2**num_sim_q,int(ngs, 2))
                ham_cs = self.contextual_subspace_hamiltonian(stabilizer_indices=o)
                best_energy, cs_vector = exact_gs_energy(ham_cs, matrix_type=m_type, initial_guess=ngs)
                if print_info:
                    print(f'Number of qubits simulated: {num_sim_q}')
                    print(f'CS-VQE error w.r.t. HF energy: {best_energy-self.hf_energy: .10f}')
                    print(f'CS-VQE error w.r.t. MP2 energy:{best_energy-self.mp_energy: .10f}')
                    if self.fci_energy is not None:
                        print(f'CS-VQE error w.r.t. FCI energy:{best_energy-self.fci_energy: .10f}')
                    print()

                optimal_energy[num_sim_q]={}
                optimal_energy[num_sim_q]['energy'] = best_energy
                if self.fci_energy is not None:
                    optimal_energy[num_sim_q]['error'] = best_energy-self.fci_energy
                else:
                    optimal_energy[num_sim_q]['error'] = 'na'
                optimal_energy[num_sim_q]['stab_indices'] = o

        return optimal_energy