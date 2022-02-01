import os
import numpy as np
from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo, FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, TensoredMeasFitter
from qiskit.algorithms import VQE
from qiskit_nature.runtime import VQEProgram
import qreduce.utils.qonversion_tools as qonvert
from qiskit.algorithms.optimizers import COBYLA, IMFIL
#from cs_vqe.circuit import cs_vqe_circuit


def get_quantum_instance(seed, noise=False, run_local=True, error_mitigation=None):
    algorithm_globals.random_seed = seed
    
    if not run_local:
        with open('data/token.txt', 'r') as file:
            TOKEN = file.read()
        IBMQ.save_account(TOKEN, overwrite=True)
        provider = IBMQ.get_provider(hub='ibm-q')
            
    if not noise:
        if run_local:
            backend = Aer.get_backend('statevector_simulator')
            qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
            
        
    else:
        device_backend = FakeRueschlikon()
        backend = Aer.get_backend('aer_simulator')
        noise_model = None
        device = QasmSimulator.from_backend(device_backend)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates
        
        qi = QuantumInstance(backend=backend,
                             shots=2**10,
                             seed_simulator=seed, 
                             seed_transpiler=seed,
                             coupling_map=coupling_map, 
                             noise_model=noise_model,
                             measurement_error_mitigation_cls=error_mitigation,
                             measurement_error_mitigation_shots=2**10,
                             cals_matrix_refresh_period=30)
        
    return qi


def vqe_simulation(ansatz, operator, init_params, noise=False, error_mitigation=None):
    
    def store_intermediate_result(eval_count, parameters, mean, std):
        """ Outputs intermediate data during VQE routine
        """
        counts.append(eval_count)
        prmset.append(parameters)
        values.append(mean)
        errors.append(std)
    
    counts=[]
    prmset=[]
    values=[]
    errors=[]
    
    qi = get_quantum_instance(seed=42, noise=noise, error_mitigation=error_mitigation)
    vqe = VQE(ansatz, 
              initial_point=init_params, 
              optimizer=IMFIL(maxiter=5000), 
              callback=store_intermediate_result, 
              quantum_instance=qi) 
    vqe_input_op = qonvert.dict_to_WeightedPauliOperator(operator)
    vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_op)
    
    return {'conval':vqe_run.optimal_value,
            'counts':counts,
            'prmset':prmset,
            'values':values,
            'errors':errors}


def remote_VQE(operator, qc, init_params):
    
    vqe_input_op = qonvert.dict_to_WeightedPauliOperator(operator)
    
    optimizer = {
        'name': 'QN-SPSA',  # leverage the Quantum Natural SPSA
        # 'name': 'SPSA',  # set to ordinary SPSA
        'maxiter': 200,
        'resamplings': {1: 200},  # 100 samples of the QFI for the first step, then 1 sample per step
    }

    provider = IBMQ.get_provider(
        hub='ibm-q',
        group='open',
        project='main'
    )

    options = {
        'backend_name': 'ibmq_qasm_simulator'
    }

    runtime_inputs = {
      'ansatz': qc,
      'operator': vqe_input_op,
      'optimizer': optimizer,
      'initial_parameters': init_params,
      'measurement_error_mitigation': True
    }

    job = provider.runtime.run(
        program_id='vqe',
        options=options,
        inputs=runtime_inputs
    )
    
    return job.result()


def remote_circuit_execution(mol_circ, num_sim_q, anz_op, backend='ibmq_qasm_simulator', maxiter=100, qfi_resamples=100, zero_params=False):
    #assert(type(operator)==dict)
    assert(type(mol_circ)==cs_vqe_circuit)
    
    ham_red     = mol_circ.ham_reduced[num_sim_q]
    qc          = mol_circ.build_circuit(anz_op, num_sim_q)
    if zero_params:
        init_params=np.zeros(qc.num_parameters)
    else:
        init_params = mol_circ.init_param

    interim_info = {'numfev': [],
                    'params': [],
                    'energy': [],
                    'stddev': []}

    def interim_result_callback(job_id, interim_result):
        interim_info['job_id'] = job_id
        numfev, params, energy, stddev = interim_result

        interim_info['numfev'].append(numfev)
        interim_info['params'].append(params)
        interim_info['energy'].append(energy)
        interim_info['stddev'].append(stddev)
    
    optimizer = {'name': 'QN-SPSA',
                 'maxiter': maxiter,
                 'resamplings': {int(qfi_resamples/100): qfi_resamples}}
    provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')
    options = {'backend_name': backend}
    
    vqe_input_op = qonvert.dict_to_WeightedPauliOperator(ham_red)
    runtime_inputs = {'ansatz': qc,
                      'operator': vqe_input_op,
                      'optimizer': optimizer,
                      'initial_parameters': init_params,
                      'measurement_error_mitigation': True}

    job = provider.runtime.run(
        program_id='vqe',
        options=options,
        inputs=runtime_inputs,
        callback=interim_result_callback)
    
    raw_data = job.result()

    parameters={}
    for index, param in enumerate([p.name for p in list(qc.parameters)]):
        parameters[param] = list(zip(*interim_info['params']))[index]

    data={'truegs':mol_circ.truegs,
          'target':mol_circ.cs_vqe_energy[num_sim_q],
          'noncon':mol_circ.noncon,
          'optmzr':optimizer['name'],
          'backnd':options['backend_name'],
          'numfev':interim_info['numfev'],
          'energy':interim_info['energy'],
          'params':parameters,
          'stddev':[t.real for t in interim_info['stddev']],
          'result':raw_data['optimal_value'],
          'opoint':raw_data['optimal_point'],
          'statev':raw_data['eigenstate'],
          'optime':raw_data['optimizer_time']}
    
    return data, raw_data