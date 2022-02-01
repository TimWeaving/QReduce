# Qondense
---

Qubit reduction techniques such as [tapering](https://arxiv.org/abs/1701.08213) and [Contextual-Subspace VQE](https://doi.org/10.22331/q-2021-05-14-456) are effected by the underlying stabilizer subspace projection mechanism; such methods may be differentiated by the approach taken to selecting the stabilizers one wishes to project over. 

With this in mind, we include the following functionality:

- The base stabilizer subspace projection class **S3_projection**;
- Various utility functions that use the symplectic Pauli operator representation wherever possible.

This facilitates implementations of the following:

- **Tapering**
    - An exact method, i.e. the energy spectrum is preserved;
    - The stablizers are chosen to be an independent generating set of a Hamiltonian symmetry;
- **CS-VQE**
    - An approximate method, however it has been demonstrated that chemical accuracy may be achieved at a saving of qubit resource;
    - Here, the stabilizers are taken to be an independent generating set of a sub-Hamiltonian symmetry (defined by a noncontextual subset of terms) with an additional contribution encapsulating the remaining anticommuting terms therein.
    - In order to deploy CS-VQE on legitimate quantum hardware, we additionally apply the stabilizer subspace projection to an Ansatz defined on the full system so that it is consistent with our CS-VQE Hamiltonians.
 
One may define their own stabilizer selection procedure and effect the corresponding projection via S3_projection, although care must be taken to ensure the resulting Hamiltonians preserve sufficient information of the target system to yield reasonable energies (i.e. to within chemical accuracy for simulating molecular systems).

