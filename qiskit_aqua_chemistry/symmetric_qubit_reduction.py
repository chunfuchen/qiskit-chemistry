
import copy
import itertools
from collections import OrderedDict

import numpy as np
from scipy import linalg as scila
from qiskit.tools.qi.pauli import label_to_pauli
from qiskit_aqua import Operator, get_algorithm_instance
import qutip as qt
from qiskit_aqua_chemistry.drivers import ConfigurationManager
from qiskit_aqua_chemistry.core import get_chemistry_operator_instance
from qiskit_aqua_chemistry import FermionicOperator


from qiskit_aqua_chemistry import AquaChemistryError, QMolecule
from pyscf import gto, scf, ao2mo
from pyscf.scf.hf import get_ovlp
import scipy

def find_good_sq_op(Pauli_symmetries):
    #symmetries: a list of Pauli object
    temp = []
    for symm in Pauli_symmetries:
        temp.append(np.concatenate(symm.v, symm.w))

    stacked_symmetries = np.stack(temp)
    symm_shape = stacked_symmetries.shape

    for row in range(symm_shape[0]):

        # Pauli_symmetries.append(Pauli(stacked_symmetries[row, : symm_shape[1] // 2],
        #                               stacked_symmetries[row, symm_shape[1] // 2:]))

        stacked_symm_del = np.delete(stacked_symmetries, (row), axis=0)
        for col in range(symm_shape[1] // 2):

            # case symmetries other than one at (row) have Z or I on col qubit
            Z_or_I = True
            for symm_idx in range(symm_shape[0] - 1):
                if not (stacked_symm_del[symm_idx, col] == 0
                        and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)):
                    Z_or_I = False
            if Z_or_I:
                if ((stacked_symmetries[row, col] == 1 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 0) or
                    (stacked_symmetries[row, col] == 1 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                    sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                           np.zeros(symm_shape[1] // 2)))
                    sq_paulis[row].v[col] = 0
                    sq_paulis[row].w[col] = 1
                    sq_list.append(col)
                    break

            # case symmetries other than one at (row) have X or I on col qubit
            X_or_I = True
            for symm_idx in range(symm_shape[0] - 1):
                if not (stacked_symm_del[symm_idx, col] in (0, 1) and
                        stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0):
                    X_or_I = False
            if X_or_I:
                if ((stacked_symmetries[row, col] == 0 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                    (stacked_symmetries[row, col] == 1 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                    sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                    sq_paulis[row].v[col] = 1
                    sq_paulis[row].w[col] = 0
                    sq_list.append(col)
                    break

            # case symmetries other than one at (row)  have Y or I on col qubit
            Y_or_I = True
            for symm_idx in range(symm_shape[0] - 1):
                if not ((stacked_symm_del[symm_idx, col] == 1 and
                         stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1)
                        or (stacked_symm_del[symm_idx, col] == 0 and
                            stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0)):
                    Y_or_I = False
            if Y_or_I:
                if ((stacked_symmetries[row, col] == 0 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                    (stacked_symmetries[row, col] == 1 and
                     stacked_symmetries[row, col + symm_shape[1] // 2] == 0)):
                    sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                    sq_paulis[row].v[col] = 1
                    sq_paulis[row].w[col] = 1
                    sq_list.append(col)
                    break

    for symm_idx, Pauli_symm in enumerate(Pauli_symmetries):
        cliffords.append(Operator([[1/np.sqrt(2), Pauli_symm], [1/np.sqrt(2), sq_paulis[symm_idx]]]))

    return sq_paulis, cliffords, sq_list


def mapping_with_symmetry_reduction(fer_op, swap_index):

    modes = fer_op.modes

    swap_index_reindexed = []
    for symmtery in swap_index:
        symmtery_list = []
        for swap_pair in symmtery:
            i, j = swap_pair
            symmtery_list.append([i % modes, j % modes])
        swap_index_reindexed.append(symmtery_list)

    for symmtery in swap_index_reindexed:
        flatten_indices = [idx for swap_pair in symmtery for idx in swap_pair]
        if len(flatten_indices) != len(set(flatten_indices)):
            raise AquaChemistryError('Do not support overlapped swap indices within one symmtery: \
                                        {}'.format(symmtery))

    # build matrix R
    r_matrix = np.eye(fer_op.modes, dtype=fer_op.h1.dtype)
    for swap_pair in swap_index_reindexed:
        for i, j in swap_pair:
            r_matrix[i, j] = r_matrix[j, i] = 1.0
            r_matrix[i, i] = r_matrix[j, j] = 0.0
    # print(r_matrix)
    # check the build r_matrix
    temp_fer_op = copy.deepcopy(fer_op)
    temp_fer_op.transform(r_matrix)
    # print(temp_fer_op.h1)
    # print(fer_op.h1)
    if temp_fer_op != fer_op:
        raise AquaChemistryError('The specificed swap index is invalid: {}'.format(swap_index))

    g_matrix = -1j * scila.logm(r_matrix)
    d_matrix, v_matrix = scila.eig(g_matrix)

    # check the build d_matrix
    d_matrix = np.around(d_matrix, 5)
    for eig in d_matrix:
        if eig != 0.0 and eig != 3.14159:
            raise AquaChemistryError('The specificed swap index is invalid. \
                                        Eigenvalues of G includes: {}'.format(eig))

    new_fer_op = copy.deepcopy(fer_op)
    new_fer_op.transform(v_matrix)

    pi_index = np.where(d_matrix == 3.14159)[0]
    s_pauli = ['I'] * modes
    s_pauli[pi_index[0]] = 'X'
    s_pauli = ''.join(s_pauli)

    s_op = Operator(paulis=[[1.0, label_to_pauli(s_pauli)]])

    clifford_pauli = ''
    for i in range(modes):
        clifford_pauli += 'I' if i not in pi_index else 'Z'
    clifford_op = Operator(paulis=[[1.0, label_to_pauli(clifford_pauli)]])
    clifford_op += s_op
    clifford_op.scaling_coeff(1.0 / np.sqrt(2))

    qubit_op = new_fer_op.mapping(map_type='jordan_wigner')

    ret_ops = []
    for coeff in itertools.product([1, -1], repeat=1):
        ret_ops.append(Operator.qubit_tapering(qubit_op, [clifford_op],
                                               [pi_index[0]], list(coeff)))
    return ret_ops, new_fer_op

def mapping_with_symmetry_reduction_new(fer_op, swap_index):

    modes = fer_op.modes

    swap_index_reindexed = []
    for symmtery in swap_index:
        symmtery_list = []
        for swap_pair in symmtery:
            i, j = swap_pair
            symmtery_list.append([i % modes, j % modes])
        swap_index_reindexed.append(symmtery_list)

    for symmtery in swap_index_reindexed:
        flatten_indices = [idx for swap_pair in symmtery for idx in swap_pair]
        if len(flatten_indices) != len(set(flatten_indices)):
            raise AquaChemistryError('Do not support overlapped swap indices within one symmtery: \
                                        {}'.format(symmtery))

    # build matrix Rs
    r_matrices = []
    for swap_pair in swap_index_reindexed:
        r_matrix = np.eye(fer_op.modes, dtype=fer_op.h1.dtype)
        for i, j in swap_pair:
            r_matrix[i, j] = r_matrix[j, i] = 1.0
            r_matrix[i, i] = r_matrix[j, j] = 0.0
        r_matrices.append(r_matrix)
    # print(r_matrix)
    # check the build r_matrix
    # for r_matrix in r_matrices:
    #     temp_fer_op = copy.deepcopy(fer_op)
    #     temp_fer_op.transform(r_matrix)
    #     # print(temp_fer_op.h1)
    #     # print(fer_op.h1)
    #     if temp_fer_op != fer_op:
    #         raise AquaChemistryError('The specificed swap index is invalid: {}'.format(swap_index))

    g_matrices = []
    for r_matrix in r_matrices:
        g_matrix = -1j * scila.logm(r_matrix)
        g_matrices.append(g_matrix)

    sim_dia = []
    for g_matrix in g_matrices:
        sim_dia.append(qt.Qobj(g_matrix))

    d_v = qt.simdiag(sim_dia)
    print(d_v)
    d_matrices = d_v[0]
    v_matrix = np.hstack([d_v[1][i].data.toarray() for i in range(modes)])

    # d_matrix, v_matrix = scila.eig(g_matrix)

    # # check the build d_matrix
    # d_matrix = np.around(d_matrix, 5)
    for eig in d_matrices.flatten():
        if not (np.isclose(eig, 0.0) or np.isclose(eig, np.pi)):
            raise AquaChemistryError('The specificed swap index is invalid. \
                                        Eigenvalues of G includes: {}'.format(eig))

    new_fer_op = copy.deepcopy(fer_op)
    new_fer_op.transform(v_matrix)

    qubit_op = new_fer_op.mapping(map_type='jordan_wigner')
    Pauli_symmetries, sq_paulis, cliffords, sq_list = qubit_op.find_Z2_symmetries()
    print(sq_list)
    for x in cliffords:
        print(x.print_operators())
    def check_commute(op1, op2):
        op3 = op1 * op2 - op2 * op1
        op3.zeros_coeff_elimination()
        return op3.is_empty()

    sq_list = []
    cliffords = []
    print(d_matrices)
    for d_idx in range(len(d_matrices)):
        pi_index = np.where(np.isclose(d_matrices[d_idx], np.pi))[0]
        s_pauli = ['I'] * modes
        s_pauli[pi_index[0]] = 'X'
        s_pauli = ''.join(s_pauli)
        s_op = Operator(paulis=[[1.0, label_to_pauli(s_pauli)]])
        sq_list.append(pi_index[0])
        print(check_commute(s_op, qubit_op))
        clifford_pauli = ''
        for i in range(modes):
            clifford_pauli += 'I' if i not in pi_index else 'Z'
        clifford_op = Operator(paulis=[[1.0, label_to_pauli(clifford_pauli)]])
        clifford_op += s_op
        clifford_op.scaling_coeff(1.0 / np.sqrt(2))
        cliffords.append(clifford_op)

    print(sq_list)
    for x in cliffords:
        print(x.print_operators())
    # pi_index = np.where(d_matrix == 3.14159)[0]
    # s_pauli = ['I'] * modes
    # s_pauli[pi_index[0]] = 'X'
    # s_pauli = ''.join(s_pauli)

    # s_op = Operator(paulis=[[1.0, label_to_pauli(s_pauli)]])

    # clifford_pauli = ''
    # for i in range(modes):
    #     clifford_pauli += 'I' if i not in pi_index else 'Z'
    # clifford_op = Operator(paulis=[[1.0, label_to_pauli(clifford_pauli)]])
    # clifford_op += s_op
    # clifford_op.scaling_coeff(1.0 / np.sqrt(2))

    # qubit_op = new_fer_op.mapping(map_type='jordan_wigner')

    ret_ops = []
    for coeff in itertools.product([1, -1], repeat=len(d_matrices)):
        ret_ops.append(Operator.qubit_tapering(qubit_op, cliffords,
                                               sq_list, list(coeff)))
    return ret_ops, new_fer_op


if __name__ == '__main__':

    is_atomic = True
    mol_string = 'H .0 .0 .0; H .0 .0 0.7414'
    basis = 'sto3g'
    cfg_mgr = ConfigurationManager()
    pyscf_cfg = OrderedDict([
        ('atom', mol_string),
        ('unit', 'Angstrom'),
        ('charge', 0),
        ('spin', 0),
        ('basis', basis),
        ('atomic', is_atomic)
    ])

    section = {'properties': pyscf_cfg}
    driver = cfg_mgr.get_driver_instance('PYSCF')
    molecule = driver.run(section)

    ee = get_algorithm_instance('ExactEigensolver')
    fer_op = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)
    ref_op = fer_op.mapping('jordan_wigner')

    # ee.init_args(ref_op, k=100)
    # ee_result = ee.run()
    # print(ee_result['eigvals'])

    if is_atomic:
        temp_int = np.einsum('ijkl->ljik', molecule._mo_eri_ints)
        two_body_temp = QMolecule.twoe_to_spin(temp_int)
        mol = gto.M(atom=mol_string, basis=basis)

        O = get_ovlp(mol)
        X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

        fer_op = FermionicOperator(h1=molecule._one_body_integrals, h2=two_body_temp)
        fer_op.transform(X)
    else:
        fer_op = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)

    ret_ops, _ = mapping_with_symmetry_reduction_new(fer_op, [[[0,1]],[[2,3]]])



    for op in ret_ops:
        ee.init_args(op, k=100)
        ee_result = ee.run()
        print(ee_result['eigvals'])