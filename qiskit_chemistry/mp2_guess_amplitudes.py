# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


def mp2_guess_amplitudes(rsgtu, e_orbitals, de_list):
    """
    Compute the MP2 guess amplitudes.

    Args:
        rsgtu (numpy.ndarray): two_body_integrals_matrix needs to be in format.
                               4-D tensor with size (mode, mode, mode, mode)
        e_orbitals (numpy.ndarray): 1e orbital energy (from HF calculation)
        de_list (list): list of double excitations

    Returns:
        list: MP2 guess amplitudes

    Note:
        #oe_spin = np.concatenate((molecule.orbital_energies, molecule.orbital_energies))
        #be aware two body integral and orbital energy should be updated with any reduction approach
        # accordingly.
    """
    excitation_mp2_list = []
    for [occ1, vir1, occ2, vir2] in de_list:
        val = (rsgtu[occ1][vir1][occ2][vir2] - rsgtu[occ1][vir2][occ2][vir1]) / \
            (e_orbitals[occ1] + e_orbitals[occ2] - e_orbitals[vir1] - e_orbitals[vir2])
        excitation_mp2_list.append(val)

    return excitation_mp2_list
