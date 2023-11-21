import numpy as np
from scipy.linalg import eig
from scipy.special import iv


# a_p in vMF equation in Trung's code
def a_p(kappa, dim):
    return iv(int(dim/2), kappa) / iv(dim/2 - 1, kappa)


# Create random hamiltonian
def create_hamiltonian(dim):
    hamiltonian = np.random.randn(dim, dim) + np.random.randn(dim, dim) * 1j
    hamiltonian = np.asmatrix(hamiltonian)
    hamiltonian = (hamiltonian + hamiltonian.H) / 2
    return np.array(hamiltonian)


def diagonal_hamiltonian(dim, gap, minimum):
    diagonal_elements = [minimum, gap+minimum, 2*gap+minimum]
    for i in range(1, dim-2):
        diagonal_elements.append(2*gap+minimum + i * (3 - 2*gap - minimum)/(dim-3))
    return np.diag(diagonal_elements)


def find_bounds(matrix):
    dim = len(matrix)
    maximum = -100000
    minimum = 100000
    for i in range(dim):
        max_candidate = 0
        min_candidate = 0
        for j in range(dim):
            max_candidate += abs(matrix[i, j])
            min_candidate -= abs(matrix[i, j])
        # min_candidate += 2 * abs(matrix[i, i])
        if max_candidate >= maximum:
            maximum = max_candidate
        if min_candidate <= minimum:
            minimum = min_candidate
    eigenvalues, eigenvectors = eig(matrix)
    eigenvalues = eigenvalues.real
    if min(eigenvalues) < minimum or max(eigenvalues) > maximum:
        print(f'lower bound is calculated by {minimum}, but the minimum of eigenvalues was {min(eigenvalues)}.')
        print(f'upper bound is calculated by {maximum}, but the minimum of eigenvalues was {max(eigenvalues)}.')
        raise ValueError
    return maximum, minimum


# Find ground state for hamiltonian exponential
def find_ground_state(exp_hamiltonian, eig_val_sheet):
    exp_eigenvalue, exp_eigenvector = eig(exp_hamiltonian)
    exp_eigenvalue = list(exp_eigenvalue.real)
    min_eigenvalue = min(exp_eigenvalue)
    eig_val_sheet.append(['eigenvalues of the Hamiltonian exponential (non-sorted -> sorted)'])
    eig_val_sheet.append(exp_eigenvalue)
    exp_eigenvalue.sort()
    eig_val_sheet.append(exp_eigenvalue)
    for j in range(len(exp_eigenvalue)):
        if exp_eigenvalue[j] == min_eigenvalue:
            ground_state = exp_eigenvector[:, j]
            break
    return ground_state


# Scale the eigenvalues of hamiltonian from 0 to pi
def hamiltonian_scaling(hamiltonian, lower_bound, upper_bound):
    first_scaled_hamiltonian = hamiltonian - lower_bound * np.eye(len(hamiltonian))
    second_scaled_hamiltonian = first_scaled_hamiltonian * np.pi / (upper_bound - lower_bound)
    return second_scaled_hamiltonian


def hydrogen_molecule_hamiltonian():
    # H_2 at an inter-atomic distance of 0.745 angstrom, 6-31G basis
    identity = [[1, 0], [0, 1]]
    pauli_x = [[0, 1], [1, 0]]
    pauli_y = [[0, -1j], [1j, 0]]
    pauli_z = [[1, 0], [0, -1]]
    coefficient = [-0.363395, -0.260044, -0.482367, -0.007374, 0.029427, -0.061555, -0.260044, -0.482367,
                   -0.007374, 0.029427, -0.061555, 0.007946, -0.001401, 0.004264, -0.001401, 0.010898, -0.011880,
                   0.004264, -0.011880, 0.025182, 0.001979, -0.004488, 0.005020, 0.006781, -0.021515, 0.044350,
                   0.010276, -0.011928, -0.011928, 0.094119, 0.005441, -0.054641, -0.016451, 0.046704, 0.001979,
                   0.006781, -0.004488, -0.021515, 0.005020, 0.044350, 0.007491, 0.010322, 0.010322, 0.080979,
                   0.005441, -0.016451, -0.054641, 0.046704, 0.032133, -0.025336, -0.025336, 0.054367]
    iiii = np.kron(identity, np.kron(identity, np.kron(identity, identity)))
    xiii = np.kron(pauli_x, np.kron(identity, np.kron(identity, identity)))
    iziz = np.kron(identity, np.kron(pauli_z, np.kron(identity, pauli_z)))
    xzii = np.kron(pauli_x, np.kron(pauli_z, np.kron(identity, identity)))
    iiiz = np.kron(identity, np.kron(identity, np.kron(identity, pauli_z)))
    iizz = np.kron(identity, np.kron(identity, np.kron(pauli_z, pauli_z)))
    ziii = np.kron(pauli_z, np.kron(identity, np.kron(identity, identity)))
    iixi = np.kron(identity, np.kron(identity, np.kron(pauli_x, identity)))
    zzii = np.kron(pauli_z, np.kron(pauli_z, np.kron(identity, identity)))
    iizi = np.kron(identity, np.kron(identity, np.kron(pauli_z, identity)))
    iixz = np.kron(identity, np.kron(identity, np.kron(pauli_x, pauli_z)))
    izzz = np.kron(identity, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))
    izzi = np.kron(identity, np.kron(pauli_z, np.kron(pauli_z, identity)))
    ziiz = np.kron(pauli_z, np.kron(identity, np.kron(identity, pauli_z)))
    zziz = np.kron(pauli_z, np.kron(pauli_z, np.kron(identity, pauli_z)))
    zizi = np.kron(pauli_z, np.kron(identity, np.kron(pauli_z, identity)))
    xiiz = np.kron(pauli_x, np.kron(identity, np.kron(identity, pauli_z)))
    zzzi = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_z, identity)))
    xizi = np.kron(pauli_x, np.kron(identity, np.kron(pauli_z, identity)))
    izii = np.kron(identity, np.kron(pauli_z, np.kron(identity, identity)))
    xizz = np.kron(pauli_x, np.kron(identity, np.kron(pauli_z, pauli_z)))
    xziz = np.kron(pauli_x, np.kron(pauli_z, np.kron(identity, pauli_z)))
    xzzi = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_z, identity)))
    xzzz = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))
    zizz = np.kron(pauli_z, np.kron(identity, np.kron(pauli_z, pauli_z)))
    zzzz = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))

    ixix = np.kron(identity, np.kron(pauli_x, np.kron(identity, pauli_x)))
    ixzx = np.kron(identity, np.kron(pauli_x, np.kron(pauli_z, pauli_x)))
    zxix = np.kron(pauli_z, np.kron(pauli_x, np.kron(identity, pauli_x)))
    zxzx = np.kron(pauli_z, np.kron(pauli_x, np.kron(pauli_z, pauli_x)))
    xxix = np.kron(pauli_x, np.kron(pauli_x, np.kron(identity, pauli_x)))
    xxzx = np.kron(pauli_x, np.kron(pauli_x, np.kron(pauli_z, pauli_x)))
    yyix = np.kron(pauli_y, np.kron(pauli_y, np.kron(identity, pauli_x)))
    yyzx = np.kron(pauli_y, np.kron(pauli_y, np.kron(pauli_z, pauli_x)))
    izxi = np.kron(identity, np.kron(pauli_z, np.kron(pauli_x, identity)))
    izxz = np.kron(identity, np.kron(pauli_z, np.kron(pauli_x, pauli_z)))
    zixi = np.kron(pauli_z, np.kron(identity, np.kron(pauli_x, identity)))
    zixz = np.kron(pauli_z, np.kron(identity, np.kron(pauli_x, pauli_z)))
    zzxi = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_x, identity)))
    zzxz = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_x, pauli_z)))
    xixi = np.kron(pauli_x, np.kron(identity, np.kron(pauli_x, identity)))
    xixz = np.kron(pauli_x, np.kron(identity, np.kron(pauli_x, pauli_z)))
    xzxi = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_x, identity)))
    xzxz = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_x, pauli_z)))
    ixxx = np.kron(identity, np.kron(pauli_x, np.kron(pauli_x, pauli_x)))
    ixyy = np.kron(identity, np.kron(pauli_x, np.kron(pauli_y, pauli_y)))
    zxxx = np.kron(pauli_z, np.kron(pauli_x, np.kron(pauli_x, pauli_x)))
    zxyy = np.kron(pauli_z, np.kron(pauli_x, np.kron(pauli_y, pauli_y)))
    xxxx = np.kron(pauli_x, np.kron(pauli_x, np.kron(pauli_x, pauli_x)))
    xxyy = np.kron(pauli_x, np.kron(pauli_x, np.kron(pauli_y, pauli_y)))
    yyxx = np.kron(pauli_y, np.kron(pauli_y, np.kron(pauli_x, pauli_x)))
    yyyy = np.kron(pauli_y, np.kron(pauli_y, np.kron(pauli_y, pauli_y)))

    hamiltonian = coefficient[0] * iiii + coefficient[1] * izii + coefficient[2] * ziii + coefficient[3] * zzii + \
        coefficient[4] * xiii + coefficient[5] * xzii + coefficient[6] * iiiz + coefficient[7] * iizi + \
        coefficient[8] * iizz + coefficient[9] * iixi + coefficient[10] * iixz + coefficient[11] * iziz + \
        coefficient[12] * izzi + coefficient[13] * izzz + coefficient[14] * ziiz + coefficient[15] * zizi + \
        coefficient[16] * zizz + coefficient[17] * zziz + coefficient[18] * zzzi + coefficient[19] * zzzz + \
        coefficient[20] * xiiz + coefficient[21] * xizi + coefficient[22] * xizz + coefficient[23] * xziz + \
        coefficient[24] * xzzi + coefficient[25] * xzzz + coefficient[26] * ixix + coefficient[27] * ixzx + \
        coefficient[28] * zxix + coefficient[29] * zxzx + coefficient[30] * xxix + coefficient[31] * xxzx + \
        coefficient[32] * yyix + coefficient[33] * yyzx + coefficient[34] * izxi + coefficient[35] * izxz + \
        coefficient[36] * zixi + coefficient[37] * zixz + coefficient[38] * zzxi + coefficient[39] * zzxz + \
        coefficient[40] * xixi + coefficient[41] * xixz + coefficient[42] * xzxi + coefficient[43] * xzxz + \
        coefficient[44] * ixxx + coefficient[45] * ixyy + coefficient[46] * zxxx + coefficient[47] * zxyy + \
        coefficient[48] * xxxx + coefficient[49] * xxyy + coefficient[50] * yyxx + coefficient[51] * yyyy
    return hamiltonian


# Convert ket vector to real vector
def ket_vec_to_real_vec(vec_input):
    ket_dim = len(vec_input)
    real_vec = []
    for i in range(ket_dim):
        real_vec.append(vec_input[i].real)
        real_vec.append(vec_input[i].imag)
    return np.array(real_vec)


def lithium_hydride_hamiltonian():
    # LiH at bond length 1.595 angstrom, sto-3g basis
    identity = [[1, 0], [0, 1]]
    pauli_x = [[0, 1], [1, 0]]
    pauli_y = [[0, -1j], [1j, 0]]
    pauli_z = [[1, 0], [0, -1]]
    coefficient = [-7.508666, -0.013941, 0.013941, 0.122001, -0.012103, 0.003241, 0.052733, 0.001838, -0.001838,
                   0.156354, 0.156354, -0.014942, 0.012103, 0.012103, 0.003241, 0.055974, 0.055974, 0.052733,
                   0.013941, -0.013941, -0.014942, 0.012103, 0.003241, 0.003241, 0.001838, 0.001838, 0.084497]
    iiii = np.kron(identity, np.kron(identity, np.kron(identity, identity)))
    ixiz = np.kron(identity, np.kron(pauli_x, np.kron(identity, pauli_z)))
    iziz = np.kron(identity, np.kron(pauli_z, np.kron(identity, pauli_z)))
    xxxz = np.kron(pauli_x, np.kron(pauli_x, np.kron(pauli_x, pauli_z)))
    xzxz = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_x, pauli_z)))
    yyxi = np.kron(pauli_y, np.kron(pauli_y, np.kron(pauli_x, identity)))
    ziii = np.kron(pauli_z, np.kron(identity, np.kron(identity, identity)))
    zxiz = np.kron(pauli_z, np.kron(pauli_x, np.kron(identity, pauli_z)))
    zzii = np.kron(pauli_z, np.kron(pauli_z, np.kron(identity, identity)))
    iizi = np.kron(identity, np.kron(identity, np.kron(pauli_z, identity)))
    ixzi = np.kron(identity, np.kron(pauli_x, np.kron(pauli_z, identity)))
    izzz = np.kron(identity, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))
    xyyi = np.kron(pauli_x, np.kron(pauli_y, np.kron(pauli_y, identity)))
    yxyi = np.kron(pauli_y, np.kron(pauli_x, np.kron(pauli_y, identity)))
    yzyi = np.kron(pauli_y, np.kron(pauli_z, np.kron(pauli_y, identity)))
    zizi = np.kron(pauli_z, np.kron(identity, np.kron(pauli_z, identity)))
    zxzi = np.kron(pauli_z, np.kron(pauli_x, np.kron(pauli_z, identity)))
    zzzi = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_z, identity)))
    ixii = np.kron(identity, np.kron(pauli_x, np.kron(identity, identity)))
    izii = np.kron(identity, np.kron(pauli_z, np.kron(identity, identity)))
    xxxi = np.kron(pauli_x, np.kron(pauli_x, np.kron(pauli_x, identity)))
    xzxi = np.kron(pauli_x, np.kron(pauli_z, np.kron(pauli_x, identity)))
    yxyz = np.kron(pauli_y, np.kron(pauli_x, np.kron(pauli_y, pauli_z)))
    yzyz = np.kron(pauli_y, np.kron(pauli_z, np.kron(pauli_y, pauli_z)))
    zizz = np.kron(pauli_z, np.kron(identity, np.kron(pauli_z, pauli_z)))
    zxzz = np.kron(pauli_z, np.kron(pauli_x, np.kron(pauli_z, pauli_z)))
    zzzz = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))

    hamiltonian = coefficient[0] * iiii + coefficient[1] * ixiz + coefficient[2] * iziz + coefficient[3] * xxxz + \
        coefficient[4] * xzxz + coefficient[5] * yyxi + coefficient[6] * ziii + coefficient[7] * zxiz + \
        coefficient[8] * zzii + coefficient[9] * iizi + coefficient[10] * ixzi + coefficient[11] * izzz + \
        coefficient[12] * xyyi + coefficient[13] * yxyi + coefficient[14] * yzyi + coefficient[15] * zizi + \
        coefficient[16] * zxzi + coefficient[17] * zzzi + coefficient[18] * ixii + coefficient[19] * izii + \
        coefficient[20] * xxxi + coefficient[21] * xzxi + coefficient[22] * yxyz + coefficient[23] * yzyz + \
        coefficient[24] * zizz + coefficient[25] * zxzz + coefficient[26] * zzzz
    return hamiltonian.real


# Convert original Hamiltonian to real-value Hamiltonian
def real_hamiltonian(exp_hamiltonian):
    mat_real_part = exp_hamiltonian.real
    mat_imaginary_part = exp_hamiltonian.imag
    real_hamiltonian_mat = np.kron(mat_real_part, np.eye(2)) + np.kron(mat_imaginary_part, np.array([[0, -1], [1, 0]]))
    return real_hamiltonian_mat


# Convert real vector to ket vector
def real_vec_to_ket_vec(vec_input):
    real_dim = len(vec_input)
    ket_vec = []
    for i in range(int(real_dim / 2)):
        ket_vec.append(vec_input[2*i] + vec_input[2*i+1] * 1j)
    return np.array(ket_vec)
