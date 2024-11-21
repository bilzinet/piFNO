"""
Created: April 2021
Updated: November 2024
@author: Bilal Thonnam Thodi (btt1@nyu.edu)

Code:
    - Generate training/testing data for Physics-Informed Fourier Neural Operator for Macroscopic traffic flow models.
    - Paper link:

Data:
    - Generated from numerical solutions for LWR model of traffic flow
    - Solutions generated using Godunov/Minium Supply Demand Scheme (Finite Volume)
"""

# =============================================================================
# Import packages
# =============================================================================

import os
import scipy
import numpy as np
import pickle as pkl

from godunov_scheme_lwr1d import fundamental_diag
from godunov_scheme_lwr1d import forward_sim

rin = np.random.randint
run = np.random.uniform
rno = np.random.normal

# =============================================================================
# Utilities
# =============================================================================


def check_directory_exists(directory):
    """
    Ensure a directory exists; create it if it does not.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_config():
    """
    Print configuration details for training and testing.
    """
    print("\n\nCONFIGURATION FOR DATA GENERATION")
    print("-" * 40)
    print(f"Training Directory: {TRAIN_DIR}")
    print(f"Testing Directory: {TEST_DIR}")
    print()
    print("Dataset Parameters:")
    print(
        f"\tTraining data size per chunk: {TRAIN_DATASET_SIZE_PER_CASE}")
    print(
        f"\tTraining data number of chunks: {NUM_TRAIN_CASES}")
    print(
        f"\tTraining data total size: {TRAIN_DATASET_SIZE_PER_CASE*NUM_TRAIN_CASES}")
    print(f"\tTesting data size per case: {TEST_DATASET_SIZE_PER_CASE}")
    print(f"\tTesting data number of cases: {NUM_TEST_CASES}")
    print(
        f"\tTesting data total size: {TEST_DATASET_SIZE_PER_CASE*NUM_TEST_CASES}")
    print(
        f"\tValidation data total size: {VALID_DATASET_SIZE_PERCENT*TRAIN_DATASET_SIZE_PER_CASE*NUM_TRAIN_CASES:.0f}")
    print()
    print("Training Cases:")
    print(f"\t{', '.join(TRAIN_CASES)}")
    print()
    print("Testing Cases:")
    print(f"\t{', '.join(TEST_CASES)}")


# =============================================================================
# Helper functions for dataset generation
# =============================================================================


def train_initconds(x_nums):
    """
    Randomly generate one-, two- and three-step initial density profiles
    """

    rin = np.random.randint
    run = np.random.uniform
    p = np.random.rand()

    # One step traffic density profile (type 1)
    if p < 0.20:
        i = rin(5, x_nums-5)
        k_initial = np.repeat(rin(80, 100), x_nums)
        k_initial[i:] = max(0, k_initial[0] - rin(20, 60))
    # One step traffic density profile (type 2)
    elif (p >= 0.20) and (p < 0.40):
        i = rin(5, x_nums-5)
        k_initial = np.repeat(rin(80, 100), x_nums)
        k_initial[:i] = max(0, k_initial[0] - rin(20, 60))
    # Two step traffic density profile
    elif (p >= 0.40) and (p < 0.65):
        i = rin(5, x_nums-20)
        j = max(i+rin(5, 20), rin(5, x_nums-10))
        k_initial = np.repeat(rin(0, 100), x_nums)
        k_initial[:i] = max(0, k_initial[0]-rin(0, 50))
        k_initial[j:] = max(0, k_initial[-1]-rin(0, 50))
    # Three step traffic density profile
    elif (p >= 0.65) and (p < 0.90):
        k_initial = np.repeat(rin(0, 50), x_nums)
        i1 = rin(5, int(x_nums/2))
        j1 = max(i1+rin(0, 5), rin(5, int(x_nums/2)))
        k_initial[i1:j1] = min(115, k_initial[0]+rin(10, 60))
        i2 = rin(int(x_nums/2), x_nums-5)
        j2 = max(i2+rin(0, 5), rin(int(x_nums/2), x_nums))
        k_initial[i2:j2] = min(115, k_initial[-1]+rin(10, 60))
    # Random traffic density profile
    else:
        k_initial = run(0, 100, x_nums)

    return k_initial


def gen_initial_condition(num_steps, num_points=50, step_height_std=20, add_noise=False):
    """
    Generate initial density profiles for traffic simulation.

    Args:
        num_steps (int): Number of steps in the density profile.
        num_points (int, optional): Total number of spatial points. Defaults to 50.
        step_height_std (int, optional): Standard deviation of step height changes. Defaults to 20.
        add_noise (bool, optional): Add random noise to the profile. Defaults to False.

    Returns:
        np.ndarray: Initial traffic density profile.
    """

    # Flat profile
    if num_steps == 0:
        k_initial = run(30, 90, num_points)
    # Multi-step profiles
    else:
        i = 0
        par_length = int(num_points/num_steps)+3
        k_initial = np.repeat(rin(30, 90), num_points)
        for _ in range(num_steps):
            j = min(num_points-1, rin(i, i+max(1, par_length)))
            step_height = rin(-step_height_std, step_height_std)
            k_initial[j:] = max(0, min(115, k_initial[j-1]+step_height))
            i = j
    if add_noise:
        rno = np.random.normal(0.0, 0.05, x_nums)
        k_initial += rno

    return k_initial.astype('float')


def gen_bound_condition(t_nums, sc_type='bc0'):
    """
    Generate boundary conditions for traffic simulation.

    Args:
        t_nums (int): Number of time steps.
        sc_type (str, optional): Boundary condition type. Defaults to 'bc0'.

    Returns:
        tuple: Entry flow and exit flow as numpy arrays.
    """

    q_entry = run(300, 1000, t_nums)
    q_exit = run(800, 1500, t_nums)

    if sc_type == 'bc1':
        q_exit[rin(50, 150):rin(150, 300)] = 0

    elif sc_type == 'bc2':
        q_exit[rin(50, 100):rin(100, 200)] = 0
        q_exit[rin(250, 350):rin(350, 450)] = 0

    elif sc_type == 'bc3':
        q_exit[rin(50, 80):rin(80, 150)] = 0
        q_exit[rin(200, 230):rin(230, 300)] = 0
        q_exit[rin(350, 380):rin(380, 450)] = 0

    elif sc_type == 'bc4':
        q_exit[rin(50, 100):rin(100, 150)] = 0
        q_exit[rin(160, 210):rin(210, 260)] = 0
        q_exit[rin(270, 320):rin(320, 370)] = 0
        q_exit[rin(380, 430):rin(430, 480)] = 0

    elif sc_type == 'bc5':
        q_exit[rin(40, 80):rin(80, 120)] = 0
        q_exit[rin(130, 170):rin(170, 210)] = 0
        q_exit[rin(220, 260):rin(260, 300)] = 0
        q_exit[rin(310, 350):rin(350, 390)] = 0
        q_exit[rin(400, 440):rin(440, 480)] = 0

    elif sc_type == 'bc6':
        q_exit[rin(40, 75):rin(75, 110)] = 0
        q_exit[rin(120, 155):rin(155, 190)] = 0
        q_exit[rin(200, 235):rin(235, 270)] = 0
        q_exit[rin(280, 315):rin(315, 350)] = 0
        q_exit[rin(360, 395):rin(395, 430)] = 0
        q_exit[rin(440, 475):rin(475, 510)] = 0

    elif sc_type == 'bc7':
        q_exit[rin(40, 70):rin(70, 100)] = 0
        q_exit[rin(110, 140):rin(140, 170)] = 0
        q_exit[rin(180, 210):rin(210, 240)] = 0
        q_exit[rin(250, 280):rin(280, 310)] = 0
        q_exit[rin(320, 350):rin(350, 380)] = 0
        q_exit[rin(390, 420):rin(420, 450)] = 0
        q_exit[rin(460, 490):rin(490, 520)] = 0

    elif sc_type == 'bc8':
        q_exit[rin(40, 67):rin(67, 95)] = 0
        q_exit[rin(105, 132):rin(132, 160)] = 0
        q_exit[rin(170, 197):rin(197, 225)] = 0
        q_exit[rin(235, 262):rin(262, 290)] = 0
        q_exit[rin(300, 327):rin(327, 355)] = 0
        q_exit[rin(365, 392):rin(392, 420)] = 0
        q_exit[rin(430, 457):rin(457, 485)] = 0
        q_exit[rin(495, 522):rin(522, 550)] = 0

    return q_entry, q_exit

# =============================================================================
# Simulation parameters
# =============================================================================


def get_fd_params(k_jam, v_free, v_b=-15, fd='Greenshield'):

    if fd == 'Greenshield':
        q_max = k_jam*v_free/4          # in vehicles/hr
        k_cr = k_jam/2                  # for Greenshield's model
    elif fd == 'Triangular':
        k_cr = 0.40*k_jam
        q_max = v_free*k_cr
    elif fd == 'NewellFrank':
        k_cr = 40
        k_arr = np.arange(0, k_jam+0.1, 0.1)
        q_arr = fundamental_diag(k_arr, k_jam, v_free, k_cr, fd)
        q_max = q_arr.max()
        q_argmax = np.argmax(q_arr)
        k_cr = k_arr[q_argmax]
        dem_fn = q_arr.copy()
        dem_fn[q_argmax:] = q_max
        sup_fn = q_arr.copy()
        sup_fn[:q_argmax] = q_max

    return q_max, k_cr


# Parameters
fd_params = {
    "k_jam": 120,
    "v_free": 60,
    "fd_type": "Greenshield"
}
q_max, k_cr = get_fd_params(
    k_jam=fd_params["k_jam"], v_free=fd_params["v_free"], fd=fd_params["fd_type"])

# Cell discretization
x_max = 1000/1000                   # road length in kilometres
t_max = 600/3600                    # time period of simulation in hours
delx = 20/1000                      # cell length in kilometres
delt = 1/3600                       # time discretization in hours
x_nums = round(x_max/delx)
t_nums = round(t_max/delt)

# Time-space mesh creation
x_ind = np.arange(0, x_nums)
t_ind = np.arange(0, t_nums)
X_ind, T_ind = np.meshgrid(x_ind, t_ind)

# Variable jam densities
k_jam_space = np.repeat(fd_params["k_jam"], x_nums)

# =============================================================================
# Constants and Configuration
# =============================================================================

# Directories
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# Simulation parameters
TRAIN_DATASET_SIZE_PER_CASE = 10
VALID_DATASET_SIZE_PERCENT = 0.05
TRAIN_DATASET_SIZE_MAX = 100
TEST_DATASET_SIZE_PER_CASE = 2
NUM_TOTAL_RUNS = 100
NUM_TEST_RUNS = 2

# Training and testing cases
TRAIN_CASES = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6']
TEST_CASES = [
    'bc0', 'bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6', 'bc7', 'bc8',
    'ic0', 'ic1', 'ic2', 'ic3', 'ic4', 'ic5', 'ic6', 'ic7', 'ic8', 'ic9',
    'ic10', 'ic15', 'ic20', 'ic25', 'ic30', 'ic40'
]
NUM_TRAIN_CASES = len(TRAIN_CASES)
NUM_TEST_CASES = len(TEST_CASES)

# Generate train/test data
generate_train_data = True
generate_test_data = True
generate_riemann_solns = False
generate_train_data_otherinit = False
print_config()

# =============================================================================
# Generate training data
# =============================================================================

if generate_train_data:
    print('\n\nGENERATING TRAINING DATA')
    print("-" * 40)

    # Ensure training directory exist
    check_directory_exists(TRAIN_DIR)

    # Generate training data
    for i, sc in enumerate(TRAIN_CASES):
        print(f'\n\tInput condition ({i+1}/{NUM_TRAIN_CASES}): {sc}')

        K_arr, Q_arr = [], []
        for n in range(TRAIN_DATASET_SIZE_PER_CASE):

            # Input conditions
            num_steps = np.random.choice([0, 1, 2, 3], p=[0.25] * 4)
            k_initial = gen_initial_condition(num_steps, num_points=x_nums)
            bt_type = np.random.choice(['bc0', 'bc1', 'bc2'], p=[
                                       0.05, 0.475, 0.475])
            q_entry, q_exit = gen_bound_condition(
                t_nums, sc_type=bt_type)

            # Run simulation
            K, Q = forward_sim(k_initial, q_entry, q_exit,
                               t_nums, x_nums, delt, delx, fd_params, k_jam_space)
            K_arr.append(K)
            Q_arr.append(Q)

        # Split into training and validation data
        TrainX, TrainY, ValidX, ValidY = [], [], [], []
        num_train_samples = TRAIN_DATASET_SIZE_PER_CASE - \
            int(TRAIN_DATASET_SIZE_PER_CASE*VALID_DATASET_SIZE_PERCENT)
        for n in range(TRAIN_DATASET_SIZE_PER_CASE):
            K, Q = K_arr[n], Q_arr[n]
            K_out = K.copy().astype('float32')
            K_inp = K.copy().astype('float32')
            K_inp[1:, 1:-1] = -1

            if n <= num_train_samples:
                TrainX.append(K_inp)
                TrainY.append(K_out)
            else:
                ValidX.append(K_inp)
                ValidY.append(K_out)

        # Save training data
        train_data_path = os.path.join(TRAIN_DIR, f'train_data-{sc}.pkl')
        with open(train_data_path, 'wb') as f:
            pkl.dump({'X': TrainX, 'Y': TrainY}, f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        print(f'\tTraining data saved to: {train_data_path}')

        # Save validation data
        valid_data_path = os.path.join(TRAIN_DIR, f'valid_data-{sc}.pkl')
        with open(valid_data_path, 'wb') as f:
            pkl.dump({'X': ValidX, 'Y': ValidY}, f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        print(f'\tValidation data saved to: {valid_data_path}')


# =============================================================================
# Generate testing data
# =============================================================================

if generate_test_data:
    print('\n\nGENERATING TESTING DATA')
    print("-" * 40)

    # Ensure testing directory exist
    check_directory_exists(TEST_DIR)

    # Generate testing data
    for i, sc in enumerate(TEST_CASES):
        print(f'\n\tInput condition ({i+1}/{len(TEST_CASES)}): {sc}')

        TestX, TestY = [], []
        for n in range(NUM_TEST_RUNS):

            # Generate input conditions
            if sc.startswith('bc'):
                num_steps = np.random.choice([0, 1, 2, 3], p=[0.25] * 4)
                k_initial = gen_initial_condition(num_steps, num_points=x_nums)
                q_entry, q_exit = gen_bound_condition(
                    t_nums, sc_type=sc)
            elif sc.startswith('ic'):
                num_steps = int(sc[2:])
                k_initial = gen_initial_condition(num_steps, num_points=x_nums)
                bt_type = np.random.choice(
                    ['bc0', 'bc1', 'bc2'], p=[0.20, 0.40, 0.40])
                q_entry, q_exit = gen_bound_condition(
                    t_nums, sc_type=bt_type)

            # Run simulation
            K, Q = forward_sim(k_initial, q_entry, q_exit,
                               t_nums, x_nums, delt, delx, fd_params, k_jam_space)

            # Create input-output pair
            K_out = K.copy().astype('float32')
            K_inp = K.copy().astype('float32')
            K_inp[1:, 1:-1] = -1
            TestX.append(K_inp)
            TestY.append(K_out)

        # Save testing data
        test_data_path = os.path.join(TEST_DIR, f'test_data-{sc}.pkl')
        with open(test_data_path, 'wb') as f:
            pkl.dump({'X': TestX, 'Y': TestY}, f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        print(f'\tTesting data saved to: {test_data_path}')


# =============================================================================
# Generate specific testing data = Step initial conditions
# =============================================================================

k_max = 120
v_max = 60
k_cr = k_max/2
q_max = (k_max*v_max)/4
def flux(k): return k*v_max*(1-k/k_max)


rin = np.random.randint
run = np.random.uniform

if generate_riemann_solns:

    K_arr = []
    Q_arr = []
    num_runs = 875
    for n in range(num_runs):
        print(f'Run = {n}')
        if run(0, 1) <= 0.5:
            # Riemann problem 1
            k_initial = np.repeat(run(70, 110), x_nums)
            k_initial[rin(10, 40):] = run(0, 30)
            q_entry = np.repeat(1800, t_nums)
            q_exit = np.repeat(1800, t_nums)
            K, Q = forward_sim(k_initial, q_entry, q_exit,
                               t_nums, x_nums, fd, k_jam_space)
        else:
            # Riemann problem 2
            k_initial = np.repeat(run(70, 110), x_nums)
            k_initial[:rin(10, 40)] = run(0, 30)
            q_entry = np.repeat(flux(k_initial[0]), t_nums)
            q_exit = np.repeat(flux(k_initial[-1]), t_nums)
            K, Q = forward_sim(k_initial, q_entry, q_exit,
                               t_nums, x_nums, fd, k_jam_space)
        K_arr.append(K)
        Q_arr.append(Q)

    # Extract input-output pairs
    num_train_runs = 750
    TrainX = []
    TrainY = []
    TestX = []
    TestY = []
    for n in range(num_runs):
        K, Q = K_arr[n], Q_arr[n]
        K_out = K.copy().astype('float32')
        K_inp = K.copy().astype('float32')
        K_inp[1:, 1:-1] = -1
        if n <= num_train_runs-1:
            TrainX.append(K_inp)
            TrainY.append(K_out)
        else:
            TestX.append(K_inp)
            TestY.append(K_out)

    # Save data offline
    data = {'X': TrainX, 'Y': TrainY}
    with open('train/train_data-part5.pkl', 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
    data = {'X': TestX, 'Y': TestY}
    with open('train/test_data-part5.pkl', 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


# =============================================================================
# Generate training data with sub-optimal initial conditions
# =============================================================================

def exponentiated_quadratic(xa, xb, sigma=1):
    """
    Exponentiated quadratic  with sigma=1
    """
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm/(sigma**2))


def gen_initial_condition_others(x_nums, init_type='uniform'):

    if init_type == 'uniform':
        # sample from a uniform distribution
        ys = np.random.uniform(30, 90, size=x_nums)

    elif init_type == 'gaussian':
        # sample from a Gaussian process distribution (https://peterroelants.github.io/posts/gaussian-process-tutorial/#:~:text=To%20sample%20functions%20from%20the,between%20each%20pair%20in%20and%20.)
        xs = np.expand_dims(np.linspace(0, 1, x_nums), 1)
        cov = exponentiated_quadratic(
            xs, xs, sigma=0.08)  # Kernel of data points
        ys = np.random.multivariate_normal(mean=np.repeat(rin(40, 80), x_nums),
                                           cov=cov, size=1)
        ys = ys[0, :]
        ys[ys > 115] = 115
        ys[ys <= 0] = 0

    elif init_type == 'sinusoid':
        # sample from a Sine wave function
        xs = np.linspace(0, 1, x_nums)
        ys = run(30, 70) + run(0, 60)*np.sin(2*np.pi*xs*rin(1, 10))
        ys[ys > 115] = 115
        ys[ys <= 0] = 0

    return ys


run = np.random.uniform
rin = np.random.randint
train_cases = ['part1', 'part2', 'part3', 'part4']
init_type = 'sinusoid'

if generate_train_data_otherinit:
    for sc in train_cases:
        print(f'Training input conditions: {sc}')
        num_runs = 1750
        K_arr = []
        Q_arr = []
        for n in range(num_runs):
            print(f'\t Run = {n}')

            # Input conditions
            k_initial = gen_initial_condition_others(
                x_nums, init_type=init_type)
            bt_type = np.random.choice(
                ['bc0', 'bc1', 'bc2'], p=[0.20, 0.40, 0.40])
            q_entry, q_exit = gen_bound_condition(
                x_nums, t_nums, sc_type=bt_type)

            # Run forward_sim scheme
            K, Q = forward_sim(k_initial, q_entry, q_exit,
                               t_nums, x_nums, fd, k_jam_space)
            K_arr.append(K)
            Q_arr.append(Q)

        # Extract input-output pairs
        num_train_runs = 1500
        TrainX = []
        TrainY = []
        TestX = []
        TestY = []
        for n in range(num_runs):
            print(n)
            K, Q = K_arr[n], Q_arr[n]
            K_out = K.copy().astype('float32')
            K_inp = K.copy().astype('float32')
            K_inp[1:, 1:-1] = -1
            if n <= num_train_runs-1:
                TrainX.append(K_inp)
                TrainY.append(K_out)
            else:
                TestX.append(K_inp)
                TestY.append(K_out)

        # Save data offline
        data = {'X': TrainX, 'Y': TrainY}
        with open('train5-{}/train_data-{}.pkl'.format(init_type, sc), 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
        data = {'X': TestX, 'Y': TestY}
        with open('train5-{}/test_data-{}.pkl'.format(init_type, sc), 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
