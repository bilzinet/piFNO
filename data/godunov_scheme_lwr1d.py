"""
Created: April 2021
@author: Bilal Thonnam Thodi (btt1@nyu.edu)

Details:
    - Generate numerical solutions for LWR model (non-linear hyperbolic PDE) of traffic flow
    - Solutions generated using Godunov/Minium Supply Demand Scheme (Finite Volume)
"""

# =============================================================================
# Import packages
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Helper functions
# =============================================================================

def plot_timespace_densitymap(K, t_max, x_max, k_jam):
    """
    Generate a heatmap of the timespace density solution.

    Args:
        K (np.ndarray): Traffic density matrix.
        t_max (float): Maximum simulation time.
        x_max (float): Maximum road length.
        k_jam (float): Jam density.
    """
    fig, ax = plt.subplots()
    ax.axis('off')
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.95, bottom=0.05, left=0.1, right=0.9, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.imshow(K.T, cmap='rainbow', extent=[0, t_max, 0, x_max],
                  origin='lower', aspect='auto', vmin=0, vmax=k_jam)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_label(r'$\rho$ ($x,t$) [vehs/km]', fontsize=10, rotation=90)
    ax.set_xlabel('Time $t$ [s]', fontsize=12)
    ax.set_ylabel('Space $x$ [m]', fontsize=12)
    ax.set_title('Density Time Space Map', fontsize=14)

    return fig, ax


def plot_initial_conditions(x_space, t_time, k_initial, q_entry, q_exit):
    """
    Plot initial density and boundary flows.

    Args:
        x_space (np.ndarray): Space discretization.
        t_time (np.ndarray): Time discretization.
        k_initial (np.ndarray): Initial traffic density.
        q_entry (np.ndarray): Entry flow boundary condition.
        q_exit (np.ndarray): Exit flow boundary condition.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 9))
    ax1.plot(x_space, k_initial)
    ax1.set_title("Initial Density", fontsize=14)
    ax1.set_xlabel("Space $x$ [m]", fontsize=12)
    ax1.set_ylabel(r"$\rho$ ($x,0$) [vehs/km]", fontsize=12)
    ax1.grid()

    ax2.plot(t_time, q_entry)
    ax2.set_title("Entering Boundary Flow", fontsize=14)
    ax2.set_xlabel("Time $t$ [s]", fontsize=12)
    ax2.set_ylabel(r"$q$ ($x_{\rm enter},t$) [vehs/hr]", fontsize=12)
    ax2.grid()

    ax3.plot(t_time, q_exit)
    ax3.set_title("Exiting Boundary Flow", fontsize=14)
    ax3.set_xlabel("Time $t$ [s]", fontsize=12)
    ax3.set_ylabel(r"$q$ ($x_{\rm exit},t$) [vehs/hr]", fontsize=12)
    ax3.grid()

    fig.tight_layout()

    return fig, (ax1, ax2, ax3)

# =============================================================================
# Godunov Scheme
# =============================================================================


def fundamental_diag(k, k_max, v_max, k_cr, fd):
    """
    Fundamenal Diagram
    """
    if fd == 'Greenshield':
        q = k*v_max*(1-k/k_max)
    elif fd == 'Triangular':
        if k <= k_cr:
            q = v_max*k
        else:
            w = k_cr*v_max/(k_max-k_cr)
            q = w*(k_max-k)
    elif fd == 'NewellFrank':
        q = k*v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
    return q


def Demandfn(k, k_max, v_max, k_cr, q_max, fd):
    """
    Traffic supply function
    """
    if fd == 'Greenshield':
        if k <= k_cr:
            q = k*v_max*(1-k/k_max)
        else:
            q = q_max
    elif fd == 'Triangular':
        q = min(v_max*k, q_max)
    elif fd == 'NewellFrank':
        if k <= k_cr:
            v = v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
            q = k*v
        else:
            q = q_max
    return q


def InvDemandfn_num(q, dem_fn, k_arr):
    qb = dem_fn[dem_fn < q][-1]
    qa = dem_fn[dem_fn >= q][0]
    kb = k_arr[dem_fn < q][-1]
    ka = k_arr[dem_fn >= q][0]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k


def InvDemandfn(q, k_max, v_max, k_cr, q_max, v_free, fd):
    """
    Inverse of traffic supply function
    """
    if fd == 'Greenshield':
        q = min(q, q_max)
        k = (k_max-np.sqrt(k_max**2-4*k_max/v_free*q))/2
    elif fd == 'Triangular':
        k = min(q/v_max, k_cr)
    elif fd == 'NewellFrank':
        q = min(q, q_max)
        k = InvDemandfn_num(q, dem_fn, k_arr)
    return k


def Supplyfn(k, k_max, v_max, k_cr, q_max, fd):
    """
    Traffic demand function
    """
    if fd == 'Greenshield':
        if k >= k_cr:
            q = k*v_max*(1-k/k_max)
        else:
            q = q_max
    elif fd == 'Triangular':
        w = k_cr*v_max/(k_max-k_cr)
        q = min(q_max, w*(k_max-k))
    elif fd == 'NewellFrank':
        if k >= k_cr:
            v = v_max*(1 - np.exp((v_b/v_max)*(k_max/k - 1)))
            q = k*v
        else:
            q = q_max
    return q


def InvSupplyfn_num(q, sup_fn, k_arr):
    qb = sup_fn[sup_fn <= q][0]
    qa = sup_fn[sup_fn > q][-1]
    kb = k_arr[sup_fn <= q][0]
    ka = k_arr[sup_fn > q][-1]
    k = kb + (ka-kb)*(q-qb)/(qa-qb)
    return k


def InvSupplyfn(q, k_max, v_max, k_cr, q_max, v_free, fd):
    """
    Inverse of traffic demand function
    """
    if fd == 'Greenshield':
        q = min(q, q_max)
        k = (k_max+np.sqrt(k_max**2-4*k_max/v_free*q))/2
    elif fd == 'Triangular':
        w = k_cr*v_max/(k_max-k_cr)
        k = min(q_max, k_max-q/w)
    elif fd == 'NewellFrank':
        q = min(q, q_max)
        k = InvSupplyfn_num(q, sup_fn, k_arr)
    return k


def bound_cond_entry(k_prev, q_en, k_max, v_max, k_cr, q_max, v_free, fd):
    """
    Boundary conditions at the link entry

    Args:
        k_prev (float): traffic density in the previous cell
        q_en (float): flow entering to the link boundary (current cell)
        k_max (float): maximum traffic density (jam density)
        v_max (float): maximum traffic speed
        k_cr (float): critical traffic density
        q_max (float): maximum traffic flow
        fd (str): type of fundamental diagram

    Returns:
        _type_: traffic density in the current cell
    """
    q_en = min(q_en, q_max)
    supply = Supplyfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_en <= supply:
        k = InvDemandfn(q_en, k_max, v_max, k_cr, q_max, v_free, fd)
    else:
        k = InvSupplyfn(q_en, k_max, v_max, k_cr, q_max, v_free, fd)

    return k


def bound_cond_exit(k_prev, q_ex, k_max, v_max, k_cr, q_max, v_free, fd):
    """
    Boundary conditions at the link exit

    Args:
        k_prev (float): traffic density in the previous cell
        q_ex (float): flow exiting from the link boundary (current cell)
        k_max (float): maximum traffic density (jam density)
        v_max (float): maximum traffic speed
        k_cr (float): critical traffic density
        q_max (float): maximum traffic flow
        fd (str): type of fundamental diagram

    Returns:
        float: traffic density in the current cell
    """
    q_ex = min(q_ex, q_max)
    demand = Demandfn(k_prev, k_max, v_max, k_cr, q_max, fd)
    if q_ex < demand:
        k = InvSupplyfn(q_ex, k_max, v_max, k_cr, q_max, v_free, fd)
    else:
        k = InvDemandfn(q_ex, k_max, v_max, k_cr, q_max, v_free, fd)

    return k


def flux_function(k_xup, k_xdn, k_cr, q_max, k_max, v_max, fd):
    """
    Calculate flux across a cell boundary (Godunov scheme)

    Args:
        k_xup (float): Density in the upstream cell.
        k_xdn (float): Density in the downstream cell.
        k_cr (float): Critical density.
        q_max (float): Maximum flow.
        k_max (float): Maximum density.
        v_max (float): Free-flow speed.
        fd_type (str): Fundamental diagram type.

    Returns:
        float: Flux across the boundary.
    """
    if (k_xdn <= k_cr) and (k_xup <= k_cr):
        q_star = fundamental_diag(k_xup, k_max, v_max, k_cr, fd)
    elif (k_xdn <= k_cr) and (k_xup > k_cr):
        q_star = q_max
    elif (k_xdn > k_cr) and (k_xup <= k_cr):
        q_star = min(fundamental_diag(k_xdn, k_max, v_max, k_cr, fd),
                     fundamental_diag(k_xup, k_max, v_max, k_cr, fd))
    elif (k_xdn > k_cr) and (k_xup > k_cr):
        q_star = fundamental_diag(k_xdn, k_max, v_max, k_cr, fd)

    return q_star


def density_update(k_x, k_xup, k_xdn, delt, delx, k_cr, q_max, k_max, v_max, fd):
    """
    Density update function
    """
    q_in = flux_function(k_xup, k_x, k_cr, q_max, k_max, v_max, fd)
    q_out = flux_function(k_x, k_xdn, k_cr, q_max, k_max, v_max, fd)
    k_x_nextt = k_x + (delt/delx)*(q_in - q_out)
    return k_x_nextt, q_out


def CFL_condition(delx, v_max):
    """
    Limit on maximum delta t for a given delta x
    """
    max_delt = delx/v_max
    return np.around(max_delt, 6)


def forward_sim(k_initial, q_entry, q_exit, t_nums, x_nums, delt, delx, fd_params, k_jam_space):
    """
    LWR numerical solution for a given initial and boundary conditions

    Args:
        k_initial (np.ndarray): Initial traffic densities.
        q_entry (np.ndarray): Entry boundary flows.
        q_exit (np.ndarray): Exit boundary flows.
        t_nums (int): Number of time discretizations.
        x_nums (int): Number of space discretizations.
        fd_params (dict): Fundamental diagram parameters.
        k_jam_space (np.ndarray): Jam densities across the road.

    Returns:
        tuple: Traffic density matrix (K) and traffic flow matrix (Q).
    """

    # FD parameters
    v_free = fd_params["v_free"]
    k_jam = fd_params["k_jam"]
    fd_type = fd_params["fd_type"]

    # Initialize time-space indices
    x_ind = np.arange(0, x_nums)
    t_ind = np.arange(0, t_nums)
    X_ind, T_ind = np.meshgrid(x_ind, t_ind)

    # Initialize K, Q matrix
    K = np.zeros((t_nums, x_nums))
    Q = np.zeros((t_nums, x_nums))

    # Runing the numerical scheme time step
    for t in range(X_ind.shape[0]):

        # Initial condition
        if t == 0:
            K[t, :] = k_initial
            continue

        # Remaining time
        for x in range(X_ind.shape[1]):

            k_jam = k_jam_space[x]
            q_max = k_jam*v_free/4
            k_cr = k_jam/2

            # Get computational stencil
            k_x = K[t-1, x]

            # Starting Boundary condition
            if x == 0:
                q_en = q_entry[t]
                k_xup = bound_cond_entry(
                    k_x, q_en, k_jam, v_free, k_cr, q_max, v_free, fd_type)
            else:
                k_xup = K[t-1, x-1]

            # Ending Boundary condition
            if x == x_nums-1:
                q_ex = q_exit[t]
                k_xdn = bound_cond_exit(
                    k_x, q_ex, k_jam, v_free, k_cr, q_max, v_free, fd_type)
            else:
                k_xdn = K[t-1, x+1]

            # Calculated and update new density
            k_x_next, q_out = density_update(k_x, k_xup, k_xdn, delt, delx,
                                             k_cr, q_max, k_jam, v_free, fd_type)
            K[t, x] = k_x_next
            Q[t, x] = q_out

    return K, Q


if __name__ == '__main__':

    print("INITIALIZE PARAMETERS")

    # Traffic parameters
    fd_params = {
        "k_jam": 120,
        "v_free": 60,
        "fd_type": "Greenshield"
    }

    # Cell discretization
    x_max = 1000/1000                   # road length in kilometres
    t_max = 600/3600                    # time period of simulation in hours
    delx = 20/1000                      # cell length in kilometres
    delt = 1/3600                       # time discretization in hours
    x_nums = round(x_max/delx)
    t_nums = round(t_max/delt)

    # Input conditions
    num_steps = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
    k_initial = np.random.uniform(0, 100, x_nums)
    q_entry = np.random.uniform(300, 1000, t_nums)
    q_exit = np.random.uniform(800, 1500, t_nums)
    q_exit[np.random.randint(50, 150):np.random.randint(150, 300)] = 0
    k_jam_space = np.repeat(fd_params["k_jam"], x_nums)

    print('\nRUNNING NUMERICAL SOLVER')

    # Generate solution
    K, Q = forward_sim(k_initial, q_entry, q_exit,
                       t_nums, x_nums, delt, delx, fd_params, k_jam_space)

    # Visualize solution
    x_space = np.linspace(0, x_max, x_nums)
    t_time = np.linspace(0, t_max, t_nums)
    save_filename = './sample_input.png'
    fig, _ = plot_initial_conditions(
        x_space, t_time, k_initial, q_entry, q_exit)
    fig.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f'\nSAMPLE INPUT SAVED TO: {save_filename}')

    save_filename = './sample_solution_densitymap.png'
    fig, _ = plot_timespace_densitymap(K, t_max, x_max, fd_params["k_jam"])
    fig.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f'\nSAMPLE SOLUTION SAVED TO: {save_filename}')
