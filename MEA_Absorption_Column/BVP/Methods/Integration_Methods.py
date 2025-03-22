import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pcsaft import InputError
from scipy.linalg import solve
from scipy.optimize import root
from MEA_Absorption_Column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium


def eulers(fxn, y, t_eval, args=None):

    n_steps = len(t_eval)
    step_size = t_eval[1] - t_eval[0]
    # Initialize storage for results
    results = np.zeros((n_steps, len(y)))
    results[0] = y  # Set initial values

    # Iteratively apply Euler's method
    for i in range(1, n_steps):
        t = t_eval[i - 1]
        y_prev = results[i-1]
        # Compute derivatives using the provided function
        dydt_scaled = np.array(fxn(t, y_prev, args))
        del chemical_equilibrium.cache
        # Update the dependent variables
        results[i] = y_prev + step_size * dydt_scaled

    return results.T, t_eval, 'Success', 'Yay'


def runge_kutta(fxn, y0, t_eval, args=None):
    """
    Solve an ODE using the 4th-order Runge-Kutta method.

    Parameters:
        f (function): The ODE function, y' = f(t, y).
        t0 (float): The initial time.
        y0 (float): The initial value of y at t0.
        t_end (float): The end time of the interval.
        h (float): The step size.

    Returns:
        t_values (numpy array): Array of time values.
        y_values (numpy array): Array of y values corresponding to t_values.
    """

    # A = np.array([
    #     [0, 0, 0, 0, 0, 0],
    #     [1 / 5, 0, 0, 0, 0, 0],
    #     [3 / 40, 9 / 40, 0, 0, 0, 0],
    #     [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
    #     [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
    #     [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
    #     [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
    # ])
    #
    # # Define the b array (7th-order weights)
    # b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    #
    # # Define the c array (time nodes)
    # c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

    # c = np.array([0, 1/3, 2/3, 1, 1/2, 1])
    # A = np.array([
    #
    #     [0,    0,    0,     0,    0,    0],
    #     [1/3,  0,    0,     0,    0,    0],
    #     [-1/3, 1,    0,     0,    0,    0],
    #     [1,   -1,    1,     0,    0,    0],
    #     [1/2,  1/2, -1/2,   1/2,  0,    0],
    #     [1/6,  1/6,  1/6,   1/6,  1/6,  0]
    # ])
    # b = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

    # A = np.array([
    #     [1 / 2, 0, 0, 0],
    #     [1 / 6, 1 / 2, 0, 0],
    #     [-1 / 2, 1 / 2, 1 / 2, 0],
    #     [3 / 2, -3 / 2, 1 / 2, 1 / 2]
    # ])
    #
    # # Define the b array
    # b = np.array([3 / 2, -3 / 2, 1 / 2, 1 / 2])
    #
    # # Define the c array
    # c = np.array([1 / 2, 2 / 3, 1 / 2, 1])
    #
    # Gauss-Legendre Method Implicit
    # A = np.array([
    #     [5 / 36                   , 2 / 9 - np.sqrt(15), 5 / 36 - np.sqrt(15) / 30],
    #     [5 / 36 + np.sqrt(15) / 24, 2 / 9              , 5 / 36 - np.sqrt(15) / 24],
    #     [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15), 5 / 36                   ],
    # ])
    #
    # # Define the b array (weights)
    # b = np.array([
    #     5 / 18, 4 / 9, 5 / 18
    # ])
    #
    # # Define the c array (time nodes)
    # c = np.array([
    #     1 / 2 - np.sqrt(15) / 10,
    #     1 / 2,
    #     1 / 2 + np.sqrt(15) / 10,
    # ])
    #
    A = np.array([
        [1 / 4, 1 / 4 - np.sqrt(3) / 6],
        [1 / 4 + np.sqrt(3) / 6, 1 / 4]
    ])
    b = np.array([1 / 2, 1 / 2])
    c = np.array([1 / 2 - np.sqrt(3) / 6, 1 / 2 + np.sqrt(3) / 6])

    A = np.array([
        [0.196815477223660, -0.065535425850198, 0.023770974348220],
        [0.394424314739087, 0.292073411665228, -0.041548752125998],
        [0.376403062700467, 0.512485826188421, 0.111111111111111]
    ])
    b = np.array([0.376403062700467, 0.512485826188421, 0.111111111111111])
    c = np.array([0.155051025721682, 0.644948974278318, 1.0])


    s = len(b)
    type = 'explicit'
    # type = 'implicit'

    # Initialize variables
    t_values = [t_eval[0]]
    t_end = t_eval[-1]
    y_values = [y0]
    h = t_eval[1] - t_eval[0]
    t = t_eval[0]
    y = np.array(y0)

    # Main RK4 loop
    while t < t_end:
        # Adjust step size if the next step exceeds t_end
        if t + h > t_end:
            h = t_end - t

        # Calculate slopes
        if type == 'explicit':
            k = np.zeros((s, len(y)))
            for i in range(s):
                k[i] = fxn(t + c[i] * h, y + h*sum([A[i, j] * k[j] for j in range(i)]), args)
        elif type == 'implicit':
            k = np.zeros((s, len(y)))

            def residual(k):
                k = k.reshape(s, -1)
                eqs = []
                for i in range(s):
                    stage = y + h * np.sum([A[i, j] * k[j] for j in range(s)])
                    eqs.append(k[i] - (fxn(t + c[i] * h, stage, *args)))
                eqs = np.array(eqs)
                return eqs.flatten()
            k = root(residual, k.flatten()).x
            k.reshape(s, -1)
            # print(k)
        else:
            raise InputError('Type must be explicit or implicit.')
        y += h * sum([b[i] * k[i] for i in range(s)])
        # print(y*args[0])
        t += h

        # Store results
        t_values.append(t)
        y_values.append(list(y))

    Y = np.array(y_values)
    Y = Y.T

    return np.array(y_values).T, t_eval, 'Success', 'Yay'


def scipy_integrate(fxn, y0, t_eval, args=None):

    obj = solve_ivp(fxn, [t_eval[0], t_eval[-1]], y0,
                    args=(args,),
                    method='RK45', t_eval=t_eval,
                    vectorized=False,
                    # options={'first_step': 1e-10,
                    #          'max_step': 1e-5,
                    #          'rtol': 1e-0,
                    #          'atol': 1e-0}
                    )

    Y_scaled = obj.y
    success = obj.success
    message = obj.message
    z = obj.t

    return Y_scaled, z, success, message


def radau(fxn, y0, t_eval, args=None):
    """
        Solve an ODE using the Radau-IIA 3-stage implicit Runge-Kutta method.

        Parameters:
            f (function): The ODE function, y' = f(t, y).
            t0 (float): The initial time.
            y0 (numpy array): Initial condition vector.
            t_end (float): The end time for the integration.
            h (float): Step size.

        Returns:
            t_values (numpy array): Array of time values.
            y_values (numpy array): Array of solution values.
        """
    # Radau-IIA 3-stage coefficients (Butcher Tableau)
    A = np.array([
        [0.196815477223660, -0.065535425850198, 0.023770974348220],
        [0.394424314739087, 0.292073411665228, -0.041548752125998],
        [0.376403062700467, 0.512485826188421, 0.111111111111111]
    ])
    b = np.array([0.376403062700467, 0.512485826188421, 0.111111111111111])
    c = np.array([0.155051025721682, 0.644948974278318, 1.0])

    # Initialization
    t_values = [t_eval[0]]
    t_end = t_eval[-1]
    y_values = [y0]
    h = t_eval[1] - t_eval[0]
    t = t_eval[0]
    y = np.array(y0)

    # Helper function to solve the implicit system
    def solve_stage_values(y, t, h):
        """
        Solve the implicit system for the stage values.
        """
        s = len(c)  # Number of stages
        g = np.zeros((s, len(y)))  # Stage values
        for i in range(s):
            g[i] = y

        # Newton's method for solving the nonlinear system
        def residual(g_flat, *args):
            g = g_flat.reshape(s, -1)
            F = np.zeros_like(g)
            for i in range(s):
                F[i] = g[i] - h * np.sum([A[i, j] * fxn(t + c[j] * h, y + g[j], *args) for j in range(s)])
                print(F[i]*args[0])

            return F.flatten()

        # def jacobian(g_flat, *args):
        #     g = g_flat.reshape(s, -1)
        #     n = len(y)
        #     J = np.zeros((s * n, s * n))
        #     for i in range(s):
        #         for j in range(s):
        #             if i != j:
        #                 y_stage = y + g[j]
        #                 df_dy = np.eye(n) - h * A[i, j] * jacobian_fd(fxn, y_stage, *args)
        #                 J[i * n:(i + 1) * n, j * n:(j + 1) * n] = -df_dy
        #     return jacobian_fd(fxn, y_stage, *args)

        # Use a nonlinear solver
        sol = root(residual, g.flatten(), method='df-sane', args=(args,))
        if not sol.success:
            raise RuntimeError("Stage solve failed: " + sol.message)
        return sol.x.reshape(s, -1)

    while t < t_end:
        # Ensure we don't step past the final time
        if t + h > t_end:
            h = t_end - t

        # Solve for stage values
        stages = solve_stage_values(y, t, h)

        # Compute the next step using stage results
        y_next = y + h * np.sum([b[i] * stages[i] for i in range(len(b))])

        print(y_next)

        # Update time and solution
        t += h
        y = y_next

        # Store results
        t_values.append(t)
        y_values.append(list(y))

    return np.array(y_values).T, t_eval, 'Success', 'Yay'


def jacobian_fd(F, y, *args):
    """
    Approximate the Jacobian matrix of F at y using finite differences.

    Parameters:
        F (function): The system of equations, F(y) = [f1(y), f2(y), ..., fn(y)].
        y (numpy array): The point at which to evaluate the Jacobian.
        h (float): Step size for finite differences.

    Returns:
        J (numpy array): The Jacobian matrix.
    """
    h = 1e-6
    n = len(y)  # Number of variables
    J = np.zeros((n, n))  # Initialize Jacobian matrix

    # Loop over each variable to compute partial derivatives
    for j in range(n):
        y_perturb_forward = y.copy()
        y_perturb_forward[j] += h  # Perturb in the positive direction

        y_perturb_backward = y.copy()
        y_perturb_backward[j] -= h  # Perturb in the negative direction

        # Compute forward and backward evaluations
        F_forward = F(h, y_perturb_forward, *args)
        F_backward = F(h, y_perturb_backward, *args)

        # Central difference for better accuracy
        J[:, j] = (F_forward - F_backward) / (2 * h)
    print(J)
    return J


def implicit_runge_kutta(f, t_eval, z0):
    """
    Implicit Runge-Kutta solver for ODEs.

    Parameters:
        f: Function defining the ODE system, f(t, z).
        t_span: Tuple (t0, t_end) for the time interval.
        z0: Initial condition (array).
        h: Step size.
        A, b, c: Butcher tableau coefficients (IRK method).

    Returns:
        t_values, z_values: Arrays of time points and solutions.
    """
    A = np.array([
        [0.196815477223660, -0.065535425850198, 0.023770974348220],
        [0.394424314739087, 0.292073411665228, -0.041548752125998],
        [0.376403062700467, 0.512485826188421, 0.111111111111111]
    ])
    b = np.array([0.376403062700467, 0.512485826188421, 0.111111111111111])
    c = np.array([0.155051025721682, 0.644948974278318, 1.0])

    t0, t_end = t_eval[0], t_eval[-1]
    h = t_eval[1] - t_eval[0]
    t_values = [t0]
    z_values = [z0]
    s = len(b)  # Number of stages

    t = t0
    z = z0

    while t < t_end:
        # Adjust step size if the next step exceeds t_end
        if t + h > t_end:
            h = t_end - t

        # Solve for stages using nonlinear solver
        def residual(K):
            K = K.reshape(s, -1)
            residuals = []
            for i in range(s):
                stage = z + h * np.sum([A[i, j] * K[j] for j in range(s)])
                residuals.append(K[i] - f(t + c[i] * h, stage))
            return np.array(residuals).flatten()

        # Initial guess for K
        K0 = np.zeros((s, len(z))).flatten()

        # Solve for stages
        sol = root(residual, K0)
        if not sol.success:
            raise RuntimeError(f"Newton's method failed: {sol.message}")
        K = sol.x.reshape(s, -1)

        # Update solution
        z = z + h * np.sum([b[i] * K[i] for i in range(s)])
        t += h

        # Store results
        t_values.append(t)
        z_values.append(z)

    return np.array(t_values), np.array(z_values)