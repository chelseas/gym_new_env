import gym
import numpy as np
import scipy
import sympy as sym
from sympy import sin, cos
import tqdm


class ControllerPendulum2():
    def __init__(self, env, method=1):
        self.env = env
        self.A, self.B, self.Q, self.R, self.vars = [None]*5 # These attributes will be initialized below.
        self.compute_matrices(method)

    def compute_matrices(self, method):
        x1, x2 = sym.symbols('x1, x2')
        u1, u2 = sym.symbols('u1, u2')
        T1, T2 = sym.symbols('T1, T2')

        # M = sym.Matrix([[2, cos(x2 - x1)],
        #                [cos(x2 - x1), 1]])
        # C = sym.Matrix([[-sin(x2 - x1)*u2**2],
        #                 [ sin(x2 - x1)*u1**2]])
        # G = self.env.g/self.env.L*sym.Matrix([[sin(x1)],
        #                                       [sin(x2)]])
        # T = 1./(self.env.m*self.env.L**2)*sym.Matrix([[T1],
        #                                               [T2]])
        # F = self.env.c/(self.env.m*self.env.L**2)*sym.Matrix([[u1],
        #                                                       [u2]])
        m, L, J, c, g = self.env.m, self.env.L, self.env.J, self.env.c, self.env.g
        M = sym.Matrix([[ m*L**2/4 + J + 2*m*L**2 + m*L**2*cos(x2) + J, m*L**2/4 + J + m*L**2*cos(x2)/2],
                       [ m*L**2/4 + m*L**2*cos(x2)/2 + J,            m*L**2/4 + J]])
        C = sym.Matrix([[-m*L**2*sin(x2)*u1*u2 - m*L**2*sin(x2)*u2**2/2],
                       [ m*L**2*sin(x2)*u1**2/2]])
        G = sym.Matrix([[ m*L*g*sin(x1)/2 + m*L*g*sin(x1) + m*L*g*sin(x1+x2)/2],
                       [ m*L*g*sin(x1+x2)/2]])
        F = sym.Matrix([[ c*u1],
                       [ c*u2]])
        T = sym.Matrix([[ T1],
                       [ T2]])


        if method == 1:
            aMat = M.inv()*(T - F - C - G)
            q0 = aMat.diff(x1)
            q1 = aMat.diff(x2)
            q2 = aMat.diff(u1)
            q3 = aMat.diff(u2)
            q4 = aMat.diff(T1)
            q5 = aMat.diff(T2)

            self.A = sym.Matrix([[0,     0,     1,     0],
                                 [0,     0,     0,     1],
                                 [q0[0], q1[0], q2[0], q3[0]],
                                 [q0[1], q1[1], q2[1], q3[1]]])

            self.B = sym.Matrix([[0,     0],
                                 [0,     0],
                                 [q4[0], q5[0]],
                                 [q4[1], q5[1]]])
        else:
            M_inv = M.inv()
            aMat = sym.sympify(M_inv*(F + C + G)*(-1))
            q0 = aMat.diff(x1)
            q1 = aMat.diff(x2)
            q2 = aMat.diff(u1)
            q3 = aMat.diff(u2)

            self.A = sym.Matrix([[0, 0, 1, 0],
                                 [0, 0, 0, 1],
                                 [q0[0], q1[0], q2[0], q3[0]],
                                 [q0[1], q1[1], q2[1], q3[1]]])

            self.B = sym.zeros(4,2)
            self.B[2, 0] = M_inv[0, 0]
            self.B[2, 1] = M_inv[0, 1]
            self.B[3, 0] = M_inv[1, 0]
            self.B[3, 1] = M_inv[1, 1]

        self.Q = sym.eye(4, 4)
        # self.Q[0, 0] = 0.1*self.env.dt*self.env.N
        # self.Q[1, 1] = 0.1*self.env.dt*self.env.N
        self.R = sym.eye(2,2)#*self.env.dt*self.env.N
        self.vars = [x1, x2, u1, u2, T1, T2]

    def compute_lqr(self, old_actions, discrete=True):
        """
        Compute the discrete-time LQR controller.
        """
        vars = self.vars.copy()
        env = self.env
        sub_list = [(vars[0], env.th[0]), # setting th1
                    (vars[1], env.th[1]), # setting th2
                    (vars[2], env.u[0]),  # setting u1
                    (vars[3], env.u[1]),  # setting u2
                    (vars[4], old_actions[0]),  # setting T1
                    (vars[5], old_actions[1])] # setting T2
        Aval = self.A.subs(sub_list)
        Bval = self.B.subs(sub_list)
        a = np.array(Aval).astype(np.float64)
        b = np.array(Bval).astype(np.float64)
        r = np.array(self.R).astype(np.float64)
        q = np.array(self.Q).astype(np.float64)
        a, b, q, r = map(np.atleast_2d, (a, b, q, r))
        p = scipy.linalg.solve_discrete_are(a, b, q, r)

        #print(a)
        # LQR gain
        if not discrete:
            k = np.linalg.solve(r, b.T.dot(p))
        else:
            # k = (b.T * p * b + r)^-1 * (b.T * p * a)
            bp = b.T.dot(p)
            tmp1 = bp.dot(b)
            tmp1 += r
            tmp2 = bp.dot(a)
            k = np.linalg.solve(tmp1, tmp2)
        return k


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def angle_from_vertical(x):
    x1 = angle_normalize(x + np.pi)
    x2 = angle_normalize(x - np.pi)
    return min(x1, x2)


env = gym.make("Pendulum2-v0")
env.reset()
controller = ControllerPendulum2(env, method=1)
N = 10000
actions = [0, 0]

for i in tqdm.trange(N):
    th, _, _, _ = env.step(actions)
    K = controller.compute_lqr(actions, discrete=False)
    state = np.array([env.th[0], env.th[1], env.u[0], env.u[1]])
    actions = np.matmul(-K, state)
    actions = np.clip(-env.max_torque, env.max_torque, actions)

#env.animate(skip=1)
env.render(skip=5)


# vars = controller.vars.copy()
# sub_list = [(vars[0], np.pi), # setting th1
#             (vars[1], 0), # setting th2
#             (vars[2], 0),  # setting u1
#             (vars[3], 0),  # setting u2
#             (vars[4], 0),  # setting T1
#             (vars[5], 0)] # setting T2
#
# Q = sym.eye(4, 4)
# Q[0, 0] = 0.001
# Q[1, 1] = 0.001
# R = sym.eye(2,2)*0.01
#
# Bval = controller.B.subs(sub_list)
# Aval = controller.A.subs(sub_list)
# a = np.array(Aval).astype(np.float64)
# b = np.array(Bval).astype(np.float64)
# r = np.array(R).astype(np.float64)
# q = np.array(Q).astype(np.float64)
# a, b, q, r = map(np.atleast_2d, (a, b, q, r))
# p = scipy.linalg.solve_discrete_are(a, b, q, r)
# k = np.linalg.solve(r+b.T.dot(p).dot(b), b.T.dot(p).dot(a))
# print("A=", Aval)
# print("B=", Bval)
# print("Q=", q)
# print("r=", r)
# print("p=", p)
# print("k=", k)