import gym
import numpy as np
from numpy import sin, cos, pi
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# This is a 3-link pendulum.

class Pendulum3Env(gym.Env):
    """
    """
    """
    this is a 3-link pendulum. The inputs are
        -m: mass of each link
        -L: length of each link
        -J: moment of inertia of each link.
        -c: frictional torque coefficient
        -mc: mass of motors mounted on joints
        -th_0: initial angles
        -u_0: initial angular velocities.
        -u1, u2: torque at joints

    The equations are extracted from Jian & Zushu, "Dynamic Model and Motion Control Analysis of Three-link
                                                    Gymnastic Robot on Horizontal Bar " ICRISSP 2003
    link: http://www.cs.cmu.edu/~akalamda/kdc/hw3/3linkmanipulator.pdf
    """
    # metadata = {
    #     'render.modes' : ['human', 'rgb_array'],
    #     'video.frames_per_second' : 30
    # }

    def __init__(self, m=[1.,1.,1.], L=[1.,2.,1.], J=[1.,1.,1.],
                       mc=[1.,1.], c=[0.,0.,0.], th_0=[0,1.5,0],
                       u_0=[0.,0.,0.5], max_torque=1000,
                       g=9.8, dt=0.005, N=20, init_guess = [0.,0.,0.]):
        self.m1,   self.m2,  self.m3         = m
        self.L1,   self.L2,  self.L3         = L
        self.J1,   self.J2,  self.J3         = J
        self.mc1,  self.mc2                  = mc
        self.c1,   self.c2,  self.c3         = c
        self.g,    self.dt,  self.N          = -g, dt, N
        self.th_0, self.u_0, self.init_guess = th_0, u_0, init_guess
        self.th,   self.u,   self.current_guess, self.history = [None]*4 # these will be updated in the reset.
        self.max_torque = max_torque
        self.reset()

        A11 = 0.25*self.m1*self.L1**2 + self.J1 + (self.m2 + self.m3 + self.mc1 + self.mc2)*self.L1**2
        A12 = (0.5*self.m2*self.L2 + (self.m3 + self.mc2)*self.L2)*self.L1
        A13 = self.m3*self.L3*self.L1*0.5
        A22 = self.m2*self.L2**2*0.25 + self.J2 + (self.m3 + self.mc2)*self.L2**2
        A23 = self.m3*self.L3*self.L2*0.5
        A33 = self.m3*self.L3**2*0.25 + self.J3
        self.A = np.array([[A11, A12, A13],
                           [A12, A22, A23],
                           [A13, A23, A33]])

        B11 = -(self.c1 + self.c2)
        B12 = self.c2 +(self.m2*self.L2*0.5 + (self.m3 + self.mc2)*self.L2)*self.L1
        B13 = self.m3*self.L3*self.L1*0.5
        B21 = self.c2 - (self.m2*self.L2*0.5 + (self.m3 + self.mc2)*self.L2)*self.L1
        B22 = -(self.c2 + self.c3)
        B23 = self.c3 + self.m3*self.L3*self.L2 *0.5
        B32 = self.c3 - self.m3*self.L3*self.L2 *0.5
        B33 = -self.c3
        self.B = np.array([[B11, B12, B13],
                           [B21, B22, B23],
                           [B13, B32, B33]])

        C1 = (self.m1*self.L1*0.5 + (self.m2 + self.m3 + self.mc1 + self.mc2)*self.L1)*self.g
        C2 = (self.m2*self.L2*0.5 + (self.m3 + self.mc2)*self.L2)*self.g
        C3 = self.m3*self.L3*0.5*self.g
        self.C = np.array([C1, C2, C3])

    def reset(self):
        self.th = np.array(self.th_0)
        self.u  = np.array(self.u_0)
        self.current_guess = self.init_guess
        self.history = [{"th": self.th.copy(), "reward": 0, "u": self.u.copy()}]

    def time_dependent_matrices(self):
        th1, th2, th3 = self.th
        u1, u2, u3 = self.u
        Ap = np.array([[1, cos(th2 - th1), cos(th3 - th1)],
                       [cos(th2 - th1), 1, cos(th3 - th2)],
                       [cos(th3 - th1), cos(th3 - th2), 1]])
        Bp = np.array([[1, sin(th2 - th1) * u2, sin(th3 - th1) * u3],
                       [sin(th2 - th1) * u1, 1, sin(th3 - th2) * u3],
                       [-sin(th3 - th1) * u1, sin(th3 - th2) * u2, 1]])
        Cp = np.array([sin(th1), sin(th2), sin(th3)])

        return Ap, Bp, Cp

    def setup_equation(self, u_p, actions):
        Ap, Bp, Cp = self.time_dependent_matrices()
        eqs = (self.A*Ap).dot(u_p) - (self.B*Bp).dot(self.u) - self.C*Cp
        eqs += np.array([-actions[0], actions[0]-actions[1], actions[1]])
        return eqs

    def step(self, actions):
        actions = np.clip(actions, -self.max_torque, self.max_torque).tolist()
        costs = 0
        for _ in range(self.N):
            u_p = fsolve(self.setup_equation, self.current_guess, args=actions)
            self.u += self.dt * u_p
            self.th += self.dt * self.u - 1 / 2 * u_p * self.dt ** 2
            costs += 1.000*np.sum(np.square([min(angle_normalize(th-pi), angle_normalize(th+pi)) for th in self.th])) #+ \
                    # 0.100*np.sum(np.square(self.u)) + \
                    # 0.001*np.sum(np.square(u_p))

        self.history.append({"th": self.th.copy(), "reward": -costs, "u": self.u.copy()})
        return self.th, -costs, False, {}

    def render(self, mode="human", close=False):
        plt.figure()
        for i in range(len(self.history)):
            plt.cla()

            th     = self.history[i]["th"]
            reward = self.history[i]["reward"]

            x0, y0 = 0, 0
            x1, y1 = x0 + self.L1*sin(th[0]), y0 + self.L1*cos(th[0])
            x2, y2 = x1 + self.L2*sin(th[1]), y1 + self.L2*cos(th[1])
            x3, y3 = x2 + self.L3*sin(th[2]), y2 + self.L3*cos(th[2])
            plt.plot([0,   -x1], [0,   -y1], color='black')
            plt.plot([-x1, -x2], [-y1, -y2], color='black', markersize=10, markerfacecolor='red', marker = "o")
            plt.plot([-x2, -x3], [-y2, -y3], color='black', markersize=10, markerfacecolor='red', marker = "o")

            limits = self.L1+self.L2+self.L3+1
            plt.xlim([-limits, limits])
            plt.ylim([-limits, limits])

            xtext, ytest = 0.1*limits, 0.8*limits
            plt.text(xtext, ytest, "th1:%0.1f, th2:%0.1f, th3:%0.1f" %(th[0], th[1], th[2]))
            xtext, ytest = 0.1*limits, 0.65*limits
            plt.text(xtext, ytest, "reward:%0.2f" %reward)
            plt.draw()
            plt.pause(0.1)
        plt.close()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)