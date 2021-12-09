import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import time

dt = 0.01
G = 0.1
sim_lim = 1000
error = 10
test3_extra_bodies = 2


def norm(vec: np.ndarray):
    return np.linalg.norm(vec, ord=2)


def accelerate(M: float, r: np.ndarray):
    return G * M * r / norm(r) ** 3


class Star:
    def __init__(self, mass: float, radius=0.):
        self.mass = mass
        self.vec_P = [0, 0, 0]
        self.radius = radius

    def __str__(self):
        return f"mass:{np.round(self.mass,2)} radius:{self.radius}"


class CosmicBody:
    def __init__(self, mass: float, vec_v: np.ndarray, vec_P: np.ndarray, r=0.):
        self.mass = mass
        self.vec_v = vec_v
        self.vec_P = vec_P
        self.coords = [[self.vec_P[0]], [self.vec_P[1]], [self.vec_P[2]]]
        self.radius = r
        self.destroy_flag = False

    def __str__(self):
        return f"m:{self.mass} v:({round(self.vec_v[0],2)}, {round(self.vec_v[1],2)}, {round(self.vec_v[2],2)}) c:({round(self.vec_P[0],2)}, {round(self.vec_P[1],2)}, {round(self.vec_P[2],2)})"

    def E_k(self):
        return self.mass * norm(self.vec_v) ** 2 / 2


def try_destroy(self_body: CosmicBody, body: [CosmicBody, Star]):
    if isinstance(body, Star):
        if norm(self_body.vec_P - body.vec_P) < 0.1 or norm(self_body.vec_P - body.vec_P) > 50:
            self_body.destroy_flag = True
    else:
        if norm(self_body.vec_P - body.vec_P) < 1:
            body.destroy_flag = True
            self_body.destroy_flag = True


def E_p(body1, body2):
    return G * body1.mass * body2.mass / norm(body1.vec_P - body2.vec_P)


def E_full(star, bodies: np.ndarray):
    E = 0
    for i in range(len(bodies)):
        E += E_p(bodies[i], star) + bodies[i].E_k()
        for j in range(i + 1, len(bodies)):
            E += E_p(bodies[i], bodies[j])
    return E


def gravitate(star, bodies: list):
    bodies_copy = copy.deepcopy(bodies)
    for i in range(len(bodies)):
        try_destroy(bodies[i], star)
        if bodies[i].destroy_flag == True:
            bodies_copy[i] = bodies[i]
        for k in range(len(bodies)):
            if k != i:
                continue
                try_destroy(bodies[i], bodies[k])
        dv = accelerate(star.mass, star.vec_P - bodies[i].vec_P) * dt
        if not bodies[i].destroy_flag:
            bodies[i].vec_v += dv
            bodies[i].vec_P += bodies[i].vec_v * dt
        for j in range(len(bodies_copy)):
            if j != i:
                dv = accelerate(
                    bodies_copy[j].mass,  bodies_copy[i].vec_P - bodies_copy[j].vec_P) * dt
                if not (bodies[i].destroy_flag and bodies_copy[j].destroy_flag):
                    bodies[i].vec_v += dv
                    bodies[i].vec_P += bodies[i].vec_v * dt
        if not bodies[i].destroy_flag:
            bodies[i].coords[0].append(bodies[i].vec_P[0])
            bodies[i].coords[1].append(bodies[i].vec_P[1])
            bodies[i].coords[2].append(bodies[i].vec_P[2])


def orbit_type(star, body):
    E = E_p(star, body) - body.E_k()
    if E > 0:
        return 'Elliptic'
    elif E < 0:
        return 'Hyperbolic'
    else:
        return 'Parabolic'


def optimal_velocity(star, body):
    e = (norm(body.vec_P) - star.radius) / (norm(body.vec_P) + star.radius)
    return np.sqrt(np.abs(2 * G * star.mass * (1 - e) / (1 + e) / (star.radius + norm(body.vec_P))))


def test1(star, bodies: np.ndarray):
    print('Test №1: calculating Energy error')
    E_initial = E_full(star, bodies)
    print('Initial system full energy:', E_initial)
    i = 0
    E_arr = []
    while np.abs(E_full(star, bodies)/E_initial - 1) < error and i < sim_lim:
        gravitate(star, bodies)
        i += 1
        E_arr.append(E_full(star, bodies))
    print('Final system full energy:', E_full(star, bodies))
    if i == sim_lim:
        print(
            f"for {i} iterations we have {round(np.abs(E_full(star,bodies)/E_initial-1)*100,1)}% error")
    else:
        print(f"{i} iterations needed to get {round(error*100,1)}% error")


def test2(star, bodies: np.ndarray):
    print('\nTest №2:')
    for body in bodies:
        print(body)
    for i in range(sim_lim):
        gravitate(star, bodies)


def test2_plot(star, bodies: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    if star.mass != 0:
        ax.scatter(0, 0, 0, marker='*', s=200)
    for body in bodies:
        ax.scatter(body.coords[0], body.coords[1],
                   body.coords[2], marker='.', s=7)
        ax.scatter(body.coords[0][0], body.coords[1]
                   [0], body.coords[2][0], color='red', label='spawn point', s=10)
    ax.set_title('Test №2')


def test3(star, bodies: np.ndarray):
    print('\nTest №3: adding bodies in random time')
    time = np.array(
        [rnd.randrange(0, sim_lim, 1) * dt for j in range(test3_extra_bodies)])
    time.sort()
    print('Random timings:', time)
    for t in range(sim_lim):
        if float(t)/sim_lim in time:
            smthng = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                np.array([rnd.randrange(-1000, 1000, 1)/100,
                                          rnd.randrange(-1000, 1000, 1)/100,
                                          rnd.randrange(-1000, 1000, 1)/100]),
                                np.array([rnd.randrange(-1000, 1000, 1)/100,
                                          rnd.randrange(-1000, 1000, 1)/100,
                                          rnd.randrange(-1000, 1000, 1)/100]), t)
            bodies = np.append(bodies, smthng)
        for body in bodies:
            gravitate(star, bodies)
    return bodies


def test3_plot(star, bodies: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    if star.mass != 0:
        ax.scatter(0, 0, 0, marker='*', s=100)
    for body in bodies:
        ax.scatter(body.coords[0], body.coords[1],
                   body.coords[2], marker='.', s=7)
        ax.scatter(body.coords[0][0], body.coords[1]
                   [0], body.coords[2][0], color='red', label='spawn point', s=10)
    ax.set_title('Test №3')


def test4(star, body):
    print('\nTest №4: testing orbyt_type function')
    print(orbit_type(star, body))
    for i in range(sim_lim):
        gravitate(star, np.array([body]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    ax.scatter(star.vec_P[0], star.vec_P[1], star.vec_P[2], marker='*', s=200)
    ax.scatter(body.coords[0], body.coords[1],
               body.coords[2], marker='.', s=10)
    ax.scatter(body.coords[0][0], body.coords[1]
               [0], body.coords[2][0], color='red', label='spawn point')
    ax.set_title('Test №4')


def test5(star, body: CosmicBody):
    body_copy = copy.deepcopy(body)
    for i in range(sim_lim):
        gravitate(star, [body])
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.scatter(0, 0, marker='*', s=100)
    ax.scatter(body.coords[0], body.coords[1], marker='.', s=7)
    ax.scatter(body.coords[0][0], body.coords[1]
               [0], color='red', label='spawn point', s=10)
    ax.set_title('Test №5')
    body_copy.vec_v = np.array([-body_copy.vec_P[1], body_copy.vec_P[0], 0]) / norm(
        body_copy.vec_P) * optimal_velocity(star, body_copy)
    for i in range(sim_lim):
        gravitate(star, [body_copy])
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.scatter(0, 0, marker='*', s=100)
    ax1.scatter(body_copy.coords[0],
                body_copy.coords[1], marker='.', s=7)
    ax1.scatter(body_copy.coords[0][0], body_copy.coords[1]
                [0], color='red', label='spawn point', s=10)
    ax1.set_title('Test №5.1')


def test6(star, bodies):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def animate(i):
        gravitate(star, bodies)
        print(sim_lim - i)
        ax.cla()
        ax.scatter(0, 0, 0, marker='*', s=100)
        for body in bodies:
            ax.scatter(body.coords[0], body.coords[1], body.coords[2], s=7)
    ani = FuncAnimation(plt.gcf(), animate, interval=100,
                        frames=sim_lim, blit=False)
    ani.save('C:/Users/timof/Desktop/smthm.mp4')

    # plt.show()
if __name__ == "__main__":
    Earth = CosmicBody(5, np.array([1., 0., 1.]), np.array([1., 1., 1.]))
    comet = CosmicBody(30, np.array([-3., 10., 5.]), np.array([6., -5., 1.]))
    comet1 = CosmicBody(30, np.array([-3., 10., 5.]), np.array([6., -5., 1.]))
    Venus = CosmicBody(6, np.array([-1., 0., 1.]), np.array([4., 1., 1.]))

    system_of_2 = np.array([CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])),
                            CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]))])
    system_of_3 = np.array([CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])),
                            CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])), CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                                                                                  np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                                                                            rnd.randrange(
                                                                                                      -1000, 1000, 1)/100,
                                                                                                      rnd.randrange(-1000, 1000, 1)/100]),
                                                                                                  np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                                                                            rnd.randrange(
                                                                                                      -1000, 1000, 1)/100,
                                                                                                      rnd.randrange(-1000, 1000, 1)/100]))])
    system_of_4 = np.array([CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])),
                            CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])),
                            CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100])),
                            CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]),
                                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                                 rnd.randrange(-1000,
                                                               1000, 1)/100,
                                                 rnd.randrange(-1000, 1000, 1)/100]))])
    ss = np.array([Earth, Venus])
    Sun = Star(10000, 2)
    zero = Star(0)

    # ------------------------TESTS------------------------

    start_time = time.time()
    whole_time=time.time()
    # Testing law of conservation of energy for system of 2 bodies
    test1(Sun, system_of_2)
    print("test1 time - %s seconds" % (time.time() - start_time))
    # Testing our gravitational model on system of 3 bodies
    test2(zero, system_of_3)
    test2_plot(zero, system_of_3)
    start_time = time.time()
    # Testing random timings adding system
    test3_res = test3(Sun, system_of_2)
    test3_plot(Sun, test3_res)
    print("test3 time - %s seconds" % (time.time() - start_time))
    # Testing orbit_type function
    test4(Sun, CosmicBody(rnd.randrange(100, 1000, 1)/10,
                          np.array([rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100]),
                          np.array([rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100])))

    # Testing optimal_velocity function
    test5(Sun, CosmicBody(rnd.randrange(100, 1000, 1)/10,
                          np.array([rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100]),
                          np.array([rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100,
                                    rnd.randrange(-1000, 1000, 1)/100])))
    start_time = time.time()
    # Testing 3d animation
    test6(Sun, system_of_4)
    print("test6 time - %s seconds" % (time.time() - start_time))
    print("tests full time - %s seconds" % (time.time() - whole_time))