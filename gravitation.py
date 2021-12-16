import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import time
import numba as nb
import timeit

dt = 100
G = 6.67e-11
sim_lim = 2000
error = 1
test3_extra_bodies = 2


def random_body():
    return CosmicBody(float(rnd.randrange(1e22, 1e23, 1.0e19)),
                      np.array([float(rnd.randrange(-100000, 100000, 100)),
                                float(rnd.randrange(-100000, 100000, 100)),
                                float(rnd.randrange(-100000, 100000, 100))]),
                      np.array([float(rnd.randrange(-1e11, 1e11, 1e8)),
                                float(rnd.randrange(-1e11, 1e11, 1e8)),
                                float(rnd.randrange(-1e11, 1e11, 1e8))]))


@nb.njit
def norm(vec: np.ndarray):
    return np.linalg.norm(vec, ord=2)


@nb.njit
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
    def __init__(self, mass: float, vec_v: np.ndarray, vec_P: np.ndarray, r: float = 0.):
        """


        Parameters
        ----------
        mass : float
            Object mass
        vec_v : np.ndarray
            Velocity vector.
        vec_P : np.ndarray
            Coordinate vector.
        r : float, optional
            Object radius. The default is 0. .
        Returns
        -------
        None.

        """

        self.mass = mass
        self.vec_v = vec_v
        self.vec_P = vec_P
        self.coords = [[self.vec_P[0]], [self.vec_P[1]], [self.vec_P[2]]]
        self.radius = r
        self.destroy_flag = False
        self.id = id(self)

    def __str__(self):
        return f"m:{self.mass} v:({round(self.vec_v[0],2)}, {round(self.vec_v[1],2)}, {round(self.vec_v[2],2)}) c:({round(self.vec_P[0],2)}, {round(self.vec_P[1],2)}, {round(self.vec_P[2],2)})"

    def E_k(self):
        """
        Returns object's kinetic energy

        """
        return self.mass * norm(self.vec_v) ** 2 / 2


def try_destroy(self_body: CosmicBody, body: [CosmicBody, Star]):
    """
    Trying to destroy (and delete from system) some objects

    Parameters
    ----------
    self_body : CosmicBody
        first body
    body : [CosmicBody, Star]
        array of bodies

    Returns
    -------
    None.

    """
    if isinstance(body, Star):
        if norm(self_body.vec_P - body.vec_P) < 1:
            self_body.destroy_flag = True
    else:
        if norm(self_body.vec_P - body.vec_P) < 1:
            body.destroy_flag = True
            self_body.destroy_flag = True


def E_p(body1, body2):
    """
    returns potential energy of 2 bodies

    Parameters
    ----------

    Returns
    -------
    TYPE
        float

    """
    return G * body1.mass * body2.mass / norm(body1.vec_P - body2.vec_P)


def E_full(star: Star, bodies: np.ndarray):
    """
    returns full system energy (potential+kinetic)

    Parameters
    ----------
    star : Star
    bodies : np.ndarray
        bodies list

    Returns
    -------
    E : float
        system full energy

    """
    E = 0
    for i in range(len(bodies)):
        E += E_p(bodies[i], star) + bodies[i].E_k()
        for j in range(i + 1, len(bodies)):
            E += E_p(bodies[i], bodies[j])
    return E


def gravitate1(star: Star, bodies: list):  # useless
    """
    Gravitation method

    Parameters
    ----------
    star : Star
        DESCRIPTION.
    bodies : list
        list of bodies

    Returns
    -------
    None.

    """
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
            k=1
            # bodies[i].coords[0].append(bodies[i].vec_P[0])
            # bodies[i].coords[1].append(bodies[i].vec_P[1])
            # bodies[i].coords[2].append(bodies[i].vec_P[2])


def gravitate(star: Star, bodies: list):
    bodies_copy = copy.deepcopy(bodies)
    for body, body_copy in zip(bodies, bodies_copy):
        try_destroy(body, star)
        body_copy.destroy_flag = body.destroy_flag
        if body.destroy_flag == True:
            continue
        dv = accelerate(star.mass, - body.vec_P) * dt
        body.vec_v += dv
        body.vec_P += body.vec_v * dt
        for dbody in bodies:
            if dbody.id != body.id:
                try_destroy(body, dbody)
                body_copy.destroy_flag = body.destroy_flag
        for body1 in bodies_copy:
            if body1.id == body.id or body1.id == True:
                continue
            dv = accelerate(body1.mass, body_copy.vec_P - body1.vec_P)*dt
            body.vec_v += dv
            body.vec_P += body.vec_v * dt
        # body.coords[0].append(body.vec_P[0])
        # body.coords[1].append(body.vec_P[1])
        # body.coords[2].append(body.vec_P[2])


def orbit_type(star: Star, body: CosmicBody):
    """
    Defines body's orbit type

    Parameters
    ----------
    star : Star
        DESCRIPTION.
    body : CosmicBody
        DESCRIPTION.

    Returns
    -------
    str
        orbit type

    """
    E = E_p(star, body) - body.E_k()
    if E > 0:
        return 'Elliptic'
    elif E < 0:
        return 'Hyperbolic'
    else:
        return 'Parabolic'
# ----------------------TESTS----------------------


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
    # ax.set_xlim3d(-1e11, 1e11)
    # ax.set_ylim3d(-1e11, 1e11)
    # ax.set_zlim3d(-1e11, 1e11)
    if star.mass != 0:
        ax.scatter(0, 0, 0, marker='*', s=200)
    for body in bodies:
        ax.scatter(body.coords[0][::4], body.coords[1][::4],
                   body.coords[2][::4], marker='.', s=7)
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
        if float(t) * dt in time:
            smthng = random_body()
            bodies = np.append(bodies, smthng)
        for body in bodies:
            gravitate(star, bodies)
    return bodies


def test3_plot(star, bodies: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(-20, 20)
    # ax.set_zlim(-20, 20)
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
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(-20, 20)
    # ax.set_zlim(-20, 20)
    ax.scatter(star.vec_P[0], star.vec_P[1], star.vec_P[2], marker='*', s=200)
    ax.scatter(body.coords[0], body.coords[1],
               body.coords[2], marker='.', s=10)
    ax.scatter(body.coords[0][0], body.coords[1]
               [0], body.coords[2][0], color='red', label='spawn point')
    ax.set_title('Test №4')


def test6(star, bodies):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlim3d(-20, 20)
    # ax.set_ylim3d(-20, 20)
    # ax.set_zlim3d(-20, 20)

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


if __name__ == "__main__":
    Earth = CosmicBody(5, np.array([1., 0., 1.]), np.array([1., 1., 1.]))
    system_of_2 = np.array([random_body(), random_body()])
    system_of_2_copy = copy.deepcopy(system_of_2)
    system_of_3 = np.array([random_body(), random_body(), random_body()])
    system_of_4 = np.array(
        [random_body(), random_body(), random_body(), random_body()])
    Sun = Star(1e31)
    zero = Star(0)

    # =========================================================

    start_time = time.time()
    whole_time = time.time()

    # # Testing law of conservation of energy for system of 2 bodies
    # test1(Sun, system_of_2)
    # print("test1 time - %s seconds" % (time.time() - start_time))

    # # Testing our gravitational model on system of 4 bodies
    test2(Sun, system_of_4)
    # test2_plot(Sun, system_of_2)
    # start_time = time.time()

    # # Testing random timings adding system
    # test3_res = test3(Sun, system_of_3)
    # test3_plot(Sun, test3_res)
    # print("test3 time - %s seconds" % (time.time() - start_time))

    # # Testing orbit_type function
    # test4(Sun, CosmicBody(rnd.randrange(100, 1000, 1)/10,
    #                       np.array([rnd.randrange(-1000, 1000, 1)/100,
    #                                 rnd.randrange(-1000, 1000, 1)/100,
    #                                 rnd.randrange(-1000, 1000, 1)/100]),
    #                       np.array([rnd.randrange(-1000, 1000, 1)/100,
    #                                 rnd.randrange(-1000, 1000, 1)/100,
    #                                 rnd.randrange(-1000, 1000, 1)/100])))

    # start_time = time.time()
    # Testing 3d animation
    # test6(Sun, system_of_4)
    print("test6 time - %s seconds" % (time.time() - start_time))
    print("tests full time - %s seconds" % (time.time() - whole_time))
