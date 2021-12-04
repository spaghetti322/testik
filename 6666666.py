import numpy as np
import random as rnd
import matplotlib.pyplot as plt

dt = 0.0001
G = 0.1
sim_lim = 1000
error = 100
test3_extra_bodies = 3


def norm(vec: np.ndarray):
    return np.linalg.norm(vec, ord=2)


def accelerate(M: float, r: np.ndarray):
    return G * M * r / norm(r) ** 3


class Star:
    def __init__(self, mass: float):
        self.mass = mass
        self.vec_P = [0, 0]

    def __str__(self):
        return f"mass:{np.round(self.mass,2)}"


class CosmicBody:
    def __init__(self, mass, vec_v: np.ndarray, vec_P: np.ndarray, s_t=0.):
        self.mass = mass
        self.vec_v = vec_v
        self.vec_P = vec_P
        self.coords = [[self.vec_P[0]], [self.vec_P[1]]]
        self.spawn_time = s_t

    def __str__(self):
        return f"m:{self.mass} v:({round(self.vec_v[0],2)}, {round(self.vec_v[1],2)}) c:({round(self.vec_P[0],2)}, {round(self.vec_P[1],2)})"

    def E_k(self):
        return self.mass * norm(self.vec_v) ** 2 / 2


def try_to_destroy(self_body, bodies):
    for body in bodies:
        if isinstance(body, Star):
            if np.allclose(self_body.vec_P[0], body.vec_P[0], 1e-3) and np.allclose(self_body.vec_P[1], body.vec_P[1], 1e-3):
                print(f"Body {self_body} have crushed into star {body}")
                self_body.__del__()
        if self_body != body and np.allclose(self_body.vec_P[0], body.vec_P[0], 1e-3) and np.allclose(self_body.vec_P[1], body.vec_P[1], 1e-3):
            print(f"Bodies {self_body} and {body} have crushed")
            self_body.__del__()
            body.__del__()


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
    bodies_copy = bodies.copy()
    for i in range(len(bodies)):
        try_to_destroy(bodies[i], bodies)
        dv = accelerate(star.mass, star.vec_P - bodies[i].vec_P) * dt
        bodies[i].vec_v += dv
        bodies[i].vec_P += bodies[i].vec_v * dt
        for j in range(len(bodies_copy)):
            if j != i:
                dv = accelerate(
                    bodies_copy[j].mass,  bodies_copy[i].vec_P - bodies_copy[j].vec_P) * dt
                bodies[i].vec_v += dv
                bodies[i].vec_P += bodies[i].vec_v * dt
        bodies[i].coords[0].append(bodies[i].vec_P[0])
        bodies[i].coords[1].append(bodies[i].vec_P[1])


def orbit_type(star, body):
    E = E_p(star, body) - body.E_k()
    if E > 0:
        return 'Elliptic'
    elif E < 0:
        return 'Hyperbolic'
    else:
        return 'Parabolic'


#------------Переделать---------------#
# def gravitate_peredelat(star, bodies: list):
#     bodies_copy = bodies.copy()
#     for body in bodies:
#         for another_body, another_body_copy in bodies, bodies_copy:
#             if body != another_body:
#                 dv = accelerate(
#                     another_body_copy.mass, another_body_copy.vec_P - body.vec_P) * dt
#                 body.vec_v += dv
#                 body.vec_P += body.vec_v * dt
#         dv = accelerate(star.mass, star.vec_P - body.vec_P) * dt
#         body.vec_v += dv
#         body.vec_P += body.vec_v * dt

#         np.append([body.coords], [body.vec_P])
#         destroy(body, bodies)
#-------------------------------------#


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
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    if star.mass != 0:
        ax.scatter(star.vec_P[0], star.vec_P[1], marker='*', s=200)
    for body in bodies:
        ax.scatter(body.coords[0], body.coords[1], marker='.', s=10)
        ax.scatter(body.coords[0][0], body.coords[1]
                   [0], color='red', label='spawn point')
    ax.set_title('Test №2')


def test3(star, bodies: np.ndarray):
    print('\nTest №3: adding bodies in random time')
    time = np.array(
        [rnd.randrange(0, dt * sim_lim * 1000, 1)/1000 for j in range(test3_extra_bodies)])
    time.sort()
    print('Random timings:', time)
    for t in np.arange(0., sim_lim * dt, dt):
        if t in time:
            smthng = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                                np.array([rnd.randrange(-1000, 1000, 1)/100,
                                          rnd.randrange(-1000, 1000, 1)/100]),
                                np.array([rnd.randrange(-1000, 1000, 1)/100, rnd.randrange(-1000, 1000, 1)/100]), t)
            bodies = np.append(bodies, smthng)
        for body in bodies:
            gravitate(star, bodies)
    return bodies


def test3_plot(star, bodies: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    if star.mass != 0:
        ax.scatter(star.vec_P[0], star.vec_P[1], marker='*', s=200)
    for body in bodies:
        ax.scatter(body.coords[0], body.coords[1], marker='.', s=10)
        ax.scatter(body.coords[0][0], body.coords[1]
                   [0], color='red', label='spawn point')
    ax.set_title('Test №3')


def test4(star, body):
    print('\nTest №4: testing orbyt_type function')
    print(orbit_type(star, body))
    for i in range(sim_lim * 100):
        gravitate(star, np.array([body]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(-20, 20)
    ax.scatter(star.vec_P[0], star.vec_P[1], marker='*', s=200)
    ax.scatter(body.coords[0], body.coords[1], marker='.', s=10)
    ax.scatter(body.coords[0][0], body.coords[1]
               [0], color='red', label='spawn point')
    ax.set_title('Test №4')


if __name__ == "__main__":
    body1 = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                 rnd.randrange(-1000, 1000, 1)/100]),
                       np.array([rnd.randrange(-1000, 1000, 1)/100, rnd.randrange(-1000, 1000, 1)/100]))
    body2 = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                 rnd.randrange(-1000, 1000, 1)/100]),
                       np.array([rnd.randrange(-1000, 1000, 1)/100, rnd.randrange(-1000, 1000, 1)/100]))
    body3 = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                       np.array([rnd.randrange(-1000, 1000, 1)/100,
                                 rnd.randrange(-1000, 1000, 1)/100]),
                       np.array([rnd.randrange(-1000, 1000, 1)/100, rnd.randrange(-1000, 1000, 1)/100]))
    body4 = CosmicBody(rnd.randrange(100, 1000, 1)/10,
                       np.array([rnd.randrange(-1000, 1000, 1)/50,
                                 rnd.randrange(-1000, 1000, 1)/50]),
                       np.array([rnd.randrange(-1000, 1000, 1)/100, rnd.randrange(-1000, 1000, 1)/100]))
    Earth = CosmicBody(5, np.array([0.6, 0.4]), np.array([0.99, 1.23]))
    comet = CosmicBody(1, np.array([0., 2.]), np.array([6., 0.]))
    Venus = CosmicBody(6, np.array([0.12, 0.17]), np.array([0.99, 1.23001]))

    system_of_2 = np.array([body1, body2])
    system_of_3 = np.array([body1, body2, body3])
    system_of_4 = np.array([body1, body2, body3, body4])
    Sun = Star(10000)
    zero = Star(0)

    # ------------------------TESTS------------------------

    # Testing law of conservation of energy for system of 2 bodies
    # test1(Sun, system_of_2)

    # Testing our gravitational model on system of 3 bodies
    # test2(Sun, system_of_3)
    # test2_plot(Sun, system_of_3)

    # Testing random timings adding system
    # test3_res = test3(Sun, system_of_2)
    # test3_plot(Sun, test3_res)

    # Testing orbit_type function
    test4(Sun, body4)
