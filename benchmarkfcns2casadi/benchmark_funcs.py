"""Module for benchmarking functions

Based on benchmarkfcns by Mazhar Ansari Ardeh. All generating functions have
sane default parameters. Parameters are named alphabetically as they occur in
the expression, and all generating functions have the dimension parameter n.
Additionally, all generating functions can have function options sent through
func_opts, and data type (SX or MX) set by setting the data_type argument.

The generating functions return:
func, default_input_domain, minima


The default input domain, minima are suggested values
and care should be taken to make sure they make sense.
Todo: Somehow specifying global vs local minima


Copyright: Mathias Hauan Arbo
Norwegian University of Science and Technology
Department of Engineering Cybernetics 2020

License: MIT License"""
import casadi as cs


def generate_ackley(n=2, a=20, b=0.2, c=2*cs.np.pi,
                    func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    sum1 = cs.sumsqrt(x)
    sum2 = cs.sum1(cs.cos(x))
    f = -a*cs.exp(-b*cs.sqrt((1./n)*sum1))-cs.exp((1./n)*sum2)+a+cs.exp(1)
    func = cs.Function("ackley", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-32., 32.]]*n, [[0.]*n]


def generate_ackleyn2(n=2, a=200, b=0.02,
                      func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("ackleyn2 is only defined for n=2")
    x = data_type.sym("x", n)
    f = -a*cs.exp(-b*cs.sqrt(x[0]**2+x[1]**2))
    func = cs.Function("ackleyn2", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-32., 32.]]*n, [[0.]*n]


def generate_ackleyn3(n=2, a=200, b=0.02, c=5., d=3.,
                      func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("ackleyn3 is only defined for n=2")
    x = data_type.sym("x", n)
    f = -a*cs.exp(-b*cs.sqrt(x[0]**2+x[1]**2))
    f += c*cs.exp(cs.cos(d*x[0])+cs.sin(d*x[1]))
    func = cs.Function("ackleyn3", [x], [f], ["x"], ["f"], func_opts)
    glob_min = [[0.682584587365898, -0.36075325513719],
                [-0.682584587365898, -0.36075325513719]]
    return func, [[-32., 32.]]*n, glob_min


def generate_ackleyn4(n=2, a=0.2, b=3., c=2.,
                      func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    f = 0.
    for i in range(n-1):
        f += cs.exp(-a)*cs.sqrt(x[i]**2+x[i+1]**2)
        f += b*cs.cos(c*x[i])+cs.sin(c*x[i+1])
    func = cs.Function("ackleyn4", [x], [f], ["x"], ["f"], func_opts)
    if n == 2:
        glob_min = [[-1.51, -0.755]]
    else:
        glob_min = []
    return func, [[-35., 35.]*n], glob_min


def generate_adjiman(n=2, func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("adjiman is onlye defined for n=2")
    x = data_type.sym("x", n)
    f = cs.cos(x[0])*cs.sin(x[1]) - (x[0])/(x[1]**2 + 1)
    func = cs.Function("adjiman", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-1., 2.], [-1., 1.]], [[0.]*n]


def generate_alpinen1(n=2, a=0.1, func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    f = 0.
    for i in range(n):
        f += cs.abs(x[i]*cs.sin(x[i]) + a*x[i])
    func = cs.Function("alpinen1", [x], [f], ["x"], ["f"], func_opts)
    return func, [[0., 10.]]*n, [[0.]*n]


def generate_alpinen2(n=2, func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    f = 1.
    for i in range(n):
        f = f*cs.sqrt(x[i])*cs.sin(x[i])
    f = -f  # Note the minus!
    func = cs.Function("alpinen2", [x], [f], ["x"], ["f"], func_opts)
    return func, [[0., 10.]]*n, [[7.917]*n]


def generate_bartelsconn(n=2, func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("bartelsconn is only defined for n=2")
    x = data_type.sym("x", n)
    f = cs.abs(cs.sumsqr(x)+x[0]*x[1])
    f += cs.abs(cs.sin(x[0]))
    f += cs.abs(cs.cos(x[1]))
    func = cs.Function("bartelsconn", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-500., 500.]]*n, [[0.]*n]


def generate_beale(n=2, a=1.5, b=2.25, c=2.625,
                   func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("beale is only defined for n=2")
    x = data_type.sym("x", n)
    f = (a - x[0] + x[0]*x[1])**2 + (b - x[0] + x[0]*x[1]**2)**2
    f += (c - x[0] + x[0]*x[1]**3)**2
    func = cs.Function("beale", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-4.5, 4.5]]*n, [[3., 0.5]]


def generate_bird(n=2, func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("bird is only defined for n=2")
    x = data_type.sym("x", n)
    f = cs.sin(x[0])*cs.exp((1 - cs.cos(x[1]))**2)
    f += cs.cos(x[1])*cs.exp((1 - cs.sin(x[0]))**2)
    f += (x[0]-x[1])**2
    func = cs.Function("bird", [x], [f], ["x"], ["f"], func_opts)
    glob_min = [[4.70104, 3.15294],
                [-1.58214, -3.13024]]
    return func, [[-2*cs.np.pi, 2*cs.np.pi]]*n, glob_min


def generate_bochavskeyn2(n=2, a=2., b=0.3, c=3*cs.np.pi, d=0.4, e=4*cs.np.pi,
                          f=0.7, func_opts={}, data_type=cs.MX):
    farg = f
    if n != 2:
        raise ValueError("bochavskeyn2 is only defined for n=2")
    x = data_type.sym("x", n)
    f = x[0]**2 + a*x[1]**2 - b*cs.cos(c*x[0]) - d*cs.sin(e*x[1]) + farg
    func = cs.Function("bochavskeyn2", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-100., 100.]]*n, [[0., 0.]]


def generate_bochavskeyn3(n=2, a=2., b=0.3, c=3*cs.np.pi, d=4*cs.np.pi, e=0.3,
                          func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("bochavskeyn3 is only defined for n=2")
    x = data_type.sym("x", n)
    f = x[0]**2 + a*x[1]**2 - b*cs.cos(c*x[0])*cs.cos(d*x[1]) + e
    func = cs.Function("bochavskeyn3", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-100., 100]]*n, [[0.]*n]


def generate_booth(n=2, a=2., b=7., c=2., d=5., func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("booth is only defined for n=2")
    x = data_type.sym("x", n)
    f = (x[0] + a*x[1] - b)**2 + (c*x[0] + x[1] - d)**2
    func = cs.Function("booth", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-10., 10.]]*n, [[1., 3.]]


def generate_brent(n=2, a=10., b=10., func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("brent is only defined for n=2")
    x = data_type.sym("x", n)
    f = (x[0] + a)**2 + (x[1]+b)**2 + cs.exp(-cs.sumsqr(x))
    func = cs.Function("brent", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-20., 0.]]*n, [[-10.]*n]


def generate_brown(n=2, func_opts={}, data_type=cs.MX):
    raise NotImplementedError()


def generate_bukinn6():
    raise NotImplementedError()


def generate_cross_in_tray():
    raise NotImplementedError()


def generate_deckkers_aarts():
    raise NotImplementedError()


def generate_drop_wave():
    raise NotImplementedError()


def generate_egg_crate():
    raise NotImplementedError()


def generate_exponential():
    raise NotImplementedError()


def generate_goldstein_price():
    raise NotImplementedError()


def generate_gramacy_lee():
    raise NotImplementedError()


def generate_griewank():
    raise NotImplementedError()


def generate_happy_cat():
    raise NotImplementedError()


def generate_himmelblau(n=2, a=11., b=7., func_opts={}, data_type=cs.MX):
    if n != 2:
        raise ValueError("himmelblau is only defined for n=2")
    x = data_type.sym("x", n)
    f = (x[0]**2 + x[1] - a)**2 + (x[0] + x[1]**2 - b)**2
    func = cs.Function("himmelblau", [x], [f], ["x"], ["f"], func_opts)
    loc_min = [[3.,2],
               [-2.805118, 3.283186],
               [-3.779310, -3.283186],
               [3.584458, -1.848126]]
    return func, [[-6.,6.]]*n, loc_min


def generate_holder_table():
    raise NotImplementedError()


def generate_keane():
    raise NotImplementedError()


def generate_leon():
    raise NotImplementedError()


def generate_levin13():
    raise NotImplementedError()


def generate_matyas():
    raise NotImplementedError()


def generate_mccormick():
    raise NotImplementedError()


def generate_periodic():
    raise NotImplementedError()


def generate_powell_sum():
    raise NotImplementedError()


def generate_qing():
    raise NotImplementedError()


def generate_quartic():
    raise NotImplementedError()


def generate_ridge():
    raise NotImplementedError()


def generate_rosenbrock(n=2, a=1., b=100., func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    f = 0.
    for i in range(n-1):
        f += (a - x[i])**2 + b*(x[i+1]-x[i]**2)**2
    func = cs.Function("rosenbrock", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-5., 10.]]*n, [[1.]*n]


def generate_salomon(n=2, a=1., b=2*cs.np.pi, c=0.1,
                     func_opts={}, data_type=cs.MX):
    x = data_type.sym("x", n)
    sumsq = cs.sumsqr(x)
    f = a - cs.cos(b*cs.sqrt(sumsq))+c*cs.sqrt(sumsq)
    func = cs.Function("salomon", [x], [f], ["x"], ["f"], func_opts)
    return func, [[-100., 100.]]*n, [[0.]*n]

def generate_schaffern1():
    raise NotImplementedError()


def generate_schaffern2():
    raise NotImplementedError()


def generate_schaffern3():
    raise NotImplementedError()


def generate_schaffern4():
    raise NotImplementedError()


def generate_schwefel220():
    raise NotImplementedError()


def generate_schwefel221():
    raise NotImplementedError()


def generate_schwefel222():
    raise NotImplementedError()


def generate_schwefel223():
    raise NotImplementedError()


def generate_schwefel():
    raise NotImplementedError()


def generate_schubert3():
    raise NotImplementedError()


def generate_schubert4():
    raise NotImplementedError()


def generate_schubert():
    raise NotImplementedError()


def generate_sphere():
    raise NotImplementedError()


def generate_styblinskitank():
    raise NotImplementedError()


def generate_sumsquares():
    raise NotImplementedError()


def generate_threehumpcamel():
    raise NotImplementedError()


def generate_wolfe():
    raise NotImplementedError()


def generate_xinsheyangn1():
    raise NotImplementedError()


def generate_xinsheyangn2():
    raise NotImplementedError()


def generate_xinsheyangn3():
    raise NotImplementedError()


def generate_xinsheyangn4():
    raise NotImplementedError()


def generate_zakharov():
    raise NotImplementedError()
