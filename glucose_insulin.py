import numpy as np
from scipy.integrate import odeint

import deepxde as dde
from deepxde.backend import tf


def IG(t, meal_t, meal_q, k=1 / 120):
    dt = t - meal_t
    return np.sum(
        meal_q * k * np.exp(-k * dt) * np.heaviside(dt, 0.5), axis=1, keepdims=True
    )


def glucose_insulin_model(
    t,
    meal_t,
    meal_q,
    Vp=3,
    Vi=11,
    Vg=10,
    E=0.2,
    tp=6,
    ti=100,
    td=12,
    k=1 / 120,
    Rm=209,
    a1=6.6,
    C1=300,
    C2=144,
    C3=100,
    C4=80,
    C5=26,
    Ub=72,
    U0=4,
    Um=90,
    Rg=180,
    alpha=7.5,
    beta=1.772,
):
    def func(y, t):
        f1 = Rm / (1 + np.exp(-y[2] / Vg / C1 + a1))
        f2 = Ub * (1 - np.exp(-y[2] / Vg / C2))
        kappa = (1 / Vi + 1 / E / ti) / C4
        f3 = (U0 + Um / (1 + (kappa * y[1]) ** (-beta))) / Vg / C3
        f4 = Rg / (1 + np.exp(alpha * (y[5] / Vp / C5 - 1)))
        IG = np.sum(
            meal_q * k * np.exp(k * (meal_t - t)) * np.heaviside(t - meal_t, 0.5)
        )
        tmp = E * (y[0] / Vp - y[1] / Vi)
        return [
            f1 - tmp - y[0] / tp,
            tmp - y[1] / ti,
            f4 + IG - f2 - f3 * y[2],
            (y[0] - y[3]) / td,
            (y[3] - y[4]) / td,
            (y[4] - y[5]) / td,
        ]

    Vp0, Vi0, Vg0 = 3, 11, 10
    y0 = [12 * Vp0, 4 * Vi0, 110 * Vg0 ** 2, 0, 0, 0]
    return odeint(func, y0, t)


def get_variable(v):
    low, up = v * 0.2, v * 1.8
    l = (up - low) / 2
    return l * tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + l + low


def pinn(data_t, data_y, meal_t=None, meal_q=None):
    if meal_t is None:
        # meal_t = tf.convert_to_tensor([300, 650, 1100], dtype=tf.float32)
        t1 = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 10 + 290
        t2 = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 10 + 640
        t3 = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 10 + 1090
        meal_t = tf.convert_to_tensor([t1, t2, t3])

        meal_q = (tf.tanh(tf.Variable([0, 0, 0], trainable=True, dtype=tf.float32)) + 1) / 2 * 900 + 100

    Vp = 3
    # Vp = tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 3
    Vi = 11
    # Vi = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 4 + 7
    Vg = 10
    # Vg = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 3 + 7
    # E = 0.2
    E = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 0.1 + 0.1
    # tp = 6
    tp = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 2 + 4
    # ti = 100
    ti = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 40 + 60
    # td = 12
    td = (tf.tanh(tf.Variable(0, trainable=True, dtype=tf.float32)) + 1) * 25 / 6 + 25 / 3
    # k = 1 / 120
    k = get_variable(0.0083)
    # Rm = 209 / 100  # scaled
    Rm = get_variable(2.09)
    # a1 = 6.6
    a1 = get_variable(6.6)
    # C1 = 300 / 100  # scaled
    C1 = get_variable(3)
    # C2 = 144 / 100  # scaled
    C2 = get_variable(1.44)
    C3 = 100 / 100  # scaled
    # C4 = 80 / 100  # scaled
    C4 = get_variable(0.8)
    # C5 = 26 / 100  # scaled
    C5 = get_variable(0.26)
    # Ub = 72 / 100  # scaled
    Ub = get_variable(0.72)
    # U0 = 4 / 100  # scaled
    U0 = get_variable(0.04)
    # Um = 90 / 100  # scaled
    Um = get_variable(0.9)
    # Rg = 180 / 100  # scaled
    Rg = get_variable(1.8)
    # alpha = 7.5
    alpha = get_variable(7.5)
    # beta = 1.772
    beta = get_variable(1.772)

    var_list = [Vp, Vi, Vg, E, tp, ti, td, k, Rm, a1, C1, C2, C3, C4, C5, Ub, U0, Um, Rg, alpha, beta]

    def ODE(t, y):
        Ip = y[:, 0:1]
        Ii = y[:, 1:2]
        G = y[:, 2:3]
        h1 = y[:, 3:4]
        h2 = y[:, 4:5]
        h3 = y[:, 5:6]

        f1 = Rm * tf.math.sigmoid(G / (Vg * C1) - a1)
        f2 = Ub * (1 - tf.math.exp(-G / (Vg * C2)))
        kappa = (1 / Vi + 1 / (E * ti)) / C4
        f3 = (U0 + Um / (1 + tf.pow(tf.maximum(kappa * Ii, 1e-3), -beta))) / (Vg * C3)
        f4 = Rg * tf.sigmoid(alpha * (1 - h3 / (Vp * C5)))
        dt = t - meal_t
        IG = tf.math.reduce_sum(
            0.5 * meal_q * k * tf.math.exp(-k * dt) * (tf.math.sign(dt) + 1),
            axis=1,
            keepdims=True,
        )
        tmp = E * (Ip / Vp - Ii / Vi)
        return [
            tf.gradients(Ip, t)[0] - (f1 - tmp - Ip / tp),
            tf.gradients(Ii, t)[0] - (tmp - Ii / ti),
            tf.gradients(G, t)[0] - (f4 + IG - f2 - f3 * G),
            tf.gradients(h1, t)[0] - (Ip - h1) / td,
            tf.gradients(h2, t)[0] - (h1 - h2) / td,
            tf.gradients(h3, t)[0] - (h2 - h3) / td,
        ]

    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Observes
    n = len(data_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=n // 5, replace=False), [0, n - 1]
    )
    ptset = dde.bc.PointSet(data_t[idx])
    inside = lambda x, _: ptset.inside(x)
    observe_y2 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 2:3]), inside, component=2
    )
    np.savetxt("glucose_input.dat", np.hstack((data_t[idx], data_y[idx, 2:3])))

    data = dde.data.PDE(geom, ODE, [observe_y2], anchors=data_t)

    net = dde.maps.FNN([1] + [128] * 3 + [6], "swish", "Glorot normal")

    def feature_transform(t):
        t = 0.01 * t
        return tf.concat(
            (t, tf.sin(t), tf.sin(2 * t), tf.sin(3 * t), tf.sin(4 * t), tf.sin(5 * t)),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        idx = 1799
        k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
        b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
            data_t[idx] - data_t[0]
        )
        linear = k * t + b
        factor = tf.math.tanh(t) * tf.math.tanh(idx - t)
        return linear + factor * tf.constant([1, 1, 1e2, 1, 1, 1]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    checkpointer = dde.callbacks.ModelCheckpoint(
        "./model/model.ckpt", verbose=1, save_better_only=True, period=1000
    )
    variable = dde.callbacks.VariableValue(
        [v for v in var_list if isinstance(v, tf.Tensor)],
        period=1000,
        filename="variables.dat",
    )
    callbacks = [checkpointer, variable]
    if isinstance(meal_t, tf.Tensor):
        variable_meal = dde.callbacks.VariableValue(
            [meal_t[0], meal_t[1], meal_t[2], meal_q[0], meal_q[1], meal_q[2]],
            period=1000,
            filename="variables_meal.dat",
            precision=3,
        )
        callbacks.append(variable_meal)

    model.compile("adam", lr=1e-3, loss_weights=[0, 0, 0, 0, 0, 0, 1e-2])
    model.train(epochs=2000, display_every=1000)
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1e-2, 1, 1, 1, 1e-2])
    losshistory, train_state = model.train(
        epochs=600000 if not isinstance(meal_t, tf.Tensor) else 1500000,
        display_every=1000,
        callbacks=callbacks,
        # model_restore_path="./model/model.ckpt-"
    )
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    var_list = [model.sess.run(v) if isinstance(v, tf.Tensor) else v for v in var_list]
    if not isinstance(meal_t, tf.Tensor):
        return var_list
    return var_list, model.sess.run(meal_t), model.sess.run(meal_q)


def main():
    meal_t = np.array([300, 650, 1100, 2000])
    meal_q = np.array([60e3, 40e3, 50e3, 100e3])
    t = np.arange(0, 3000, 1)[:, None]
    hidden_nutrition = False

    # Data
    y = glucose_insulin_model(np.ravel(t), meal_t, meal_q)
    np.savetxt("glucose.dat", np.hstack((t, y, IG(t, meal_t, meal_q))))

    # Train
    if not hidden_nutrition:
        var_list = pinn(t[:1800], y[:1800] / 100, meal_t, meal_q / 100)
    else:
        var_list, meal_t, meal_q = pinn(t[:1800], y[:1800] / 100)
        meal_t = np.append(meal_t, 2000)
        meal_q = np.append(100 * meal_q, 100e3)

    # Prediction
    for i in [8, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        var_list[i] = var_list[i] * 100
    y = glucose_insulin_model(np.ravel(t), meal_t, meal_q, *var_list)
    np.savetxt(
        "glucose_pred.dat", np.hstack((t, y, IG(t, meal_t, meal_q, k=var_list[7])))
    )


if __name__ == "__main__":
    main()
