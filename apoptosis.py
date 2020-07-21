import numpy as np
from scipy.integrate import odeint

import deepxde as dde
from deepxde.backend import tf


def apoptosis_model(
    t, x0, k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6,
):
    def func(x, t):
        v4_1 = kd1 * x[4]
        v4_2 = kd2 * x[4]
        v5_3 = kd3 * x[5]
        v5_4 = kd4 * x[5]
        v7_5 = kd5 * x[7]
        v7_6 = kd6 * x[7]
        v03 = k1 * x[3] * x[0]
        v12 = k3 * x[1] * x[2]
        v36 = k5 * x[6] * x[3]

        return [
            -v03 + v4_1,
            v4_2 - v12 + v5_3 + v5_4,
            -v12 + v5_3,
            v5_4 - v03 + v4_1 - v36 + v7_5 + v4_2,
            -v4_2 + v03 - v4_1,
            -v5_4 + v12 - v5_3,
            -v36 + v7_5 + v7_6,
            v36 - v7_5 - v7_6,
        ]

    return odeint(func, x0, t)


def pinn(data_t, data_y):
    k1 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32))
    kd1 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 10
    kd2 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 10
    k3 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 10
    kd3 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 100
    kd4 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32))
    k5 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 1e4
    kd5 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 0.01
    kd6 = tf.math.softplus(tf.Variable(1, trainable=True, dtype=tf.float32)) * 0.1

    var_list = [k1, kd1, kd2, k3, kd3, kd4, k5, kd5, kd6]

    def ODE(t, y):
        v4_1 = kd1 * y[:, 4:5]
        v4_2 = kd2 * y[:, 4:5]
        v5_3 = kd3 * y[:, 5:6]
        v5_4 = kd4 * y[:, 5:6]
        v7_5 = kd5 * y[:, 7:8]
        v7_6 = kd6 * y[:, 7:8]
        v03 = k1 * y[:, 3:4] * y[:, 0:1]
        v12 = k3 * y[:, 1:2] * y[:, 2:3]
        v36 = k5 * y[:, 6:7] * y[:, 3:4]

        return [
            tf.gradients(y[:, 0:1], t)[0] - (-v03 + v4_1),
            tf.gradients(y[:, 1:2], t)[0] - (v4_2 - v12 + v5_3 + v5_4),
            tf.gradients(y[:, 2:3], t)[0] - (-v12 + v5_3),
            tf.gradients(y[:, 3:4], t)[0] - (v5_4 - v03 + v4_1 - v36 + v7_5 + v4_2),
            tf.gradients(y[:, 4:5], t)[0] - (-v4_2 + v03 - v4_1),
            tf.gradients(y[:, 5:6], t)[0] - (-v5_4 + v12 - v5_3),
            tf.gradients(y[:, 6:7], t)[0] - (-v36 + v7_5 + v7_6),
            tf.gradients(y[:, 7:8], t)[0] - (v36 - v7_5 - v7_6),
        ]

    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Right point
    def boundary(x, _):
        return np.isclose(x[0], data_t[len(data_t) // 2, 0])

    y1 = data_y[len(data_t) // 2]
    bc0 = dde.DirichletBC(geom, lambda X: y1[0], boundary, component=0)
    bc1 = dde.DirichletBC(geom, lambda X: y1[1], boundary, component=1)
    bc2 = dde.DirichletBC(geom, lambda X: y1[2], boundary, component=2)
    bc3 = dde.DirichletBC(geom, lambda X: y1[3], boundary, component=3)
    bc4 = dde.DirichletBC(geom, lambda X: y1[4], boundary, component=4)
    bc5 = dde.DirichletBC(geom, lambda X: y1[5], boundary, component=5)
    bc6 = dde.DirichletBC(geom, lambda X: y1[6], boundary, component=6)
    bc7 = dde.DirichletBC(geom, lambda X: y1[7], boundary, component=7)

    # Observes
    n = len(data_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=n // 5, replace=False), [0, n - 1]
    )
    ptset = dde.bc.PointSet(data_t[idx])
    inside = lambda x, _: ptset.inside(x)
    observe_y3 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 3:4]), inside, component=3
    )
    np.savetxt("apoptosis_input.dat", np.hstack((data_t[idx], data_y[idx, 3:4])))

    data = dde.data.PDE(
        geom, ODE, [bc0, bc1, bc2, bc3, bc4, bc5, bc6, bc7, observe_y3], anchors=data_t,
    )

    net = dde.maps.FNN([1] + [256] * 4 + [8], "swish", "Glorot normal")

    def feature_transform(t):
        t = 0.1 * t
        return tf.concat((t, tf.exp(-t)), axis=1,)

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        return (
            data_y[0]
            + tf.math.tanh(t) * tf.constant([1, 1, 1, 1, 0.01, 0.1, 0.01, 0.01]) * y
        )

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    checkpointer = dde.callbacks.ModelCheckpoint(
        "./model/model.ckpt", verbose=1, save_better_only=True, period=1000
    )
    variable = dde.callbacks.VariableValue(
        var_list, period=1000, filename="variables.dat", precision=3,
    )
    callbacks = [checkpointer, variable]

    model.compile(
        "adam",
        lr=1e-3,
        # loss_weights=[1, 1, 1, 1, 1e3, 1, 1, 1] + [1, 1, 1, 1, 100, 10, 100, 100, 1e2],  # noiseless
        loss_weights=[1, 1, 1, 1, 1e3, 1, 1, 1] + [1, 1, 1, 1, 100, 10, 100, 100, 10],  # death noise
        # loss_weights=[1, 1, 1, 1, 1e3, 1, 1, 1] + [1, 1, 1, 1, 100, 10, 100, 100, 1],  # survival noise
    )
    losshistory, train_state = model.train(
        # epochs=700000,  # death noiseless
        epochs=1500000,  # death noise
        # epochs=1500000,  # survival
        display_every=1000,
        callbacks=callbacks,
    )
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    var_list = [model.sess.run(v) for v in var_list]
    return var_list


def main():
    t = np.arange(0, 60, 0.1)[:, None]
    t_scale = 3600
    c_scale = 1e5
    noise = 0.05

    # Data
    x0 = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e3, 0]  # death
    # x0 = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e4, 0]  # survival
    x0 = [x / c_scale for x in x0]
    var_list = [2.67e-9, 1e-2, 8e-3, 6.8e-8, 5e-2, 1e-3, 7e-5, 1.67e-5, 1.67e-4]
    var_list = [v * t_scale for v in var_list]
    for i in [0, 3, 6]:
        var_list[i] *= c_scale
    y = apoptosis_model(np.ravel(t), x0, *var_list)
    np.savetxt("apoptosis.dat", np.hstack((t, y)))
    # Add noise
    if noise > 0:
        std = noise * y.std(0)
        tmp = np.copy(y[len(t) // 2])
        y[1:, :] += np.random.normal(0, std, (y.shape[0] - 1, y.shape[1]))
        y[len(t) // 2] = tmp
        np.savetxt("apoptosis_noise.dat", np.hstack((t, y)))

    # Train
    var_list = pinn(t, y)

    # Prediction
    y = apoptosis_model(np.ravel(t), x0, *var_list)
    np.savetxt("apoptosis_pred.dat", np.hstack((t, y)))
    var_list = [v / t_scale for v in var_list]
    for i in [0, 3, 6]:
        var_list[i] /= c_scale
    print(dde.utils.list_to_str(var_list))


if __name__ == "__main__":
    main()
