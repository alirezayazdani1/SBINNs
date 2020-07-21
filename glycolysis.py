import numpy as np
from scipy.integrate import odeint

import deepxde as dde
from deepxde.backend import tf


def glycolysis_model(
    t,
    J0=2.5,
    k1=100,
    k2=6,
    k3=16,
    k4=100,
    k5=1.28,
    k6=12,
    k=1.8,
    kappa=13,
    q=4,
    K1=0.52,
    psi=0.1,
    N=1,
    A=4,
):
    def func(x, t):
        v1 = k1 * x[0] * x[5] / (1 + (x[5] / K1) ** q)
        v2 = k2 * x[1] * (N - x[4])
        v3 = k3 * x[2] * (A - x[5])
        v4 = k4 * x[3] * x[4]
        v5 = k5 * x[5]
        v6 = k6 * x[1] * x[4]
        v7 = k * x[6]
        J = kappa * (x[3] - x[6])
        return [
            J0 - v1,
            2 * v1 - v2 - v6,
            v2 - v3,
            v3 - v4 - J,
            v2 - v4 - v6,
            -2 * v1 + 2 * v3 - v5,
            psi * J - v7,
        ]

    x0 = [
        0.50144272,
        1.95478666,
        0.19788759,
        0.14769148,
        0.16059078,
        0.16127341,
        0.06404702,
    ]
    return odeint(func, x0, t)


def pinn(data_t, data_y, noise):
    J0 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    k1 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 100
    k2 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    k3 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 10
    k4 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 100
    k5 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    k6 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 10
    k = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    kappa = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 10
    q = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    K1 = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    psi = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1
    N = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))
    A = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32))

    var_list = [J0, k1, k2, k3, k4, k5, k6, k, kappa, q, K1, psi, N, A]

    def ODE(t, y):
        v1 = k1 * y[:, 0:1] * y[:, 5:6] / (1 + tf.maximum(y[:, 5:6] / K1, 1e-3) ** q)
        v2 = k2 * y[:, 1:2] * (N - y[:, 4:5])
        v3 = k3 * y[:, 2:3] * (A - y[:, 5:6])
        v4 = k4 * y[:, 3:4] * y[:, 4:5]
        v5 = k5 * y[:, 5:6]
        v6 = k6 * y[:, 1:2] * y[:, 4:5]
        v7 = k * y[:, 6:7]
        J = kappa * (y[:, 3:4] - y[:, 6:7])
        return [
            tf.gradients(y[:, 0:1], t)[0] - (J0 - v1),
            tf.gradients(y[:, 1:2], t)[0] - (2 * v1 - v2 - v6),
            tf.gradients(y[:, 2:3], t)[0] - (v2 - v3),
            tf.gradients(y[:, 3:4], t)[0] - (v3 - v4 - J),
            tf.gradients(y[:, 4:5], t)[0] - (v2 - v4 - v6),
            tf.gradients(y[:, 5:6], t)[0] - (-2 * v1 + 2 * v3 - v5),
            tf.gradients(y[:, 6:7], t)[0] - (psi * J - v7),
        ]

    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Right point
    def boundary(x, _):
        return np.isclose(x[0], data_t[-1, 0])

    y1 = data_y[-1]
    bc0 = dde.DirichletBC(geom, lambda X: y1[0], boundary, component=0)
    bc1 = dde.DirichletBC(geom, lambda X: y1[1], boundary, component=1)
    bc2 = dde.DirichletBC(geom, lambda X: y1[2], boundary, component=2)
    bc3 = dde.DirichletBC(geom, lambda X: y1[3], boundary, component=3)
    bc4 = dde.DirichletBC(geom, lambda X: y1[4], boundary, component=4)
    bc5 = dde.DirichletBC(geom, lambda X: y1[5], boundary, component=5)
    bc6 = dde.DirichletBC(geom, lambda X: y1[6], boundary, component=6)

    # Observes
    n = len(data_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=n // 4, replace=False), [0, n - 1]
    )
    ptset = dde.bc.PointSet(data_t[idx])
    inside = lambda x, _: ptset.inside(x)
    observe_y4 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 4:5]), inside, component=4
    )
    observe_y5 = dde.DirichletBC(
        geom, ptset.values_to_func(data_y[idx, 5:6]), inside, component=5
    )
    np.savetxt("glycolysis_input.dat", np.hstack((data_t[idx], data_y[idx, 4:5], data_y[idx, 5:6])))

    data = dde.data.PDE(
        geom,
        ODE,
        [bc0, bc1, bc2, bc3, bc4, bc5, bc6, observe_y4, observe_y5],
        anchors=data_t,
    )

    net = dde.maps.FNN([1] + [128] * 3 + [7], "swish", "Glorot normal")

    def feature_transform(t):
        return tf.concat(
            (
                t,
                tf.sin(t),
                tf.sin(2 * t),
                tf.sin(3 * t),
                tf.sin(4 * t),
                tf.sin(5 * t),
                tf.sin(6 * t),
            ),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        return (
            data_y[0] + tf.math.tanh(t) * tf.constant([1, 1, 0.1, 0.1, 0.1, 1, 0.1]) * y
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

    bc_weights = [1, 1, 10, 10, 10, 1, 10]
    if noise >= 0.1:
        bc_weights = [w * 10 for w in bc_weights]
    data_weights = [1e3, 1]
    # Large noise requires small data_weights
    if noise >= 0.1:
        data_weights = [w / 10 for w in data_weights]
    model.compile("adam", lr=1e-3, loss_weights=[0] * 7 + bc_weights + data_weights)
    model.train(epochs=1000, display_every=1000)
    ode_weights = [1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-3, 1]
    # Large noise requires large ode_weights
    if noise > 0:
        ode_weights = [10 * w for w in ode_weights]
    model.compile("adam", lr=1e-3, loss_weights=ode_weights + bc_weights + data_weights)
    losshistory, train_state = model.train(
        epochs=900000 if noise == 0 else 2000000,
        display_every=1000,
        callbacks=callbacks,
        disregard_previous_best=True,
        # model_restore_path="./model/model.ckpt-"
    )
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    var_list = [model.sess.run(v) for v in var_list]
    return var_list


def main():
    t = np.arange(0, 10, 0.005)[:, None]
    noise = 0.1

    # Data
    y = glycolysis_model(np.ravel(t))
    np.savetxt("glycolysis.dat", np.hstack((t, y)))
    # Add noise
    if noise > 0:
        std = noise * y.std(0)
        y[1:-1, :] += np.random.normal(0, std, (y.shape[0] - 2, y.shape[1]))
        np.savetxt("glycolysis_noise.dat", np.hstack((t, y)))

    # Train
    var_list = pinn(t, y, noise)

    # Prediction
    y = glycolysis_model(np.ravel(t), *var_list)
    np.savetxt("glycolysis_pred.dat", np.hstack((t, y)))


if __name__ == "__main__":
    main()
