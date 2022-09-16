""" Tools for calculating Gradient Descent for ||Ax-b||. """
import matplotlib.pyplot as plt
import numpy as np


def main():
    ################################################################################
    # TODO: Fill in and change as needed
    A = np.array([1, 1])
    b = np.array([6, 4.5])
    initial_position = np.array([0, 0])  # position at iteration 0
    total_step_count = 1300  # number of GD steps to take
    step_size = lambda i: 1/(i+1)  # function returning step size at iteration i
    ################################################################################

    # computes desired number of steps of gradient descent
    positions = compute_updates(A, b, initial_position, total_step_count, step_size)

    # print out the values of the x_i
    print(positions)
    print("AAAA")
    print(A)
    print(np.dot(A, b))

    # plot the values of the x_i
    plt.scatter(positions[:, 0], positions[:, 1], c='blue')
    plt.scatter(np.dot(A, b),
                np.dot(A, b), c='red')
    plt.plot()
    plt.show()


def compute_gradient(A, b, x):
    """Computes the gradient of ||Ax-b|| with respect to x."""
    # TODO: Fill in
    return (x-b)/np.linalg.norm(b-x, 2)


def compute_update(A, b, x, step_count, step_size):
    """Computes the new point after the update at x."""
    return x - step_size(step_count) * compute_gradient(A, b, x)


def compute_updates(A, b, p, total_step_count, step_size):
    """Computes several updates towards the minimum of ||Ax-b|| from p.

    Params:
        b: in the equation ||Ax-b||
        p: initialization point
        total_step_count: number of iterations to calculate
        step_size: function for determining the step size at step i
    """
    positions = [np.array(p)]
    for k in range(total_step_count):
        positions.append(compute_update(A, b, positions[-1], k, step_size))
    return np.array(positions)

main()
