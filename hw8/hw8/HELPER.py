import numpy as np
import sys
import matplotlib.pyplot as plt


def normal(size, mean, std):
    return np.random.normal(loc=mean, scale=std, size=size)


def rrf(*, head_dim, proj_dim):
    return normal(size=(proj_dim, head_dim), mean=0., std=1)


def orf(*, head_dim, proj_dim):
    assert proj_dim >= head_dim
    g = normal(size=(proj_dim, head_dim), mean=0., std=1.)
    q, _ = np.linalg.qr(g)
    s = np.sqrt(np.random.chisquare(df=proj_dim, size=proj_dim))
    for i in range(q.shape[0]):
        q[i, :] *= s[i]
    return q


def normalize(x):
    x_norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / x_norm


def random_proj(q, k, random_matrix):
    q = np.einsum("bd,hd->bh", q, random_matrix)
    k = np.einsum("bd,hd->bh", k, random_matrix)
    q_sin, q_cos = np.sin(q), np.cos(q)
    k_sin, k_cos = np.sin(k), np.cos(k)
    scale = q_sin.shape[-1] ** -0.5
    phi_q = np.concatenate([q_sin, q_cos], axis=-1) * scale
    phi_k = np.concatenate([k_sin, k_cos], axis=-1) * scale
    return phi_q, phi_k


def test(head_dim, proj_dim, random_matrix):
    num_ites = 100
    tau, b, err = 0.7, 100, 0.
    random_matrix /= tau
    scale = 1. / tau ** 2
    scale = np.exp(scale)
    for i in range(num_ites):
        q = 3 * (np.random.normal(loc=0., scale=1., size=(b, head_dim)) - 0.5)
        k = 10 * (np.random.normal(loc=0., scale=1., size=(b, head_dim)) - 0.7)
        q, k = normalize(q), normalize(k)
        gold = (1. / tau ** 2) * np.einsum("bd,bd->b", q, k)
        gold = np.exp(gold)

        phi_q, phi_k = random_proj(q, k, random_matrix)
        pred = np.einsum("bd,bd->b", phi_q, phi_k) * scale
        e = np.linalg.norm(gold - pred, ord=2, axis=-1) / np.linalg.norm(gold, ord=2, axis=-1)
        err += e
    # print(err / num_ites)
    return err / num_ites


def construct_random_matrices(feature_func, num_random_matrices, head_dim, proj_dim):
    random_matrices = []
    for _ in range(num_random_matrices):
        random_matrix = feature_func(head_dim=head_dim, proj_dim=proj_dim)
        random_matrices.append(random_matrix)
    random_matrices = np.stack(random_matrices, axis=0)
    return random_matrices

Ds = [i+1 for i in range(20)] # Fill in the dimension of the features #
Ws = [i+1 for i in range(20)] # Fill in the dimensions of the random matrix #
feature_func = rrf

for d in Ds:
    errors = []
    for w in Ws:
        random_matrix = feature_func(head_dim=d, proj_dim=w)
        error = test( d, w, random_matrix )
        errors.append(error)
    plt.plot(Ws, errors, label=f"Dim_of_feature={d}")

plt.ylabel("Error")
plt.xlabel('Dimension of random matrix')
plt.legend()
plt.show()