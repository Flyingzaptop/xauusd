"""PCGrad in a small Gram matrix; no approximation to the vector projection."""
import numpy as np


def coefficients(gram, orders):
    gram = np.asarray(gram, dtype=np.float64)
    m = len(gram)
    c = np.eye(m)
    for i in range(m):
        for j in orders[i]:
            if i == j or gram[j, j] <= 0:
                continue
            dot = c[i] @ gram[:, j]
            if dot < 0:
                c[i, j] -= dot / gram[j, j]
    return c


def diagnostics(gram, c, weights):
    m = len(weights)
    w = np.asarray(weights)
    a = w @ c
    diag = np.maximum(np.diag(gram), 0)
    denominator = np.sqrt(diag[:, None] * diag[None, :])
    cos = gram / np.maximum(denominator, 1e-30)
    pairs = cos[np.triu_indices(m, 1)]
    plain = gram @ w
    after = gram @ a
    tolerance = 1e-7 * np.maximum(diag, 1e-20)
    normal = np.sqrt(max(w @ gram @ w, 0))
    correction = np.sqrt(max((a-w) @ gram @ (a-w), 0))
    return dict(conflict_fraction=float(np.mean(pairs < -1e-6)) if len(pairs) else 0.,
                cancellation_ratio=float(normal/max(w @ np.sqrt(diag), 1e-30)),
                plain_harm_fraction=float(np.mean(plain < -tolerance)),
                projected_harm_fraction=float(np.mean(after < -tolerance)),
                correction_ratio=float(correction/max(normal, 1e-30)))


def fixtures():
    rng = np.random.default_rng(13)
    maximum_error = 0.
    for g in [np.array([[1., 0.], [-1., 0.]]), np.array([[1., 2.], [2., 4.]]),
              np.array([[1., 0.], [-1., 1.]]), np.zeros((2, 3))] + [rng.normal(size=(4, 19)) for _ in range(20)]:
        orders = [rng.permutation(len(g)) for _ in g]
        explicit = g.copy()
        for i in range(len(g)):
            for j in orders[i]:
                dot = explicit[i] @ g[j]
                if i != j and dot < 0 and g[j] @ g[j] > 0:
                    explicit[i] -= dot / (g[j] @ g[j]) * g[j]
        actual = coefficients(g @ g.T, orders) @ g
        maximum_error = max(maximum_error, float(abs(actual-explicit).max()))
        assert np.allclose(actual, explicit, atol=1e-12)
    opposing=np.array([[1.,0.],[-1.,0.]])
    c=coefficients(opposing@opposing.T, [[1],[0]])
    assert np.allclose(c@opposing,0), 'PCGrad must not invent a solution to exact opposition'
    return dict(explicit_reference_max_error=maximum_error, exactly_opposing_gradients_remain_unsolved=True)

if __name__ == '__main__':
    print(fixtures())
