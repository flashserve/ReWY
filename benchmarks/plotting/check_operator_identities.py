"""CPU-only editorial checks of operator identities; no accelerator benchmark."""
from pathlib import Path
from paper_paths import check_output
import json
import numpy as np

rng = np.random.default_rng(20261025)
errors = {"gated_factorization": 0.0, "feature_adjoint": 0.0,
          "scaling_invariance": 0.0, "rectangular_reductions": 0.0}

def check(name, actual, expected):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    errors[name] = max(errors[name], error)
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11), (name, error)

for n, d in [(1, 3), (4, 7), (7, 2)]:
    for masked in [False, True]:
        C = rng.normal(size=(n, n))
        if masked:
            C = np.tril(C, -1)
        G = rng.normal(scale=0.5, size=(n, d))
        X, Z = rng.normal(size=(2, n, d))
        gp, gm = np.exp(G), np.exp(-G)
        pair = np.einsum("ij,jk,ijk->ik", C, X,
                         np.exp(G[:, None, :] - G[None, :, :]))
        factored = gp * (C @ (gm * X))
        check("gated_factorization", factored, pair)
        adjoint = gm * (C.T @ (gp * Z))
        check("feature_adjoint", np.sum(pair * Z), np.sum(X * adjoint))

        Q, K, Be = rng.normal(size=(3, n, d))
        H, D = rng.normal(size=(2, n, n))
        H, D = np.tril(H), np.tril(D, -1)
        center = rng.normal(size=d)
        alpha, zeta = np.exp(G-center), np.exp(center-G)
        def outputs(q, e, y, p, inv):
            uq, ue, uy = H@y, D@y, H.T@q+D.T@e
            return (p*uq, p*Be*ue+inv*uy, p*K*ue,
                    q*uq+e*ue-y*uy)
        original = outputs(Q*gp, Be*K*gp, K*gm, gp, gm)
        centered = outputs(Q*alpha, Be*K*alpha, K*zeta, alpha, zeta)
        for a, b in zip(original, centered):
            check("scaling_invariance", a, b)

for m, n, d in [(1, 3, 2), (4, 7, 3), (5, 2, 8), (6, 6, 4)]:
    X = rng.normal(size=(m, d))
    Y = rng.normal(size=(n, d))
    C = rng.normal(size=(m, n))
    J = C * (X @ Y.T)
    check("rectangular_reductions", J.sum(axis=1), (X*(C@Y)).sum(axis=1))
    check("rectangular_reductions", J.sum(axis=0), (Y*(C.T@X)).sum(axis=1))

result = {"purpose": "editorial equation verification, not performance evidence",
          "status": "PASS", "max_absolute_errors": errors}
path = check_output("operator_identity_check.json")
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
