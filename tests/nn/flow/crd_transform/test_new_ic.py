import torch
import numpy as np

from bgtorch.nn.flow.crd_transform.ic_helper import (
    outer,
    det2x2,
    det3x3,
    orientation,
    dist_deriv,
    angle_deriv,
    torsion_deriv,
)
from bgtorch.nn.flow.crd_transform.new_ic import (
    ic2xy0_deriv,
    ic2xyz_deriv,
)

# TODO Floating point precision is brittle!
#      Revision should include numerically more robust
#      implementations - especially for angular values.

# TODO missing test for global IC layer

torch.set_default_tensor_type = torch.DoubleTensor

N_REPETITIONS = 500


def rad2deg(x):
    return x * 180.0 / np.pi


def deg2rad(x):
    return x * np.pi / 180.0


def test_outer(atol=1e-6, rtol=1e-5):
    for _ in range(N_REPETITIONS):
        x, y = torch.Tensor(2, 5, 7, 3).normal_()
        A = outer(x, y).view(-1)
        B = []
        for i in range(5):
            for j in range(7):
                for k in range(3):
                    for l in range(3):
                        B.append(x[i, j, k] * y[i, j, l])
        B = torch.Tensor(B)
        assert torch.allclose(A, B, atol=atol, rtol=rtol)


def test_det2x2(atol=1e-6, rtol=1e-5):
    for _ in range(N_REPETITIONS):
        x = torch.Tensor(7, 5, 3, 2, 2).normal_()
        assert torch.allclose(det2x2(x), x.det(), atol=atol, rtol=rtol)


def test_det3x3(atol=1e-6, rtol=1e-5):
    for _ in range(N_REPETITIONS):
        x = torch.Tensor(7, 5, 3, 3, 3).normal_()
        if not torch.allclose(det3x3(x), x.det(), atol=atol, rtol=rtol):
            print(det3x3(x) - x.det())
        assert torch.allclose(det3x3(x), x.det(), atol=atol, rtol=rtol)


def test_dist_deriv(atol=1e-6, rtol=1e-5):
    x1 = torch.Tensor([0, 0, 0])
    x2 = torch.Tensor([1, 1, 0])
    d, J = dist_deriv(x1, x2)
    sqrt2 = torch.Tensor([2]).sqrt()
    assert torch.allclose(d, sqrt2, atol=atol, rtol=rtol)
    assert torch.allclose(J, -x2 / sqrt2)


# TODO Floating point precision!
def test_angle_deriv(atol=1e-4, rtol=1e-4):

    # check 45 deg angle
    x1 = torch.Tensor([0, 1, 0])
    x2 = torch.Tensor([0, 0, 0])
    x3 = torch.Tensor([1, 1, 0])
    a, J = angle_deriv(x1, x2, x3)
    assert torch.allclose(a, torch.tensor(deg2rad(45.0)))
    assert torch.allclose(J, torch.Tensor([-1, 0, 0]))

    # check random angle
    for i in range(N_REPETITIONS):

        # random reference angle
        a_ref = np.random.uniform(0, np.pi)
        x1 = torch.Tensor([np.cos(a_ref), np.sin(a_ref), 0]).requires_grad_(True)

        # construct system in standard basis
        x2 = torch.Tensor([0, 0, 0])
        x3 = torch.Tensor([1, 0, 0])

        # apply random rotation to system
        R = torch.Tensor(np.linalg.qr(np.random.uniform(size=(3, 3)))[0])
        x1, x2, x3 = (x @ R for x in (x1, x2, x3))

        a, J = angle_deriv(x1, x2, x3)

        # compute Jacobian with autograd
        J_ref = torch.autograd.grad(a.sum(), x1)[0]

        assert torch.allclose(a, torch.tensor(a_ref), atol=atol, rtol=rtol)
        assert torch.allclose(J, J_ref, atol=atol, rtol=rtol)


def test_torsion_deriv(atol=1e-6, rtol=1e-5):
    for i in range(N_REPETITIONS):

        # random reference angle
        a_ref = np.random.uniform(0, np.pi)

        # construct system in standard basis
        x1 = torch.Tensor([np.cos(a_ref), np.sin(a_ref), 1]).requires_grad_(True)
        x2 = torch.Tensor([0, 0, 1]).unsqueeze(0)
        x3 = torch.Tensor([0, 0, -1]).unsqueeze(0)
        x4 = torch.Tensor([1, 0, -1]).unsqueeze(0)

        # apply random rotation to system
        R = torch.Tensor(np.linalg.qr(np.random.uniform(size=(3, 3)))[0])
        x1, x2, x3, x4 = (x @ R for x in (x1, x2, x3, x4))

        # torsion angle should be invariant under rotation
        a, J = torsion_deriv(x1, x2, x3, x4)

        # compute Jacobian with autograd
        J_ref = torch.autograd.grad(a.sum(), x1)[0]

        assert torch.allclose(a, torch.tensor(a_ref), atol=atol, rtol=rtol)
        assert torch.allclose(J, J_ref, atol=atol, rtol=rtol)


# TODO Floating point precision!
def test_ic2xyz_deriv(atol=1e-5, rtol=1e-4):
    for i in range(N_REPETITIONS):

        d12 = torch.tensor(np.random.uniform(0.5, 1.5))
        d23 = torch.tensor(np.random.uniform(0.5, 1.5))
        a = torch.tensor(np.random.uniform(0, np.pi))

        # first point placed in origin
        x1 = torch.zeros(1, 1, 3)

        # second point placed in z-axis
        x2 = torch.zeros_like(x1)
        x2[..., 2] = d12

        # third point placed wrt to p0 and p1
        x3, _ = ic2xy0_deriv(x1, x2, d23, a)

        # fourth point placed randomly
        x4 = torch.Tensor(1, 1, 3).normal_()

        # compute ic
        d, Jd = dist_deriv(x4, x3)
        a, Ja = angle_deriv(x4, x3, x2)
        t, Jt = torsion_deriv(x4, x3, x2, x1)

        # reconstruct 4th point
        x4_new, J = ic2xyz_deriv(x3, x2, x1, d, a, t)

        assert torch.allclose(x4_new, x4, atol=atol, rtol=rtol)


def test_global_ic_transform():
    pass