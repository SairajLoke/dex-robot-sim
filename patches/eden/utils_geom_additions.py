# -- Missing from this repo's vendored Eden snapshot (see managers_terms_utils.py in this same
# -- directory for the same situation). shared_terms.py imports these four; reconstructed here
# -- using the exact same (w, x, y, z) quaternion convention already established in this file and
# -- ported directly from the vendored Genesis's own (numpy-only)
# -- genesis.utils.geom.quat_to_rotvec / genesis.utils.geom.inv_quat /
# -- genesis.utils.geom._tc_quat_mul, which this file already imports other functions from.
#
# Apply by (1) adding `inv_quat` to the existing
# `from genesis.utils.geom import (...)` block at the top of eden/utils/geom.py, and
# (2) appending the three functions below to the end of that file.
#
# Sanity-checked standalone against known rotations (see
# docs/tactile-genesis-verification.md): quat_mul(id, id) == id, axis_angle_from_quat(90deg
# about z) == [0, 0, pi/2], quat_error_magnitude(45deg, 90deg about z) == pi/4, and
# quat_mul(q, inv_quat(q)) == id.


def quat_mul(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batches of (w, x, y, z) quaternions. Ported from
    genesis.utils.geom._tc_quat_mul (same formula, public name)."""
    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out = torch.empty(qq.shape + (4,), dtype=qq.dtype, device=qq.device)
    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)
    out /= torch.linalg.vector_norm(out, ord=2, dim=-1, keepdim=True)
    return out


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Angle-axis vector (angle * unit axis) of a batch of (w, x, y, z) quaternions. Ported
    from genesis.utils.geom.quat_to_rotvec's numpy formula to torch."""
    q_w, q_vec = quat[..., :1], quat[..., 1:]
    s2 = torch.linalg.vector_norm(q_vec, ord=2, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s2, torch.abs(q_w))
    inv_sinc = angle / torch.clamp(s2, min=eps)
    sign = torch.where(q_w < 0.0, -1.0, 1.0)
    return sign * inv_sinc * q_vec


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation-angle magnitude (radians) between two batches of (w, x, y, z) quaternions."""
    quat_diff = quat_mul(q1, inv_quat(q2))
    return torch.linalg.vector_norm(axis_angle_from_quat(quat_diff), ord=2, dim=-1)
