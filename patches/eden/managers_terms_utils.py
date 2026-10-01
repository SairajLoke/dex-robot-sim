"""Shared helpers for MDP terms.

Missing from this repo's vendored Eden snapshot (the top-level README describes Eden as
"stripped to the modules this project imports", which evidently dropped this one even though
``shared_terms.py`` still imports it). Reconstructed from its single call site
(``shared_terms.dofs_pos_limits_penalty``), which documents it as "Penalty for joint positions
within ``margin`` (rad) of their mechanical limits" using a ``band`` of ``[low + margin, high -
margin]``, and sums its result over the last dim as a per-env scalar.
"""

import torch


def soft_dof_pos_violation(dofs_pos: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Per-dof violation of a ``[low, high]`` soft position band.

    Parameters
    ----------
    dofs_pos : torch.Tensor, shape (..., n_dofs)
        Current joint positions.
    band : torch.Tensor, shape (n_dofs, 2)
        Per-dof ``[low, high]`` soft limits (already margined off the hard mechanical limits).

    Returns
    -------
    torch.Tensor, shape (..., n_dofs)
        ``0`` inside the band; the (always non-negative) distance past whichever side of the
        band ``dofs_pos`` violates, otherwise.
    """
    low, high = band[..., 0], band[..., 1]
    return (low - dofs_pos).clamp(min=0.0) + (dofs_pos - high).clamp(min=0.0)
