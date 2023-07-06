import torch
from torch import Tensor
from torch_geometric.typing import OptTensor

from typing import Optional


def knn(x: torch.Tensor,
        y: torch.Tensor,
        k: int,
        batch_x: Optional[torch.Tensor] = None,
        batch_y: Optional[torch.Tensor] = None,
        *args,
        **kwargs):
    # pylint: disable=unused-argument, keyword-arg-before-vararg
    r"""Finds for each element in `y` the `k` nearest points in `x`.

    Args:
        x (torch.Tensor): Node feature matrix
        y (torch.Tensor): Node feature matrix
        k (int): The number of neighbors.
        batch_x (torch.Tensor, optional): Batch vector which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_y = torch.tensor([0, 0])
        >>> assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.int32)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.int32)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    # Rescale x and y.
    min_xy = torch.min(torch.min(x), torch.min(y))
    x, y = x - min_xy, y - min_xy

    max_xy = torch.max(torch.max(x), torch.max(y))
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([
        x, 2 * x.size(1) * batch_x.view(
            -1, 1).to(torch.int32 if x.dtype == torch.long else x.dtype)
    ],
                  dim=-1)
    y = torch.cat([
        y, 2 * y.size(1) * batch_y.view(
            -1, 1).to(torch.int32 if y.dtype == torch.long else y.dtype)
    ],
                  dim=-1)

    x_expanded = x.expand(y.size(0), *x.shape)
    y_expanded = y.reshape(y.size(0), 1, y.size(1))

    diff = x_expanded - y_expanded
    norm = torch.norm(diff, dim=-1)
    dist, col = torch.topk(norm,
                           k=k,
                           dim=-1,
                           largest=False,
                           sorted=True)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)

    distance_upper_bound = x.size(1)

    row = torch.where(dist > distance_upper_bound, -1, row).view(-1)
    col = torch.where(dist > distance_upper_bound, -1, col).view(-1)

    return torch.stack([row, col], dim=0)


def radius(
        x: Tensor,
        y: Tensor,
        r: float,
        batch_x=None,
        batch_y=None,
        max_num_neighbors: int = 32,
        *args,
        **kwargs,
) -> Tensor:
    # pylint: disable=unused-argument, keyword-arg-before-vararg
    r"""
    Copied from version 3.3 of the poptorch_geometric repo.
    https://github.com/graphcore/poptorch/blob/sdk-release-3.3/poptorch_geometric/python/ops/radius.py
    Computes graph edges to all points within a given distance.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor` with static shape, where not found neighbours
            are marked by -1
    """
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    if not torch.is_floating_point(x):
        x = x.float()

    if not torch.is_floating_point(y):
        y = y.float()

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)
    distance_upper_bound = r + 1e-8

    dist = torch.cdist(y, x)
    k = min(dist.size(-1), max_num_neighbors)
    dist, col = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    row = torch.cat([torch.full(col.shape[1:], i) for i in range(col.size(0))],
                    dim=0)
    col = torch.where(dist < distance_upper_bound, col, -1)
    col = torch.flatten(col)
    row = torch.where(col == -1, -1, row)

    return torch.stack([row, col], dim=0)


def knn_graph(x: torch.Tensor,
              k: int,
              batch: OptTensor = None,
              loop: bool = False,
              flow: str = 'source_to_target',
              cosine: bool = False,
              num_workers: int = 1) -> torch.Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    .. code-block:: python

        import torch
        from torch_geometric.nn import knn_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)

    :rtype: :class:`torch.Tensor`
    """
    assert flow in ['source_to_target', 'target_to_source']
    edge_index = knn(x, x, k if loop else k + 1, batch, batch, cosine,
                     num_workers)

    if not loop:
        edge_index = edge_index.reshape(2, -1, k + 1)[:, :, 1:].reshape(2, -1)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    return torch.stack([row, col], dim=0)
