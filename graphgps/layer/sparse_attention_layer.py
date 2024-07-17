import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pykeops.torch import LazyTensor

class SparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        kq_dim: int,
        val_dim: int,
        num_heads: int,
        k: int,
    ):
        """
        Args:
            dim: the dimension of the input
            kq_dim: the dimension of the key and query space
            val_dim: the dimension of the value space
            num_heads: the number of heads, so also the number of different graphs that are generated
            k: for each query, the number of nearest keys to consider
        """
        super().__init__()

        self.dim = dim
        self.kq_dim = kq_dim
        self.val_dim = val_dim
        self.num_heads = num_heads
        self.k = k

        self.MQs = nn.Linear(dim, kq_dim * num_heads)  # query transformations
        self.MKs = nn.Linear(dim, kq_dim * num_heads)  # key transformations
        self.MVs = nn.Linear(dim, val_dim * num_heads)  # value transformations
        self.MO = nn.Linear(val_dim * num_heads, dim)  # output transformations

    def reset_parameters(self):
        self.MQs.reset_parameters()
        self.MKs.reset_parameters()
        self.MVs.reset_parameters()
        self.MO.reset_parameters()

    def nearest_k_keys(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N, k] tensor.

        Args:
            queries: [**, num_heads, N, kq_dim]
            keys: [**, num_heads, N, kq_dim]

        Returns:
            [**, num_heads, N, k]
        """
        # --- Compute the nearest k keys: [**, num_heads, N, k] ---

        # [**, num_heads, N, 1, kq_dim]
        queries_extended: LazyTensor = LazyTensor(queries[..., :, :, None, :])
        #  [**, num_heads, 1, N, kq_dim]
        keys_extended: LazyTensor = LazyTensor(keys[..., :, None, :, :])
        # [**, num_heads, N, N]
        full_attention_weights: LazyTensor = (queries_extended * keys_extended).sum(
            -1
        ) / math.sqrt(self.kq_dim)

        # [**, num_heads, N, k]
        ndims = len(full_attention_weights.shape)
        assert ndims in [3,4]
        nearest_key_indices = (-full_attention_weights).argKmin(self.k, dim=ndims-2)

        return nearest_key_indices

    def sparse_self_attention(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [**, num_heads, N, kq_dim]
            keys: [**, num_heads, N, kq_dim]
            values: [**, num_heads, N, val_dim]

        Returns:
            [**, num_heads, N, val_dim]
        """

        trailing_dimensions = queries.shape[:-3]
        N = queries.shape[-2]
        assert queries.shape == (*trailing_dimensions, self.num_heads, N, self.kq_dim)
        assert keys.shape == (*trailing_dimensions, self.num_heads, N, self.kq_dim)
        assert values.shape == (*trailing_dimensions, self.num_heads, N, self.val_dim)
        # assert queries.is_contiguous()

        # --- Compute the attention weights: [**, num_heads, N, k] ---

        # [**, num_heads, N, k]
        nearest_key_indices = self.nearest_k_keys(queries, keys)
        assert nearest_key_indices.shape == (
            *trailing_dimensions,
            self.num_heads,
            N,
            self.k,
        )
        # assert nearest_key_indices.is_contiguous()

        # the k keys nearest to each query
        # [**, num_heads, N, k, kq_dim]
        nearest_keys = torch.gather(
            input=keys.unsqueeze(-2).expand(
                *keys.shape[:-1], self.k, self.kq_dim
            ),  # [**, num_heads, N, k, kq_dim]
            dim=-3,
            index=nearest_key_indices.unsqueeze(-1).expand(
                *nearest_key_indices.shape, self.kq_dim
            ),  # [**, num_heads, N, k, kq_dim]
            # sparse_grad=True,
        )
        assert nearest_keys.shape == (
            *trailing_dimensions,
            self.num_heads,
            N,
            self.k,
            self.kq_dim,
        )
        # assert nearest_keys.is_contiguous()

        # the values corresponding to those keys
        # [**, num_heads, N, k, val_dim]
        nearest_values = torch.gather(
            input=values.unsqueeze(-2).expand(
                *keys.shape[:-1], self.k, self.val_dim
            ),  # [**, num_heads, N, k, kq_dim]
            dim=-3,
            index=nearest_key_indices.unsqueeze(-1).expand(
                *nearest_key_indices.shape, self.val_dim
            ),  # [**, num_heads, N, k, kq_dim]
            # sparse_grad=True,
        )
        assert nearest_values.shape == (
            *trailing_dimensions,
            self.num_heads,
            N,
            self.k,
            self.val_dim,
        )
        # assert nearest_values.is_contiguous()

        # [**, num_heads, N, k, kq_dim]
        queries_extended = queries.unsqueeze(-2).expand(
            *queries.shape[:-1], self.k, self.kq_dim
        )
        assert queries_extended.shape == (
            *trailing_dimensions,
            self.num_heads,
            N,
            self.k,
            self.kq_dim,
        )
        # assert queries_extended.is_contiguous()

        # [**, num_heads, N, k]
        largest_attention_weights = (queries_extended * nearest_keys).sum(
            -1
        ) / math.sqrt(self.kq_dim)
        largest_attention_weights = F.softmax(largest_attention_weights, dim=-1)
        assert largest_attention_weights.shape == (
            *trailing_dimensions,
            self.num_heads,
            N,
            self.k,
        )
        # assert largest_attention_weights.is_contiguous()

        # Apply the attention weights to the values
        # [**, num_heads, N, val_dim]
        out = (largest_attention_weights.unsqueeze(-1) * nearest_values).sum(
            dim=-2
        )  # sum over k nearest keys for each query
        assert out.shape == (*trailing_dimensions, self.num_heads, N, self.val_dim)
        # assert out.is_contiguous()

        return out

    def split_heads(self, x: torch.Tensor, small_dim: int):
        """
        Rearrange the input from [**, N, num_heads*small_dim] to [**, num_heads, N, small_dim]

        The important cases are small_dim = kq_dim and small_dim = val_dim
        """
        assert x.shape[-1] == self.num_heads * small_dim
        assert x.ndim >= 2
        x = (
            x.reshape(*x.shape[:-1], self.num_heads, small_dim)
            .transpose(-3, -2)
            .contiguous()
        )
        assert x.is_contiguous()  # more efficient for pykeops
        return x

    def combine_heads(self, x: torch.Tensor, small_dim: int):
        """
        Rearrange the input from [**, num_heads, N, small_dim] to [**, N, num_heads * small_dim]
        """
        assert x.shape[-3] == self.num_heads
        assert x.shape[-1] == small_dim
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], self.num_heads * small_dim)
        assert x.is_contiguous()  # more efficient for pykeops
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [**, N, dim]

        Returns:
            [**, N, dim]

        TODO: implement masking to ignore padding nodes
        TODO: try attention dropout
        """

        # [**, num_heads, N, kq_dim]
        queries = self.split_heads(self.MQs(x), small_dim=self.kq_dim)
        keys = self.split_heads(self.MKs(x), small_dim=self.kq_dim)
        # [**, num_heads, N, val_dim]
        values = self.split_heads(self.MVs(x), small_dim=self.val_dim)

        # [**, num_heads, N, val_dim]
        x = self.sparse_self_attention(queries, keys, values)
        # [**, N, num_heads * val_dim]
        x = self.combine_heads(x, small_dim=self.val_dim)
        # [**, N, dim]
        x = self.MO(x)

        return x