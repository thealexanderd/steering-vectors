from __future__ import annotations


applied_steering_vectors = []

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, mul, nn
from torch.utils.hooks import RemovableHandle

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from .torch_utils import get_module, untuple_tensor
from .steering_vector import SteeringPatchHandle

@dataclass
class SteeringVectorField:
    """
    A local, softmax based steering vector field that adapts to input activations.
    """

    layer_activations: dict[int, list[tuple[Tensor, Tensor]]]
    layer_type: LayerType = "decoder_block"

    def to(self, *args: Any, **kwargs: Any) -> SteeringVectorField:
        moved: dict[int, list[tuple[Tensor, Tensor]]] = {}
        for layer, pairs in self.layer_activations.items():
            moved[layer] = [(p.to(*args, **kwargs), n.to(*args, **kwargs))
                            for p, n in pairs]
        return replace(self, layer_activations=moved)

    def patch_activations(
        self,
        model: nn.Module,
        layer_config: ModelLayerConfig | None = None,
        *,
        n_neighbors: int = 5,
        multiplier: float = 1.0,
        min_token_index: int | None = None,
        token_indices: list[int] | slice | Tensor | None = None,
        temperature: float = 1.0,
    ) -> SteeringPatchHandle:
        assert (min_token_index is None) or (token_indices is None), (
            "Pass either `min_token_index` or `token_indices`, not both"
        )
        if isinstance(token_indices, Tensor):
            assert torch.all((token_indices == 0) | (token_indices == 1)), (
                "`token_indices` tensor must be a 0/1 mask"
            )

        token_indices = token_indices if token_indices is not None else slice(min_token_index, None)

        layer_config = guess_and_enhance_layer_config(model, layer_config, self.layer_type)
        if self.layer_type not in layer_config:
            raise ValueError(f"layer_type {self.layer_type!r} not found in layer_config")

        matcher = layer_config[self.layer_type]
        matching_layers = collect_matching_layers(model, matcher)

        hooks: list[RemovableHandle] = []
        for layer_idx, pairs in self.layer_activations.items():
            if layer_idx >= len(matching_layers):
                raise IndexError(f"layer {layer_idx} not found (only {len(matching_layers)} matched)")
            module_name = matching_layers[layer_idx]
            module = get_module(model, module_name)

            handle = module.register_forward_hook(
                _create_field_hook(
                    pairs,
                    token_indices=token_indices,
                    n_neighbors=n_neighbors,
                    multiplier=multiplier,
                    temperature=temperature,
                )
            )
            hooks.append(handle)

        return SteeringPatchHandle(hooks)

    @contextmanager
    def apply(
        self,
        model: nn.Module,
        layer_config: ModelLayerConfig | None = None,
        *,
        n_neighbors: int = 5,
        multiplier: float = 1.0,
        min_token_index: int = 0,
        token_indices: list[int] | slice | Tensor | None = None,
        temperature: float = 1.0,
    ) -> Generator[None, None, None]:
        handle = self.patch_activations(
            model,
            layer_config,
            n_neighbors=n_neighbors,
            multiplier=multiplier,
            min_token_index=min_token_index,
            token_indices=token_indices,
            temperature=temperature,
        )
        try:
            yield
        finally:
            handle.remove()


def _create_field_hook(
    pairs: list[tuple[Tensor, Tensor]],
    *,
    token_indices: list[int] | slice | Tensor,
    n_neighbors: int,
    multiplier: float,
    temperature: float,
) -> Any:
    neg_stack = torch.stack([n for n, _ in pairs])
    diff_stack = torch.stack([delta for _, delta in pairs])
    k = min(n_neighbors, len(pairs))

    def hook_fn(_module: Any, _inputs: Any, outputs: Any) -> Any:
        original = untuple_tensor(outputs)
        device = original.device
        neg, diff = neg_stack.to(device), diff_stack.to(device)

        print(original.shape)
        # print(original)

        query = original.mean(dim=1)  

        print(query.shape)
        # print(query)

        dists = torch.cdist(query, neg, p=2)  # L2 distances: [batch, pairs]# 
        weights = torch.softmax(-dists / temperature, dim=-1)  # [batch, pairs]
        steering = torch.einsum("bp,pd->bd", weights, diff)  # [batch, hidden_dim]

        steering = steering * multiplier  # Scale the steering vector

        # Store the steering vector for inspection
        global applied_steering_vectors
        applied_steering_vectors.append(steering.detach().cpu())
        # steering = steering * multiplier
        steering = steering.unsqueeze(1)  # [batch, 1, hidden_dim]

        if isinstance(token_indices, Tensor):
            mask = token_indices.to(device)
        else:
            mask = torch.zeros(original.shape[1], device=device)
            mask[token_indices] = 1
        mask = mask.view(1, -1, 1)

        original[:] = original + mask * steering
        return outputs

    return hook_fn