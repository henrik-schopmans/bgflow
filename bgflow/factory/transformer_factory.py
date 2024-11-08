"""Factory for flow transformations."""

import torch
from ..nn.flow.inverted import InverseFlow
from ..nn.flow.transformer.affine import AffineTransformer
from ..nn.flow.transformer.spline import ConditionalSplineTransformer

__all__ = ["make_transformer"]


def make_transformer(
    transformer_type, what, shape_info, conditioners, inverse=False, **kwargs
):
    """Factory function.

    Parameters
    ----------
    transformer_type : bgflow.
    """
    factory = TRANSFORMER_FACTORIES[transformer_type]

    if "spline_disable_identity_transform" in kwargs:
        spline_disable_identity_transform = kwargs["spline_disable_identity_transform"]
        del kwargs["spline_disable_identity_transform"]
    else:
        spline_disable_identity_transform = False

    transformer = factory(
        what=what, shape_info=shape_info, conditioners=conditioners, **kwargs
    )

    if spline_disable_identity_transform:
        transformer._default_settings["enable_identity_init"] = False

    if inverse:
        transformer = InverseFlow(transformer)
    return transformer


def _make_spline_transformer(what, shape_info, conditioners, **kwargs):
    return ConditionalSplineTransformer(
        is_circular=shape_info.is_circular(what), **conditioners, **kwargs
    )


def _make_affine_transformer(what, shape_info, conditioners, **kwargs):
    if shape_info.dim_circular(what) not in [0, shape_info[what[0]][-1]]:
        raise NotImplementedError(
            "Circular affine transformers are currently "
            "not supported for partly circular indices."
        )
    return AffineTransformer(
        **conditioners, is_circular=shape_info.dim_circular(what) > 0, **kwargs
    )


TRANSFORMER_FACTORIES = {
    ConditionalSplineTransformer: _make_spline_transformer,
    AffineTransformer: _make_affine_transformer,
}
