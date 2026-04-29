"""Shared training dataset options."""

from __future__ import annotations

import argparse
from typing import cast

from src.data.freihand import VARIANTS, VariantName


DEFAULT_TRAIN_VARIANTS = VARIANTS
DEFAULT_VAL_VARIANTS: tuple[VariantName, ...] = ("gs",)


def parse_variant_selection(value: str | tuple[VariantName, ...]) -> tuple[VariantName, ...]:
    """Parse a CLI variant selection.

    Accepts `all`, a single FreiHAND variant, or a comma-separated list such as
    `gs,hom`.
    """
    if isinstance(value, tuple):
        return value

    value = value.strip()
    if value == "all":
        return VARIANTS

    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    if not variants:
        raise argparse.ArgumentTypeError("variant selection cannot be empty.")

    invalid = [variant for variant in variants if variant not in VARIANTS]
    if invalid:
        valid_values = ", ".join((*VARIANTS, "all"))
        raise argparse.ArgumentTypeError(
            f"invalid FreiHAND variant(s): {', '.join(invalid)}. "
            f"Valid values: {valid_values}.",
        )

    duplicates = sorted({variant for variant in variants if variants.count(variant) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            f"duplicate FreiHAND variant(s): {', '.join(duplicates)}.",
        )

    return cast(tuple[VariantName, ...], variants)


def add_variant_args(parser: argparse.ArgumentParser) -> None:
    valid_values = ", ".join((*VARIANTS, "all"))
    parser.add_argument(
        "--train-variants",
        type=parse_variant_selection,
        default=DEFAULT_TRAIN_VARIANTS,
        help=(
            "FreiHAND image variants to use for training. Use 'all' for every "
            f"provided variant, or choose from: {valid_values}. "
            "Comma-separated lists such as 'gs,hom' are also supported. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--val-variants",
        type=parse_variant_selection,
        default=DEFAULT_VAL_VARIANTS,
        help=(
            "FreiHAND image variants to use for validation. Defaults to 'gs' "
            "so validation remains comparable across runs."
        ),
    )


def variant_names(variants: tuple[VariantName, ...]) -> list[str]:
    return list(variants)


__all__ = [
    "DEFAULT_TRAIN_VARIANTS",
    "DEFAULT_VAL_VARIANTS",
    "add_variant_args",
    "parse_variant_selection",
    "variant_names",
]
