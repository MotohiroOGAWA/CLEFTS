from __future__ import annotations

import math

CE_RANGE_LABELS = (
    "non-finite",
    "min-to-q1",
    "q1-to-median",
    "median-to-q3",
    "q3-to-max",
)


def tensorboard_label(value: str) -> str:
    """Sanitize a group label so it is safe to use as a TensorBoard tag segment."""
    return value.strip().replace("/", "∕").replace(" ", "_")


def ce_range_label(value: float, *, q1: float, median: float, q3: float) -> str:
    """Bin a collision-energy value into one of ``CE_RANGE_LABELS``.

    ``q1``/``median``/``q3`` are quartile cuts computed over whatever
    collection of collision-energy values the caller is grouping (a training
    batch or a validation split); the same value can therefore fall into a
    different bucket depending on the surrounding population.
    """
    if not math.isfinite(value):
        return "non-finite"
    bin_index = sum(value > cut for cut in (q1, median, q3))
    return CE_RANGE_LABELS[bin_index + 1]


def tensorboard_metric_card(namespace: str, metric: str, scope: str = '') -> tuple[str, str]:
    """Return the processing-stage card and optional condition series label."""
    # Older absolute-score diagnostics encode conditions as path prefixes.
    if metric.startswith(('by_adduct/', 'by_ce_range/')):
        kind, label, metric = metric.split('/', 2)
        scope = f'{kind}:{label}'
    rollout = ''
    if metric.startswith(('rollout_step_', 'rollout_final/')):
        rollout, metric = metric.split('/', 1)
    stage, _, remainder = metric.partition('/')
    depth_groups = {
        'edge_selection': 'edge_selection',
        'expanded_edge_selection': 'expanded_edge_selection',
        'next_cleavage': 'next_cleavage',
        'fragment_selection': 'fragment_selection',
        'path_coverage': 'path_coverage',
    }
    if stage in depth_groups:
        card_scope = scope or 'overall'
        prefix = 'validation_scope/' if namespace.startswith('validation_scope/') else ''
        prefix += f'{rollout}/' if rollout else ''
        return f'{depth_groups[stage]}/{prefix}{remainder}/{card_scope}', ''
    leaf = metric.rsplit('/', 1)[-1]
    if 'loss' in leaf:
        group = 'loss'
    elif leaf == 'pairwise_ranking_accuracy' or 'absolute_score' in metric:
        group = 'edge_absolute_score'
    elif metric.startswith('precursor/') or leaf.startswith('precursor_detection_'):
        group = 'precursor_selection'
    elif 'cosine' in leaf or 'intensity' in leaf:
        group = 'intensity'
    elif namespace == 'peak_selection' or leaf.startswith(('peak_selection_', 'top')):
        group = 'peak_selection'
    else:
        group = 'edge_selection'
    condition = ''
    if scope.startswith('by_adduct:'):
        card_scope, condition = 'by_adduct', scope.split(':', 1)[1]
    elif scope.startswith('by_ce_range:'):
        card_scope, condition = 'by_ce_range', scope.split(':', 1)[1]
    else:
        card_scope = scope or 'overall'
    prefix = 'validation_scope/' if namespace.startswith('validation_scope/') else ''
    return f'{group}/{prefix}{metric}/{card_scope}', condition
