"""Compare GPU-batched selection and gradients with the original scalar path."""
import unittest
from types import SimpleNamespace

import torch
from torch import Tensor
from typing import List

from clefts.ml.specgen.fragment_tree_candidate_selector import (
    FragmentIonCandidate, FragmentTreeCandidateSelector,
)

def reference_select(self, *, features, sample_tree_batch, keep_logit: Tensor, edge_cleave_logit: Tensor, ion_logit: Tensor, unsaturation_logit: Tensor, radical_logit: Tensor) -> List[FragmentIonCandidate]:
    structure = features.structure
    device = keep_logit.device
    kept_sample_ids = sample_tree_batch.kept_sample_ids.to(device).long()
    graph_index_by_node = sample_tree_batch.batch.to(device).long()
    candidates: List[FragmentIonCandidate] = []

    # The precursor-root node competes for keep-probability mass on the
    # same footing as every fragment node: it is frequently the dominant
    # peak in the observed spectrum and must remain eligible for
    # selection here, not just supervised during training.
    for graph_index in range(int(kept_sample_ids.numel())):
        sample_node_index = (graph_index_by_node == graph_index).nonzero(as_tuple=False).view(-1)
        if sample_node_index.numel() == 0:
            continue
        # All fragment-producing edges in one spectrum compete for a
        # finite probability mass.  The destination node keep score and
        # its incoming cleavage-edge score jointly define that mass.
        node_score = keep_logit[sample_node_index].clone()
        if sample_tree_batch.edge_index.numel() > 0:
            edge_dst = sample_tree_batch.edge_index[1].to(device).long()
            for local_index, batch_node_index in enumerate(sample_node_index):
                incoming = (edge_dst == batch_node_index).nonzero(as_tuple=False).view(-1)
                if incoming.numel() > 0:
                    node_score[local_index] = node_score[local_index] + torch.logsumexp(
                        edge_cleave_logit[incoming], dim=0
                    )
        node_probability = torch.softmax(node_score, dim=0)
        k = int(node_score.numel())
        if self.max_nodes_for_ion_candidates is not None:
            k = min(int(self.max_nodes_for_ion_candidates), k)
        node_order = sample_node_index[torch.topk(node_score, k=k).indices]
        sample_candidates: List[FragmentIonCandidate] = []
        for batch_node_index_tensor in node_order:
            batch_node_index = int(batch_node_index_tensor.detach().cpu().item())
            sample_candidates.extend(
                self._score_joint_ion_candidates_for_node(
                    structure=structure,
                    sample_tree_batch=sample_tree_batch,
                    batch_node_index=batch_node_index,
                    global_node_id=int(sample_tree_batch.node_id_global[batch_node_index].detach().cpu().item()),
                    sample_id=int(kept_sample_ids[graph_index].detach().cpu().item()),
                    keep_logit=keep_logit,
                    ion_logit=ion_logit,
                    unsaturation_logit=unsaturation_logit,
                    radical_logit=radical_logit,
                    edge_probability=node_probability[
                        (sample_node_index == batch_node_index).nonzero(as_tuple=False).view(-1)[0]
                    ],
                )
            )
        sample_candidates.sort(key=lambda item: item.score, reverse=True)
        candidates.extend(sample_candidates[: self.max_fragment_ion_candidates])

    return candidates


def fixture(device, node_limit=None, empty_states=False, no_edges=False):
    selector = FragmentTreeCandidateSelector.__new__(FragmentTreeCandidateSelector)
    torch.nn.Module.__init__(selector)
    mask = torch.tensor([[[1, 0, 1], [0, 1, 1]], [[1, 1, 0], [1, 0, 1]]],
                        dtype=torch.bool, device=device)
    if empty_states:
        mask[1, 1] = False
    selector.feature_model = SimpleNamespace(
        ion_candidate_valid_mask_by_role_adduct=mask,
        unsaturation_candidate_valid_mask_by_role_adduct=torch.ones(2, 2, 2, dtype=torch.bool, device=device),
        radical_candidate_valid_mask_by_role_adduct=torch.tensor(
            [[[1, 0], [1, 1]], [[1, 1], [0, 1]]], dtype=torch.bool, device=device),
        formula_tensorizer=SimpleNamespace(tensor_to_formula=lambda row: tuple(row.tolist())),
    )
    selector.max_fragment_ion_candidates = 4
    selector.max_nodes_for_ion_candidates = node_limit
    g = torch.Generator().manual_seed(901)
    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device).requires_grad_()
    batch = SimpleNamespace(
        kept_sample_ids=torch.tensor([3, 7, 9, 11], device=device),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 2], device=device),
        node_id_global=torch.tensor([6, 4, 2, 5, 1, 0, 3], device=device),
        node_main_adduct_type_index=torch.tensor([0, 0, 1, 1, 1, 0, 1], device=device),
        node_is_precursor_root=torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.bool, device=device),
        edge_index=torch.tensor([[0, 0, 1, 3, 3], [1, 2, 2, 4, 5]], device=device),
    )
    if no_edges:
        batch.edge_index = batch.edge_index[:, :0]
    structure = SimpleNamespace(
        node_formula=torch.arange(21, device=device).reshape(7, 3).float(),
        ion_formula_delta=torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], device=device),
        unsaturation_formula_delta=torch.tensor([[0, 0, 0], [0, -1, 0]], device=device),
        radical_formula_delta=torch.tensor([[0, 0, 0], [0, 0, -1]], device=device),
    )
    return selector, dict(features=SimpleNamespace(structure=structure), sample_tree_batch=batch,
        keep_logit=rand(7), edge_cleave_logit=rand(batch.edge_index.size(1)),
        ion_logit=rand(7, 3), unsaturation_logit=rand(7, 2), radical_logit=rand(7, 2))


class TestBatchedFragmentIonSelection(unittest.TestCase):
    def check_equivalence(self, device, block_elements=None):
        for limit, empty, no_edges in [(None, False, False), (1, False, False),
                                      (3, True, False), (None, False, True)]:
            with self.subTest(device=device, limit=limit, empty=empty, no_edges=no_edges):
                selector, args = fixture(device, limit, empty, no_edges)
                if block_elements is not None:
                    selector._ION_CANDIDATE_SCORE_BLOCK_ELEMENTS = block_elements
                expected = reference_select(selector, **args)
                actual = selector._select_fragment_ion_candidates(**args)
                self.assertEqual(len(actual), len(expected))
                for got, want in zip(actual, expected):
                    for field in ('sample_id', 'batch_node_index', 'global_node_id', 'ion_index',
                                  'unsaturation_index', 'radical_index', 'formula'):
                        self.assertEqual(getattr(got, field), getattr(want, field), field)
                    for field in ('score', 'keep_logit', 'candidate_logit', 'probability',
                                  'edge_probability', 'adduct_probability'):
                        self.assertAlmostEqual(getattr(got, field), getattr(want, field), places=5)
                    torch.testing.assert_close(got.formula_tensor, want.formula_tensor)
                    self.assertEqual(got.probability_tensor.device.type, device)
                tensors = [args[key] for key in ('keep_logit', 'edge_cleave_logit', 'ion_logit',
                                                'unsaturation_logit', 'radical_logit')]
                def loss(candidates):
                    return sum((i + 1) * (c.probability_tensor + 0.1 * c.score_tensor)
                               for i, c in enumerate(candidates))
                old_grad = torch.autograd.grad(loss(expected), tensors, retain_graph=True, allow_unused=True)
                new_grad = torch.autograd.grad(loss(actual), tensors, allow_unused=True)
                for old, new in zip(old_grad, new_grad):
                    if old is None:
                        self.assertTrue(new is None or torch.count_nonzero(new) == 0)
                    else:
                        torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-5)

    def test_cpu_matches_original_values_and_gradients(self):
        self.check_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_original_values_and_gradients(self):
        self.check_equivalence('cuda')

    def test_chunked_scores_preserve_values_and_gradients(self):
        self.check_equivalence('cpu', block_elements=1)

    def test_no_valid_states(self):
        selector, args = fixture('cpu')
        selector.feature_model.ion_candidate_valid_mask_by_role_adduct.zero_()
        self.assertEqual(selector._select_fragment_ion_candidates(**args), [])

    def test_empty_batch(self):
        selector, args = fixture('cpu')
        args['keep_logit'] = torch.empty(0)
        self.assertEqual(selector._select_fragment_ion_candidates(**args), [])

    def test_ties_and_large_logits_are_finite_and_deterministic(self):
        selector, args = fixture('cpu')
        for name in ('keep_logit', 'edge_cleave_logit', 'ion_logit', 'unsaturation_logit', 'radical_logit'):
            args[name] = torch.full_like(args[name], 10000., requires_grad=True)
        first = selector._select_fragment_ion_candidates(**args)
        second = selector._select_fragment_ion_candidates(**args)
        self.assertEqual([(c.sample_id, c.batch_node_index, c.ion_index) for c in first],
                         [(c.sample_id, c.batch_node_index, c.ion_index) for c in second])
        self.assertTrue(all(torch.isfinite(c.score_tensor) for c in first))
        sum(c.probability_tensor for c in first).backward()
        self.assertTrue(torch.isfinite(args['keep_logit'].grad).all())

    def test_invalid_adduct_raises(self):
        selector, args = fixture('cpu')
        args['sample_tree_batch'].node_main_adduct_type_index[0] = -1
        with self.assertRaises(IndexError):
            selector._select_fragment_ion_candidates(**args)
