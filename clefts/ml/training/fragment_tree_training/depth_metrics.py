"""Depth diagnostics against saved supervision; no fragmentation or teacher forcing."""
from collections import defaultdict, deque
import math


def _pairs(tensor):
    return {tuple(map(int, pair)) for pair in tensor.detach().cpu().t().tolist() if pair[1] >= 0}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def selection_metrics(available, selected, truth):
    selected = selected & available
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    return {
        'precision': _ratio(tp, tp + fp),
        'recall': _ratio(tp, tp + fn),
        'f1': _ratio(2 * tp, 2 * tp + fp + fn),
        'accuracy': _ratio(len(available) - len((selected ^ truth) & available), len(available)),
        'candidate_coverage': _ratio(len(available & truth), len(truth)),
        'conditional_recall': _ratio(tp, len(available & truth)),
        'true_positive_count': float(tp),
        'false_positive_count': float(fp),
        'false_negative_count': float(fn),
        'target_count': float(len(truth)),
        'candidate_count': float(len(available)),
        'selected_count': float(len(selected)),
    }


class DepthEvaluation:
    def __init__(self, target):
        self.edges = [tuple(map(int, edge)) for edge in target.edge_index.detach().cpu().t().tolist()]
        adjacency = defaultdict(list)
        destinations = set()
        for source, destination in self.edges:
            adjacency[source].append(destination)
            destinations.add(destination)
        self.node_depth = {node: 0 for node in range(int(target.num_nodes)) if node not in destinations}
        queue = deque(self.node_depth)
        while queue:
            source = queue.popleft()
            for destination in adjacency[source]:
                if destination not in self.node_depth:
                    self.node_depth[destination] = self.node_depth[source] + 1
                    queue.append(destination)
        self.edge_depth = {i: self.node_depth[source] + 1 for i, (source, _) in enumerate(self.edges) if source in self.node_depth}
        self.expandable = set(adjacency)
        self.outgoing = defaultdict(set)
        for edge, (source, _) in enumerate(self.edges):
            self.outgoing[source].add(edge)
        self.truth_edges = _pairs(target.target_edge_index)
        self.truth_fragments = set(zip(
            map(int, target.target_sample_index.detach().cpu().tolist()),
            map(int, target.target_terminal_node_index.detach().cpu().tolist()),
        ))
        self.truth_nodes = set()
        self.paths = []
        samples = target.target_sample_index.detach().cpu().tolist()
        nodes = target.target_expand_node_index.detach().cpu().tolist()
        ptr = target.terminal_expand_ptr.detach().cpu().tolist()
        paths = target.target_path_edge_index.detach().cpu().tolist()
        path_ptr = target.terminal_path_ptr.detach().cpu().tolist()
        for assignment, sample in enumerate(samples):
            self.truth_nodes.update((int(sample), int(node)) for node in nodes[ptr[assignment]:ptr[assignment + 1]] if self.node_depth.get(int(node), 0) > 0 and int(node) in self.expandable)
            path = {(int(sample), int(edge)) for edge in paths[path_ptr[assignment]:path_ptr[assignment + 1]] if edge >= 0}
            if path:
                self.paths.append(path)
        self.path_edges = set().union(*self.paths) if self.paths else set()
        self.groups = defaultdict(set)
        for (sample, edge), group in zip(target.target_edge_index.detach().cpu().t().tolist(), target.target_edge_group_index.detach().cpu().tolist()):
            if edge >= 0:
                self.groups[(int(sample), int(group))].add((int(sample), int(edge)))

    @staticmethod
    def edge_decisions(output):
        batch = output.sample_tree_batch
        sample_by_graph = batch.kept_sample_ids.detach().cpu().tolist()
        graph_by_node = batch.batch.detach().cpu().tolist()
        edge_sources = batch.edge_index[0].detach().cpu().tolist()
        edge_ids = batch.edge_id_global.detach().cpu().tolist()
        logits = output.edge_absolute_logit.detach().cpu().tolist()
        available, selected = set(), set()
        for source, edge, score in zip(edge_sources, edge_ids, logits):
            if edge < 0:
                continue
            pair = (int(sample_by_graph[graph_by_node[source]]), int(edge))
            available.add(pair)
            if math.isfinite(score) and score > 0:
                selected.add(pair)
        return available, selected

    def edge_metrics(self, available, selected, *, stage='edge_selection', eligible=None):
        result = {}
        truth = self.truth_edges
        if eligible is not None:
            available, selected, truth = available & eligible, selected & eligible, truth & eligible
        depths = sorted({self.edge_depth[e] for _, e in available | truth if e in self.edge_depth})
        for depth in depths:
            at_depth = lambda pairs: {p for p in pairs if self.edge_depth.get(p[1]) == depth}
            candidates, kept, targets = map(at_depth, (available, selected, truth))
            values = selection_metrics(candidates, kept, targets)
            values['retained_precision'] = _ratio(len(candidates & targets), len(candidates))
            groups = [at_depth(group) for group in self.groups.values()]
            groups = [group & truth for group in groups if group & targets]
            values['target_group_recall'] = _ratio(sum(bool(group & kept) for group in groups), len(groups))
            prefix = f'{stage}/depth_{depth}'
            result.update({f'{prefix}/{name}': value for name, value in values.items()})
        return result

    def evaluate(self, output, *, pending_only=False):
        available, selected = self.edge_decisions(output)
        result = self.edge_metrics(available, selected)
        batch = output.sample_tree_batch
        node_ids = batch.node_id_global.detach().cpu().tolist()
        graph_ids = batch.batch.detach().cpu().tolist()
        samples = batch.kept_sample_ids.detach().cpu().tolist()
        precursor = batch.node_is_precursor_root.detach().cpu().tolist()
        fragment_candidates = {(int(samples[g]), int(n)) for n, g, root in zip(node_ids, graph_ids, precursor)
                               if not root and self.node_depth.get(n, 0) > 0}
        fragments = {(int(c.sample_id), int(c.global_node_id)) for c in output.kept_candidates}
        for depth in sorted({self.node_depth[n] for _, n in fragment_candidates | self.truth_fragments if self.node_depth.get(n, 0) > 0}):
            at_depth = lambda pairs: {p for p in pairs if self.node_depth.get(p[1]) == depth}
            values = selection_metrics(*map(at_depth, (fragment_candidates, fragments, self.truth_fragments)))
            result.update({f'fragment_selection/depth_{depth}/{name}': value for name, value in values.items()})
        candidates = {(int(samples[g]), int(n)) for n, g, root in zip(node_ids, graph_ids, precursor) if not root and n in self.expandable and self.node_depth.get(n, 0) > 0}
        chosen = {(int(c.sample_id), int(c.global_node_id)) for c in output.next_cleavage_candidates}
        truth_nodes = self.truth_nodes
        if pending_only:
            seen = _pairs(output.features.structure.sample_edge_index)
            needed = self.path_edges - seen
            truth_nodes = {(sample, node) for sample, node in truth_nodes
                           if any(s == sample and self.edges[edge][0] == node for s, edge in needed)}
        for depth in sorted({self.node_depth[n] for _, n in candidates | truth_nodes}):
            at_depth = lambda pairs: {p for p in pairs if self.node_depth.get(p[1]) == depth}
            local, predicted, truth = map(at_depth, (candidates, chosen, truth_nodes))
            values = selection_metrics(local, predicted, truth)
            top = {}
            for candidate in output.next_cleavage_candidates:
                pair = (int(candidate.sample_id), int(candidate.global_node_id))
                if pair in local and (pair[0] not in top or candidate.score > top[pair[0]].score):
                    top[pair[0]] = candidate
            truth_samples = {s for s, _ in truth}
            values['hit_at_k'] = _ratio(sum(any(s == sample for s, _ in predicted & truth) for sample in truth_samples), len(truth_samples))
            values['top1_accuracy'] = _ratio(sum((s, int(top[s].global_node_id)) in truth for s in truth_samples if s in top), len(truth_samples))
            result.update({f'next_cleavage/depth_{depth}/{name}': value for name, value in values.items()})
        for depth in sorted(set(self.edge_depth.values())):
            paths = [path for path in self.paths if max(self.edge_depth.get(e, -1) for _, e in path) == depth]
            if paths:
                result[f'path_coverage/depth_{depth}/complete_path_recall'] = _ratio(sum(path <= available for path in paths), len(paths))
                needed = set().union(*paths)
                result[f'path_coverage/depth_{depth}/path_edge_coverage'] = _ratio(len(needed & available), len(needed))
        return result

    def after_expansion(self, before, after):
        # Evaluate every possible outgoing edge of the chosen parents, including
        # edges removed by depth budgets. Missing children remain false negatives.
        previous = _pairs(before.features.structure.sample_edge_index)
        parents = {(int(c.sample_id), int(c.global_node_id)) for c in before.next_cleavage_candidates}
        eligible = {(sample, edge) for sample, node in parents for edge in self.outgoing[node]} - previous
        available, selected = self.edge_decisions(after)
        return self.edge_metrics(available, selected, stage='expanded_edge_selection', eligible=eligible)
