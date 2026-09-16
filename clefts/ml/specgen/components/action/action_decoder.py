"""Fixed-pool, tensor-only normalized action-set decoder."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .action_state_encoder import ActionStateEncoder
from .action_compatibility import ActionCompatibilityEngine, ActionExpansion


@dataclass(frozen=True)
class ActionPool:
    action_index: Tensor
    valid: Tensor
    embeddings: Tensor
    absolute_logits: Tensor
    conflict: Tensor
    precedence: Tensor
    dominance: Tensor
    retained: Tensor
    source_atom_valid: Tensor
    positive_recall: Tensor


def positive_mask(ptr: Tensor, index: Tensor, rows: int, columns: int) -> Tensor:
    owner = torch.repeat_interleave(torch.arange(rows, device=ptr.device), ptr[1:] - ptr[:-1])
    result = torch.zeros((rows, columns), device=ptr.device, dtype=torch.bool)
    result[owner, index] = True
    return result


def build_action_pool(action_h: Tensor, absolute: Tensor, data: object, *, top_k: int,
                      max_k: int, threshold: float, training: bool) -> ActionPool:
    s, a = absolute.shape
    device = absolute.device
    eligible = data.action_tree_index[None, :] == data.sample_tree_index[:, None]
    filtered = absolute.masked_fill(~eligible | (absolute < threshold), -torch.inf)
    ids = torch.argsort(filtered,dim=1,descending=True,stable=True)[:,:min(top_k,a)]
    values = filtered.gather(1,ids)
    inference = torch.zeros_like(eligible)
    inference.scatter_(1, ids, torch.isfinite(values))
    positive = positive_mask(data.sample_positive_action_ptr, data.sample_positive_action_index, s, a)
    recall = (inference & positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    selection = inference | positive if training else inference
    if training and torch.any(selection.sum(dim=1) > max_k):
        raise ValueError("Training pool union exceeds 128; split samples or reduce first-stage top-k")
    width = max_k if training else top_k
    priority = absolute.masked_fill(~selection, -torch.inf)
    chosen_ids = torch.argsort(priority,dim=1,descending=True,stable=True)[:,:min(width,a)]
    chosen_values = priority.gather(1,chosen_ids)
    valid = torch.isfinite(chosen_values)
    missing = width - chosen_ids.shape[1]
    chosen_ids = F.pad(chosen_ids, (0, missing), value=a)
    valid = F.pad(valid, (0, missing), value=False)
    chosen_ids = chosen_ids.masked_fill(~valid, a)
    padded_h = torch.cat((action_h, action_h.new_zeros((1, action_h.shape[1]))))
    padded_absolute = torch.cat((absolute, absolute.new_full((s, 1), -torch.inf)), dim=1)
    inverse = torch.full((s, a + 1), -1, dtype=torch.long, device=device)
    slots = torch.arange(width, device=device).expand(s, -1)
    inverse.scatter_(1, chosen_ids, slots)
    inverse[:, a] = -1
    sample = torch.arange(s, device=device)[:, None]

    def dense(pairs: Tensor) -> Tensor:
        matrix = torch.zeros((s, width, width), dtype=torch.bool, device=device)
        left, right = inverse[:, pairs[0]], inverse[:, pairs[1]]
        okay = (left >= 0) & (right >= 0)
        rows = sample.expand_as(left)[okay]
        matrix[rows, left[okay], right[okay]] = True
        return matrix

    # Packed Source-local retained regions avoid [expansions, atoms] memory.
    capacity=data.source_atom_capacity or data.source_graph.num_nodes
    words=(capacity+63)//64
    starts=data.source_graph.ptr[data.sample_tree_index]
    sizes=data.source_graph.ptr[data.sample_tree_index+1]-starts
    word_bits=(sizes[:,None]-64*torch.arange(words,device=device)[None,:]).clamp(0,64)
    atom_valid=(torch.ones_like(word_bits)<<word_bits.clamp_max(63))-1
    atom_valid=atom_valid.masked_fill(word_bits==64,-1)
    retained=torch.zeros((s,width,words),dtype=torch.long,device=device)
    pair=data.action_retained_index
    local=inverse[:,pair[0]]
    atom=pair[1][None,:]-starts[:,None]
    okay=local>=0
    rows=sample.expand_as(local)[okay]
    slots=local[okay]
    atom=atom[okay]
    bits=torch.ones_like(atom) << (atom%64)
    retained.index_put_((rows,slots,atom//64),bits,accumulate=True)
    return ActionPool(chosen_ids.masked_fill(~valid, -1), valid, padded_h[chosen_ids],
                      padded_absolute.gather(1, chosen_ids), dense(data.action_conflict_index),
                      dense(data.action_precedence_index), dense(data.action_dominance_index),
                      retained, atom_valid, recall)


def deduplicate(sample: Tensor, state: Tensor, score: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    keys = torch.cat((sample[:, None], state), dim=1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    best = score.new_full((unique.shape[0],), -torch.inf)
    best.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
    row = torch.arange(score.numel(), device=score.device)
    representative = torch.full((unique.shape[0],), score.numel(), dtype=torch.long, device=score.device)
    representative.scatter_reduce_(0, inverse, row.masked_fill(score != best[inverse], score.numel()), reduce="amin", include_self=True)
    return unique[:, 0], unique[:, 1:], best, representative


def grouped_topk(values: Tensor, groups: Tensor, k: int, num_groups: int) -> Tensor:
    order = torch.argsort(values, descending=True, stable=True)
    order = order[torch.argsort(groups[order], stable=True)]
    count = torch.bincount(groups, minlength=num_groups)
    start = count.cumsum(0) - count
    rank = torch.arange(order.numel(), device=order.device) - start[groups[order]]
    return order[rank < k]


@dataclass(frozen=True)
class ActionDecoderOutput:
    state_sample_index: Tensor
    state_action_index: Tensor
    state_log_score: Tensor
    terminal: Tensor
    parent_state_index: Tensor
    added_action_index: Tensor
    pool: ActionPool


class ActionSequenceDecoder(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, max_action_count: int,
                 beam_size: int = 32, num_heads: int = 4, max_decode_steps: int = 16) -> None:
        super().__init__()
        if beam_size < 1 or max_decode_steps < 1:
            raise ValueError("beam_size and max_decode_steps must be positive")
        self.state_encoder = ActionStateEncoder(hidden_dim, num_heads=num_heads)
        self.compatibility = ActionCompatibilityEngine(max_action_count)
        self.query = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.eos_head = nn.Linear(hidden_dim + condition_dim, 1)
        self.scale = hidden_dim ** -0.5
        self.max_action_count = max_action_count
        self.beam_size = beam_size
        self.max_decode_steps = max_decode_steps

    def score_states(self, pool: ActionPool, conditions: Tensor, state: Tensor,
                     sample: Tensor) -> tuple[Tensor, ActionExpansion]:
        h = self.state_encoder(pool.embeddings, state, sample)
        context = torch.cat((h, conditions[sample]), dim=1)
        action = pool.absolute_logits[sample] + torch.einsum("ph,pkh->pk", self.query(context), self.key(pool.embeddings[sample])) * self.scale
        expansion = self.compatibility.expand(state_action_index=state, state_sample_index=sample,
            pool_valid=pool.valid, pool_conflict=pool.conflict, pool_precedence=pool.precedence,
            pool_dominance=pool.dominance, pool_retained=pool.retained, source_atom_valid=pool.source_atom_valid)
        valid = expansion.valid.reshape(state.shape[0], -1) & ((state >= 0).sum(dim=1) < self.max_action_count)[:, None]
        logits = torch.cat((action.masked_fill(~valid, -torch.inf), self.eos_head(context)), dim=1)
        return logits, expansion

    def forward(self, pool: ActionPool, conditions: Tensor) -> ActionDecoderOutput:
        s, k = pool.valid.shape
        device = conditions.device
        state = torch.full((s, self.max_action_count), -1, dtype=torch.long, device=device)
        sample = torch.arange(s, device=device)
        score = conditions.new_zeros(s)
        global_parent = torch.full((s,), -1, dtype=torch.long, device=device)
        added = torch.full_like(global_parent, -1)
        all_sample, all_state, all_score, all_parent, all_added, all_terminal = [], [], [], [], [], []
        offset = 0
        raw_history_score = []
        # Tensor batches per generation iteration, never loops over states/actions.
        for step in range(self.max_decode_steps + 1):
            if sample.numel() == 0:
                break
            row = torch.arange(sample.numel(), device=device) + offset
            raw_history_score.append(score)
            logits, expansion = self.score_states(pool, conditions, state, sample)
            probabilities = F.log_softmax(logits, dim=1)
            all_sample.append(sample)
            all_state.append(state)
            all_score.append(score + probabilities[:, -1])
            all_parent.append(global_parent)
            all_added.append(added)
            all_terminal.append(torch.ones_like(sample, dtype=torch.bool))
            offset += sample.numel()
            if step == self.max_decode_steps:
                break
            parent = expansion.parent_state_index
            candidate = expansion.candidate_action_index
            candidate_score = score[parent] + probabilities[parent, candidate]
            valid = expansion.valid & torch.isfinite(candidate_score)
            parent, candidate, candidate_score = parent[valid], candidate[valid], candidate_score[valid]
            child_sample, child, child_score, representative = deduplicate(sample[parent], expansion.child_action_index[valid], candidate_score)
            old_keys = torch.cat((torch.cat(all_sample)[:, None], torch.cat(all_state)), dim=1)
            new_keys = torch.cat((child_sample[:, None], child), dim=1)
            unique, inverse = torch.unique(torch.cat((old_keys, new_keys)), dim=0, return_inverse=True)
            old_best = child_score.new_full((unique.shape[0],), -torch.inf)
            old_best.scatter_reduce_(0, inverse[:old_keys.shape[0]], torch.cat(raw_history_score), reduce="amax", include_self=True)
            improved = child_score > old_best[inverse[old_keys.shape[0]:]]
            child_sample, child, child_score, representative = child_sample[improved], child[improved], child_score[improved], representative[improved]
            parents = row[parent[representative]]
            actions = candidate[representative]
            selected = grouped_topk(child_score, child_sample, self.beam_size, s)
            sample, state, score = child_sample[selected], child[selected], child_score[selected]
            global_parent, added = parents[selected], actions[selected]
        samples, states, scores = torch.cat(all_sample), torch.cat(all_state), torch.cat(all_score)
        terminal_sample, terminal_state, terminal_score, representative = deduplicate(samples, states, scores)
        chosen = grouped_topk(terminal_score, terminal_sample, self.beam_size, s)
        terminal = torch.zeros_like(samples, dtype=torch.bool)
        terminal[representative[chosen]] = True
        # All generated states form a bounded superset of terminal ancestor
        # closure; materialization computes the actual closure outside Torch.
        return ActionDecoderOutput(samples, states, scores, terminal, torch.cat(all_parent), torch.cat(all_added), pool)
