"""Explicit post-decoding RDKit boundary; never called by training forward."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence
from rdkit import Chem
import torch
from torch import Tensor
from torch_geometric.data import Batch
from clefts.libs.mmkit.mmkit import Compound
from clefts.domain.fragment.cleavage import CleavageAction, CleavageActionSequence
from clefts.domain.fragment.tree.FragmentTree import FragmentTree
from clefts.domain.fragment.tree.FragmentNode import FragmentNode
from clefts.domain.fragment.tree.FragmentEdge import FragmentEdge
from clefts.domain.fragment.tree.CleavageActionTransition import CleavageActionTransition
from ..mol.graph_builder import MolGraphBuilder
from .components.action.action_decoder import ActionDecoderOutput


@dataclass(frozen=True)
class DecodedFragmentTreeBatch:
    node_graph: Batch
    edge_index: Tensor
    node_sample_index: Tensor
    edge_added_action_index: Tensor
    terminal_node_index: Tensor
    node_log_score: Tensor
    trees: tuple[FragmentTree, ...]
    compounds: tuple[Compound, ...]
    rdkit_run_count: int
    failed_effect_count: int
    materialized_state_index: Tensor = field(default_factory=lambda:torch.empty(0,dtype=torch.long),kw_only=True)
    materialized_node_index: Tensor = field(default_factory=lambda:torch.empty(0,dtype=torch.long),kw_only=True)

    def to(self, device: str | torch.device) -> DecodedFragmentTreeBatch:
        from dataclasses import replace
        return replace(self, node_graph=self.node_graph.to(device),
            **{name:getattr(self,name).to(device) for name in ("edge_index", "node_sample_index", "edge_added_action_index", "terminal_node_index", "node_log_score", "materialized_state_index", "materialized_node_index")})


def materialize_action_states(output: ActionDecoderOutput, sources: Sequence[Compound],
                              actions_by_tree: Sequence[tuple[CleavageAction, ...]], sample_tree_index: Tensor,
                              graph_builder: MolGraphBuilder) -> DecodedFragmentTreeBatch:
    # CPU transfer and domain object reconstruction happen only after decoding.
    sample = output.state_sample_index.detach().cpu().tolist()
    state = output.state_action_index.detach().cpu().tolist()
    parent = output.parent_state_index.detach().cpu().tolist()
    added = output.added_action_index.detach().cpu().tolist()
    score = output.state_log_score.detach().cpu().tolist()
    terminal = output.terminal.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    pool = output.pool.action_index.detach().cpu().tolist()
    tree_by_sample = sample_tree_index.detach().cpu().tolist()
    closure = set(terminal)
    pending = list(terminal)
    while pending:
        ancestor = parent[pending.pop()]
        if ancestor >= 0 and ancestor not in closure:
            closure.add(ancestor)
            pending.append(ancestor)
    action_offsets = [sum(len(row) for row in actions_by_tree[:tree]) for tree in range(len(actions_by_tree))]
    effects: dict[tuple[int, object], Compound | None] = {}
    nodes_by_sample = [[FragmentNode(0,-1,sources[tree].smiles)] for tree in tree_by_sample]
    compounds_by_sample = [[sources[tree]] for tree in tree_by_sample]
    index_by_smiles = [{source[0].smiles:0} for source in nodes_by_sample]
    edges_by_sample: list[dict[tuple[int,int],FragmentEdge]] = [{} for _ in tree_by_sample]
    state_node, sequences = {}, {}
    runs = failures = 0
    terminal_set = set(terminal)
    terminal_nodes: list[tuple[int,int]] = []
    scores_by_sample = [[-float('inf')] for _ in tree_by_sample]
    for row in sorted(closure):
        s, tree = sample[row], tree_by_sample[sample[row]]
        ids = [pool[s][i] - action_offsets[tree] for i in state[row] if i >= 0]
        sequence = CleavageActionSequence(actions_by_tree[tree][i] for i in ids) if ids else None
        sequences[row] = sequence
        if sequence is None:
            node = 0
        else:
            effect = (tree,sequence.effect_key)
            if effect not in effects:
                try:
                    reaction = sequence.compile(sources[tree])
                    runs += 1
                    products = reaction.run(sources[tree])
                    if len(products)!=1:
                        raise ValueError("Source action sequence must materialize one molecule")
                    effects[effect] = Compound(products[0])
                except Chem.rdchem.MolSanitizeException:
                    failures += 1
                    effects[effect] = None
            compound = effects[effect]
            if compound is None:
                continue
            node = index_by_smiles[s].get(compound.smiles)
            if node is None:
                node = len(nodes_by_sample[s])
                nodes_by_sample[s].append(FragmentNode(node,-1,compound.smiles))
                compounds_by_sample[s].append(compound)
                index_by_smiles[s][compound.smiles] = node
                scores_by_sample[s].append(-float('inf'))
        state_node[row] = node
        scores_by_sample[s][node] = max(scores_by_sample[s][node],score[row])
        if row in terminal_set:
            terminal_nodes.append((s,node))
        previous = parent[row]
        if previous >= 0 and previous in state_node and sequence is not None:
            src = state_node[previous]
            if src != node:
                key = (src,node)
                action = actions_by_tree[tree][pool[s][added[row]] - action_offsets[tree]]
                transition = CleavageActionTransition(action,sequence,sequences[previous])
                edges = edges_by_sample[s]
                if key not in edges:
                    edges[key] = FragmentEdge(len(edges),-1,src,node,-1,-1)
                edges[key] = edges[key].with_transition(transition)
    trees=[]
    all_compounds=[]
    sample_rows,edge_rows,edge_actions,node_scores=[],[],[],[]
    offsets=[]
    for s, nodes in enumerate(nodes_by_sample):
        offset=len(all_compounds)
        offsets.append(offset)
        edges=tuple(edges_by_sample[s].values())
        trees.append(FragmentTree.from_nodes_and_edges(smiles=nodes[0].smiles,nodes=tuple(nodes),edges=edges))
        all_compounds.extend(compounds_by_sample[s])
        sample_rows.extend([s]*len(nodes))
        node_scores.extend(scores_by_sample[s])
        for edge in edges:
            edge_rows.append((edge.source_index+offset,edge.target_index+offset))
            action=edge.transitions[0].added_action
            edge_actions.append(action_offsets[tree_by_sample[s]] + actions_by_tree[tree_by_sample[s]].index(action))
    graph=Batch.from_data_list([graph_builder.build(compound) for compound in all_compounds])
    return DecodedFragmentTreeBatch(graph,torch.tensor(edge_rows,dtype=torch.long).reshape(-1,2).T,
        torch.tensor(sample_rows,dtype=torch.long),torch.tensor(edge_actions,dtype=torch.long),
        torch.tensor(sorted({offsets[s]+node for s,node in terminal_nodes}),dtype=torch.long),
        torch.tensor(node_scores,dtype=torch.float32),tuple(trees),tuple(all_compounds),runs,failures,
        materialized_state_index=torch.tensor(sorted(state_node),dtype=torch.long),
        materialized_node_index=torch.tensor([offsets[sample[row]]+state_node[row] for row in sorted(state_node)],dtype=torch.long))
