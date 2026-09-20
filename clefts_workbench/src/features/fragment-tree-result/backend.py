"""Backend for inspecting generated schema-v6 action training structures."""
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import torch


def _teacher_states(structure) -> list[dict]:
    """One row per (sample, normalized action-set) teacher state, with its
    positive branching actions and, if materialized, the resulting fragment SMILES."""
    state_action_ptr = structure.teacher_node_action_ptr.tolist()
    state_action_index = structure.teacher_node_action_index.tolist()
    positive_ptr = structure.teacher_positive_action_ptr.tolist()
    positive_index = structure.teacher_positive_action_index.tolist()
    eos = structure.teacher_node_observed.tolist()
    sample_index = structure.teacher_node_sample_index.tolist()
    fragment_node_index = structure.state_fragment_node_index.tolist()
    compounds = structure.downstream.decoded.compounds if structure.downstream is not None else None

    rows = []
    for row in range(len(sample_index)):
        actions = state_action_index[state_action_ptr[row]:state_action_ptr[row + 1]]
        next_actions = positive_index[positive_ptr[row]:positive_ptr[row + 1]]
        node = fragment_node_index[row]
        rows.append({
            "sample": sample_index[row],
            "actions": actions,
            "terminal": not next_actions,
            "observed": bool(eos[row]),
            "ms2Depth": int(structure.teacher_node_ms2_depth[row]),
            "precursorRow": int(structure.teacher_node_precursor_row_index[row]),
            "positiveWeights": structure.teacher_positive_action_weight[positive_ptr[row]:positive_ptr[row+1]].tolist(),
            "fragmentSmiles": str(compounds[node].smiles) if compounds is not None and node >= 0 else None,
            "positiveBranchActions": next_actions,
        })
    return rows


def _transitions(structure) -> list[dict]:
    parents = structure.transition_parent_state_index.tolist()
    children = structure.transition_child_state_index.tolist()
    added = structure.transition_added_action_index.tolist()
    return [
        {"parent": parent, "child": child, "addedAction": action}
        for parent, child, action in zip(parents, children, added)
    ]


def _precursor_rows(structure) -> list[list[list[int]]]:
    """Per sample, every alternative precursor action set (empty = Source
    itself is the precursor for that sample)."""
    sample_ptr = structure.sample_precursor_row_ptr.tolist()
    row_action_ptr = structure.precursor_row_action_ptr.tolist()
    row_action_index = structure.precursor_row_action_index.tolist()
    rows = []
    for sample in range(len(sample_ptr) - 1):
        alternatives = []
        for row in range(sample_ptr[sample], sample_ptr[sample + 1]):
            alternatives.append(row_action_index[row_action_ptr[row]:row_action_ptr[row + 1]])
        rows.append(alternatives)
    return rows


def _node_view(structure, samples, source_smiles, file_path):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from clefts.libs.mmkit.mmkit import Compound
    source=Compound.from_smiles(source_smiles)
    mol=source.mapped_mol
    drawer=rdMolDraw2D.MolDraw2DSVG(620,360)
    drawer.DrawMolecule(mol);drawer.FinishDrawing()
    registry=[];original_actions=[];params={}
    for parent in file_path.parents:
        config=parent/'preparation_config.json'
        if config.exists():
            params=json.loads(config.read_text());break
    model=params.get('model_config',{})
    fragmenter_params=model.get('fragmenter_params',params.get('fragmenterParams'))
    if fragmenter_params:
        from clefts.domain.fragment.fragmenter import Fragmenter
        original_actions=list(Fragmenter.from_dict(fragmenter_params).fragment_ion_tree_builder.create_cleavage_actions(source))
        if len(original_actions)!=len(structure.action_type): original_actions=[]
    atoms=list(mol.GetAtoms())
    if len(atoms)!=structure.source_graph.num_nodes:
        atoms=[atom for atom in atoms if atom.GetAtomicNum()!=1]
    ptr=structure.action_source_atom_ptr.tolist();index=structure.action_source_atom_index.tolist()
    for number,category in enumerate(structure.action_type.tolist()):
        original=original_actions[number] if original_actions else None
        definitions=(fragmenter_params or {}).get('fragment_ion_tree_builder',{}).get('cleavage_pattern_set',{}).get('patterns',[])
        definition=definitions[category[0]] if category[0]<len(definitions) else {}
        registry.append(dict(id=number,cleavagePatternId=category[0],reactionId=category[1],reactantId=category[1],productMoleculeId=category[2],
            cleavagePatternName=definition.get('name',''),reactantSmarts=definition.get('reactant_smarts',''),
            sourceAtomMaps=list(original.source_atom_maps) if original else [atoms[i].GetAtomMapNum() for i in index[ptr[number]:ptr[number+1]]],
            retainedAtomMaps=sorted(original.retained_atom_maps) if original else [],
            discardedAtomMaps=sorted(original.discarded_atom_maps) if original else [],
            cutBondMaps=sorted(original.cut_bond_maps) if original else [],
            changedBondMaps=sorted(original.changed_bond_maps) if original else []))
    def action_id(action):
        if action is None:return None
        if original_actions:
            try:return original_actions.index(action)
            except ValueError:pass
        for entry in registry:
            if (entry['cleavagePatternId'],entry['reactantId'],entry['productMoleculeId'])==(action.cleavage_pattern_id,action.reaction_id,action.product_molecule_id) and entry['sourceAtomMaps']==list(action.source_atom_maps):return entry['id']
        raise ValueError('Saved transition cannot be matched to an action registry entry.')
    decoded=structure.downstream.decoded if structure.downstream is not None else None
    offset=0
    totals=dict(nodes=0,edges=0,edgeTransitions=0)
    if decoded:
        for sample,tree in zip(samples,decoded.trees):
            tree_nodes=[tree.get_node(index) for index in range(tree.num_nodes)]
            ids=[node.id for node in tree_nodes]
            use_ids=len(set(ids))==len(ids) and all(number>=0 for number in ids)
            ids=ids if use_ids else [node.index for node in tree_nodes]
            states_for_node={}
            for state in sample['states']:
                global_node=int(structure.state_fragment_node_index[state['id']].item())
                states_for_node.setdefault(global_node,[]).append(state)
            nodes=[]
            for node in tree_nodes:
                histories=[];eos=False;next_actions=set()
                for state in states_for_node.get(offset+node.index,[]):
                    if state['actions'] not in histories:histories.append(state['actions'])
                    eos=eos or state['observed'];next_actions.update(state['positiveBranchActions'])
                precursor=any(set(history)==set(alternative) for history in histories for alternative in sample['precursorAlternatives'])
                if node.index==0 and (not sample['precursorAlternatives'] or [] in sample['precursorAlternatives']):precursor=True
                nodes.append(dict(id=ids[node.index],smiles=node.smiles,actionSets=histories,precursor=precursor,observed=eos,positiveBranchActions=sorted(next_actions)))
            edges=[]
            for edge in (tree.get_edge(index) for index in range(tree.num_edges)):
                transitions=[]
                for transition in edge.transitions:
                    transitions.append(dict(addedAction=action_id(transition.added_action),
                        parentActions=[action_id(action) for action in transition.parent_action_sequence.actions] if transition.parent_action_sequence else [],
                        targetActions=[action_id(action) for action in transition.action_sequence.actions]))
                edges.append(dict(id=edge.index,source=ids[edge.source_index],target=ids[edge.target_index],transitions=transitions))
            for peak in sample.get('peaks',[]):
                for match in peak['matches']:
                    match['nodeIds']=[ids[index-offset] for index in match.get('nodeIndices',[]) if offset<=index<offset+len(nodes)]
            sample.update(nodes=nodes,edges=edges,nodeIdSource='stored ID' if use_ids else 'local fragment node index',
                          nodeCount=len(nodes),edgeCount=len(edges),edgeTransitionCount=sum(len(edge['transitions']) for edge in edges))
            totals['nodes']+=len(nodes);totals['edges']+=len(edges);totals['edgeTransitions']+=sample['edgeTransitionCount']
            offset+=len(nodes)
    return registry,drawer.GetDrawingText(),totals,fragmenter_params


def inspect_structure(file_path: Path) -> dict:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "structure" not in payload:
        raise TypeError("This .preft.pt file has no saved structure.")
    structure = payload["structure"]
    metadata = dict(payload.get("metadata") or {})
    required = ("teacher_node_action_ptr", "sample_precursor_row_ptr", "action_type")
    if not all(hasattr(structure, name) for name in required):
        raise TypeError("This .preft.pt file is not a schema-v6 Source/action training structure.")

    source_smiles = str(metadata.get("smiles", ""))
    conditions = structure.condition_features.tolist()
    states = _teacher_states(structure)
    transitions = _transitions(structure)
    precursor_rows = _precursor_rows(structure)

    states_by_sample: dict[int, list[dict]] = {}
    for local_id, state in enumerate(states):
        state = {**state, "id": local_id}
        states_by_sample.setdefault(state["sample"], []).append(state)
    transitions_by_sample: dict[int, list[dict]] = {}
    for transition in transitions:
        sample = states[transition["parent"]]["sample"] if transition["parent"] < len(states) else None
        if sample is not None:
            transitions_by_sample.setdefault(sample, []).append(transition)

    num_samples = int(structure.num_samples)
    annotations=getattr(structure,'sample_annotations',()) or metadata.get('sample_annotations',[])
    samples = []
    for sample_id in range(num_samples):
        adduct_index, collision_energy = (conditions[sample_id] if sample_id < len(conditions) else (None, None))
        if sample_id<len(annotations): collision_energy=annotations[sample_id]["collisionEnergy"]
        display_ce=str(Decimal(str(collision_energy if sample_id<len(annotations) else round(collision_energy,5))).quantize(Decimal("0.1"),rounding=ROUND_HALF_UP)) if collision_energy is not None else "Unavailable"
        samples.append({
            **(annotations[sample_id] if sample_id<len(annotations) else {}),
            "id": sample_id,
            "adductIndex": adduct_index,
            "collisionEnergy": collision_energy,
            "collisionEnergyDisplay":display_ce,
            "precursorAlternatives": precursor_rows[sample_id] if sample_id < len(precursor_rows) else [],
            "states": states_by_sample.get(sample_id, []),
            "transitions": transitions_by_sample.get(sample_id, []),
        })

    actions,source_svg,totals,fragmenter_params=_node_view(structure,samples,source_smiles,file_path)
    if fragmenter_params:
        from clefts.domain.fragment.fragmenter import Fragmenter
        from clefts.libs.mmkit.mmkit import Adduct
        fragmenter=Fragmenter.from_dict(fragmenter_params)
        for sample in samples:
            if not sample.get('adduct'):
                for parent in file_path.parents:
                    config=parent/'preparation_config.json'
                    if not config.exists():continue
                    model=json.loads(config.read_text()).get('model_config',{})
                    types=model.get('adduct_type_strs',[str(adduct) for adduct in fragmenter.adduct_types])
                    index=int(sample['adductIndex'])
                    if 0<=index<len(types):
                        sample['adduct']=types[index]
                        sample['mainAdduct']=str(fragmenter._resolve_main_adduct_type(Adduct.parse(types[index])))
                    break
    return {
        "actions":actions,"sourceSvg":source_svg,"fragmenterParams":fragmenter_params,
        "file": str(file_path),
        "metadata": metadata,
        "sourceSmiles": source_smiles,
        "summary": {
            **totals,
            "samples": num_samples,
            "primitiveActions": int(structure.action_type.shape[0]),
            "teacherNodes": len(states),
            "positiveBranches": structure.transition_added_action_index.numel(),
            "precursorCandidates": structure.precursor_row_action_ptr.numel()-1,
            "maxMs2Depth": int(structure.teacher_node_ms2_depth.max()) if states else 0,
            "transitions": len(transitions),
        },
        "samples": samples,
    }


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": inspect_structure(Path(request["path"]).resolve())}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
