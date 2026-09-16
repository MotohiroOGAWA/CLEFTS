from __future__ import annotations
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from clefts.domain.fragment.cleavage import CleavageAction, CleavageActionSequence
from clefts.domain.fragment.cleavage.CleavageActionRelations import CleavageActionRelations
from clefts.domain.fragment.cleavage.CleavageActionSearch import CleavageActionSearch
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.libs.mmkit.mmkit import Compound
from clefts.ml.input.source_action_structure import SourceActionStructure, prepare_source_actions, coo
from clefts.ml.mol.mol_encoder import MolEncoder
from clefts.ml.specgen.source_action_feature_model import SourceActionFeatureModel
from clefts.ml.specgen.components.action import ActionCompatibilityEngine, ActionStateEncoder
from clefts.ml.training.fragment_tree_training.action_model import ActionFragmentTreeTrainingModel, multi_positive_loss


def model_and_data() -> tuple[SourceActionFeatureModel, SourceActionStructure, tuple[CleavageAction, ...], Compound]:
    fragmenter = Fragmenter.from_json('clefts/domain/fragment/presets/fragmenter_single_bond_pos.json')
    source = Compound.from_smiles('CC(O)N')
    actions = fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
    targets = [result.action_sequence for result in fragmenter.fragment_ion_tree_builder.cleave_all(source, max_action_count=2)]
    mol = MolEncoder(symbols=('C','O','N','H'), node_dim=16, graph_dim=16, num_layers=1, num_heads=4, dropout=0.)
    data = prepare_source_actions(source=source, actions=actions, graph_builder=mol.graph_builder,
        condition_features=torch.tensor([[20.,1.],[40.,1.]]), max_action_count=2, target_sequences=(targets, targets))
    model = SourceActionFeatureModel(mol, (1,1,2), 2, hidden_dim=16, condition_dim=16, max_action_count=2,
        action_prefilter_top_k=8, action_prefilter_max_k=16, beam_size=4, max_decode_steps=4)
    return model, data, actions, source


class TestSourceAction(unittest.TestCase):
    def test_conflict_cycles_replacement_and_noop(self) -> None:
        universe = frozenset((1,2,3))
        def action(i: int, source: tuple[int,...], retained: frozenset[int], cuts: frozenset = frozenset()) -> CleavageAction:
            return CleavageAction(cleavage_pattern_id=i, reaction_id=0, product_molecule_id=0,
                source_atom_maps=source, retained_atom_maps=retained, discarded_atom_maps=universe-retained,
                matched_bond_maps=cuts, cut_bond_maps=cuts)
        cases = (
            ((action(0,(1,2),universe,frozenset(((1,2),))), action(1,(1,2),universe,frozenset(((1,2),)))), (0,), 1, 'hard_conflict'),
            ((action(0,(1,),frozenset((1,3))), action(1,(2,),frozenset((2,3)))), (0,), 1, 'precedence_cycle'),
            ((action(0,(1,),frozenset((1,2))), action(1,(2,),frozenset((2,3))), action(2,(3,),frozenset((1,3)))), (0,1), 2, 'precedence_cycle'),
            ((action(0,(1,),universe), action(1,(2,),frozenset((1,2)))), (0,), 1, None),
            ((action(0,(1,),universe), action(1,(2,),frozenset((1,2)))), (1,), 0, 'noop'),
        )
        for actions, parent, candidate, rejection in cases:
            with self.subTest(rejection=rejection):
                relation = CleavageActionRelations.from_actions(actions)
                matrices = []
                for pairs in (relation.conflict_pairs, relation.precedence_pairs, relation.dominance_pairs):
                    matrix = torch.zeros((1,len(actions),len(actions)),dtype=torch.bool)
                    if pairs:
                        index = coo(pairs)
                        matrix[0,index[0],index[1]] = True
                    matrices.append(matrix)
                output = ActionCompatibilityEngine(3).expand(
                    state_action_index=torch.tensor([[*parent,*([-1]*(3-len(parent)))]]),
                    state_sample_index=torch.tensor([0]), pool_valid=torch.ones((1,len(actions)),dtype=torch.bool),
                    pool_conflict=matrices[0], pool_precedence=matrices[1], pool_dominance=matrices[2],
                    pool_retained=torch.tensor([[[v in a.retained_atom_maps for v in universe] for a in actions]]),
                    source_atom_valid=torch.ones((1,3),dtype=torch.bool))
                self.assertEqual(output.valid[candidate].item(), rejection is None)
                if rejection == 'noop':
                    self.assertTrue(output.no_op[candidate])
                elif rejection:
                    self.assertEqual(relation.rejection((*parent,candidate)), rejection)
                else:
                    self.assertEqual(output.child_action_index[candidate].tolist(), [1,-1,-1])
                    self.assertEqual(output.child_action_count[candidate].item(), 1)

    def test_set_state_permutation_invariance_and_bos(self) -> None:
        torch.manual_seed(10)
        encoder = ActionStateEncoder(16).eval()
        actions = torch.randn(2,5,16)
        sample = torch.tensor([0,1,0])
        first = torch.tensor([[0,2,4],[1,3,-1],[-1,-1,-1]])
        second = torch.tensor([[4,0,2],[-1,3,1],[-1,-1,-1]])
        torch.testing.assert_close(encoder(actions,first,sample), encoder(actions,second,sample), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(encoder(actions,first,sample)).all())

    def test_multi_positive_likelihood(self) -> None:
        logits = torch.tensor([[1.,2.,3.,4.]], requires_grad=True)
        positive = torch.tensor([[True,True,False,True]])
        expected = -torch.log(torch.softmax(logits,dim=1)[positive].sum())
        torch.testing.assert_close(multi_positive_loss(logits, positive), expected)
        multi_positive_loss(logits,positive).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_teacher_forward_no_rdkit_and_target_blindness(self) -> None:
        from dataclasses import replace
        from rdkit.Chem import rdChemReactions
        model, data, _, _ = model_and_data()
        model.eval()
        training = ActionFragmentTreeTrainingModel(model)
        with patch.object(Compound, 'from_smiles', side_effect=AssertionError('RDKit in forward')), patch.object(CleavageActionSequence, 'compile', side_effect=AssertionError('RDKit in forward')), patch.object(rdChemReactions.ChemicalReaction, 'RunReactants', side_effect=AssertionError('RDKit in forward')):
            result = training(data)
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            other = model(replace(data, downstream=torch.randn(9,9)),training_pool=True)
            torch.testing.assert_close(result.absolute_logits, other.absolute_logits)
            torch.testing.assert_close(result.next_action_logits, training(replace(data,downstream=torch.zeros(3))).next_action_logits)
            decoded = model.decoder(other.pool,other.condition_h)
            self.assertTrue(decoded.terminal.any())
        self.assertIsNotNone(model.action_encoder.roles.weight.grad)

    def test_schema_roundtrip_and_collate_offsets(self) -> None:
        model, data, _, _ = model_and_data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new.preft.pt'
            data.save(path)
            loaded = SourceActionStructure.load(path)
            torch.testing.assert_close(loaded.action_conflict_index,data.action_conflict_index)
            torch.save({'schema_version':3,'structure':data},path)
            with self.assertRaises(ValueError): SourceActionStructure.load(path)
        batch = SourceActionStructure.from_structures((data,data))
        self.assertEqual(batch.source_graph.num_graphs,2)
        self.assertEqual(batch.sample_tree_index.tolist(),[0,0,1,1])
        self.assertEqual(batch.action_type.shape[0],data.action_type.shape[0]*2)
        result = ActionFragmentTreeTrainingModel(model)(batch)
        self.assertTrue(torch.isfinite(result.loss))

    def test_torch_domain_randomized_10000_expansions(self) -> None:
        rng = random.Random(713)
        universe = frozenset(range(1,8))
        actions = []
        for i in range(24):
            retained = frozenset(v for v in universe if rng.random() < .65) or frozenset((1,))
            matched = frozenset((v,v+1) for v in range(1,7) if rng.random()<.15)
            cuts = frozenset(edge for edge in matched if rng.random()<.4)
            actions.append(CleavageAction(cleavage_pattern_id=i,reaction_id=0,product_molecule_id=0,
                source_atom_maps=tuple(sorted(set(rng.sample(sorted(universe),2)) | {v for edge in matched for v in edge})), retained_atom_maps=retained,
                discarded_atom_maps=universe-retained,matched_bond_maps=matched,cut_bond_maps=cuts))
        actions=tuple(sorted(actions,key=lambda a:a.key))
        relation=CleavageActionRelations.from_actions(actions)
        parent_rows=[]
        expected=[]
        # 432 states x 24 candidates = 10,368 transitions.
        for _ in range(432):
            while True:
                raw=tuple(rng.sample(range(24),rng.randrange(4)))
                if relation.rejection(raw): continue
                seq=CleavageActionSequence(actions[i] for i in raw) if raw else None
                if seq and not seq.retained_atom_maps: continue
                break
            ids=tuple(actions.index(a) for a in seq.actions) if seq else ()
            parent_rows.append([*ids,*([-1]*(3-len(ids)))])
            for candidate in range(24):
                raw=(*ids,candidate)
                rejection=relation.rejection(tuple(dict.fromkeys(raw)))
                child=None if rejection else CleavageActionSequence(actions[i] for i in raw)
                valid=(candidate not in ids and not rejection and child!=seq and bool(child.retained_atom_maps) and len(child.actions)<=3)
                expected.append((bool(valid), tuple(actions.index(a) for a in child.actions) if child else (),child.retained_atom_maps if child else frozenset()))
        dense=[]
        for pairs in (relation.conflict_pairs,relation.precedence_pairs,relation.dominance_pairs):
            matrix=torch.zeros((1,24,24),dtype=torch.bool)
            if pairs: matrix[0,coo(pairs)[0],coo(pairs)[1]]=True
            dense.append(matrix)
        retained=torch.tensor([[[v in a.retained_atom_maps for v in universe] for a in actions]])
        engine=ActionCompatibilityEngine(3)
        inputs=dict(state_action_index=torch.tensor(parent_rows),state_sample_index=torch.zeros(432,dtype=torch.long),pool_valid=torch.ones((1,24),dtype=torch.bool),pool_conflict=dense[0],pool_precedence=dense[1],pool_dominance=dense[2],pool_retained=retained,source_atom_valid=torch.ones((1,7),dtype=torch.bool))
        output=engine.expand(**inputs)
        self.assertEqual(output.valid.tolist(),[value[0] for value in expected])
        for i,(valid,ids,atoms) in enumerate(expected):
            if valid:
                self.assertEqual(tuple(output.child_action_index[i][output.child_action_index[i]>=0].tolist()),ids)
                self.assertEqual(output.child_action_count[i].item(),len(ids))
                self.assertEqual(frozenset(v for v,flag in zip(universe,output.retained_atom_mask[i]) if flag),atoms)
        if torch.cuda.is_available():
            gpu=engine.expand(**{name:value.cuda() for name,value in inputs.items()})
            torch.testing.assert_close(gpu.valid.cpu(),output.valid)
            torch.testing.assert_close(gpu.child_action_index.cpu(),output.child_action_index)
