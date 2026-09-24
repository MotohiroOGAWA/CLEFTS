import math
from unittest.mock import patch
import pytest
import torch

from clefts.libs.mmkit.mmkit import Compound,Adduct
from clefts.domain.fragment.cleavage import CleavageActionSequence
from clefts.domain.fragment.cleavage.CleavageActionRelations import CleavageActionRelations
from clefts.domain.fragment.ion_tree import FragmentIonTree
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.input.source_action_structure import SourceActionStructure
from clefts.ml.input.source_action_structure import prepare_source_actions
from clefts.ml.mol.graph_builder import MolGraphBuilder
from clefts.ml.specgen import materialization
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from clefts.ml.specgen.post_materialization_model import normalized_attention_pool
from clefts.ml.specgen.components.action.action_compatibility import ActionCompatibilityEngine
from clefts.ml.training.fragment_tree_training.model import ActionFragmentTreeTrainingModel,smooth_max_mil
from .TestActionPipeline import config


def prepared_two_ce():
    generator=create_spectrum_generator(config())
    source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    data,_=ActionStructureBuilder(generator).build(source,[adduct,adduct],[10.,40.],
        [[18.033826,44.049476],[18.033826,44.049476]],[[.5,1.],[1.,.5]])
    return generator,data


def test_ce_spectra_share_one_branch_group_and_tree():
    generator,data=prepared_two_ce()
    assert data.num_samples==2 and data.num_branch_groups==1
    assert data.sample_branch_group_index.tolist()==[0,0]
    assert len(data.downstream.decoded.trees)==1
    generator.eval();source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    output=generator.predict([source,source],[adduct,adduct],[10.,40.])
    assert len(output.fragments.trees)==1
    assert output.downstream.tree_graph.num_graphs==1
    first=output.spectra.intensity[output.spectra.sample_index==0]
    second=output.spectra.intensity[output.spectra.sample_index==1]
    assert first.shape==second.shape and not torch.equal(first,second)


def test_log_path_probability_and_zero_threshold():
    scores=torch.logit(torch.tensor([.8,.5]))
    log_probability=torch.nn.functional.logsigmoid(scores).sum()
    assert log_probability.item()==pytest.approx(math.log(.4))
    generator=create_spectrum_generator(config())
    assert generator.feature_model.decoder.branch_log_threshold==-math.inf


def test_candidate_mask_requires_mutual_action_center_retention():
    engine=ActionCompatibilityEngine(max_action_count=3)
    # A is selected. B preserves A and is preserved by A. C is invalidated by
    # A; D invalidates A. Because actions are simultaneous, both directions
    # make the composite invalid rather than establishing an execution order.
    invalidation=torch.zeros((1,4,4),dtype=torch.bool)
    invalidation[0,0,2]=True
    invalidation[0,3,0]=True
    expansion=engine.expand(
        state_action_index=torch.tensor([[0,-1,-1]]),state_sample_index=torch.tensor([0]),
        pool_valid=torch.ones((1,4),dtype=torch.bool),
        pool_conflict=torch.zeros((1,4,4),dtype=torch.bool),
        pool_invalidation=invalidation,
        pool_dominance=torch.zeros((1,4,4),dtype=torch.bool),
        pool_retained=torch.full((1,4,1),15,dtype=torch.long),
        source_atom_valid=torch.full((1,1),15,dtype=torch.long))
    assert expansion.valid.tolist()==[False,True,False,False]


def test_positive_mil_is_normalized_and_smooth():
    paths=torch.tensor([math.log(.8),math.log(.4)])
    value=smooth_max_mil(paths,torch.tensor([0,2]),temperature=.1)
    expected=.1*(torch.logsumexp(paths/.1,0)-math.log(2))
    torch.testing.assert_close(value,expected[None])


def test_multi_depth_states_and_cross_ce_negative_protection():
    generator=create_spectrum_generator(config())
    source=Compound.from_smiles('CC(O)N')
    actions=generator.fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
    relations=CleavageActionRelations.from_actions(actions)
    pair=None
    for first in range(len(actions)):
        for second in range(len(actions)):
            if first==second or relations.rejection((first,second)):continue
            sequence=CleavageActionSequence((actions[first],actions[second]))
            if sequence.retained_atom_maps and len(sequence.actions)==2:
                try:
                    if len(sequence.compile(source).run(source))==1:
                        pair=(first,second,sequence);break
                except Exception:pass
        if pair:break
    assert pair is not None
    first,second,sequence=pair
    prefix=CleavageActionSequence((actions[first],))
    shallow=((None,None),(prefix,actions[first]))
    deep=((*shallow,(sequence,actions[second])))
    data=prepare_source_actions(source=source,actions=actions,graph_builder=generator.mol_encoder.graph_builder,
        condition_features=torch.tensor([[0.,10.],[0.,40.]]),max_action_count=3,
        sample_branch_group_index=torch.tensor([0,0]),branch_group_adduct_index=torch.tensor([0]),
        teacher_pathways=[[[shallow]],[[deep]]])
    assert {0,1,2}<=set(data.teacher_node_ms2_depth.tolist())
    prefix_row=next(row for row,depth in enumerate(data.teacher_node_ms2_depth.tolist()) if depth==1)
    start,stop=data.weak_negative_action_ptr[prefix_row:prefix_row+2]
    assert second not in data.weak_negative_action_index[start:stop].tolist()
    assert torch.all(data.state_transition_next_state_index>=0)


def test_fragment_node_budget_is_shared_across_collision_energies():
    value=config();value['action_model_params']['max_fragment_nodes']=2
    generator=create_spectrum_generator(value).eval();source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    outputs=list(generator.predict_batches([source,source],[adduct,adduct],[10.,40.],max_samples=1))
    assert len(outputs)==1
    output=outputs[0]
    assert len(output.fragments.trees)==1
    assert output.selection.decoded.state_sample_index.numel()<=2


def test_physical_candidates_preserve_equivalent_explanations_without_sum_bias():
    _,data=prepared_two_ce();downstream=data.downstream
    counts=downstream.physical_candidate_explanation_ptr[1:]-downstream.physical_candidate_explanation_ptr[:-1]
    assert torch.any(counts>1)
    candidate=int((counts>1).nonzero()[0]);start,stop=downstream.physical_candidate_explanation_ptr[candidate:candidate+2]
    triples=set(zip(downstream.explanation_ion_type_index[start:stop].tolist(),downstream.explanation_unsaturation[start:stop].tolist(),downstream.explanation_radical[start:stop].tolist()))
    assert len(triples)==int(counts[candidate])
    values=torch.tensor([[2.,-1.],[2.,-1.]])
    attention=torch.nn.Linear(2,1,bias=False);torch.nn.init.zeros_(attention.weight)
    one=normalized_attention_pool(values[:1],torch.tensor([0]),1,attention)
    duplicate=normalized_attention_pool(values,torch.tensor([0,0]),1,attention)
    torch.testing.assert_close(one,duplicate)


def test_existing_hydrogen_state_equation_is_preserved():
    generator,_=prepared_two_ce()
    store=generator.fragmenter.fragment_ion_tree_builder._build_hydrogen_state_candidate_store()
    expected=-2*store.candidate_states[:,0]-store.candidate_states[:,1]
    assert (store.candidate_delta_h==expected).all()


def test_multiple_physical_ions_and_training_step_use_no_chemistry(tmp_path):
    generator,data=prepared_two_ce();downstream=data.downstream
    per_node=torch.bincount(downstream.physical_candidate_node_index)
    assert int(per_node.max())>1
    path=tmp_path/'strict.preft.pt';data.save(path);data=SourceActionStructure.load(path)
    trainer=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model)
    optimizer=torch.optim.Adam(trainer.parameters(),lr=1e-4)
    with patch.object(Compound,'from_smiles',side_effect=AssertionError('RDKit in training')), \
         patch.object(CleavageActionSequence,'compile',side_effect=AssertionError('fragment materialization in training')), \
         patch.object(FragmentIonTree,'from_fragment_tree',side_effect=AssertionError('ion chemistry in training')), \
         patch.object(MolGraphBuilder,'build',side_effect=AssertionError('graph construction in training')), \
         patch.object(materialization,'materialize_action_states',side_effect=AssertionError('fragment materialization in training')):
        optimizer.zero_grad();result=trainer(data);result.loss.backward();optimizer.step()
    assert torch.isfinite(result.loss)
    assert {'branch_positive_mil_loss','branch_negative_loss','num_physical_ion_candidates','num_ion_explanations'}<=result.metrics.keys()


def test_prepared_payload_has_no_version_migration_fields(tmp_path):
    _,data=prepared_two_ce();path=tmp_path/'data.preft.pt';data.save(path)
    payload=torch.load(path,weights_only=False)
    assert set(payload)=={'schema','structure'}
    loaded=SourceActionStructure.load(path)
    assert loaded.num_branch_groups==1
