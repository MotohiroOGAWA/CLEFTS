"""Branching semantics, sparse storage, chemistry boundaries and real inference."""
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
import pytest
from clefts.libs.mmkit.mmkit import Compound,Adduct
from clefts.domain.fragment.cleavage import CleavageActionSequence
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.domain.fragment.cleavage.CleavageActionSearch import CleavageActionSearch
from clefts.ml.input.source_action_structure import SourceActionStructure,prepare_source_actions
from clefts.ml.input.action_batching import select_samples
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.mol.mol_encoder import MolEncoder
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from clefts.ml.training.fragment_tree_training.model import multi_positive_loss,ActionFragmentTreeTrainingModel
from clefts.ml.training.fragment_tree_training.spectrum_validation import matched_cosine,validate_spectra
from tests.ml.action.TestActionPipeline import config


def fixture():
    f=Fragmenter.from_json('clefts/domain/fragment/presets/fragmenter_single_bond_pos.json')
    source=Compound.from_smiles('CCOCCN')
    actions=f.fragment_ion_tree_builder.create_cleavage_actions(source)
    tree=f.build_fragment_ion_tree(source)
    chains=[]
    for edge in (tree.get_edge(i) for i in range(tree.num_edges)):
        for transition in edge.transitions:
            parent=transition.parent_action_sequence
            if parent is not None and len(parent.actions)==1 and transition.added_action is not None and len(transition.action_sequence.actions)==2:
                chains.append(((None,None),(parent,parent.actions[0]),(transition.action_sequence,transition.added_action)))
    assert chains
    mol=MolEncoder(symbols=('C','N','O'),node_dim=16,graph_dim=16,num_layers=1,num_heads=4,dropout=0.)
    return dict(source=source,actions=actions,graph_builder=mol.graph_builder,condition_features=torch.tensor([[0.,20.]]),max_action_count=3),chains


def rows(data):
    return [data.teacher_positive_action_index[a:b].tolist() for a,b in zip(data.teacher_positive_action_ptr[:-1],data.teacher_positive_action_ptr[1:])]


def test_sparse_ancestor_closure_salience_and_retention():
    kwargs,chains=fixture();chain=chains[0]
    with patch.object(CleavageActionSearch,'enumerate',side_effect=AssertionError('full enumeration')):
        d=prepare_source_actions(**kwargs,teacher_pathways=[[(chain,.8)]])
    assert d.teacher_node_ms2_depth.tolist()==[0,1,2]
    assert d.teacher_positive_action_weight.tolist()==pytest.approx([.8,.8])
    assert d.transition_added_action_index.numel()==2
    assert d.precursor_next_index.numel()==0
    assert d.precursor_row_action_index.numel()==0
    assert set(d.sample_positive_action_index.tolist())=={kwargs['actions'].index(chain[1][1]),kwargs['actions'].index(chain[2][1])}
    assert not hasattr(d,'teacher_positive_eos')


def test_multiple_positive_and_ambiguous_pathways():
    kwargs,chains=fixture()
    paths=[(chain,.3 if i==0 else .9) for i,chain in enumerate(chains[:8])]
    d=prepare_source_actions(**kwargs,teacher_pathways=[paths])
    expected={kwargs['actions'].index(chain[1][1]) for chain,_ in paths}
    assert set(rows(d)[0])==expected
    assert len(expected)>1
    assert d.teacher_node_sample_index.numel()<=1+2*len(paths)
    assert set(d.transition_added_action_index.tolist())==set(d.sample_positive_action_index.tolist())


def test_nonroot_precursor_and_independent_candidates():
    kwargs,chains=fixture();chain=chains[0];seed=chain[1][0]
    other=next(c[1][0] for c in chains if c[1][0]!=seed)
    d=prepare_source_actions(**kwargs,precursor_sequences=[(seed,other)],teacher_pathways=[[(chain,.8)]])
    assert d.sample_precursor_row_ptr.tolist()==[0,2]
    assert d.precursor_row_action_ptr.tolist()==[0,1,2]
    assert d.teacher_node_ms2_depth.tolist()==[0,1,0]
    assert d.teacher_node_precursor_row_index.tolist()==[0,0,1]
    assert kwargs['actions'].index(chain[1][1]) not in d.sample_positive_action_index.tolist()
    assert d.transition_added_action_index.numel()==1


def test_save_load_reject_v5_and_collation_offsets():
    kwargs,chains=fixture()
    first=prepare_source_actions(**kwargs,teacher_pathways=[[(chains[0],.8)]])
    second=prepare_source_actions(**kwargs,precursor_sequences=[(chains[0][1][0],)],teacher_pathways=[[(chains[0],.4)]])
    d=SourceActionStructure.from_structures([first,second]);offset=len(kwargs['actions'])
    assert d.teacher_node_sample_index.tolist()==[0,0,0,1,1]
    assert d.teacher_node_precursor_row_index.tolist()==[0,0,0,1,1]
    assert d.teacher_node_parent_index.tolist()==[-1,0,1,-1,3]
    assert d.teacher_node_added_action_index[-1]==second.teacher_node_added_action_index[-1]+offset
    assert d.precursor_row_action_index[-1]==second.precursor_row_action_index[-1]+offset
    subset=select_samples(d,[1])
    assert subset.teacher_node_parent_index.tolist()==[-1,0]
    assert subset.teacher_node_precursor_row_index.tolist()==[0,0]
    torch.testing.assert_close(subset.teacher_positive_action_weight,second.teacher_positive_action_weight)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'x.pt';d.save(path)
        loaded=SourceActionStructure.load(path)
        torch.testing.assert_close(loaded.teacher_node_ms2_depth,d.teacher_node_ms2_depth)
        payload=torch.load(path,weights_only=False);payload['schema_version']=5;torch.save(payload,path)
        with pytest.raises(ValueError,match='regenerate'):SourceActionStructure.load(path)


def test_independent_bce_weak_negatives_invalid_mask_and_empty_leaf():
    logits=torch.tensor([[0.,0.,0.,float('-inf')]],requires_grad=True)
    positive=torch.tensor([[True,True,False,False]])
    loss=multi_positive_loss(logits,positive,torch.tensor([[.8,.2,0.,0.]]))
    loss.backward()
    assert (logits.grad[0,:2]<0).all() and logits.grad[0,2]>0 and logits.grad[0,3]==0
    assert logits.grad[0,0].abs()>logits.grad[0,1].abs()
    leaf=torch.tensor([[0.,float('-inf')]],requires_grad=True)
    multi_positive_loss(leaf,torch.zeros_like(leaf,dtype=torch.bool)).backward()
    assert leaf.grad[0,0]>0 and leaf.grad[0,1]==0


def test_tensor_only_forward_materialization_and_free_running_validation(tmp_path):
    c=config();c['action_model_params'].update(action_prefilter_threshold_logit=-10,prediction_threshold=.01)
    generator=create_spectrum_generator(c)
    d,_=ActionStructureBuilder(generator).build(Compound.from_smiles('CC(O)N'),[Adduct.parse('[M+H]+')],[20.],[[18.033826,44.049476,62.06004]],[[.5,.8,1.]])
    assert (d.state_fragment_node_index>=0).all()
    combined=SourceActionStructure.from_structures([d,d])
    n=d.teacher_node_sample_index.numel()
    torch.testing.assert_close(combined.state_fragment_node_index[n:],d.state_fragment_node_index+len(d.downstream.decoded.compounds))
    torch.testing.assert_close(select_samples(combined,[1]).state_fragment_node_index,d.state_fragment_node_index)
    trainer=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model)
    with patch.object(Compound,'from_smiles',side_effect=AssertionError('RDKit in forward')),patch.object(CleavageActionSequence,'compile',side_effect=AssertionError('chemistry in forward')):
        result=trainer(d);result.loss.backward()
    assert torch.isfinite(result.loss)
    generator.eval();features=generator.feature_model(d)
    decoded=generator.feature_model.decoder(features.pool,features.condition_h)
    assert (decoded.parent_state_index==0).sum()>1
    assert not hasattr(generator.feature_model.decoder,'eos_head')
    # No teacher data is supplied to predict_batches: it prepares and decodes from source/conditions.
    with patch.object(generator,'predict_batches',wraps=generator.predict_batches) as predict:
        summary=validate_spectra(generator,[d],tmp_path,'epoch')
        assert predict.call_count==1
    assert summary['spectrum_nonempty_fraction']==1
    import json
    saved=json.loads((tmp_path/'spectrum_validation/epoch.json').read_text())
    # A precursor that branches still emits its observed precursor peak.
    precursor=Adduct.parse('[M+H]+').apply_to_formula(Compound.from_smiles('CC(O)N').formula).normalized.exact_mass
    assert any(abs(mz-precursor)<.01 for mz in saved['spectra'][0]['generated_mz'])


def test_cosine_penalizes_unassigned_peaks_and_no_double_matching():
    f=Fragmenter.from_json('clefts/domain/fragment/presets/fragmenter_single_bond_pos.json')
    assert matched_cosine([100.],[1.],[{'mz':100.,'intensity':1.},{'mz':200.,'intensity':1.}],f.mass_tolerance)==pytest.approx(2**-.5)
    assert matched_cosine([],[],[{'mz':100.,'intensity':1.}],f.mass_tolerance)==0


def test_preparation_cli_argument_contract():
    import json
    from clefts.ml.data_preparation.fragment_tree.create_training_data import build_arg_parser
    contract=[dict(flags=x.option_strings,dest=x.dest,default=x.default,required=x.required,nargs=x.nargs,choices=x.choices) for x in build_arg_parser()._actions]
    assert json.loads(json.dumps(contract))==json.loads(Path(__file__).with_name('preparation_cli_contract.json').read_text())


def test_zero_intensity_samples_have_finite_post_gradients():
    generator=create_spectrum_generator(config())
    d,_=ActionStructureBuilder(generator).build(Compound.from_smiles('CCO'),[Adduct.parse('[M+H]+')]*2,[20.,40.],[[47.04914],[999.]],[[1.],[1.]])
    # Second sample has no assignable formula, a common real-data case.
    trainer=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model)
    trainer(d).loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in trainer.parameters() if p.grad is not None)
    sliced=select_samples(d,[1])
    assert sliced.sample_annotations[0]['peaks'][0]['mz']==999.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_branching_backward_uses_no_chemistry():
    generator=create_spectrum_generator(config())
    d,_=ActionStructureBuilder(generator).build(Compound.from_smiles('CC(O)N'),[Adduct.parse('[M+H]+')],[20.],[[18.033826,44.049476]],[[.5,.8]])
    trainer=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model).cuda()
    with patch.object(CleavageActionSequence,'compile',side_effect=AssertionError('chemistry in forward')),patch.object(Compound,'from_smiles',side_effect=AssertionError('chemistry in forward')):
        result=trainer(d.to('cuda'));result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(p.grad).all() for p in trainer.parameters() if p.grad is not None)
