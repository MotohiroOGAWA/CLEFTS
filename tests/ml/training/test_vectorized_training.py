from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from clefts.ml.training.fragment_tree_training.model import PairwiseEdgeIntensityRankingLoss, FragmentTreeSelectionTrainingLoss
from clefts.ml.specgen.fragment_tree_formula_intensity_model import FragmentTreeFormulaIntensityPredictor
from ._scalar_training_reference import ScalarRanking, ScalarSelection, ScalarIntensity

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def fixture(device):
    torch.manual_seed(892)
    def tensor(value): return torch.tensor(value, device=device)
    def rand(*shape): return torch.randn(*shape, device=device, requires_grad=True)
    batch = SimpleNamespace(batch=tensor([0,0,0,1,1,1]), kept_sample_ids=tensor([2,5]),
        node_id_global=tensor([0,1,2,0,3,4]), node_is_precursor_root=tensor([True,False,False,True,False,False]),
        node_main_adduct_type_index=tensor([0,0,0,1,1,1]), edge_id_global=tensor([0,1,2,0,3,4]),
        edge_index=tensor([[0,0,1,3,3,4],[1,2,2,4,5,5]]), edge_ptr=tensor([0,3,6]),
        x=rand(6,8), edge_attr=rand(6,4))
    out = SimpleNamespace(sample_tree_batch=batch, keep_logit=rand(6), cleave_logit=rand(6),
        edge_cleave_logit=rand(6), edge_absolute_logit=rand(6), ion_logit=rand(6,3),
        unsaturation_logit=rand(6,2), radical_logit=rand(6,2))
    for name, size in [('ion',3),('unsaturation',2),('radical',2)]:
        mask=torch.ones(2,2,size,dtype=torch.bool,device=device)
        mask[1,0,-1]=False
        setattr(out,name+'_valid_mask_by_role_adduct',mask)
    target = SimpleNamespace(target_sample_index=tensor([2,2,2,2,5,5,5,2]),
        target_node_index=tensor([0,1,1,2,3,4,0,99]),
        target_peak_index=tensor([0,1,1,1,2,3,4,5]), target_intensity=tensor([1.,.5,.5,.5,.8,0.,1.,.4]),
        target_ion_index=tensor([0,0,1,0,0,1,0,0]),
        target_unsaturation_index=tensor([0,0,1,0,0,1,0,0]),
        target_radical_index=tensor([0,0,1,0,0,1,0,0]),
        target_edge_index=tensor([[2,2,2,5,5,2],[0,1,1,0,3,99]]),
        target_edge_group_index=tensor([0,0,1,2,3,4]), formula_peak_index=tensor([0,1,2,3,4]),
        sample_peak_intensity=tensor([1.,.3,.9,0.,.5]), sample_peak_ptr=tensor([0,0,0,2,2,2,5]),
        target_expand_node_index=tensor([1,2,3]), terminal_expand_ptr=tensor([0,0,1,1,2,3,3,3,3]))
    return out,target


def assert_gradients(a,b,tensors):
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    ga=torch.autograd.grad(a,tensors,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(b,tensors,retain_graph=True,allow_unused=True)
    for x,y in zip(ga,gb):
        if x is None or y is None:
            other=y if x is None else x
            assert other is None or torch.count_nonzero(other)==0
        else:
            torch.testing.assert_close(x,y,atol=3e-6,rtol=5e-5)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('top,near,far,background,threshold',[(10,1,3,10,.05),(1,0,0,2,.0),(2,3,3,0,.2),(4,0,0,0,.05)])
def test_ranking_values_and_gradients(device,top,near,far,background,threshold):
    out,target=fixture(device)
    options=dict(top_n=top,nearest_lower_partners=near,extended_lower_partners=far,
                 background_partners=background,intensity_threshold=threshold)
    old,new=ScalarRanking(**options),PairwiseEdgeIntensityRankingLoss(**options)
    assert_gradients(old(out,target),new(out,target),[out.edge_absolute_logit])
    with patch.object(new,'forward',side_effect=AssertionError('loss was recomputed')):
        metrics=new.metrics(out,target,loss=torch.tensor(.75,device=device))
        assert metrics['edge_ranking_loss']==.75


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('zero_intensity',[False,True])
def test_selection_values_and_gradients(device,zero_intensity):
    out,target=fixture(device)
    if zero_intensity: target.target_intensity.zero_()
    old,new=ScalarSelection(),FragmentTreeSelectionTrainingLoss()
    tensors=[out.keep_logit,out.cleave_logit,out.edge_cleave_logit,out.ion_logit,out.unsaturation_logit,out.radical_logit]
    for name in ['_peak_fragment_loss','_precursor_keep_loss','_state_loss','_keep_negative_loss','_edge_negative_loss','_edge_group_coverage_loss']:
        assert_gradients(getattr(old,name)(out,target,device=device),getattr(new,name)(out,target,device=device),tensors)
    for x,y in zip(old._build_cleave_targets(out,target,device=device),new._build_cleave_targets(out,target,device=device)):
        torch.testing.assert_close(x,y)
    assert_gradients(old(out,target),new(out,target),tensors)
    torch.testing.assert_close(new.last_precursor_keep_loss,old._precursor_keep_loss(out,target,device=device))
    assert not new.last_precursor_keep_loss.requires_grad


@pytest.mark.parametrize('device',DEVICES)
def test_intensity_values_and_gradients(device):
    out,_=fixture(device)
    feature=SimpleNamespace(formula_tensorizer=SimpleNamespace(dim=3),tree_encoder=SimpleNamespace(hidden_dim=8),
        fragment_edge_encoder=SimpleNamespace(feature_dim=4),ion_flat_candidates=range(3),
        unsaturation_flat_candidates=range(2),radical_flat_candidates=range(2))
    old=ScalarIntensity(feature,num_encoder_layers=1,num_attention_heads=2).to(device).eval()
    new=FragmentTreeFormulaIntensityPredictor(feature,num_encoder_layers=1,num_attention_heads=2).to(device).eval()
    new.load_state_dict(old.state_dict())
    probability=torch.tensor([.1,.3,.6,.25,.75],device=device,requires_grad=True)
    out.kept_candidates=[SimpleNamespace(sample_id=sample,batch_node_index=node,ion_index=ion,
        unsaturation_index=0,radical_index=0,formula_tensor=torch.tensor(formula),probability=.5,
        probability_tensor=probability[i]) for i,(sample,node,ion,formula) in enumerate([
        (2,0,0,[1.,2.,1.]),(5,4,1,[2.,2.,1.]),(2,1,1,[1.,2.,1.]),
        (5,5,0,[1.,2.,1.]),(2,2,0,[3.,2.,1.])])]
    a,b=old.forward_candidate_output(out),new.forward_candidate_output(out)
    for name in ['sample_index','formula_tensor','logit','presence_logit','abundance_logit']:
        torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=3e-6,rtol=3e-5)
    def loss(x):
        weight=torch.arange(1,x.logit.numel()+1,device=device)
        return (weight*(x.logit+x.presence_logit*.1+x.abundance_logit*.2)).sum()
    tensors=[out.sample_tree_batch.x,out.sample_tree_batch.edge_attr,out.edge_absolute_logit,out.edge_cleave_logit,probability]
    assert_gradients(loss(a),loss(b),tensors)
    ga=torch.autograd.grad(loss(a),tuple(old.parameters()),allow_unused=True)
    gb=torch.autograd.grad(loss(b),tuple(new.parameters()),allow_unused=True)
    for x,y in zip(ga,gb):
        if x is None: assert y is None
        else: torch.testing.assert_close(x,y,atol=5e-6,rtol=1e-4)
