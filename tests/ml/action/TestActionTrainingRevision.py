from __future__ import annotations
import json
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from clefts.libs.mmkit.mmkit import Compound,Adduct,Formula
from clefts.ml.input.source_action_structure import prepare_source_actions
from clefts.ml.input.action_batching import select_samples
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from clefts.ml.specgen.post_materialization_model import deduplicate_molecular_graphs
from clefts.ml.training.fragment_tree_training.model import ActionFragmentTreeTrainingModel,multi_positive_loss
from clefts.ml.training.fragment_tree_training.training import train_actions
from .TestActionPipeline import config


def fixture():
    torch.manual_seed(42)
    value=config();value['action_model_params']['action_prefilter_threshold_logit']=1.
    generator=create_spectrum_generator(value).eval()
    source=Compound.from_smiles('CCCO')
    adducts=(Adduct.parse('[M+H]+'),Adduct.parse('[M+H-H2O]+'))
    peaks=((Formula.parse('C3H9O+').exact_mass,Formula.parse('C2H5+').exact_mass),
           (Formula.parse('C3H7+').exact_mass,Formula.parse('C2H5+').exact_mass))
    data=ActionStructureBuilder(generator).build(source,adducts,(20.,40.),peaks,((1.,.2),(1.,.3)))
    return value,generator,source,adducts,data


class TestActionTrainingRevision(unittest.TestCase):
    def test_precursor_candidates_and_strict_capacity(self):
        _,generator,source,adducts,_=fixture()
        data,*_=generator.prepare([source,source],adducts,[20.,40.])
        # Root is an explicit empty precursor alternative, sample 1 is dehydrated.
        self.assertEqual(data.sample_precursor_row_ptr.tolist(),[0,1,2])
        self.assertEqual(data.precursor_row_action_ptr[1].item(),0)
        generator.feature_model.top_k=1
        required=data.precursor_row_action_index.tolist()
        generator.feature_model.threshold=1e6
        features=generator.feature_model(data)
        from clefts.ml.specgen.components.action.action_decoder import build_action_pool
        at_threshold=build_action_pool(features.action_h,torch.ones_like(features.absolute_logits),data,top_k=1,max_k=128,threshold=1.,training=False,max_action_count=3)
        self.assertEqual(at_threshold.valid.sum(1).tolist(),[0,1])
        self.assertTrue(set(required)<=set(features.pool.action_index[1].tolist()))
        self.assertTrue((features.pool.valid.sum(1)<=1).all())
        rows=features.pool.precursor_row_action_index
        logits,expansion=generator.feature_model.decoder.score_states(features.pool,features.condition_h,rows,features.pool.precursor_row_sample_index)
        self.assertTrue(torch.isfinite(logits[:,-1]).all())
        with self.assertRaisesRegex(ValueError,'No valid precursor'):
            prepare_source_actions(source=source,actions=generator.fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source),graph_builder=generator.mol_encoder.graph_builder,condition_features=torch.tensor([[0.,20.]]),max_action_count=3,precursor_sequences=[()])

    def test_chunked_predictions_preserve_indices_and_share_static_encoding(self):
        _,generator,source,adducts,_=fixture()
        sources=[source]*4;types=[adducts[0],adducts[1],adducts[0],adducts[1]];energies=[20.,40.,25.,45.]
        full=generator.predict(sources,types,energies)
        with patch.object(generator.feature_model.action_encoder,'forward',wraps=generator.feature_model.action_encoder.forward) as encode:
            chunks=list(generator.predict_batches(sources,types,energies,max_samples=1))
            self.assertEqual(encode.call_count,1)
        self.assertEqual(sorted(i for item in chunks for i in item.sample_input_index),list(range(4)))
        for item in chunks:
            self.assertEqual(item.selection.features.absolute_logits.shape[0],1)
            original=item.sample_input_index[0]
            internal=full.sample_input_index.index(original)
            mask=full.spectra.sample_index==internal
            torch.testing.assert_close(item.spectra.mz,full.spectra.mz[mask])
            torch.testing.assert_close(item.spectra.intensity,full.spectra.intensity[mask],atol=2e-5,rtol=2e-5)

    def test_multi_action_precursor_seed_edges_and_capacity(self):
        value=config();value['fragmenter_params']['precursor_candidate_max_action_count']=2
        value['adduct_type_strs'].append('[M+H-C4H8]+')
        generator=create_spectrum_generator(value).eval()
        source=Compound.from_smiles('CCOCC');adduct=Adduct.parse('[M+H-C4H8]+')
        output=generator.predict([source],[adduct],[20.])
        self.assertGreater(output.spectra.mz.numel(),0)
        self.assertEqual(output.fragments.edge_seed_action_index.shape[1],2)
        self.assertIn(Formula.parse('H3O+').exact_mass,output.spectra.mz.tolist())
        stored=ActionStructureBuilder(generator).build(source,[adduct],[20.],[[Formula.parse('H3O+').exact_mass]],[[1.]])
        self.assertEqual(stored.downstream.decoded.edge_seed_action_index.shape[1],2)
        generator.feature_model.top_k=1
        with self.assertRaisesRegex(ValueError,'Mandatory precursor actions exceed'):
            generator.predict([source],[adduct],[20.])

    def test_changed_reactant_bond_is_masked_on_cuda(self):
        from clefts.domain.fragment.cleavage import CleavageAction
        from clefts.domain.fragment.cleavage.CleavageActionRelations import CleavageActionRelations
        from clefts.ml.specgen.components.action.action_compatibility import ActionCompatibilityEngine
        from clefts.ml.input.source_action_structure import coo
        universe=frozenset((1,2,3))
        first=CleavageAction(cleavage_pattern_id=0,reaction_id=0,product_molecule_id=0,source_atom_maps=(1,2),retained_atom_maps=universe,discarded_atom_maps=frozenset(),matched_bond_maps=frozenset(((1,2),)),cut_bond_maps=frozenset(((1,2),)),changed_bond_maps=frozenset(((1,2),)))
        second=CleavageAction(cleavage_pattern_id=1,reaction_id=0,product_molecule_id=0,source_atom_maps=(1,2,3),retained_atom_maps=universe,discarded_atom_maps=frozenset(),matched_bond_maps=frozenset(((1,2),(2,3))),cut_bond_maps=frozenset(((2,3),)),changed_bond_maps=frozenset(((2,3),)))
        relation=CleavageActionRelations.from_actions((first,second))
        self.assertIsNone(relation.rejection((0,1)))
        matrices=[]
        for pairs in (relation.conflict_pairs,relation.precedence_pairs,relation.dominance_pairs):
            dense=torch.zeros((1,2,2),dtype=torch.bool)
            if pairs:dense[0,coo(pairs)[0],coo(pairs)[1]]=True
            matrices.append(dense)
        inputs=dict(state_action_index=torch.tensor([[0,-1,-1],[1,-1,-1]]),state_sample_index=torch.tensor([0,0]),pool_valid=torch.ones((1,2),dtype=torch.bool),pool_conflict=matrices[0],pool_precedence=matrices[1],pool_dominance=matrices[2],pool_retained=torch.ones((1,2,3),dtype=torch.bool),source_atom_valid=torch.ones((1,3),dtype=torch.bool))
        engine=ActionCompatibilityEngine(3,enforce_reactant_order=True)
        valid=engine.expand(**inputs).valid
        self.assertFalse(valid[1]);self.assertTrue(valid[2])
        if torch.cuda.is_available():torch.testing.assert_close(valid,engine.expand(**{key:value.cuda() for key,value in inputs.items()}).valid.cpu())

    def test_empty_teacher_and_legacy_precursor_upgrade(self):
        from clefts.ml.input.action_batching import upgrade_precursor_metadata
        _,generator,source,adducts,data=fixture()
        bare,*_=generator.prepare([source],[adducts[0]],[20.])
        result=ActionFragmentTreeTrainingModel(generator.feature_model)(bare)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        expected=data.precursor_next_index.clone()
        object.__delattr__(data,'precursor_next_index')
        upgraded=upgrade_precursor_metadata(data)
        self.assertEqual(set(map(tuple,upgraded.precursor_next_index.t().tolist())),set(map(tuple,expected.t().tolist())))

    def test_intensity_ranking_prefers_high_target(self):
        positive=torch.tensor([[True,True,False]])
        weight=torch.tensor([[1.,.1,0.]])
        good=multi_positive_loss(torch.tensor([[3.,0.,-3.]]),positive,weight)
        bad=multi_positive_loss(torch.tensor([[0.,3.,-3.]]),positive,weight)
        self.assertLess(good,bad)
        zero=multi_positive_loss(torch.zeros((1,3)),torch.zeros((1,3),dtype=torch.bool),torch.zeros((1,3)))
        self.assertEqual(zero.item(),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_forward_backward_without_chemistry_and_graph_dedup(self):
        _,generator,source,_,data=fixture()
        generator=generator.cuda();generator.feature_model.freeze_mol_encoder()
        downstream=deduplicate_molecular_graphs(data.downstream,(source.smiles,))
        self.assertLess(downstream.unique_source_index.numel(),len(data.downstream.decoded.compounds))
        data=replace(data,downstream=downstream).to('cuda')
        model=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model).cuda()
        with patch.object(Chem,'MolToSmiles',side_effect=AssertionError('chemistry during forward')),patch.object(rdChemReactions.ChemicalReaction,'RunReactants',side_effect=AssertionError('chemistry during training')),patch.object(generator.mol_encoder.graph_builder,'build',side_effect=AssertionError('graph preparation during training')):
            result=model(data);result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in generator.mol_encoder.parameters()))
        for i in range(2):
            one=select_samples(data.to('cpu'),[i]);self.assertEqual(one.num_samples,1)
            self.assertTrue((one.downstream.formula_sample_index==0).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_training_curve_tensorboard_resume_and_reports(self):
        value,generator,_,_,data=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            encoder=root/'encoder.pt'
            torch.save({'mol_encoder_params':value['mol_encoder_params'],'mol_encoder_state_dict':generator.mol_encoder.state_dict()},encoder)
            value['mol_encoder_checkpoint']=str(encoder)
            for name in ('train','validation'):
                (root/name).mkdir();data.save(root/name/'source.preft.pt')
                (root/name/'action_statistics.json').write_text(json.dumps({'model_config':value}))
            kwargs=dict(model_config=value,train_dir=root/'train',val_dir=root/'validation',output_dir=root/'run',device='cuda',lr=.003,warmup_steps=0,early_stopping_patience=0,max_samples=1,gradient_clip=1.)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(rdChemReactions.ChemicalReaction,'RunReactants',side_effect=AssertionError('chemistry in training')):
                report=train_actions(**kwargs,epochs=120)
            first=report['history'][0]['train_loss'];last=report['history'][-1]['train_loss']
            self.assertLess(last,first*.8)
            print('Learning check:',first,last,'cosine',report['history'][-1]['validation/spectrum_cosine_similarity'],'filter',report['history'][-1]['validation/intensity_recall_at_filter'])
            self.assertGreater(report['history'][-1]['validation/spectrum_cosine_similarity'],.8)
            self.assertGreater(report['history'][-1]['validation/intensity_recall_at_filter'],.8)
            self.assertEqual(report['schema_version'],5)
            self.assertTrue(all(row['train/samples']==2 and row['train/batches']==2 for row in report['history']))
            for name in ('training_args.json','training_config.json','dataset_summary.json','last.pt','best.pt','metrics.tsv','metrics.json','training_report.json','training_report.md'):
                self.assertTrue((root/'run'/name).is_file(),name)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator=EventAccumulator(str(root/'run'/'tensorboard'));accumulator.Reload()
            self.assertIn('validation/spectrum_cosine_similarity',accumulator.Tags()['scalars'])
            with contextlib.redirect_stdout(io.StringIO()):resumed=train_actions(**kwargs,epochs=1,resume=root/'run'/'last.pt')
            self.assertEqual(resumed['history'][-1]['epoch'],121)
            self.assertEqual(resumed['history'][-1]['global_step'],242)
            self.assertEqual(len(resumed['history']),121)
            print(f'CUDA overfit check: train loss {first:.6f} -> {last:.6f}; filter recall {report["history"][-1]["validation/intensity_recall_at_filter"]:.6f}; validation cosine {report["history"][-1]["validation/spectrum_cosine_similarity"]:.6f}')

    def test_cpu_training_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'requires CUDA'):
                train_actions(model_config={},train_dir='train',val_dir='val',output_dir=temporary,device='cpu')

    def test_validation_subset_samples_records_reproducibly(self):
        import random
        from clefts.ml.training.fragment_tree_training.training import validation_subset
        _,_,_,_,data=fixture()
        pieces=[data]*10
        rng_state=random.getstate()
        selected,indices=validation_subset(pieces,.1,42)
        self.assertEqual(sum(piece.num_samples for piece in selected),2)
        self.assertEqual(indices,validation_subset(pieces,.1,42)[1])
        self.assertEqual(random.getstate(),rng_state)
        self.assertEqual(len(set(indices)),2)
        self.assertEqual(sum(piece.num_samples for piece in validation_subset(pieces,1.,42)[0]),20)
        self.assertEqual(sum(piece.num_samples for piece in validation_subset(pieces,.001,42)[0]),1)
        for fraction in (0,-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):validation_subset(pieces,fraction,42)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_intermediate_validation_step_cadence_reports_and_resume(self):
        value,generator,_,_,data=fixture()
        calls=[]
        original=ActionFragmentTreeTrainingModel.forward
        def forward(model,batch):
            calls.append((model.training,torch.is_grad_enabled(),batch.num_samples))
            self.assertTrue(batch.condition_features.is_cuda)
            return original(model,batch)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            encoder=root/'encoder.pt'
            torch.save({'mol_encoder_params':value['mol_encoder_params'],'mol_encoder_state_dict':generator.mol_encoder.state_dict()},encoder)
            value['mol_encoder_checkpoint']=str(encoder)
            for name in ('train','validation'):
                (root/name).mkdir()
                saved=data if name=='train' else type(data).from_structures([data]*5)
                saved.save(root/name/'source.preft.pt')
                (root/name/'action_statistics.json').write_text(json.dumps({'model_config':value}))
            kwargs=dict(model_config=value,train_dir=root/'train',val_dir=root/'validation',output_dir=root/'run',
                        device='cuda',max_samples=1,warmup_steps=0,early_stopping_patience=0,
                        validation_interval_steps=3,validation_fraction=.1)
            logs=io.StringIO()
            with contextlib.redirect_stdout(logs),patch.object(ActionFragmentTreeTrainingModel,'forward',forward),patch.object(rdChemReactions.ChemicalReaction,'RunReactants',side_effect=AssertionError('chemistry in training')):
                report=train_actions(**kwargs,epochs=2)
                resumed=train_actions(**kwargs,epochs=1,resume=root/'run'/'last.pt')
            self.assertEqual([row['global_step'] for row in report['intermediate_validation_history']],[3])
            self.assertEqual([row['global_step'] for row in resumed['intermediate_validation_history']],[3,6])
            self.assertEqual(report['intermediate_validation_history'][0]['epoch'],2)
            self.assertEqual(resumed['history'][-1]['global_step'],6)
            self.assertTrue(all(row['validation/samples']==10 for row in resumed['history']))
            self.assertTrue(all(row['validation_samples']==1 and row['total_validation_samples']==10 for row in resumed['intermediate_validation_history']))
            self.assertEqual(sum(training for training,_,_ in calls),6)
            self.assertTrue(all(training==grad_enabled for training,grad_enabled,_ in calls))
            self.assertEqual(sum(not training for training,_,_ in calls),32)
            events=[json.loads(line) for line in logs.getvalue().splitlines()]
            self.assertEqual([event['global_step'] for event in events if event['event']=='intermediate_validation_end'],[3,6])
            for filename in ('intermediate_validation.json','intermediate_validation.tsv','intermediate_validation_subset.json'):
                self.assertTrue((root/'run'/filename).is_file())
            self.assertEqual(len(json.loads((root/'run'/'intermediate_validation.json').read_text())['history']),2)
            self.assertIn('## Intermediate validation',(root/'run'/'training_report.md').read_text())
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator=EventAccumulator(str(root/'run'/'tensorboard'/'iterations'));accumulator.Reload()
            self.assertEqual([event.step for event in accumulator.Scalars('intermediate_validation/validation_loss')],[3,6])
