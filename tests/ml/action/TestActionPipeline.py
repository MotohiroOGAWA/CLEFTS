from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit.Chem import rdChemReactions

from clefts.domain.fragment import Fragmenter
from clefts.domain.fragment.cleavage import CleavageActionSequence
from clefts.libs.mmkit.mmkit import Adduct, Compound, Formula
from clefts.ml.input.action_structure_builder import ActionStructureBuilder
from clefts.ml.input.source_action_structure import prepare_source_actions
from clefts.ml.specgen.action_materialization import materialize_action_states
from clefts.ml.specgen.fragment_tree_spectrum_predictor import create_spectrum_generator
from clefts.ml.specgen.components.action.action_decoder import build_action_pool, deduplicate
from clefts.ml.specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from clefts.ml.training.fragment_tree_training.action_model import ActionFragmentTreeTrainingModel
from clefts.ml.training.fragment_tree_training.action_training import train_actions
from clefts.ml.training.fragment_tree_training.action_fine_tuning import prepare_action_model_config, parameter_report
from .TestSourceAction import model_and_data


def config() -> dict:
    value = json.loads(Path('clefts/presets/spectrum_generator_params/source_anchored_pos_model_config.json').read_text())
    value['mol_encoder_params'].update(node_dim=16, graph_dim=16, num_layers=1, num_heads=4, dropout=0.)
    value['action_model_params'].update(hidden_dim=16, condition_dim=16, action_prefilter_top_k=8,
        action_prefilter_max_k=16, beam_size=16, max_decode_steps=4)
    value['post_model_params'].update(hidden_dim=16, num_layers=1)
    return value


class TestActionPipeline(unittest.TestCase):
    def test_static_action_encoding_order_and_shared_conditions(self) -> None:
        model, data, actions, source = model_and_data()
        model.eval()
        reversed_data = prepare_source_actions(source=source, actions=tuple(reversed(actions)),
            graph_builder=model.mol_encoder.graph_builder, condition_features=data.condition_features,
            max_action_count=2)
        with patch.object(model.action_encoder, 'forward', wraps=model.action_encoder.forward) as encode:
            first = model(data)
            self.assertEqual(encode.call_count, 1)
        second = model(reversed_data)
        torch.testing.assert_close(first.action_h, second.action_h.flip(0))
        torch.testing.assert_close(first.absolute_logits, second.absolute_logits.flip(1))

    def test_prefilter_forces_all_positives_and_reports_unforced_recall(self) -> None:
        model, data, _, _ = model_and_data()
        encoded = model(data)
        absolute = torch.full_like(encoded.absolute_logits, -10.)
        pool = build_action_pool(encoded.action_h, absolute, data, top_k=1, max_k=16, threshold=0., training=True, max_action_count=model.max_action_count)
        self.assertEqual(pool.positive_recall.tolist(), [0., 0.])
        for sample in range(data.num_samples):
            start, stop = data.sample_positive_action_ptr[sample:sample+2]
            self.assertTrue(set(data.sample_positive_action_index[start:stop].tolist()) <= set(pool.action_index[sample][pool.valid[sample]].tolist()))
        with self.assertRaises(ValueError):
            build_action_pool(encoded.action_h, absolute, data, top_k=1, max_k=1, threshold=0., training=True, max_action_count=model.max_action_count)
        empty = build_action_pool(encoded.action_h, absolute, data, top_k=1, max_k=16, threshold=0., training=False, max_action_count=model.max_action_count)
        output = model.decoder(empty, encoded.condition_h)
        self.assertEqual(output.terminal.sum().item(), data.num_samples)
        self.assertTrue((output.state_action_index == -1).all())

    def test_terminal_dedup_uses_max_score(self) -> None:
        samples, states, scores, representatives = deduplicate(torch.tensor([0,0,0]),
            torch.tensor([[0,-1],[0,-1],[1,-1]]), torch.tensor([-3.,-1.,-2.]))
        self.assertEqual(scores.tolist(), [-1.,-2.])
        self.assertEqual(representatives.tolist(), [1,2])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cpu_cuda_beam_parity(self) -> None:
        import copy
        torch.manual_seed(837)
        model, data, _, _ = model_and_data()
        model.eval()
        cpu_features = model(data, training_pool=True)
        cpu = model.decoder(cpu_features.pool, cpu_features.condition_h)
        gpu_model = copy.deepcopy(model).cuda().eval()
        gpu_features = gpu_model(data.to('cuda'), training_pool=True)
        gpu = gpu_model.decoder(gpu_features.pool, gpu_features.condition_h)
        torch.testing.assert_close(cpu_features.absolute_logits, gpu_features.absolute_logits.cpu(), atol=2e-5, rtol=2e-5)
        for name in ('state_sample_index','state_action_index','terminal','parent_state_index','added_action_index'):
            torch.testing.assert_close(getattr(cpu,name),getattr(gpu,name).cpu())
        # state_log_score accumulates log-softmax outputs over several decode
        # steps, so float32 CPU/GPU matmul non-associativity compounds beyond
        # the single-shot absolute_logits tolerance above; the discrete beam
        # decisions above already assert exact cross-device parity.
        torch.testing.assert_close(cpu.state_log_score, gpu.state_log_score.cpu(), atol=2e-4, rtol=2e-4)

    def test_materialization_matches_domain_and_caches_effects_across_ce(self) -> None:
        model, data, actions, source = model_and_data()
        model.eval()
        encoded = model(data, training_pool=True)
        output = model.decoder(encoded.pool, encoded.condition_h)
        decoded = materialize_action_states(output, (source,), (actions,), data.sample_tree_index, model.mol_encoder.graph_builder)
        effects = set()
        for row, node in zip(decoded.materialized_state_index.tolist(), decoded.materialized_node_index.tolist()):
            sample = output.state_sample_index[row].item()
            ids = [encoded.pool.action_index[sample, index].item() for index in output.state_action_index[row].tolist() if index >= 0]
            if not ids:
                self.assertEqual(decoded.compounds[node].smiles, source.smiles)
                continue
            sequence = CleavageActionSequence(actions[index] for index in ids)
            expected = Compound(sequence.compile(source).run(source)[0])
            self.assertEqual(decoded.compounds[node].smiles, expected.smiles)
            effects.add(sequence.effect_key)
        self.assertEqual(decoded.rdkit_run_count, len(effects))
        self.assertEqual(decoded.failed_effect_count, 0)

    def test_dehydrated_precursor_stored_graph_training_and_checkpoint(self) -> None:
        generator = create_spectrum_generator(config()).eval()
        source = Compound.from_smiles('CCCO')
        adducts = (Adduct.parse('[M+H]+'), Adduct.parse('[M+H-H2O]+'))
        mzs = [[Formula.parse('C3H9O+').exact_mass], [Formula.parse('C3H7+').exact_mass, Formula.parse('C2H5+').exact_mass]]
        data = ActionStructureBuilder(generator).build(source, adducts, (20.,40.), mzs, ((1.,), (1.,.5)))
        self.assertTrue(data.teacher_positive_eos.any())
        self.assertTrue((data.state_fragment_node_index >= 0).any())
        self.assertTrue(any(compound.smiles == 'CCC' for compound in data.downstream.decoded.compounds))
        for sample, peaks in enumerate(mzs):
            for mz in peaks:
                mask = data.downstream.formula_sample_index == sample
                self.assertTrue((torch.abs(data.downstream.formula_mz[mask] - mz) < .01).any())
        training = ActionFragmentTreeTrainingModel(generator.feature_model, downstream_model=generator.post_model)
        with patch.object(CleavageActionSequence, 'compile', side_effect=AssertionError('RDKit in forward')), patch.object(rdChemReactions.ChemicalReaction, 'RunReactants', side_effect=AssertionError('RDKit in forward')):
            result = training(data)
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            training.eval()
            self.assertTrue(torch.isfinite(training(data).loss))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('train','val'):
                (root/name).mkdir()
                data.save(root/name/'source.preft.pt')
            report = train_actions(model_config=config(), train_dir=root/'train', val_dir=root/'val', output_dir=root/'run', epochs=1)
            self.assertEqual(report['schema_version'], 4)
            self.assertTrue((root/'run'/'last.pt').is_file())
            resumed = train_actions(model_config=config(), train_dir=root/'train', val_dir=root/'val', output_dir=root/'run', epochs=1, resume=root/'run'/'last.pt')
            self.assertEqual(resumed['history'][0]['epoch'], 2)
            from clefts.ml.specgen.predict_spectrum import load_generator
            loaded = load_generator(model_path=str(root/'run'/'last.pt'), params_path=None,
                                    device=torch.device('cpu'), strict=True)
            self.assertEqual(loaded.architecture, generator.architecture)

    def test_predict_seeds_beam_decoding_from_precursor_state(self) -> None:
        # Real MS2 fragmentation happens on the selected precursor ion, never
        # on the bare Source: for [M+H-H2O]+, every further fragment must be
        # explained as a descendant of the dehydrated ion (CCC), and the
        # decoder must never be free to skip straight from Source to it.
        generator = create_spectrum_generator(config()).eval()
        source = Compound.from_smiles('CCCO')
        result = generator.predict([source, source],
            [Adduct.parse('[M+H]+'), Adduct.parse('[M+H-H2O]+')], [20., 40.])
        pool = result.selection.features.pool
        self.assertEqual(pool.precursor_row_sample_index.tolist(), [1])
        decoded = result.fragments
        sample_index = decoded.node_sample_index.tolist()
        compounds = [compound.smiles for compound in decoded.compounds]
        dehydrated_sample = [i for i, s in enumerate(sample_index) if s == 1]
        source_node = next(i for i in dehydrated_sample if compounds[i] == 'CCCO')
        edges = decoded.edge_index.t().tolist()
        self.assertFalse(any(src == source_node and compounds[dst] != 'CCC'
                             for src, dst in edges if src in dehydrated_sample))

    def test_action_fine_tuning_freezes_base_and_trains_new_pattern(self) -> None:
        base_config = config()
        base_generator = create_spectrum_generator(deepcopy(base_config)).eval()
        source = Compound.from_smiles('CCCO')
        adducts = (Adduct.parse('[M+H]+'),)
        mzs = [[Formula.parse('C3H9O+').exact_mass]]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_data = ActionStructureBuilder(base_generator).build(source, adducts, (20.,), mzs, ((1.,),))
            for name in ('base_train', 'base_val'):
                (root/name).mkdir()
                base_data.save(root/name/'source.preft.pt')
            train_actions(model_config=base_config, train_dir=root/'base_train', val_dir=root/'base_val',
                output_dir=root/'base_run', epochs=1)
            base_checkpoint = root/'base_run'/'last.pt'

            # A new pattern set with one extra, earlier pattern; the old
            # pattern's id shifts from 0 to 1, exactly like a real expansion.
            expanded_fragmenter = Fragmenter.from_dict(base_config['fragmenter_params']).to_dict()
            pattern_set = expanded_fragmenter['fragment_ion_tree_builder']['cleavage_pattern_set']
            pattern_set['patterns'][0]['pattern_id'] = 1
            pattern_set['patterns'].insert(0, dict(pattern_id=0, name='carbon_oxygen',
                reactant_smarts='[#6:1]-[#8:2]', products=[dict(name='carbon_oxygen', smarts='[#6:1]')]))
            pattern_file = root/'new.clevageset.json'
            pattern_file.write_text(json.dumps({'cleavage_pattern_set': pattern_set}))
            new_config = deepcopy(base_config)
            new_config['fragmenter_params'] = expanded_fragmenter
            new_params_file = root/'new_params.json'
            new_params_file.write_text(json.dumps(new_config))

            fine_tuned_config = prepare_action_model_config(checkpoint_path=base_checkpoint,
                pattern_set_path=pattern_file, new_params_path=new_params_file, width=2)
            self.assertEqual(fine_tuned_config['fine_tuning']['width'], 2)
            self.assertIn('categories.0', fine_tuned_config['fine_tuning']['category_mapping'])

            # Regenerate training data with the expanded action universe.
            expanded_generator = create_spectrum_generator(deepcopy(new_config)).eval()
            expanded_data = ActionStructureBuilder(expanded_generator).build(source, adducts, (20.,), mzs, ((1.,),))
            for name in ('ft_train', 'ft_val'):
                (root/name).mkdir()
                expanded_data.save(root/name/'source.preft.pt')

            train_actions(model_config=fine_tuned_config, train_dir=root/'ft_train', val_dir=root/'ft_val',
                output_dir=root/'ft_run', epochs=1)
            tuned_checkpoint = root/'ft_run'/'last.pt'
            self.assertTrue(tuned_checkpoint.is_file())

            tuned_generator = create_spectrum_generator(deepcopy(fine_tuned_config))
            report = parameter_report(tuned_generator)
            self.assertGreater(report['frozen_parameters'], 0)
            self.assertGreater(report['trainable_parameters'], 0)
            self.assertTrue(all(not p.requires_grad for p in tuned_generator.mol_encoder.parameters()))

            mismatched_config = deepcopy(fine_tuned_config)
            mismatched_config['fine_tuning']['width'] = 3
            with self.assertRaises(ValueError):
                train_actions(model_config=mismatched_config, train_dir=root/'ft_train', val_dir=root/'ft_val',
                    output_dir=root/'ft_run', epochs=1, resume=tuned_checkpoint)
            # Resuming with the exact original fine-tuning config succeeds.
            resumed = train_actions(model_config=fine_tuned_config, train_dir=root/'ft_train', val_dir=root/'ft_val',
                output_dir=root/'ft_run', epochs=1, resume=tuned_checkpoint)
            self.assertEqual(resumed['history'][0]['epoch'], 2)

            # Prediction also works from the fine-tuned checkpoint.
            checkpoint = torch.load(tuned_checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['model_config'], fine_tuned_config)
            from clefts.ml.specgen.predict_spectrum import load_generator
            predicting_generator = load_generator(model_path=str(tuned_checkpoint), params_path=None,
                                                  device=torch.device('cpu'), strict=True)
            result = predicting_generator.predict([source], [Adduct.parse('[M+H]+')], [20.])
            self.assertGreater(len(result.fragments.compounds), 0)
