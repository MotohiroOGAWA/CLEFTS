"""Regression checks for GUI/CLI configuration precedence and forwarding."""
import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from clefts.ml.specgen.config_options import configure_model_options, resolve_model_options, namespace_argv

def preparation_dataset(smiles_column='SMILES'):
    import pandas as pd
    import numpy as np
    from clefts.libs.msentity.msentity import MSDataset
    from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
    return MSDataset(pd.DataFrame({smiles_column:['CCO'],'AdductType':['[M+H]+'],
                                   'CollisionEnergy':[20],'PrecursorMZ':[47]}),
                     PeakSeries(np.array([[47.,1.]]),np.array([0,1],dtype=np.int64)))

class TestConfigOptions(unittest.TestCase):
    def parser(self):
        parser=argparse.ArgumentParser()
        configure_model_options(parser)
        return parser

    def test_adduct_embedding_order_is_not_a_cli_parameter(self):
        parser = self.parser()
        self.assertNotIn('--adduct-types-json', parser._option_string_actions)
        resolved = resolve_model_options(parser.parse_args([
            '--params-json', '{"adduct_type_strs":["[M+K]+"]}']))
        self.assertNotIn('adduct_type_strs', resolved)

    def test_file_inline_named_and_individual_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'config.json'
            file.write_text(json.dumps({'fragmenter_params':{'mass_tolerance':'1Da'},'action_model_params':{'max_fragment_nodes':30}}))
            args=self.parser().parse_args(['--params',str(file),'--params-json','{"action_model_params":{"max_fragment_nodes":40}}',
                '--mass-tolerance','0.02Da','--max-fragment-nodes','50','--set','action_model_params.max_fragment_nodes=60',
                '--set','fragmenter_params.fragment_ion_tree_builder.cleavage_pattern_set.patterns.0.name="custom"'])
            config=resolve_model_options(args)
            self.assertEqual(config['action_model_params']['max_fragment_nodes'],60)
            self.assertEqual(config['fragmenter_params']['mass_tolerance'],'0.02Da')
            self.assertEqual(config['fragmenter_params']['fragment_ion_tree_builder']['cleavage_pattern_set']['patterns'][0]['name'],'custom')
            self.assertEqual(config['action_model_params']['branch_path_threshold'],0)

    def test_fragmenter_file_without_model_wrapper(self):
        parser=self.parser()
        result=resolve_model_options(parser.parse_args(['--params-json','{"fragment_ion_tree_builder":{"max_action_count":2},"mass_tolerance":"0.1Da"}','--max-action-count','1']))
        self.assertEqual(result['fragmenter_params']['fragment_ion_tree_builder']['max_action_count'],1)
        self.assertIn('mol_encoder_params',result)

    def test_repeat_options_survive_official_cli_forwarding(self):
        parser=self.parser()
        args=parser.parse_args(['--set','action_model_params.max_fragment_nodes=40','--set','action_model_params.branch_path_threshold=0.2'])
        forwarded=namespace_argv(parser,args)
        self.assertEqual(forwarded.count('--set'),2)
        self.assertEqual(resolve_model_options(parser.parse_args(forwarded))['action_model_params']['branch_path_threshold'],0.2)

    def test_malformed_override_is_rejected(self):
        with self.assertRaises(ValueError):resolve_model_options(self.parser().parse_args(['--set','beam_size']))

    def test_official_preparation_command_forwards_parameters_and_columns(self):
        from clefts.cli.train.commands.create_fragment_tree_data import CreateFragmentTreeDataCommand
        from clefts.ml.data_preparation.fragment_tree import create_training_data as preparation
        parser=preparation.build_arg_parser()
        args=parser.parse_args(['--input','input.msds','--output-dir','out',
                               '--max-action-count','2','--max-node','500','--max-edge','1000','--num-workers','2','--chunk-size','3','--symbols-json','["C","O"]','--smiles-column','CanonicalSMILES',
                               '--set','action_model_params.max_fragment_nodes=70'])
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation.MSDataset,'load',return_value=preparation_dataset('CanonicalSMILES')), patch.object(preparation,'create_action_training_data') as run:
            args.output_dir=directory
            CreateFragmentTreeDataCommand().run(args)
        kwargs=run.call_args.kwargs
        self.assertEqual(kwargs['smiles_column'],'CanonicalSMILES')
        self.assertEqual(kwargs['max_node'],500)
        self.assertEqual(kwargs['max_edge'],1000)
        self.assertEqual(kwargs['num_workers'],2)
        self.assertEqual(kwargs['chunk_size'],3)
        self.assertEqual(kwargs['model_config']['mol_encoder_params']['symbols'],['C','O'])
        self.assertEqual(kwargs['model_config']['fragmenter_params']['fragment_ion_tree_builder']['max_action_count'],2)
        self.assertEqual(kwargs['model_config']['action_model_params']['max_fragment_nodes'],70)

    def test_official_training_command_forwards_explicit_options(self):
        from clefts.cli.train.commands.fragment_tree import FragmentTreeTrainCommand
        from clefts.ml.training.fragment_tree_training import training
        args=training.build_arg_parser().parse_args(['--train-dir','train','--val-dir','val','--output-dir','out',
            '--mol-encoder-checkpoint','encoder.pt','--max-fragment-nodes','80','--train-mol-encoder',
            '--weight-decay','0','--intensity-weight','0.5',
            '--validation-interval-steps','1000','--validation-fraction','0.1'])
        with tempfile.TemporaryDirectory() as directory, patch('clefts.ml.training.fragment_tree_training.sources.inherit_model_config',side_effect=lambda config,*args,**kwargs:config), patch.object(training,'train_actions',return_value={}) as run, patch('builtins.print'):
            args.output_dir=directory
            FragmentTreeTrainCommand().run(args)
        self.assertEqual(run.call_args.kwargs['model_config']['action_model_params']['max_fragment_nodes'],80)
        self.assertEqual(run.call_args.kwargs['weight_decay'],0)
        self.assertEqual(run.call_args.kwargs['intensity_weight'],0.5)
        self.assertTrue(run.call_args.kwargs['train_mol_encoder'])
        self.assertEqual(run.call_args.kwargs['validation_interval_steps'],1000)
        self.assertEqual(run.call_args.kwargs['validation_fraction'],0.1)

    def test_preparation_file_limits_and_symbols_are_overridden_only_explicitly(self):
        from clefts.ml.data_preparation.fragment_tree import create_training_data as preparation
        import pandas as pd
        raw={'max_node':20,'max_edge':40,'symbols':['C','O']}
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation,'load_spectrum_dataset',return_value=preparation_dataset()), patch.object(preparation,'create_action_training_data') as run:
            preparation.main(['--input','test.msds','--output-dir',directory,'--params-json',json.dumps(raw)])
            self.assertEqual(run.call_args.kwargs['max_node'],20)
            self.assertEqual(run.call_args.kwargs['max_edge'],40)
            self.assertEqual(run.call_args.kwargs['model_config']['mol_encoder_params']['symbols'],['C','O'])
            preparation.main(['--input','test.msds','--output-dir',directory,'--params-json',json.dumps(raw),'--max-node','10','--max-edge','15','--symbols-json','["C","N"]'])
            self.assertEqual(run.call_args.kwargs['max_node'],10)
            self.assertEqual(run.call_args.kwargs['max_edge'],15)
            self.assertEqual(run.call_args.kwargs['model_config']['mol_encoder_params']['symbols'],['C','N'])
