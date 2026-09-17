"""Real dataset inspection and train/validation preparation regressions."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from clefts.libs.msentity.msentity import MSDataset
from clefts.libs.msentity.msentity.core.PeakSeries import PeakSeries
from clefts.libs.mmkit.mmkit import Compound
from clefts.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset, split_by_smiles
from clefts.ml.data_preparation.fragment_tree.create_training_data import main
from .TestActionPipeline import config

spec=importlib.util.spec_from_file_location('workbench_dataset_backend',Path('clefts_workbench/src/workbench/dataset_backend.py'))
backend=importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

class TestPreparationWorkflow(unittest.TestCase):
    def dataset(self):
        smiles=['CCO','CCO','CCN','CCC','COC']
        mzs=[Compound.from_smiles(value).formula.exact_mass+1.007276466 for value in smiles]
        metadata=pd.DataFrame({'SMILES':smiles,'AdductType':['[M+H]+']*5,'CollisionEnergy':[20]*5,'PrecursorMZ':mzs})
        peaks=np.array([[mz,1] for mz in mzs],dtype=float)
        return MSDataset(metadata,PeakSeries(peaks,np.arange(6,dtype=np.int64)))

    def test_split_is_deterministic_and_keeps_source_groups(self):
        data=self.dataset()
        train,validation=split_by_smiles(data,'SMILES',0.5,7)
        train2,validation2=split_by_smiles(data,'SMILES',0.5,7)
        self.assertFalse(set(train['SMILES'])&set(validation['SMILES']))
        self.assertEqual(len(train)+len(validation),len(data))
        self.assertEqual(validation['SMILES'].tolist(),validation2['SMILES'].tolist())

    def test_csv_loader_uses_official_spectrum_table_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'spectra.csv'
            pd.DataFrame({'SMILES':['CCO'],'Peak':['100,0.5;150,1']}).to_csv(file,index=False)
            data=load_spectrum_dataset(file)
            self.assertEqual(data[0].peaks.mz.tolist(),[100,150])
            self.assertEqual(data[0].peaks.intensity.tolist(),[0.5,1])

    def test_preview_and_column_validation_use_actual_data(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'dataset.msds';self.dataset().save(str(file))
            preview=backend.preview({'path':str(file),'limit':2})
            self.assertFalse(preview['valuesChecked'])
            self.assertEqual(preview['validation']['smilesColumn']['valid'],0)
            checked=backend.preview({'path':str(file),'validateValues':True})
            self.assertTrue(checked['valuesChecked'])
            self.assertEqual(checked['validation']['smilesColumn']['valid'],5)
            self.assertEqual(preview['records'],5)
            self.assertEqual(preview['summary']['uniqueSmiles'],4)
            self.assertEqual(len(preview['rows']),2)
            self.assertEqual(preview['size'],file.stat().st_size)
            self.assertFalse(preview['errors'])
            invalid=backend.preview({'path':str(file),'mapping':{'smilesColumn':'missing'}})
            self.assertFalse(invalid['validation']['smilesColumn']['exists'])
            self.assertTrue(invalid['errors'])

    def test_invalid_metadata_is_inspectable_and_excluded_from_preparation(self):
        data=self.dataset()
        metadata=pd.DataFrame({'SMILES':['CCO','invalid','CCN','CCC','COC'],
            'AdductType':['[M+H]+','[M+H]+','invalid','[M+H]+','[M+H]+'],
            'CollisionEnergy':['20','20','20','invalid','20'],
            'PrecursorMZ':['100','100','100','100','invalid']})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'invalid.msds'
            MSDataset(metadata,data.peaks).save(str(file))
            result=backend.preview({'path':str(file),'validateValues':True,'fragmenterParams':config()['fragmenter_params']})
            self.assertFalse(result['errors'])
            self.assertEqual(result['eligibleRecords'],1)
            self.assertEqual(result['excludedRecords'],4)
            self.assertEqual([row['index'] for row in result['invalidRecords']],[1,2,3,4])
            self.assertEqual(result['invalidRecords'][1]['values']['adductTypeColumn'],'invalid')
            self.assertIn('main adduct',result['invalidRecords'][1]['issues'][0]['reason'])
            from clefts.ml.data_preparation.fragment_tree.create_training_data import create_action_training_data
            model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
            with contextlib.redirect_stdout(io.StringIO()):
                files=create_action_training_data(dataset=MSDataset(metadata,data.peaks),model_config=model,output_dir=root/'direct')
                main(['--input',str(file),'--output-dir',str(root/'cli'),'--params-json',json.dumps(model)])
            self.assertEqual(len(files),1)
            self.assertEqual(len(json.loads((root/'direct/invalid_records.json').read_text())),4)
            self.assertEqual(len(json.loads((root/'cli/invalid_records.json').read_text())['train']),4)
            self.assertEqual(len(list((root/'cli').glob('*.preft.pt'))),1)
            saved=json.loads((root/'cli/preparation_config.json').read_text())
            self.assertEqual(saved['input'],str(file))
            self.assertEqual(saved['max_node'],-1)

    def test_tree_limits_skip_only_the_source_and_continue_serial_and_parallel(self):
        from clefts.ml.data_preparation.fragment_tree.create_training_data import create_action_training_data
        metadata=pd.DataFrame({'SMILES':['CCO','CCO','C'],'AdductType':['[M+H]+']*3,
                               'CollisionEnergy':[20]*3,'PrecursorMZ':[47,47,17]})
        data=MSDataset(metadata,PeakSeries(np.array([[47.,1.],[47.,1.],[17.,1.]]),np.arange(4,dtype=np.int64)))
        model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
        with tempfile.TemporaryDirectory() as directory:
            for workers in (1,2):
                for limit in ({'max_node':1},{'max_edge':0}):
                    with self.subTest(workers=workers,limit=limit):
                        output=Path(directory)/f'{workers}-{next(iter(limit))}'
                        stderr=io.StringIO()
                        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(stderr):
                            files=create_action_training_data(dataset=data,model_config=model,output_dir=output,
                                num_workers=workers,chunk_size=3,**limit)
                        self.assertEqual(len(files),1)
                        skipped=json.loads((output/'skipped_sources.json').read_text())
                        self.assertEqual(skipped[0]['smiles'],'CCO')
                        self.assertEqual(skipped[0]['record_indexes'],[0,1])
                        self.assertIn('limit exceeded',skipped[0]['reason'])
                        stats=json.loads((output/'action_statistics.json').read_text())
                        self.assertEqual(stats['num_samples'],1)
                        self.assertEqual(stats['num_metadata_valid_records'],3)
                        self.assertEqual(stats['num_skipped_sources'],1)
                        self.assertEqual(stats['num_skipped_records'],2)
                        self.assertTrue((output/'preparation_config.json').is_file())
                        self.assertIn('100%',stderr.getvalue())

    def test_split_cli_creates_both_structure_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'dataset.msds';self.dataset().save(str(file))
            model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
            with contextlib.redirect_stdout(io.StringIO()) as log:
                main(['--input',str(file),'--output-dir',str(root/'out'),'--validation-ratio','0.5',
                      '--params-json',json.dumps(model),'--minimum-relative-intensity','0.1','--normalize-intensities','1','--overwrite','0'])
            self.assertEqual(len(list((root/'out/train_structures').glob('*.preft.pt'))),2)
            self.assertEqual(len(list((root/'out/validation_structures').glob('*.preft.pt'))),2)
            self.assertTrue((root/'out/preparation_config.json').is_file())
            self.assertIn('"event": "progress"',log.getvalue())
            with self.assertRaises(FileExistsError),contextlib.redirect_stdout(io.StringIO()):
                main(['--input',str(file),'--output-dir',str(root/'out'),'--validation-ratio','0.5',
                      '--params-json',json.dumps(model),'--overwrite','0'])

    def test_preflight_rejects_leaking_validation_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'dataset.msds';self.dataset().save(str(file))
            with self.assertRaisesRegex(ValueError,'share SMILES'):
                backend.validate({'modelConfig':config(),'input':str(file),'validationInput':str(file)})

    def test_explicit_validation_uses_main_adduct_and_metadata_parsers(self):
        data=self.dataset()
        # Construct actual metadata with parsable strings, an unsupported main adduct,
        # a neutral loss that resolves to registered [M+H]+, and invalid values.
        metadata=pd.DataFrame({'SMILES':['CCO','CCO','CCN','CCC','invalid'],
            'AdductType':['[M+H-H2O]+','[M+K]+','invalid','[M+H]+','[M+H]+'],
            'CollisionEnergy':['20 eV','30%','invalid','nan','40'],
            'PrecursorMZ':['100.1','200','invalid','300','inf']})
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'data.msds';MSDataset(metadata,data.peaks).save(str(file))
            result=backend.preview({'path':str(file),'validateValues':True})
            self.assertEqual(result['validation']['smilesColumn']['valid'],4)
            self.assertEqual(result['validation']['adductTypeColumn']['valid'],3)
            self.assertEqual(result['validation']['collisionEnergyColumn']['valid'],3)
            self.assertEqual(result['validation']['precursorMzColumn']['valid'],3)
            self.assertEqual(result['excludedRecords'],4)
            self.assertEqual(len(result['invalidRecords'][1]['issues']),3)
            self.assertFalse(result['errors'])

    def test_preparation_context_ignores_neural_dimensions(self):
        from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context
        model=config();model['action_model_params']={'hidden_dim':'unused'}
        model['mol_encoder_params']['node_dim']='unused'
        context=create_preparation_context(model,observed_adducts=['[M+H-H2O]+'])
        from clefts.libs.mmkit.mmkit import Adduct
        self.assertIn(str(Adduct.parse('[M+H-H2O]+')),context.adduct_type_strs)
        self.assertEqual(set(context.mol_encoder.symbols),set(model['mol_encoder_params']['symbols']))
        self.assertFalse(hasattr(context,'feature_model'))

    def test_preflight_accepts_fragmenter_and_symbols_without_a_model(self):
        model=config()
        result=backend.validate({'fragmenterParams':model['fragmenter_params'],'symbols':model['mol_encoder_params']['symbols'],'maxNode':20,'maxEdge':40})
        self.assertTrue(result['valid'])
        with self.assertRaisesRegex(ValueError,'Max node'):
            backend.validate({'fragmenterParams':model['fragmenter_params'],'symbols':['C'],'maxNode':0})

    def test_parallel_preparation_matches_serial_outputs(self):
        from clefts.ml.data_preparation.fragment_tree.create_training_data import create_action_training_data,build_arg_parser
        from clefts.ml.input.source_action_structure import SourceActionStructure
        import torch
        self.assertEqual(build_arg_parser().parse_args(['--input','x.msds','--output-dir','out']).normalize_intensities,1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
            serial=create_action_training_data(dataset=self.dataset(),model_config=model,output_dir=root/'serial')
            parallel=create_action_training_data(dataset=self.dataset(),model_config=model,output_dir=root/'parallel',num_workers=2,chunk_size=2)
            self.assertEqual([file.name for file in serial],[file.name for file in parallel])
            for left,right in zip(serial,parallel):
                first=SourceActionStructure.load(left);second=SourceActionStructure.load(right)
                self.assertTrue(torch.equal(first.condition_features,second.condition_features))
                self.assertTrue(torch.equal(first.teacher_positive_action_index,second.teacher_positive_action_index))
                self.assertTrue(torch.equal(first.downstream.target_intensity,second.downstream.target_intensity))
            stats=json.loads((root/'parallel/action_statistics.json').read_text())
            self.assertEqual(stats['num_workers'],2)
            self.assertTrue(stats['normalize_intensities'])
