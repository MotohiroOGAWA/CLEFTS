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
            self.assertEqual(len(list((root/'cli').rglob('*.preft.pt'))),1)
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
                        manifest=pd.read_csv(output/'manifest.tsv',sep='\t')
                        rejected=manifest[manifest.status=='skipped'].iloc[0]
                        self.assertEqual(rejected.rejected_sample_count,2)
                        self.assertEqual(rejected.num_nodes,0)
                        self.assertEqual(rejected.num_edges,0)
                        self.assertTrue((output/rejected.rejection_log).is_file())
                        self.assertEqual(skipped[0]['record_indexes'],[0,1])
                        self.assertIn('limit exceeded',skipped[0]['reason'])
                        stats=json.loads((output/'action_statistics.json').read_text())
                        self.assertEqual(stats['num_samples'],1)
                        self.assertEqual(stats['num_metadata_valid_records'],3)
                        self.assertEqual(stats['num_skipped_sources'],1)
                        self.assertEqual(stats['num_skipped_records'],2)
                        self.assertTrue((output/'preparation_config.json').is_file())
                        self.assertIn('100%',stderr.getvalue())

    def test_explicit_hydrogen_preparation_preserves_graph_and_action_alignment(self):
        from clefts.ml.input.structure_builder import ActionStructureBuilder
        from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
        from clefts.libs.mmkit.mmkit import Adduct
        import torch
        model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
        generator=create_spectrum_generator(model).eval()
        source=Compound.from_smiles('C([2H])O')
        from clefts.ml.input.source_action_structure import prepare_source_actions
        actions=generator.fragmenter.fragment_ion_tree_builder.create_cleavage_actions(source)
        data=prepare_source_actions(source=source,actions=actions,graph_builder=generator.mol_encoder.graph_builder,
            condition_features=torch.tensor([[0.,20.]]),max_action_count=1)
        self.assertEqual(data.source_graph.num_nodes,2)
        self.assertEqual(data.source_atom_capacity,2)
        self.assertTrue(torch.all(data.action_source_atom_index<2))
        self.assertEqual(data.action_source_atom_features.shape[0],data.action_source_atom_index.numel())
        batch,indices=generator.mol_encoder.encode_batch([source,Compound.from_smiles('[H]')])
        self.assertEqual(indices.tolist(),[0,1])
        self.assertEqual(batch.embeddings.shape[0],2)

    def test_shared_subprocess_runner_reports_completion_order(self):
        import sys
        from clefts.utils.parallel_subprocess import run_parallel_subprocesses
        commands=[[sys.executable,'-c','import time; time.sleep(0.3)'],[sys.executable,'-c','pass']]
        completed=[]
        with contextlib.redirect_stderr(io.StringIO()):
            run_parallel_subprocesses(commands,max_workers=2,on_complete=completed.append)
        self.assertEqual(completed,[commands[1],commands[0]])

    def test_overwrite_removes_all_previous_outputs_and_preserves_complete_root_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);input_file=root/'train.msds';validation_file=root/'validation.msds'
            data=self.dataset();data[[0,1,2]].save(str(input_file));data[[3,4]].save(str(validation_file))
            output=root/'out';(output/'train_structures/stale').mkdir(parents=True)
            (output/'train_structures/stale/old.bin').write_text('stale')
            (output/'orphan.txt').write_text('old')
            (output/'fragment-tree.pft.json').write_text(json.dumps({'validationRatio':0.25}))
            model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main(['--input',str(input_file),'--validation-input',str(validation_file),'--output-dir',str(output),
                    '--params-json',json.dumps(model),'--max-node','100','--max-edge','200','--overwrite','1',
                    '--normalize-intensities','0','--minimum-relative-intensity','0.2','--validation-seed','7'])
            self.assertFalse((output/'orphan.txt').exists())
            self.assertFalse((output/'train_structures/stale').exists())
            saved=json.loads((output/'train_structures/fragment-tree.pft.json').read_text())
            self.assertEqual(saved['validationInput'],str(validation_file))
            self.assertEqual(saved['validationRatio'],0.25)
            self.assertEqual(saved['validationSeed'],7)
            self.assertEqual(saved['maxNode'],100)
            self.assertEqual(saved['maxEdge'],200)
            self.assertFalse(saved['normalizeIntensities'])
            self.assertEqual(saved['minimumRelativeIntensity'],0.2)
            self.assertFalse(list(output.glob('*/preparation_config.json')))

    def test_assignment_scores_handle_precursor_only_and_zero_intensity(self):
        from clefts.libs.mmkit.mmkit import Adduct, Formula
        from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context
        from clefts.ml.input.structure_builder import ActionStructureBuilder
        from clefts.ml.input.source_action_structure import SourceActionStructure
        model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
        builder=ActionStructureBuilder(create_preparation_context(model))
        source=Compound.from_smiles('CCO');adduct=Adduct.parse('[M+H]+')
        mz=Formula.parse('C2H7O+').exact_mass
        data=builder.build(source,[adduct,adduct],[20.,20.],[[mz],[mz]],[[1.],[0.]])
        first,second=data.sample_annotations
        self.assertEqual(first['assignmentScore'],1.)
        self.assertIsNone(first['assignmentScoreWithoutPrecursor'])
        self.assertIsNone(second['assignmentScore'])
        self.assertIsNone(second['assignmentScoreWithoutPrecursor'])
        self.assertGreater(len(first['peaks'][0]['matches']),0)
        collated=SourceActionStructure.from_structures([data,data])
        self.assertEqual(len(collated.sample_annotations),4)
        offset=len(data.downstream.decoded.compounds)
        original=first['peaks'][0]['matches'][0]['nodeIndices']
        shifted=collated.sample_annotations[2]['peaks'][0]['matches'][0]['nodeIndices']
        self.assertEqual(shifted,[node+offset for node in original])

    def test_result_inspection_uses_unique_fragment_nodes_and_action_references(self):
        from clefts.libs.mmkit.mmkit import Formula
        from clefts.ml.data_preparation.fragment_tree.create_training_data import create_action_training_data
        result_spec=importlib.util.spec_from_file_location('result_backend','clefts_workbench/src/features/fragment-tree-result/backend.py')
        result_backend=importlib.util.module_from_spec(result_spec);result_spec.loader.exec_module(result_backend)
        peaks=np.array([[Formula.parse(value).exact_mass,10. if value=='C2H7O+' else 1.] for value in ['C2H7O+','CH5O+','H3O+','C2H7+']]+[[1000.,2.]])
        data=MSDataset(pd.DataFrame({'SMILES':['CCO'],'AdductType':['[M+H]+'],'CollisionEnergy':[20.25],'PrecursorMZ':[47]}),PeakSeries(peaks,np.array([0,5],dtype=np.int64)))
        model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            files=create_action_training_data(dataset=data,model_config=model,output_dir=directory)
            result=result_backend.inspect_structure(files[0])
            scores=pd.read_csv(Path(directory)/'assignment_scores.tsv',sep='\t')
            self.assertEqual(len(scores),1)
            manifest=pd.read_csv(Path(directory)/'manifest.tsv',sep='\t').iloc[0]
            self.assertEqual(manifest.num_nodes,4)
            self.assertEqual(manifest.num_edges,3)
            self.assertEqual(manifest.rejected_sample_count,0)
            self.assertAlmostEqual(manifest.assignment_score,13/15)
            self.assertAlmostEqual(manifest.assignment_score_without_precursor,3/5)
            recovered=backend.structure_manifest({'directory':directory})[0]
            self.assertEqual(recovered['num_nodes'],4)
            self.assertEqual(recovered['num_edges'],3)
            self.assertAlmostEqual(recovered['assignment_score'],13/15)
            self.assertAlmostEqual(scores.assignment_score.iloc[0],13/15)
            self.assertAlmostEqual(scores.assignment_score_without_precursor.iloc[0],3/5)
        sample=result['samples'][0]
        self.assertEqual(sample['adduct'],'[M+H]+')
        self.assertEqual(sample['mainAdduct'],'[M+H]+')
        self.assertEqual(sample['collisionEnergyDisplay'],'20.3')
        self.assertEqual(len(sample['peaks']),5)
        self.assertEqual(sample['peaks'][-1]['matches'],[])
        self.assertTrue(sample['peaks'][0]['precursor'])
        self.assertAlmostEqual(sample['assignmentScore'],13/15)
        self.assertAlmostEqual(sample['assignmentScoreWithoutPrecursor'],3/5)
        self.assertTrue(all('hydrogenShift' in match and match['nodeIds'] for peak in sample['peaks'][:-1] for match in peak['matches']))
        self.assertEqual(len({node['id'] for node in sample['nodes']}),sample['nodeCount'])
        self.assertGreater(sample['edgeCount'],0)
        self.assertTrue(next(node for node in sample['nodes'] if node['id']==0)['precursor'])
        self.assertEqual(result['summary']['nodes'],sample['nodeCount'])
        self.assertEqual(result['summary']['edgeTransitions'],sample['edgeTransitionCount'])
        self.assertTrue(all(isinstance(node['actionSets'],list) for node in sample['nodes']))
        registry={action['id'] for action in result['actions']}
        self.assertTrue(all(transition['addedAction'] in registry for edge in sample['edges'] for transition in edge['transitions']))
        self.assertTrue(all(action['sourceAtomMaps'] for action in result['actions']))
        self.assertIn('<svg',result['sourceSvg'])

    def test_split_cli_creates_both_structure_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'dataset.msds';self.dataset().save(str(file))
            model=config();model['fragmenter_params']['fragment_ion_tree_builder']['max_action_count']=1
            with contextlib.redirect_stdout(io.StringIO()) as log:
                main(['--input',str(file),'--output-dir',str(root/'out'),'--validation-ratio','0.5',
                      '--params-json',json.dumps(model),'--minimum-relative-intensity','0.1','--normalize-intensities','1','--overwrite','0'])
            self.assertEqual(len(list((root/'out/train_structures').rglob('*.preft.pt'))),2)
            self.assertEqual(len(list((root/'out/validation_structures').rglob('*.preft.pt'))),2)
            self.assertTrue((root/'out/preparation_config.json').is_file())
            self.assertFalse((root/'out/train_structures/preparation_config.json').exists())
            self.assertFalse((root/'out/validation_structures/preparation_config.json').exists())
            self.assertFalse((root/'out/fragment-tree.pft.json').exists())
            self.assertTrue((root/'out/train_structures/fragment-tree.pft.json').exists())
            self.assertTrue((root/'out/validation_structures/fragment-tree.pft.json').exists())
            for split in ('train','validation'):
                folder=root/'out'/f'{split}_structures'
                self.assertEqual(len(list((folder/'data').glob('*.preft.pt'))),2)
                rows=pd.read_csv(folder/'manifest.tsv',sep='\t')
                self.assertEqual(len(rows),2)
                self.assertTrue(all((folder/'data'/file).exists() for file in rows['file']))
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
