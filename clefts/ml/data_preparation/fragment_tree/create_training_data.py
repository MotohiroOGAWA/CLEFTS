"""Regenerate schema-v6 branching teachers directly from original MSDataset."""
from __future__ import annotations
import argparse,json,math,sys,os,pickle,tempfile,shutil,csv
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from clefts.utils.parallel_subprocess import run_parallel_subprocesses
from clefts.domain.fragment.tree.FragmentTreeBuilder import FragmentTreeLimitExceeded
from clefts.libs.mmkit.mmkit import Compound,Adduct
from clefts.libs.msentity.msentity import MSDataset
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.input.source_action_structure import save_fragment_tree_structure,make_structure_file_stem,UnresolvedPrecursorError
from .manifest_summary import structure_manifest_fields
from .context import create_preparation_context, validate_limits
from clefts.ml.specgen.config_options import configure_model_options, resolve_model_options
from .datasets import load_spectrum_dataset, split_by_smiles, dedupe_validation
from .record_validation import inspect_records


def _reset_output(output, overwrite, protected_paths=()):
    output=Path(output).resolve()
    if overwrite and output.exists():
        for protected in [Path(__file__).resolve(),*map(lambda value:Path(value).resolve(),protected_paths)]:
            if protected.is_relative_to(output):
                raise ValueError('Output directory contains input or application files; choose a separate output directory.')
        shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)


def _prepare_group(task, builder, options):
    tree,smiles,rows,dataset=task
    output=Path(options['output_dir'])
    adduct_type_column=options['adduct_type_column'];collision_energy_column=options['collision_energy_column'];precursor_mz_column=options['precursor_mz_column']
    minimum_relative_intensity=options['minimum_relative_intensity'];normalize_intensities=options['normalize_intensities']
    adducts=[];energies=[];mzs=[];intensities=[]
    for index in range(len(dataset)):
        record=dataset[index]
        adducts.append(Adduct.parse(str(dataset[adduct_type_column].iloc[index])))
        precursor_mz=float(dataset[precursor_mz_column].iloc[index])
        if not math.isfinite(precursor_mz) or precursor_mz<=0: raise ValueError(f'Invalid precursor m/z at record {rows[index]}')
        ev=parse_ce_to_ev(dataset[collision_energy_column].iloc[index],precursor_mz)
        if ev is None or not math.isfinite(float(ev)) or float(ev)<0:raise ValueError(f'Cannot convert CE at record {rows[index]}')
        energies.append(float(ev))
        peaks=record.peaks
        values=[(float(peak.mz),float(peak.intensity)) for peak in peaks]
        if any(not math.isfinite(mz) or mz<=0 or not math.isfinite(value) or value<0 for mz,value in values):
            raise ValueError(f'Invalid peak values at record {rows[index]}')
        maximum=max((value for _,value in values),default=0.0)
        retained=[(mz,value) for mz,value in values if value>=maximum*minimum_relative_intensity]
        mzs.append([mz for mz,_ in retained])
        intensities.append([value/maximum if normalize_intensities and maximum>0 else value for _,value in retained])
    structure,kept=builder.build(Compound.from_smiles(smiles),adducts,energies,mzs,intensities)
    kept_rows=[rows[index] for index in kept]
    path=output/'data'/(make_structure_file_stem(smiles,index=tree)+'.preft.pt')
    save_fragment_tree_structure(structure=structure,output_file=path,metadata=dict(smiles=smiles,record_indexes=kept_rows,sample_annotations=structure.sample_annotations))
    return path,kept_rows,structure.sample_annotations,structure_manifest_fields(structure)


def _prepare_group_safely(task, builder, options):
    try:
        path,kept_rows,annotations,summary=_prepare_group(task,builder,options)
        scores=[dict(structure_file=path.name,sample_index=index,record_index=record_index,
                     assignment_score=sample['assignmentScore'],assignment_score_without_precursor=sample['assignmentScoreWithoutPrecursor'])
                for index,(record_index,sample) in enumerate(zip(kept_rows,annotations))]
        rejected=len(task[2])-len(kept_rows)
        reason=f'{rejected} of {len(task[2])} records dropped: unresolved precursor action sequence' if rejected else ''
        return dict(path=path,record_count=len(kept_rows),input_record_count=len(task[2]),rejected_record_count=rejected,
                    smiles=task[1],record_indexes=kept_rows,assignment_scores=scores,summary=summary,reason=reason)
    except (FragmentTreeLimitExceeded,UnresolvedPrecursorError) as error:
        return dict(skipped=dict(source_index=task[0],smiles=task[1],record_indexes=task[2],reason=str(error)))


def create_action_training_data(*, dataset: MSDataset, model_config: dict, output_dir: str | Path,
                                smiles_column: str = 'SMILES', adduct_type_column: str = 'AdductType',
                                collision_energy_column: str = 'CollisionEnergy', precursor_mz_column: str = 'PrecursorMZ',
                                minimum_relative_intensity: float = 0.0, normalize_intensities: bool = True,
                                overwrite: bool = True, split: str = 'dataset', max_node: int = -1, max_edge: int = -1, num_workers: int = 1, chunk_size: int = 1, preparation_config: dict | None = None,
                                _check_existing: bool = True) -> list[Path]:
    if not math.isfinite(minimum_relative_intensity) or not 0<=minimum_relative_intensity<=1:
        raise ValueError('Minimum relative intensity must be between 0 and 1.')
    validate_limits(max_node,max_edge)
    if type(num_workers) is not int or num_workers<1 or type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Worker processes and chunk size must be positive integers.')
    inspection=inspect_records(dataset,create_preparation_context(model_config).fragmenter,
        dict(smilesColumn=smiles_column,adductTypeColumn=adduct_type_column,
             collisionEnergyColumn=collision_energy_column,precursorMzColumn=precursor_mz_column))
    original_rows=inspection['validIndexes']
    dataset=dataset[original_rows]
    if not len(dataset): raise ValueError('The dataset contains no valid records to prepare.')
    generator=create_preparation_context(model_config,observed_adducts=dataset[adduct_type_column].unique())
    builder=ActionStructureBuilder(generator,max_node=max_node,max_edge=max_edge)
    grouped={}
    for index,smiles in enumerate(dataset[smiles_column].tolist()):grouped.setdefault(str(smiles),[]).append(index)
    output=Path(output_dir)
    # main() already verified (and, with --overwrite, cleared) the whole output tree
    # before any split started computing; skip this redundant re-check there via
    # _check_existing=False so it can never abort a run mid-way, after training's
    # (possibly expensive) split has already completed, over a directory main()
    # already accounted for. overwrite still stays False for that caller so this
    # call's own _reset_output does not delete the split config main() just wrote.
    if _check_existing and not overwrite:
        existing=list(output.rglob('*.preft.pt'))
        if existing: raise FileExistsError('Output already contains training structures. Enable overwrite or choose another directory.')
    _reset_output(output,overwrite)
    (output/'invalid_records.json').write_text(json.dumps(inspection['invalidRecords'],indent=2))
    options=dict(output_dir=str(output),adduct_type_column=adduct_type_column,collision_energy_column=collision_energy_column,
        precursor_mz_column=precursor_mz_column,minimum_relative_intensity=minimum_relative_intensity,
        normalize_intensities=normalize_intensities,max_node=max_node,max_edge=max_edge)
    tasks=((tree,smiles,[original_rows[index] for index in rows],dataset[rows]) for tree,(smiles,rows) in enumerate(grouped.items()))
    if preparation_config is None:
        (output/'preparation_config.json').write_text(json.dumps(dict(split=split,model_config=model_config,
            smiles_column=smiles_column,num_workers=num_workers,chunk_size=chunk_size,overwrite=overwrite,**options),indent=2))
    files=[];skipped=[];prepared_records=0;manifest_rows=[];score_rows=[]
    completed_sources=0
    def collect_result(result):
        nonlocal prepared_records,completed_sources
        completed_sources+=1
        if 'skipped' in result:
            source=result['skipped']
            manifest_rows.append(dict(file='',smiles=source['smiles'],record_indexes=json.dumps(source['record_indexes']),num_input_records=len(source['record_indexes']),num_valid_samples=0,rejected_sample_count=len(source['record_indexes']),rejection_log='skipped_sources.json',num_nodes=0,num_edges=0,assignment_score=None,assignment_score_without_precursor=None,status='skipped',reason=source['reason']))
            skipped.append(result['skipped'])
            print(json.dumps(dict(event='source_skipped',split=split,**result['skipped'])),flush=True)
        else:
            files.append(Path(result['path']));prepared_records+=result['record_count']
            score_rows.extend(result['assignment_scores'])
            manifest_rows.append(dict(file=Path(result['path']).name,smiles=result['smiles'],record_indexes=json.dumps(result['record_indexes']),num_input_records=result['input_record_count'],num_valid_samples=result['record_count'],rejected_sample_count=result['rejected_record_count'],rejection_log='',**result['summary'],status='completed',reason=result['reason']))
        print(json.dumps(dict(event='progress',split=split,current=completed_sources,total=len(grouped),
                              prepared=len(files),skipped=len(skipped))),flush=True)

    if num_workers==1 or len(grouped)<2:
        with tqdm(total=len(grouped),desc=f'Fragment trees ({split})',unit='tree',file=sys.stderr) as progress:
            for task in tasks:
                result=_prepare_group_safely(task,builder,options)
                collect_result(result)
                progress.set_postfix(prepared=len(files),skipped=len(skipped),refresh=False)
                progress.update(1)
    else:
        # Use the shared subprocess runner; only temporary task files cross the boundary.
        with tempfile.TemporaryDirectory(prefix='clefts-preparation-') as directory:
            commands=[]
            effective_chunk_size=min(chunk_size,max(1,len(grouped)//num_workers))
            while True:
                chunk=list(islice(tasks,effective_chunk_size))
                if not chunk: break
                task_file=Path(directory)/f'task-{len(commands)}.pkl'
                result_file=task_file.with_suffix('.json')
                with task_file.open('wb') as stream:
                    pickle.dump(dict(tasks=chunk,model_config=model_config,observed_adducts=generator.adduct_type_strs,
                                     options=options),stream,pickle.HIGHEST_PROTOCOL)
                commands.append([sys.executable,'-m','clefts.ml.data_preparation.fragment_tree.subprocess_worker',
                                 '--task',str(task_file),'--result',str(result_file)])
            def on_complete(command):
                for result in json.loads(Path(command[-1]).read_text()): collect_result(result)
            env={**os.environ,'PYTHONUNBUFFERED':'1'}
            root=str(Path(__file__).resolve().parents[4])
            env['PYTHONPATH']=os.pathsep.join(filter(None,[root,env.get('PYTHONPATH')]))
            run_parallel_subprocesses(commands,max_workers=min(num_workers,len(commands)),print_output=False,
                                      env=env,desc=f'Fragment trees ({split})',unit='chunk',on_complete=on_complete)
    with (output/'manifest.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['file','smiles','record_indexes','num_input_records','num_valid_samples','rejected_sample_count','rejection_log','num_teacher_nodes','num_positive_transitions','num_precursor_candidates','max_ms2_depth','num_nodes','num_edges','assignment_score','assignment_score_without_precursor','status','reason'],delimiter='\t')
        writer.writeheader();writer.writerows(sorted(manifest_rows,key=lambda row:row['smiles']))
    with (output/'assignment_scores.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['structure_file','sample_index','record_index','assignment_score','assignment_score_without_precursor'],delimiter='\t')
        writer.writeheader();writer.writerows(sorted(score_rows,key=lambda row:(row['structure_file'],row['sample_index'])))
    (output/'skipped_sources.json').write_text(json.dumps(skipped,indent=2))
    (output/'action_statistics.json').write_text(json.dumps(dict(schema_version=6,fragmentation_schema=generator.architecture,
        num_teacher_nodes=sum(row.get('num_teacher_nodes',0) or 0 for row in manifest_rows),
        num_positive_transitions=sum(row.get('num_positive_transitions',0) or 0 for row in manifest_rows),
        mean_teacher_nodes_per_sample=sum(row.get('num_teacher_nodes',0) or 0 for row in manifest_rows)/max(prepared_records,1),
        mean_positive_edges_per_sample=sum(row.get('num_positive_transitions',0) or 0 for row in manifest_rows)/max(prepared_records,1),
        mean_precursor_candidates=sum(row.get('num_precursor_candidates',0) or 0 for row in manifest_rows)/max(prepared_records,1),
        max_teacher_ms2_depth=max((row.get('max_ms2_depth',0) or 0 for row in manifest_rows),default=0),
        num_sources=len(files),num_samples=prepared_records,num_metadata_valid_records=len(dataset),num_skipped_sources=len(skipped),
        num_skipped_records=sum(len(source['record_indexes']) for source in skipped),
        num_rejected_records=sum(row['rejected_sample_count'] for row in manifest_rows if row['status']=='completed'),
        num_excluded_records=inspection['excludedRecords'],num_workers=num_workers,chunk_size=chunk_size,model_config=model_config,minimum_relative_intensity=minimum_relative_intensity,normalize_intensities=normalize_intensities),indent=2))
    return sorted(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output-dir',required=True)
    configure_model_options(parser)
    for name,default in (('smiles','SMILES'),('adduct-type','AdductType'),('collision-energy','CollisionEnergy'),('precursor-mz','PrecursorMZ')):
        parser.add_argument('--'+name+'-column',default=default,help='Dataset metadata column.')
        parser.add_argument('--validation-'+name+'-column',help='Validation dataset column, if it differs from --'+name+'-column.')
    parser.add_argument('--symbols-json',help='JSON array of selected element symbols; overrides imported symbols.')
    parser.add_argument('--max-node',type=int,help='Maximum unique fragment nodes, including source; -1 is unlimited.')
    parser.add_argument('--max-edge',type=int,help='Maximum total distinct cleavage transitions, including multiple routes between merged nodes; -1 is unlimited.')
    parser.add_argument('--num-workers',type=int,default=1,help='Parallel worker processes per split; 1 runs serially.')
    parser.add_argument('--chunk-size',type=int,default=1,help='SMILES groups dispatched per worker chunk.')
    parser.add_argument('--validation-input',help='Optional held-out spectrum dataset. Produces train_structures and validation_structures.')
    parser.add_argument('--validation-ratio',type=float,help='Split unique SMILES when validation input is omitted. With --validation-input, instead caps validation to this fraction of training\'s unique SMILES count after removing molecules shared with training.')
    parser.add_argument('--validation-seed',type=int,default=0)
    parser.add_argument('--minimum-relative-intensity',type=float,default=0.0)
    parser.add_argument('--normalize-intensities',type=int,choices=(0,1),default=1)
    parser.add_argument('--overwrite',type=int,choices=(0,1),default=0,help='Allow replacing existing structure files (0 or 1).')
    return parser


def main(argv: list[str] | None = None) -> None:
    args=build_arg_parser().parse_args(argv)
    dataset=load_spectrum_dataset(args.input)
    config=resolve_model_options(args)
    imported_max_node=config.pop('max_node',-1);imported_max_edge=config.pop('max_edge',-1)
    args.max_node=args.max_node if args.max_node is not None else imported_max_node
    args.max_edge=args.max_edge if args.max_edge is not None else imported_max_edge
    validate_limits(args.max_node,args.max_edge)
    imported_symbols=config.pop('symbols',None)
    if args.symbols_json: config['mol_encoder_params']['symbols']=json.loads(args.symbols_json)
    elif imported_symbols is not None: config['mol_encoder_params']['symbols']=imported_symbols
    # Exclude invalid metadata before collecting adducts or splitting molecules.
    validation_dataset=load_spectrum_dataset(args.validation_input) if args.validation_input else None
    checker=create_preparation_context(config).fragmenter
    mapping=dict(smilesColumn=args.smiles_column,adductTypeColumn=args.adduct_type_column,
                 collisionEnergyColumn=args.collision_energy_column,precursorMzColumn=args.precursor_mz_column)
    # Falls back to the training column name for anything not explicitly
    # overridden; only takes effect when validation is a separate dataset
    # (--validation-input) since an auto ratio split reuses the same columns.
    validation_mapping=dict(smilesColumn=args.validation_smiles_column or args.smiles_column,
        adductTypeColumn=args.validation_adduct_type_column or args.adduct_type_column,
        collisionEnergyColumn=args.validation_collision_energy_column or args.collision_energy_column,
        precursorMzColumn=args.validation_precursor_mz_column or args.precursor_mz_column)
    reports={}
    for name,data in (('train',dataset),('validation',validation_dataset)):
        if data is None: continue
        report=inspect_records(data,checker,mapping if name=='train' else validation_mapping)
        reports[name]=report['invalidRecords']
        filtered=data[report['validIndexes']]
        if not len(filtered): raise ValueError(name+' dataset contains no valid records to prepare.')
        if name=='train': dataset=filtered
        else: validation_dataset=filtered
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(event='excluded_records',counts={name:len(rows) for name,rows in reports.items()})),flush=True)
    observed=list(dataset[args.adduct_type_column].unique())
    if validation_dataset is not None: observed.extend(validation_dataset[validation_mapping['adductTypeColumn']].unique())
    config['adduct_type_strs']=list(create_preparation_context(config,observed_adducts=observed).adduct_type_strs)
    kwargs=dict(model_config=config,smiles_column=args.smiles_column,adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,precursor_mz_column=args.precursor_mz_column,
        minimum_relative_intensity=args.minimum_relative_intensity,normalize_intensities=bool(args.normalize_intensities),
        overwrite=bool(args.overwrite),max_node=args.max_node,max_edge=args.max_edge,num_workers=args.num_workers,chunk_size=args.chunk_size)
    preparation_config=dict(input=args.input,validation_input=args.validation_input,
        validation_ratio=args.validation_ratio,validation_seed=args.validation_seed,output_dir=args.output_dir,
        # Blank means "same as the training column"; only a real override is persisted,
        # so reloading a saved configuration keeps showing it as inherited, not explicit.
        validation_smiles_column=args.validation_smiles_column or '',validation_adduct_type_column=args.validation_adduct_type_column or '',
        validation_collision_energy_column=args.validation_collision_energy_column or '',validation_precursor_mz_column=args.validation_precursor_mz_column or '',
        **kwargs)
    kwargs['preparation_config']=preparation_config
    validation_kwargs={**kwargs,'smiles_column':validation_mapping['smilesColumn'],'adduct_type_column':validation_mapping['adductTypeColumn'],
        'collision_energy_column':validation_mapping['collisionEnergyColumn'],'precursor_mz_column':validation_mapping['precursorMzColumn']}
    if args.validation_input:
        train=dataset
        # A model must not be scored on a molecule it trained on, so overlapping
        # SMILES are always dropped from validation rather than failing the run;
        # --validation-ratio (optional here) then caps validation to a size
        # proportional to training instead of using every remaining record.
        validation,overlap_report=dedupe_validation(train,validation_dataset,args.smiles_column,
            validation_mapping['smilesColumn'],ratio=args.validation_ratio,seed=args.validation_seed)
        if overlap_report['removedOverlapRecords']:
            print(json.dumps(dict(event='validation_overlap_removed',**overlap_report)),flush=True)
        if 'targetSmiles' in overlap_report and overlap_report['remainingSmiles']<overlap_report['targetSmiles']:
            print(json.dumps(dict(event='validation_below_target',**overlap_report,
                note='Validation has fewer molecule-disjoint SMILES than --validation-ratio requests; using all available records instead of failing.')),flush=True)
        if not len(validation):
            print(json.dumps(dict(event='validation_empty',
                note='No validation records remain after removing molecules shared with training; continuing with training only.')),flush=True)
            validation=None
    elif args.validation_ratio is not None:
        train,validation=split_by_smiles(dataset,args.smiles_column,args.validation_ratio,args.validation_seed)
    else:
        train,validation=dataset,None
    if not args.overwrite and (list(output.rglob('*.preft.pt')) or any(list((output/name).rglob('*.preft.pt')) for name in ('train_structures','validation_structures'))):
        raise FileExistsError('Output already contains training structures. Enable overwrite or choose another directory.')
    workbench_path=output/'fragment-tree.pft.json'
    try:
        if workbench_path.exists(): previous=json.loads(workbench_path.read_text())
        elif (output/'preparation_config.json').exists(): previous=json.loads((output/'preparation_config.json').read_text()).get('workbench_config',{})
        elif (output/'train_structures/fragment-tree.pft.json').exists(): previous=json.loads((output/'train_structures/fragment-tree.pft.json').read_text())
        else: previous={}
        if not isinstance(previous,dict): previous={}
    except (OSError,json.JSONDecodeError):
        previous={}
    _reset_output(output,bool(args.overwrite),[value for value in (args.input,args.validation_input,args.params) if value])
    workbench_path.unlink(missing_ok=True)
    aliases={'validation_input':'validationInput','validation_ratio':'validationRatio','validation_seed':'validationSeed',
        'output_dir':'outputDir','smiles_column':'smilesColumn','adduct_type_column':'adductTypeColumn',
        'collision_energy_column':'collisionEnergyColumn','precursor_mz_column':'precursorMzColumn',
        'validation_smiles_column':'validationSmilesColumn','validation_adduct_type_column':'validationAdductTypeColumn',
        'validation_collision_energy_column':'validationCollisionEnergyColumn','validation_precursor_mz_column':'validationPrecursorMzColumn',
        'minimum_relative_intensity':'minimumRelativeIntensity','normalize_intensities':'normalizeIntensities',
        'max_node':'maxNode','max_edge':'maxEdge','num_workers':'numWorkers','chunk_size':'chunkSize'}
    restored={**previous,'application':'fragment-tree-data-preparation'}
    for key,value in preparation_config.items():
        if key=='model_config': continue
        if key=='validation_ratio' and value is None: continue
        restored[aliases.get(key,key)]=value
    restored.update(fragmenterParams=config['fragmenter_params'],symbols=config['mol_encoder_params']['symbols'])
    restored.pop('modelConfig',None);restored.pop('params',None)
    preparation_config['workbench_config']=restored
    (output/'preparation_config.json').write_text(json.dumps(preparation_config,indent=2))
    splits=[('train',train)]
    if validation is not None: splits.append(('validation',validation))
    for name,split_dataset in splits:
        directory=output/(name+'_structures')
        directory.mkdir(parents=True,exist_ok=True)
        split_config={**restored,'split':name,'status':'running'}
        (directory/'fragment-tree.pft.json').write_text(json.dumps(split_config,indent=2))
        # The split is newly empty after resetting Output Directory. Do not delete its configuration.
        create_action_training_data(dataset=split_dataset,output_dir=directory,split=name,**{**(kwargs if name=='train' else validation_kwargs),'overwrite':False},_check_existing=False)
        split_config['status']='completed'
        (directory/'fragment-tree.pft.json').write_text(json.dumps(split_config,indent=2))
    (output/'invalid_records.json').write_text(json.dumps(reports,indent=2))
    (output/'preparation_config.json').write_text(json.dumps(preparation_config,indent=2))

if __name__=='__main__':main()
