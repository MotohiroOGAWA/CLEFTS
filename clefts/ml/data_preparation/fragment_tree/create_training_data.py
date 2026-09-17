"""Regenerate schema-v4 action teachers directly from original MSDataset."""
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
from clefts.ml.input.source_action_structure import save_fragment_tree_structure,make_structure_file_stem
from .context import create_preparation_context, validate_limits
from clefts.ml.specgen.config_options import configure_model_options, resolve_model_options
from .datasets import load_spectrum_dataset, split_by_smiles
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
    structure=builder.build(Compound.from_smiles(smiles),adducts,energies,mzs,intensities)
    path=output/'data'/(make_structure_file_stem(smiles,index=tree)+'.preft.pt')
    save_fragment_tree_structure(structure=structure,output_file=path,metadata=dict(smiles=smiles,record_indexes=rows))
    return path


def _prepare_group_safely(task, builder, options):
    try:
        return dict(path=_prepare_group(task,builder,options),record_count=len(task[2]),smiles=task[1],record_indexes=task[2])
    except FragmentTreeLimitExceeded as error:
        return dict(skipped=dict(source_index=task[0],smiles=task[1],record_indexes=task[2],reason=str(error)))


def create_action_training_data(*, dataset: MSDataset, model_config: dict, output_dir: str | Path,
                                smiles_column: str = 'SMILES', adduct_type_column: str = 'AdductType',
                                collision_energy_column: str = 'CollisionEnergy', precursor_mz_column: str = 'PrecursorMZ',
                                minimum_relative_intensity: float = 0.0, normalize_intensities: bool = True,
                                overwrite: bool = True, split: str = 'dataset', max_node: int = -1, max_edge: int = -1, num_workers: int = 1, chunk_size: int = 1, preparation_config: dict | None = None) -> list[Path]:
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
    if not overwrite:
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
    files=[];skipped=[];prepared_records=0;manifest_rows=[]
    completed_sources=0
    def collect_result(result):
        nonlocal prepared_records,completed_sources
        completed_sources+=1
        if 'skipped' in result:
            source=result['skipped']
            manifest_rows.append(dict(file='',smiles=source['smiles'],record_indexes=json.dumps(source['record_indexes']),num_input_records=len(source['record_indexes']),num_valid_samples=0,status='skipped',reason=source['reason']))
            skipped.append(result['skipped'])
            print(json.dumps(dict(event='source_skipped',split=split,**result['skipped'])),flush=True)
        else:
            files.append(Path(result['path']));prepared_records+=result['record_count']
            manifest_rows.append(dict(file=Path(result['path']).name,smiles=result['smiles'],record_indexes=json.dumps(result['record_indexes']),num_input_records=result['record_count'],num_valid_samples=result['record_count'],status='completed',reason=''))
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
        writer=csv.DictWriter(stream,fieldnames=['file','smiles','record_indexes','num_input_records','num_valid_samples','status','reason'],delimiter='\t')
        writer.writeheader();writer.writerows(sorted(manifest_rows,key=lambda row:row['smiles']))
    (output/'skipped_sources.json').write_text(json.dumps(skipped,indent=2))
    (output/'action_statistics.json').write_text(json.dumps(dict(schema_version=4,fragmentation_schema=generator.architecture,
        num_sources=len(files),num_samples=prepared_records,num_metadata_valid_records=len(dataset),num_skipped_sources=len(skipped),
        num_skipped_records=sum(len(source['record_indexes']) for source in skipped),num_excluded_records=inspection['excludedRecords'],num_workers=num_workers,chunk_size=chunk_size,model_config=model_config,minimum_relative_intensity=minimum_relative_intensity,normalize_intensities=normalize_intensities),indent=2))
    return sorted(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output-dir',required=True)
    configure_model_options(parser)
    for name,default in (('smiles','SMILES'),('adduct-type','AdductType'),('collision-energy','CollisionEnergy'),('precursor-mz','PrecursorMZ')):
        parser.add_argument('--'+name+'-column',default=default,help='Dataset metadata column.')
    parser.add_argument('--symbols-json',help='JSON array of selected element symbols; overrides imported symbols.')
    parser.add_argument('--max-node',type=int,help='Maximum unique fragment nodes, including source; -1 is unlimited.')
    parser.add_argument('--max-edge',type=int,help='Maximum total distinct cleavage transitions, including multiple routes between merged nodes; -1 is unlimited.')
    parser.add_argument('--num-workers',type=int,default=1,help='Parallel worker processes per split; 1 runs serially.')
    parser.add_argument('--chunk-size',type=int,default=1,help='SMILES groups dispatched per worker chunk.')
    parser.add_argument('--validation-input',help='Optional held-out spectrum dataset. Produces train_structures and validation_structures.')
    parser.add_argument('--validation-ratio',type=float,help='Split unique SMILES when validation input is omitted.')
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
    reports={}
    for name,data in (('train',dataset),('validation',validation_dataset)):
        if data is None: continue
        report=inspect_records(data,checker,mapping)
        reports[name]=report['invalidRecords']
        filtered=data[report['validIndexes']]
        if not len(filtered): raise ValueError(name+' dataset contains no valid records to prepare.')
        if name=='train': dataset=filtered
        else: validation_dataset=filtered
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(event='excluded_records',counts={name:len(rows) for name,rows in reports.items()})),flush=True)
    observed=list(dataset[args.adduct_type_column].unique())
    if validation_dataset is not None: observed.extend(validation_dataset[args.adduct_type_column].unique())
    config['adduct_type_strs']=list(create_preparation_context(config,observed_adducts=observed).adduct_type_strs)
    kwargs=dict(model_config=config,smiles_column=args.smiles_column,adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,precursor_mz_column=args.precursor_mz_column,
        minimum_relative_intensity=args.minimum_relative_intensity,normalize_intensities=bool(args.normalize_intensities),
        overwrite=bool(args.overwrite),max_node=args.max_node,max_edge=args.max_edge,num_workers=args.num_workers,chunk_size=args.chunk_size)
    if args.validation_input and args.validation_ratio is not None:
        raise ValueError('Use a separate validation input or a SMILES split, not both.')
    preparation_config=dict(input=args.input,validation_input=args.validation_input,
        validation_ratio=args.validation_ratio,validation_seed=args.validation_seed,output_dir=args.output_dir,**kwargs)
    kwargs['preparation_config']=preparation_config
    if args.validation_input:
        train,validation=dataset,validation_dataset
    elif args.validation_ratio is not None:
        train,validation=split_by_smiles(dataset,args.smiles_column,args.validation_ratio,args.validation_seed)
    else:
        train,validation=dataset,None
    if validation is not None and set(train[args.smiles_column].astype(str)) & set(validation[args.smiles_column].astype(str)):
        raise ValueError('Training and validation datasets share SMILES. Choose molecule-disjoint datasets.')
    if not args.overwrite and (list(output.rglob('*.preft.pt')) or any(list((output/name).rglob('*.preft.pt')) for name in ('train_structures','validation_structures'))):
        raise FileExistsError('Output already contains training structures. Enable overwrite or choose another directory.')
    workbench_path=output/'fragment-tree.pft.json'
    try:
        previous=json.loads(workbench_path.read_text()) if workbench_path.exists() else {}
        if not isinstance(previous,dict): previous={}
    except (OSError,json.JSONDecodeError):
        previous={}
    _reset_output(output,bool(args.overwrite),[value for value in (args.input,args.validation_input,args.params) if value])
    aliases={'validation_input':'validationInput','validation_ratio':'validationRatio','validation_seed':'validationSeed',
        'output_dir':'outputDir','smiles_column':'smilesColumn','adduct_type_column':'adductTypeColumn',
        'collision_energy_column':'collisionEnergyColumn','precursor_mz_column':'precursorMzColumn',
        'minimum_relative_intensity':'minimumRelativeIntensity','normalize_intensities':'normalizeIntensities',
        'max_node':'maxNode','max_edge':'maxEdge','num_workers':'numWorkers','chunk_size':'chunkSize'}
    restored={**previous,'application':'fragment-tree-data-preparation'}
    for key,value in preparation_config.items():
        if key=='model_config': continue
        if key=='validation_ratio' and value is None: continue
        restored[aliases.get(key,key)]=value
    restored.update(fragmenterParams=config['fragmenter_params'],symbols=config['mol_encoder_params']['symbols'])
    restored.pop('modelConfig',None);restored.pop('params',None)
    (output/'fragment-tree.pft.json').write_text(json.dumps(restored,indent=2))
    (output/'preparation_config.json').write_text(json.dumps(preparation_config,indent=2))
    if validation is None:
        kwargs['overwrite']=False
        create_action_training_data(dataset=train,output_dir=output,**kwargs)
    else:
        for name,split_dataset in (('train',train),('validation',validation)):
            create_action_training_data(dataset=split_dataset,output_dir=output/(name+'_structures'),split=name,**kwargs)
    (output/'fragment-tree.pft.json').write_text(json.dumps(restored,indent=2))
    (output/'invalid_records.json').write_text(json.dumps(reports,indent=2))
    (output/'preparation_config.json').write_text(json.dumps(preparation_config,indent=2))

if __name__=='__main__':main()
