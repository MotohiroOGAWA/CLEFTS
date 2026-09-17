"""Regenerate schema-v4 action teachers directly from original MSDataset."""
from __future__ import annotations
import argparse,json,math,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from itertools import islice
from pathlib import Path
from clefts.libs.mmkit.mmkit import Compound,Adduct
from clefts.libs.msentity.msentity import MSDataset
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.input.structure_builder import ActionStructureBuilder
from clefts.ml.input.source_action_structure import save_fragment_tree_structure,make_structure_file_stem
from .context import create_preparation_context, validate_limits
from clefts.ml.specgen.config_options import configure_model_options, resolve_model_options
from .datasets import load_spectrum_dataset, split_by_smiles


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
    path=output/(make_structure_file_stem(smiles,index=tree)+'.preft.pt')
    save_fragment_tree_structure(structure=structure,output_file=path,metadata=dict(smiles=smiles,record_indexes=rows))
    return path


def _initialize_worker(model_config, observed_adducts, options):
    import torch
    torch.set_num_threads(1)
    global _worker_builder, _worker_options
    generator=create_preparation_context(model_config,observed_adducts=observed_adducts)
    _worker_builder=ActionStructureBuilder(generator,max_node=options['max_node'],max_edge=options['max_edge'])
    _worker_options=options


def _prepare_worker_chunk(tasks):
    return [_prepare_group(task,_worker_builder,_worker_options) for task in tasks]

def _parallel_results(pool,tasks,chunk_size,num_workers):
    # Bound pending subsets instead of retaining a second copy of the whole dataset.
    pending=deque()
    def submit():
        chunk=list(islice(tasks,chunk_size))
        if chunk: pending.append(pool.submit(_prepare_worker_chunk,chunk))
    for _ in range(num_workers*2): submit()
    while pending:
        result=pending.popleft().result()
        submit()
        yield from result


def create_action_training_data(*, dataset: MSDataset, model_config: dict, output_dir: str | Path,
                                smiles_column: str = 'SMILES', adduct_type_column: str = 'AdductType',
                                collision_energy_column: str = 'CollisionEnergy', precursor_mz_column: str = 'PrecursorMZ',
                                minimum_relative_intensity: float = 0.0, normalize_intensities: bool = True,
                                overwrite: bool = True, split: str = 'dataset', max_node: int = -1, max_edge: int = -1, num_workers: int = 1, chunk_size: int = 1) -> list[Path]:
    if not math.isfinite(minimum_relative_intensity) or not 0<=minimum_relative_intensity<=1:
        raise ValueError('Minimum relative intensity must be between 0 and 1.')
    validate_limits(max_node,max_edge)
    if type(num_workers) is not int or num_workers<1 or type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Worker processes and chunk size must be positive integers.')
    generator=create_preparation_context(model_config,observed_adducts=dataset[adduct_type_column].unique())
    builder=ActionStructureBuilder(generator,max_node=max_node,max_edge=max_edge)
    grouped={}
    for index,smiles in enumerate(dataset[smiles_column].tolist()):grouped.setdefault(str(smiles),[]).append(index)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if not overwrite:
        existing=list(output.glob('*.preft.pt'))
        if existing: raise FileExistsError('Output already contains training structures. Enable overwrite or choose another directory.')
    options=dict(output_dir=str(output),adduct_type_column=adduct_type_column,collision_energy_column=collision_energy_column,
        precursor_mz_column=precursor_mz_column,minimum_relative_intensity=minimum_relative_intensity,
        normalize_intensities=normalize_intensities,max_node=max_node,max_edge=max_edge)
    tasks=((tree,smiles,rows,dataset[rows]) for tree,(smiles,rows) in enumerate(grouped.items()))
    files=[]
    def collect(results):
        for current,path in enumerate(results,1):
            files.append(path)
            print(json.dumps(dict(event='progress',split=split,current=current,total=len(grouped))),flush=True)
    if num_workers==1 or len(grouped)<2:
        collect(_prepare_group(task,builder,options) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=min(num_workers,len(grouped)),mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker,initargs=(model_config,generator.adduct_type_strs,options)) as pool:
            collect(_parallel_results(pool,tasks,min(chunk_size,max(1,len(grouped)//num_workers)),min(num_workers,len(grouped))))
    (output/'action_statistics.json').write_text(json.dumps(dict(schema_version=4,fragmentation_schema=generator.architecture,
        num_sources=len(files),num_samples=len(dataset),num_workers=num_workers,chunk_size=chunk_size,model_config=model_config,minimum_relative_intensity=minimum_relative_intensity,normalize_intensities=normalize_intensities),indent=2))
    return files


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
    # Stable condition indices across training and validation, including neutral-loss precursors.
    observed=list(dataset[args.adduct_type_column].unique())
    if args.validation_input: observed.extend(load_spectrum_dataset(args.validation_input)[args.adduct_type_column].unique())
    config['adduct_type_strs']=list(create_preparation_context(config,observed_adducts=observed).adduct_type_strs)
    kwargs=dict(model_config=config,smiles_column=args.smiles_column,adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,precursor_mz_column=args.precursor_mz_column,
        minimum_relative_intensity=args.minimum_relative_intensity,normalize_intensities=bool(args.normalize_intensities),
        overwrite=bool(args.overwrite),max_node=args.max_node,max_edge=args.max_edge,num_workers=args.num_workers,chunk_size=args.chunk_size)
    if args.validation_input and args.validation_ratio is not None:
        raise ValueError('Use a separate validation input or a SMILES split, not both.')
    if args.validation_input:
        train,validation=dataset,load_spectrum_dataset(args.validation_input)
    elif args.validation_ratio is not None:
        train,validation=split_by_smiles(dataset,args.smiles_column,args.validation_ratio,args.validation_seed)
    else:
        create_action_training_data(dataset=dataset,output_dir=args.output_dir,**kwargs)
        return
    if set(train[args.smiles_column].astype(str)) & set(validation[args.smiles_column].astype(str)):
        raise ValueError('Training and validation datasets share SMILES. Choose molecule-disjoint datasets.')
    output=Path(args.output_dir)
    # Check both destinations before producing either split.
    if not args.overwrite and any((output/name).exists() and list((output/name).glob('*.preft.pt')) for name in ('train_structures','validation_structures')):
        raise FileExistsError('Output already contains training structures. Enable overwrite or choose another directory.')
    for name,split_dataset in (('train',train),('validation',validation)):
        create_action_training_data(dataset=split_dataset,output_dir=output/(name+'_structures'),split=name,**kwargs)
    (output/'preparation_config.json').write_text(json.dumps(dict(input=args.input,validation_input=args.validation_input,
        validation_ratio=args.validation_ratio,validation_seed=args.validation_seed,**kwargs),indent=2))

if __name__=='__main__':main()
