"""MSDataset adapter for Source/action inference, with no full-tree precompute."""
from __future__ import annotations
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.utils.parallel_subprocess import run_parallel_subprocesses
from .prepare_worker import prepare_compound_cache
from .source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .spectrum_generator import GeneratedMassSpectrum,GeneratedSpectrumPeak,FragmentSpectrumGeneratorOutput,fragment_spectrum_output_to_msdataset


def build_prediction_cache(groups: dict[str,list[str]], prep_config: dict, *,
                           num_workers: int = 1, chunk_size: int = 1, desc: str = 'Preparing molecules',
                           keep_temp: bool = False, temp_dir: str | None = None) -> dict[str,dict]:
    """groups: {smiles: [adduct_str, ...]} for every unique compound in the
    batch. Returns {smiles: result} (see prepare_worker.prepare_chunk) for
    every compound RDKit could process -- a compound left out (or marked
    ok=False) is resolved inline instead, by
    SourceAnchoredFragmentSpectrumGenerator.prepare()'s own fallback path.

    num_workers==1 runs in-process, with per-compound progress; num_workers>1
    fans out via clefts.utils.parallel_subprocess (real OS processes -- RDKit
    does not parallelize well across Python threads). The per-chunk task and
    result files those subprocesses exchange live under a temporary
    directory that is removed once every chunk has completed, unless
    keep_temp is set (for inspecting a chunk's inputs/outputs directly)."""
    if type(num_workers) is not int or num_workers<1:raise ValueError('num_workers must be a positive integer')
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('chunk_size must be a positive integer')
    if not groups:return {}
    from clefts.ml.data_preparation.fragment_tree.context import create_preparation_context
    if num_workers==1 or len(groups)<2:
        fragmenter=create_preparation_context(prep_config).fragmenter
        results={}
        for smiles,adduct_strs in tqdm(groups.items(),total=len(groups),desc=desc):
            try:
                results[smiles]={"ok":True,**prepare_compound_cache(smiles,adduct_strs,fragmenter)}
            except Exception as error:
                results[smiles]={"ok":False,"error":f'{type(error).__name__}: {error}'}
        return results
    items=list(groups.items())
    effective_chunk_size=min(chunk_size,max(1,len(items)//num_workers))
    results={}
    directory=tempfile.mkdtemp(prefix='clefts-predict-prepare-',dir=temp_dir)
    try:
        commands=[]
        for offset in range(0,len(items),effective_chunk_size):
            chunk=dict(items[offset:offset+effective_chunk_size])
            task_file=Path(directory)/f'task-{len(commands)}.pkl'
            result_file=task_file.with_suffix('.result.pkl')
            with task_file.open('wb') as stream:
                pickle.dump({'groups':chunk,'prep_config':prep_config},stream,pickle.HIGHEST_PROTOCOL)
            commands.append([sys.executable,'-m','clefts.ml.specgen.prepare_worker','--task',str(task_file),'--result',str(result_file)])
        def on_complete(command):
            with Path(command[-1]).open('rb') as stream:
                results.update(pickle.load(stream))
        env={**os.environ,'PYTHONUNBUFFERED':'1'}
        root=str(Path(__file__).resolve().parents[3])
        env['PYTHONPATH']=os.pathsep.join(filter(None,[root,env.get('PYTHONPATH')]))
        run_parallel_subprocesses(commands,max_workers=min(num_workers,len(commands)),print_output=False,
                                  env=env,desc=desc,unit='chunk',on_complete=on_complete)
    finally:
        if keep_temp:
            print(f"kept prepare task/result files: {directory}",file=sys.stderr)
        else:
            shutil.rmtree(directory,ignore_errors=True)
    return results


def predict_source_msdataset(dataset: object, generator: SourceAnchoredFragmentSpectrumGenerator, *,
                             smiles_column: str, adduct_type_column: str, collision_energy_column: str,
                             precursor_mz_column: str, instrument_column: str | None,
                             include_formula_annotation: bool = True,
                             num_workers: int = 1, chunk_size: int = 1,
                             keep_temp: bool = False, temp_dir: str | None = None,
                             validate_desc: str = 'Validating records', prepare_desc: str = 'Preparing molecules',
                             predict_desc: str = 'Predicting spectra') -> tuple[object,pd.DataFrame]:
    total = len(dataset)
    smiles_values = dataset[smiles_column].tolist()
    adduct_values = dataset[adduct_type_column].tolist()
    ce_values = dataset[collision_energy_column].tolist()
    mz_values = dataset[precursor_mz_column].tolist()
    instrument_values = None if instrument_column is None else dataset[instrument_column].tolist()

    # Phase 1: cheap, RDKit-free per-row validation only (adduct syntax, this
    # model's known adduct vocabulary, collision energy). The RDKit-heavy
    # per-compound work -- SMILES parsing and fragmentation -- happens once
    # per UNIQUE compound below, not once per row, since a real dataset
    # typically repeats the same compound across many adducts/energies.
    candidates,failures = [],[]
    for i in tqdm(range(total),total=total,desc=validate_desc):
        try:
            adduct=Adduct.parse(str(adduct_values[i]))
            if str(adduct) not in generator.adduct_type_strs:
                # A syntactically valid adduct this particular model was
                # never trained on (unlike training's validation split, real
                # input data isn't limited to the model's known adduct
                # vocabulary). Left unchecked, this only fails deep inside
                # generator.prepare()'s conditions tensor, after grouping,
                # crashing every other sample sharing that compound.
                raise ValueError(f"Adduct {adduct} is not one of this model's supported adduct types: {generator.adduct_type_strs}")
            mz=float(mz_values[i])
            instrument=None if instrument_values is None else instrument_values[i]
            ev=parse_ce_to_ev(ce_values[i],mz,instrument)
            if ev is None:
                raise ValueError(f"Cannot convert collision energy {ce_values[i]!r} to eV")
        except Exception as error:
            # Broad on purpose: a malformed record can fail in ways this
            # function cannot enumerate up front (a bad adduct string or a
            # bug surfaced deep in a third-party parser). Any of those should
            # be skipped and reported, never abort every other spectrum.
            failures.append({'index':i,'smiles':smiles_values[i],'error':f'{type(error).__name__}: {error}'})
            continue
        candidates.append((i,str(smiles_values[i]),adduct,float(ev)))
    print(f"{validate_desc}: {len(candidates)}/{total} records valid, {len(failures)} failed",file=sys.stderr)

    # Phase 2: group by unique compound and precompute cleavage actions plus
    # candidate precursor sequences once per compound, optionally in
    # parallel. This is the RDKit-bound step generator.prepare() would
    # otherwise run synchronously, per compound, inside predict_batches().
    groups: dict[str,list[str]] = {}
    for _,smiles,adduct,_ in candidates:
        requested=groups.setdefault(smiles,[])
        if str(adduct) not in requested:requested.append(str(adduct))
    prep_config={
        'fragmenter_params':generator.fragmenter.to_dict(),
        'symbols':list(generator.mol_encoder.symbols),
        'adduct_type_strs':list(generator.adduct_type_strs),
    }
    cache=build_prediction_cache(groups,prep_config,num_workers=num_workers,chunk_size=chunk_size,
                                 desc=prepare_desc,keep_temp=keep_temp,temp_dir=temp_dir)

    # Phase 3: a compound RDKit could not fragment, or an adduct with no
    # valid precursor action sequence for this source (an empty tuple --
    # source_action_structure.prepare_source_actions raises
    # UnresolvedPrecursorError on that, deep inside the batched prediction
    # loop, taking every other sample sharing this compound down with it),
    # fails only its own rows instead of crashing
    # generator.predict_batches() for the rest of the dataset.
    # Compound.from_smiles is re-run once per unique compound here (already
    # validated by the cache above) rather than pickling RDKit Mol objects
    # back across the subprocess boundary.
    sources,adducts,energies,valid_indices,precomputed = [],[],[],[],{}
    compound_cache: dict[str,Compound] = {}
    for i,smiles,adduct,ev in candidates:
        result=cache.get(smiles)
        if result is None or not result.get('ok'):
            error=(result or {}).get('error','RDKit fragmentation produced no result for this compound')
            failures.append({'index':i,'smiles':smiles,'error':error})
            continue
        if not result['precursor_sequences'].get(str(adduct)):
            failures.append({'index':i,'smiles':smiles,'error':f'No valid precursor action sequence found for adduct {adduct} on this compound'})
            continue
        if smiles not in compound_cache:
            try:
                compound_cache[smiles]=Compound.from_smiles(smiles)
            except Exception as error:
                failures.append({'index':i,'smiles':smiles,'error':f'{type(error).__name__}: {error}'})
                continue
            precomputed[smiles]={'actions':result['actions'],'precursor_sequences':result['precursor_sequences']}
        sources.append(compound_cache[smiles]);adducts.append(adduct);energies.append(ev);valid_indices.append(i)

    spectra=[GeneratedMassSpectrum(k,[]) for k in range(len(sources))]
    threshold=generator.post_model.ion_prediction_threshold
    if sources:
        progress=tqdm(total=len(sources),desc=predict_desc)
        try:
            for output in generator.predict_batches(sources,adducts,energies,precomputed=precomputed):
                prediction=output.spectra
                for sample,formula,mz,intensity,confidence in zip(prediction.sample_index.detach().cpu().tolist(),prediction.formula_tensor.detach().cpu(),prediction.mz.detach().cpu().tolist(),prediction.intensity.detach().cpu().tolist(),prediction.confidence.detach().cpu().tolist()):
                    # Below the trained ion confidence threshold: the same
                    # unrelated adduct/hydrogen-shift candidate the model learns
                    # to push toward zero, kept out of the reported spectrum.
                    if confidence<threshold:
                        continue
                    original=output.sample_input_index[sample]
                    spectra[original].peaks.append(GeneratedSpectrumPeak(mz=mz,intensity=max(float(intensity),0.),sample_id=original,
                        formula=str(generator.tensorizer.tensor_to_formula(formula)) if include_formula_annotation else None))
                progress.update(len(output.sample_input_index))
        finally:
            progress.close()
    for spectrum in spectra:
        if spectrum.peaks:
            maximum=max(peak.intensity for peak in spectrum.peaks)
            for peak in spectrum.peaks: peak.intensity=peak.intensity/maximum if maximum>0 else 0.
            spectrum.peaks.sort(key=lambda peak:peak.mz)

    metadata=dataset.metadata.iloc[valid_indices].reset_index(drop=True)
    # Values actually fed to the model, not re-derived afterward from the raw
    # columns: the resolved main adduct (the base ion rule the model used,
    # e.g. [M+H-H2O]+ -> [M+H]+) and the numeric collision energy in eV.
    metadata['MainAdduct']=[str(generator.fragmenter._resolve_main_adduct_type(adduct)) for adduct in adducts]
    metadata['ModelCollisionEnergyEV']=list(energies)
    predicted=fragment_spectrum_output_to_msdataset(FragmentSpectrumGeneratorOutput(spectra),metadata=metadata)
    return predicted,pd.DataFrame(failures,columns=['index','smiles','error'])
