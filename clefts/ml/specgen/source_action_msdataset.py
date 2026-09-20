"""MSDataset adapter for Source/action inference, with no full-tree precompute."""
from __future__ import annotations
import pandas as pd
import torch
from tqdm.auto import tqdm
from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from .source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .spectrum_generator import GeneratedMassSpectrum,GeneratedSpectrumPeak,FragmentSpectrumGeneratorOutput,fragment_spectrum_output_to_msdataset


def predict_source_msdataset(dataset: object, generator: SourceAnchoredFragmentSpectrumGenerator, *,
                             smiles_column: str, adduct_type_column: str, collision_energy_column: str,
                             precursor_mz_column: str, instrument_column: str | None,
                             include_formula_annotation: bool = True) -> tuple[object,pd.DataFrame]:
    total = len(dataset)
    smiles_values = dataset[smiles_column].tolist()
    adduct_values = dataset[adduct_type_column].tolist()
    ce_values = dataset[collision_energy_column].tolist()
    mz_values = dataset[precursor_mz_column].tolist()
    instrument_values = None if instrument_column is None else dataset[instrument_column].tolist()

    # One bad record (an unparseable adduct, an invalid SMILES, a collision
    # energy that doesn't resolve to eV) must not abort every other spectrum in
    # the dataset: skip it, record why, and keep going. valid_indices maps each
    # surviving source/adduct/energy back to its original row for metadata.
    sources,adducts,energies,valid_indices,failures = [],[],[],[],[]
    for i in tqdm(range(total),total=total,desc='Parsing records'):
        try:
            source=Compound.from_smiles(str(smiles_values[i]))
            adduct=Adduct.parse(str(adduct_values[i]))
            mz=float(mz_values[i])
            instrument=None if instrument_values is None else instrument_values[i]
            ev=parse_ce_to_ev(ce_values[i],mz,instrument)
            if ev is None:
                raise ValueError(f"Cannot convert collision energy {ce_values[i]!r} to eV")
        except Exception as error:
            # Broad on purpose: a malformed record can fail in ways this
            # function cannot enumerate up front (a bad adduct string, an
            # unsupported SMILES, or a bug surfaced deep in a third-party
            # parser). Any of those should be skipped and reported, never
            # abort every other spectrum in the dataset.
            failures.append({'index':i,'smiles':smiles_values[i],'error':f'{type(error).__name__}: {error}'})
            continue
        sources.append(source);adducts.append(adduct);energies.append(float(ev));valid_indices.append(i)

    spectra=[GeneratedMassSpectrum(k,[]) for k in range(len(sources))]
    threshold=generator.post_model.ion_prediction_threshold
    if sources:
        progress=tqdm(total=len(sources),desc='Predicting spectra')
        try:
            for output in generator.predict_batches(sources,adducts,energies):
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
