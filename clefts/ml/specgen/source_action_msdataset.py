"""MSDataset adapter for Source/action inference, with no full-tree precompute."""
from __future__ import annotations
import pandas as pd
import torch
from clefts.libs.mmkit.mmkit import Adduct, Compound
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from .source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .spectrum_generator import GeneratedMassSpectrum,GeneratedSpectrumPeak,FragmentSpectrumGeneratorOutput,fragment_spectrum_output_to_msdataset


def predict_source_msdataset(dataset: object, generator: SourceAnchoredFragmentSpectrumGenerator, *,
                             smiles_column: str, adduct_type_column: str, collision_energy_column: str,
                             precursor_mz_column: str, instrument_column: str | None,
                             include_formula_annotation: bool = True) -> tuple[object,pd.DataFrame]:
    sources=[Compound.from_smiles(str(value)) for value in dataset[smiles_column].tolist()]
    adducts=[Adduct.parse(str(value)) for value in dataset[adduct_type_column].tolist()]
    energies=[]
    for i,value in enumerate(dataset[collision_energy_column].tolist()):
        mz=float(dataset[precursor_mz_column].iloc[i])
        instrument=None if instrument_column is None else dataset[instrument_column].iloc[i]
        ev=parse_ce_to_ev(value,mz,instrument)
        if ev is None: raise ValueError(f"Cannot convert collision energy at record {i}")
        energies.append(float(ev))
    spectra=[GeneratedMassSpectrum(i,[]) for i in range(len(sources))]
    for output in generator.predict_batches(sources,adducts,energies):
        prediction=output.spectra
        for sample,formula,mz,intensity in zip(prediction.sample_index.detach().cpu().tolist(),prediction.formula_tensor.detach().cpu(),prediction.mz.detach().cpu().tolist(),prediction.intensity.detach().cpu().tolist()):
            original=output.sample_input_index[sample]
            spectra[original].peaks.append(GeneratedSpectrumPeak(mz=mz,intensity=max(float(intensity),0.),sample_id=original,
                formula=str(generator.tensorizer.tensor_to_formula(formula)) if include_formula_annotation else None))
    for spectrum in spectra:
        if spectrum.peaks:
            maximum=max(peak.intensity for peak in spectrum.peaks)
            for peak in spectrum.peaks: peak.intensity=peak.intensity/maximum if maximum>0 else 0.
            spectrum.peaks.sort(key=lambda peak:peak.mz)
    predicted=fragment_spectrum_output_to_msdataset(FragmentSpectrumGeneratorOutput(spectra),metadata=dataset.metadata)
    return predicted,pd.DataFrame(columns=['smiles','error'])
