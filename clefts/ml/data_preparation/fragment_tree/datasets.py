"""Dataset loading shared by preparation and Workbench inspection."""
from __future__ import annotations
import csv
import io
import random
from pathlib import Path
from clefts.libs.msentity.msentity import MSDataset

SUPPORTED_SUFFIXES = ('.msds', '.msp', '.mgf', '.tsv', '.csv', '.parquet')

def load_spectrum_dataset(path: str | Path) -> MSDataset:
    path=Path(path)
    suffix=path.suffix.lower()
    if suffix=='.msds': return MSDataset.load(str(path),load_peak_metadata=False)
    if suffix=='.msp':
        from clefts.libs.msentity.msentity.io.msp import read_msp
        return read_msp(str(path),show_progress=False,normalize_intensity=False)
    if suffix=='.mgf':
        from clefts.libs.msentity.msentity.io.mgf import read_mgf
        return read_mgf(str(path),show_progress=False,normalize_intensity=False)
    from clefts.libs.msentity.msentity.io.tsv import read_tsv,read_tsv_text
    if suffix=='.tsv': return read_tsv(path,show_progress=False)
    if suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.reader(handle))
        output=io.StringIO();writer=csv.writer(output,delimiter='\t');writer.writerows(rows)
        return read_tsv_text(output.getvalue(),source_name=str(path))
    if suffix=='.parquet':
        import pandas as pd
        return read_tsv_text(pd.read_parquet(path).to_csv(sep='\t',index=False),source_name=str(path))
    raise ValueError('Unsupported dataset format. Use MSDataset, MSP, MGF or a TSV / CSV / Parquet spectrum table with a Peak column.')

def split_by_smiles(dataset: MSDataset, column: str, ratio: float, seed: int) -> tuple[MSDataset,MSDataset]:
    if not 0<ratio<1: raise ValueError('Validation ratio must be between 0 and 1.')
    smiles=list(dict.fromkeys(dataset[column].astype(str).tolist()))
    if len(smiles)<2: raise ValueError('Automatic validation split requires at least two unique SMILES. Supply a separate validation dataset.')
    random.Random(seed).shuffle(smiles)
    count=min(len(smiles)-1,max(1,round(len(smiles)*ratio)))
    held_out=set(smiles[:count])
    mask=dataset[column].astype(str).isin(held_out)
    return dataset[~mask],dataset[mask]
