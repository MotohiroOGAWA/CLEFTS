"""Regenerate schema-v4 action teachers directly from original MSDataset."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from clefts.libs.mmkit.mmkit import Compound,Adduct
from clefts.libs.msentity.msentity import MSDataset
from clefts.domain.mass.parse_ce import parse_ce_to_ev
from clefts.ml.input.action_structure_builder import ActionStructureBuilder
from clefts.ml.input.fragment_tree_training_data import save_fragment_tree_structure,make_structure_file_stem
from clefts.ml.specgen.fragment_tree_spectrum_predictor import create_spectrum_generator


def create_action_training_data(*, dataset: MSDataset, model_config: dict, output_dir: str | Path,
                                smiles_column: str = 'SMILES', adduct_type_column: str = 'AdductType',
                                collision_energy_column: str = 'CollisionEnergy', precursor_mz_column: str = 'PrecursorMZ') -> list[Path]:
    generator=create_spectrum_generator(model_config.get('params',model_config))
    if getattr(generator,'architecture',None)!='source-anchored-action-autoregressive-v1':
        raise ValueError('Schema v4 preparation requires an action model config')
    builder=ActionStructureBuilder(generator)
    grouped={}
    for index,smiles in enumerate(dataset[smiles_column].tolist()):grouped.setdefault(str(smiles),[]).append(index)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    files=[]
    for tree,(smiles,rows) in enumerate(grouped.items()):
        adducts=[];energies=[];mzs=[];intensities=[]
        for index in rows:
            record=dataset[index]
            adducts.append(Adduct.parse(str(dataset[adduct_type_column].iloc[index])))
            ev=parse_ce_to_ev(dataset[collision_energy_column].iloc[index],float(dataset[precursor_mz_column].iloc[index]))
            if ev is None:raise ValueError(f'Cannot convert CE at record {index}')
            energies.append(float(ev))
            peaks=record.peaks
            mzs.append([float(peak.mz) for peak in peaks]);intensities.append([float(peak.intensity) for peak in peaks])
        structure=builder.build(Compound.from_smiles(smiles),adducts,energies,mzs,intensities)
        path=output/(make_structure_file_stem(smiles,index=tree)+'.preft.pt')
        save_fragment_tree_structure(structure=structure,output_file=path,metadata=dict(smiles=smiles,record_indexes=rows))
        files.append(path)
    (output/'action_statistics.json').write_text(json.dumps(dict(schema_version=4,fragmentation_schema=generator.architecture,
        num_sources=len(files),num_samples=len(dataset)),indent=2))
    return files


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--params',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    create_action_training_data(dataset=MSDataset.load(args.input),model_config=json.loads(Path(args.params).read_text()),output_dir=args.output_dir)

if __name__=='__main__':main()
