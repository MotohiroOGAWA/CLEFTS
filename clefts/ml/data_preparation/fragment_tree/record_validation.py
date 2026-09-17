"""Shared metadata checks for inspection and exclusion during preparation."""
import math
from rdkit import Chem
from clefts.libs.mmkit.mmkit import Adduct
from clefts.domain.mass.parse_ce import parse_ce_to_ev

DEFAULT_MAPPING = {'smilesColumn': 'SMILES', 'adductTypeColumn': 'AdductType',
                   'collisionEnergyColumn': 'CollisionEnergy', 'precursorMzColumn': 'PrecursorMZ'}

def inspect_records(dataset, fragmenter, mapping=None):
    mapping = {**DEFAULT_MAPPING, **(mapping or {})}
    missing = [name for name in mapping.values() if name not in dataset.columns]
    if missing:
        raise ValueError('Missing dataset columns: ' + ', '.join(missing))
    validation = {key: {'column': name, 'exists': True, 'valid': 0, 'total': len(dataset)}
                  for key, name in mapping.items()}
    invalid = []
    accepted = []
    smiles_cache = {}
    reasons = {'smilesColumn': 'Invalid RDKit SMILES',
               'adductTypeColumn': 'Invalid adduct or main adduct not registered in the fragmenter',
               'collisionEnergyColumn': 'Cannot parse collision energy to finite non-negative eV',
               'precursorMzColumn': 'Cannot parse precursor m/z to a finite positive number'}
    values = {key: dataset[name].tolist() for key, name in mapping.items()}
    ids = dataset['SpecID'].astype(str).tolist() if 'SpecID' in dataset.columns else None
    for index in range(len(dataset)):
        failures = []
        for key, item in validation.items():
            value = values[key][index]
            valid = False
            try:
                if key == 'smilesColumn':
                    text = str(value)
                    if text not in smiles_cache:
                        smiles_cache[text] = bool(text.strip()) and Chem.MolFromSmiles(text) is not None
                    valid = smiles_cache[text]
                elif key == 'adductTypeColumn':
                    fragmenter.get_index_by_adduct_type(Adduct.parse(str(value)))
                    valid = True
                elif key == 'precursorMzColumn':
                    valid = math.isfinite(float(value)) and float(value) > 0
                else:
                    try:
                        mz = float(values['precursorMzColumn'][index])
                    except (ValueError, TypeError):
                        mz = 0.0
                    energy = parse_ce_to_ev(value, mz)
                    valid = energy is not None and math.isfinite(float(energy)) and float(energy) >= 0
            except Exception:
                valid = False
            item['valid'] += int(valid)
            if not valid:
                failures.append({'field': key, 'column': item['column'], 'reason': reasons[key]})
        if failures:
            invalid.append({'index': index, 'id': ids[index] if ids else str(index),
                            'values': {key: str(items[index]) for key, items in values.items()},
                            'issues': failures})
        else:
            accepted.append(index)
    return {'validation': validation, 'invalidRecords': invalid, 'validIndexes': accepted,
            'eligibleRecords': len(accepted), 'excludedRecords': len(invalid),
            'eligibleUniqueSmiles': len({str(values['smilesColumn'][index]) for index in accepted})}
