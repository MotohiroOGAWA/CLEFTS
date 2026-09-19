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
    # Reference libraries (e.g. NIST) repeat a handful of distinct adduct/CE
    # strings across hundreds of thousands of rows; caching by the raw string
    # (like smiles_cache already does) turns this loop from O(records) calls
    # into O(unique values) calls to Adduct.parse/parse_ce_to_ev, which is
    # where nearly all the wall-clock time otherwise goes.
    smiles_cache = {}
    adduct_cache = {}
    ce_cache = {}
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
                    text = str(value)
                    if text not in adduct_cache:
                        try:
                            fragmenter.get_index_by_adduct_type(Adduct.parse(text))
                            adduct_cache[text] = True
                        except Exception:
                            adduct_cache[text] = False
                    valid = adduct_cache[text]
                elif key == 'precursorMzColumn':
                    valid = math.isfinite(float(value)) and float(value) > 0
                else:
                    try:
                        mz = float(values['precursorMzColumn'][index])
                    except (ValueError, TypeError):
                        mz = 0.0
                    text = str(value)
                    # parse_ce_to_ev only depends on precursor m/z for a "%"
                    # (NCE) value; every other form can be cached by string
                    # alone (this caller never passes an instrument, so the
                    # other mz-dependent branch is unreachable here).
                    if '%' in text:
                        energy = parse_ce_to_ev(value, mz)
                    else:
                        if text not in ce_cache:
                            ce_cache[text] = parse_ce_to_ev(value, mz)
                        energy = ce_cache[text]
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
