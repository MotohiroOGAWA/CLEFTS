"""Run with the CLEFTS Python environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clefts.domain.molecule import smarts_search as module

result = module.count_matches(['CCO', 'OCC', 'CC', None, '', 'invalid'], '[#8]')
assert result == dict(total=6, valid=3, matched=2, missing=2, invalid=1,
                      unique=2, uniqueMatched=1, percent=200 / 3, uniquePercent=50)
assert module.count_matches([], 'C')['percent'] is None
assert module.count_matches(['CC'], 'N')['matched'] == 0
assert module.count_matches(['OCCO'], 'O')['matched'] == 1
for query in ('', '['):
    try:
        module.count_matches([], query)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid SMARTS accepted')
print('SMARTS search checks passed.')

listed = module.count_matches(['CCO', 'OCC', 'CC', None, '', 'invalid'], '[#8]', include_compounds=True)
assert listed['compounds'] == [
    dict(smiles='CCO', matched=True, records=2),
    dict(smiles='CC', matched=False, records=1),
]
assert sum(c['records'] for c in listed['compounds']) == listed['valid']
assert sum(c['records'] for c in listed['compounds'] if c['matched']) == listed['matched']
assert module.count_matches([], 'C', include_compounds=True)['compounds'] == []
