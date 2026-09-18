"""Exercise SMILES inspection and Mol Training preflight without starting training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location('workbench_backend', Path(__file__).resolve().parents[1] / 'src/workbench/dataset_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
from clefts.domain.molecule.descriptors import DEFAULT_DESCRIPTOR_NAMES
catalog = json.loads((Path(__file__).resolve().parents[1] / 'content/mol-descriptors.json').read_text())
assert tuple(item['name'] for item in catalog) == DEFAULT_DESCRIPTOR_NAMES
with tempfile.TemporaryDirectory() as directory:
    smiles = Path(directory) / 'molecules.smi'
    smiles.write_text('CCO\nnot-a-molecule\nC\n\nCCO\n')
    result = backend.mol_smiles({'paths': [str(smiles)]})
    assert (result['count'], result['checked'], result['valid'], result['unique']) == (4, 4, 3, 2)
    assert result['symbols'] == ['C', 'O']
    assert '<svg' in result['preview'][0]['svg']
    config = {'trainSmiles': [str(smiles)], 'valSmiles': [str(smiles)], 'outputDir': directory,
              'symbols': 'C,O', 'nodeDim': '64', 'graphDim': '128', 'numLayers': '4',
              'numHeads': '8', 'maxDegree': '4', 'maxSpatialDist': '3', 'maxEdgeDist': '3', 'device': 'cpu'}
    assert backend.mol_preflight({'config': config})['combinations'] == 1
    for change in ({'graphDim': '32'}, {'numHeads': '7'}, {'symbols': 'Unknown'}, {'descriptorNames': 'UnknownDescriptor'}):
        try:
            backend.mol_preflight({'config': {**config, **change}})
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid configuration accepted: {change}')
    smiles.write_text(' \n\n')
    try:
        backend.mol_preflight({'config': config})
    except ValueError:
        pass
    else:
        raise AssertionError('Empty SMILES input accepted')
print('SMILES inspection, molecule drawings, Mol Training preflight and invalid configuration checks passed.')
