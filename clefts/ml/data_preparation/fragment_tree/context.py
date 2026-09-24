"""Preparation features and chemistry without constructing a neural model."""
from types import SimpleNamespace
from rdkit import Chem
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.libs.mmkit.mmkit import Adduct
from clefts.ml.mol.graph_builder import MolGraphBuilder
from clefts.ml.mol.formula_encoder import FormulaTensorizer

ARCHITECTURE = 'fragment-tree-physical-ion'

def validate_limits(max_unique_fragment_smiles=-1, max_cleavage_combinations=-1):
    """Per-source search limits; see README.md. Each is -1 (unlimited) or positive."""
    if type(max_unique_fragment_smiles) is not int or not (max_unique_fragment_smiles == -1 or max_unique_fragment_smiles > 0):
        raise ValueError('Max unique fragment SMILES must be -1 or a positive integer.')
    if type(max_cleavage_combinations) is not int or not (max_cleavage_combinations == -1 or max_cleavage_combinations > 0):
        raise ValueError('Max cleavage combinations must be -1 or a positive integer.')

def create_preparation_context(config, *, observed_adducts=()):
    config=config.get('params',config)
    fragmenter=Fragmenter.from_dict(config.get('fragmenter_params',config))
    symbols=config.get('symbols',config.get('mol_encoder_params',{}).get('symbols'))
    if not symbols: raise ValueError('Select at least one element symbol.')
    if not isinstance(symbols,(list,tuple)) or any(not isinstance(value,str) or value not in {Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1,119)} for value in symbols):
        raise ValueError('Symbols must be a list of valid element symbols.')
    # The embedding order is resolved from chemistry and the actual datasets;
    # it is not a user-configurable preparation or training parameter.
    types=list(dict.fromkeys(str(value) for value in fragmenter.adduct_types))
    for value in observed_adducts:
        adduct=Adduct.parse(str(value))
        fragmenter.get_index_by_adduct_type(adduct)
        if str(adduct) not in types: types.append(str(adduct))
    graph_builder=MolGraphBuilder(symbols=symbols)
    return SimpleNamespace(architecture=ARCHITECTURE,fragmenter=fragmenter,adduct_type_strs=tuple(types),
        mol_encoder=SimpleNamespace(graph_builder=graph_builder,symbols=graph_builder.symbols),
        tensorizer=FormulaTensorizer.from_symbols_and_adducts(symbols=graph_builder.symbols,adducts=fragmenter.adduct_types))
