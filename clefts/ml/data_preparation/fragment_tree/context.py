"""Preparation features and chemistry without constructing a neural model."""
from types import SimpleNamespace
from rdkit import Chem
from clefts.domain.fragment.fragmenter import Fragmenter
from clefts.libs.mmkit.mmkit import Adduct
from clefts.ml.mol.graph_builder import MolGraphBuilder
from clefts.ml.mol.formula_encoder import FormulaTensorizer

ARCHITECTURE = 'fragment-tree-physical-ion'

def validate_limits(max_node=-1, max_edge=-1):
    if type(max_node) is not int or not (max_node == -1 or max_node > 0):
        raise ValueError('Max node must be -1 or a positive integer.')
    if type(max_edge) is not int or max_edge < -1:
        raise ValueError('Max edge must be -1 or a non-negative integer.')

def create_preparation_context(config, *, observed_adducts=()):
    config=config.get('params',config)
    fragmenter=Fragmenter.from_dict(config.get('fragmenter_params',config))
    symbols=config.get('symbols',config.get('mol_encoder_params',{}).get('symbols'))
    if not symbols: raise ValueError('Select at least one element symbol.')
    if not isinstance(symbols,(list,tuple)) or any(not isinstance(value,str) or value not in {Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1,119)} for value in symbols):
        raise ValueError('Symbols must be a list of valid element symbols.')
    types=list(dict.fromkeys(str(Adduct.parse(str(value))) for value in config.get('adduct_type_strs',fragmenter.adduct_types)))
    for value in observed_adducts:
        adduct=Adduct.parse(str(value))
        fragmenter.get_index_by_adduct_type(adduct)
        if str(adduct) not in types: types.append(str(adduct))
    graph_builder=MolGraphBuilder(symbols=symbols)
    return SimpleNamespace(architecture=ARCHITECTURE,fragmenter=fragmenter,adduct_type_strs=tuple(types),
        mol_encoder=SimpleNamespace(graph_builder=graph_builder,symbols=graph_builder.symbols),
        tensorizer=FormulaTensorizer.from_symbols_and_adducts(symbols=graph_builder.symbols,adducts=fragmenter.adduct_types))
