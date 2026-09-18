"""Direct reaction locations and products, tested against actual RDKit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'src/features/cleavage-pattern-set'))
import reaction_preview as preview
from rdkit import Chem

def complete_preview(payload):
    result = preview.reaction_preview(payload)
    for pattern, definition in zip(result['patterns'], payload['patterns']):
        for match in pattern['matches']:
            products = preview.reaction_preview_products({'smiles':payload['smiles'], 'pattern':definition,
                                                          'atoms':match['atoms'], 'bonds':match['bonds']})
            match.update(products)
    return result

split = {'name': 'C–O cleavage', 'reactant_smarts': '[#6:4]-[#8:9]',
         'products': [{'name': 'Split', 'smarts': '[#6:4].[#8:9]'},
                      {'name': 'Carbon side', 'smarts': '[#6:4]'}]}
# Initial rendering must perform matching only: exactly one input drawing and no SMIRKS compilation.
original_drawing, original_compile = preview.drawing, preview.rdChemReactions.ReactionFromSmarts
calls = []
def input_drawing(*args, **kwargs):
    calls.append(1)
    return original_drawing(*args, **kwargs)
def forbidden_compile(*args, **kwargs):
    raise AssertionError('Initial matching must not compile reactions')
preview.drawing = input_drawing
preview.rdChemReactions.ReactionFromSmarts = forbidden_compile
try:
    matches_only = preview.reaction_preview({'smiles':'CCOCCO','patterns':[split]})
    assert calls == [1]
    assert len(matches_only['patterns'][0]['matches']) == 3
    assert not matches_only['patterns'][0]['errors']
    assert all(not m['loaded'] and not m['products'] for m in matches_only['patterns'][0]['matches'])
finally:
    preview.drawing, preview.rdChemReactions.ReactionFromSmarts = original_drawing, original_compile

x = complete_preview({'smiles': 'CCOCCO', 'patterns': [split]})
assert not x['truncated']
assert len(x['patterns'][0]['matches']) == 3
assert [m['atoms'] for m in x['patterns'][0]['matches']] == [[1,2],[2,3],[4,5]]
for match in x['patterns'][0]['matches']:
    assert len(match['products']) == 2 and not match['errors']
    split_product = next(p for p in match['products'] if p['name'] == 'Split')
    assert len(split_product['molecules']) == 2
    assert sorted(i for m in split_product['molecules'] for i in m['sourceAtoms']) == list(range(6))
    for m in split_product['molecules']:
        assert Chem.MolFromSmiles(m['smiles']) is not None
        assert m['formula'] and m['exactMass'] > 0 and '<svg' in m['drawing']['svg']
    carbon = next(p for p in match['products'] if p['name'] == 'Carbon side')
    assert len(carbon['molecules']) == 1
    oxygen_index = next(i for i in match['atoms'] if i in (2, 5))
    assert oxygen_index not in carbon['molecules'][0]['sourceAtoms']
assert '<svg' in x['drawing']['svg'] and len(x['drawing']['positions']) == 6
assert 'fill:#FFFFFF' not in x['drawing']['svg']
assert x['patterns'][0]['matches'][0]['bonds'] == [1]

symmetric = {'name':'Symmetric C–C', 'reactant_smarts':'[#6:1]-[#6:2]',
             'products':[{'name':'One side','smarts':'[#6:1]'}]}
x = complete_preview({'smiles':'CC(C)C', 'patterns':[symmetric]})
assert len(x['patterns'][0]['matches']) == 3
for match in x['patterns'][0]['matches']:
    assert match['assignments'] == 2
    assert {p['smiles'] for p in match['products']} == {'C','CCC'}

cyclize = {'name':'New bond', 'reactant_smarts':'[#6:1]-[#6:2]-[#6:3]',
           'products':[{'name':'Ring','smarts':'[#6:1]1-[#6:2]-[#6:3]-1'}]}
x = complete_preview({'smiles':'CCC', 'patterns':[cyclize]})
assert len(x['patterns'][0]['matches']) == 1
assert [p['smiles'] for p in x['patterns'][0]['matches'][0]['products']] == ['C1CC1']

unchanged = {'name':'Keep source types', 'reactant_smarts':'[!#1:1]~[!#1:2]',
             'products':[{'name':'Unchanged','smarts':'[!#1:1]~[!#1:2]'}]}
x = complete_preview({'smiles':'C=O', 'patterns':[unchanged]})
assert x['patterns'][0]['matches'][0]['products'][0]['smiles'] == 'C=O'
ring = {'name':'Ring bond', 'reactant_smarts':'[#6:1]-;@[#6:2]', 'products':[]}
x = complete_preview({'smiles':'CC1CCC1', 'patterns':[ring]})
assert len(x['patterns'][0]['matches']) == 4
source = Chem.MolFromSmiles('CC1CCC1')
assert all(source.GetBondWithIdx(m['bonds'][0]).IsInRing() for m in x['patterns'][0]['matches'])

bad = {'name':'Invalid', 'reactant_smarts':'[broken', 'products':[]}
x = complete_preview({'smiles':'CCO', 'patterns':[bad,split,symmetric]})
assert x['patterns'][0]['errors'] and len(x['patterns'][1]['matches']) == 1
assert len(x['patterns'][2]['matches']) == 1
assert not preview.reaction_preview({'smiles':'CC', 'patterns':[split]})['patterns'][0]['matches']

invalid_product = {'name':'Valence', 'reactant_smarts':'[#6:1]-[#8:2]',
                   'products':[{'name':'Invalid oxygen','smarts':'[#6:1]#[#8:2]'}]}
x = complete_preview({'smiles':'CO', 'patterns':[invalid_product]})
assert x['patterns'][0]['matches'][0]['errors']
assert not x['patterns'][0]['matches'][0]['products']

for smiles, patterns in [('not smiles',[split]), ('',[split]), ('CC',[])]:
    try:
        preview.reaction_preview({'smiles':smiles,'patterns':patterns})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid request accepted')

preview.LIMIT = 2
assert preview.reaction_preview({'smiles':'CCCC','patterns':[symmetric]})['truncated']
print('Direct reaction preview: exact source locations, omitted maps, symmetric assignments, multiple rules, dot-separated products, new bonds, RDKit SVG, validation and limits passed.')
