from typing import Tuple, List, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from bidict import bidict
import re
import json
import inspect

from ....libs.mmkit.mmkit import Compound
from .CleavageResult import CleavageResult
from .CleavageProduct import CleavageProduct

class CleavagePattern:
    SUPPORTED_CHARGE_MODES = {"positive1", "negative1", "neutral", "any"}
    def __init__(self, smirks: str, name: str="", charge_mode: str="any"):
        """
        Define a cleavage pattern using a SMIRKS reaction.
        Args:
            smirks (str): SMIRKS reaction string.
            name (str): Pattern name (e.g. "amide bond cleavage").
            charge_mode (str): Allowed charge states (can be multiple separated by '|').
                Examples:
                    - "positive1"
                    - "negative1"
                    - "neutral"
                    - "any"
                    - "positive1|neutral"
        """
        self.version = 1.0
        self.smirks = smirks
        self.name = name
        charge_modes = [cm.strip().lower() for cm in charge_mode.split("|")]
        assert all(cm in self.SUPPORTED_CHARGE_MODES for cm in charge_modes), \
            f"Unsupported charge_mode: {charge_mode}"
        self.charge_mode = "|".join(set(charge_modes))
        self.rxn = rdChemReactions.ReactionFromSmarts(smirks)

        # Extract the reactant SMARTS part to use for substructure matching
        reactant_smarts = smirks.split(">>")[0]
        product_smarts = smirks.split(">>")[1]
        self.reactant_query = Chem.MolFromSmarts(reactant_smarts)

        # --- Basic validation (only single-reactant/product reactions are supported)
        assert len(self.rxn.GetReactants()) == 1, "Only single-reactant patterns are supported."
        assert len(self.rxn.GetProducts()) == 1, "Only single-reactant patterns are supported."

        # --- Initialize reactant and product templates
        self.react_temp = self.rxn.GetReactants()[0]
        self.prod_temp = self.rxn.GetProducts()[0]

        # --- Validate atom map numbers in reactants
        react_map_nums = [a.GetAtomMapNum() for a in self.react_temp.GetAtoms() if a.GetAtomMapNum() > 0]
        # assert all(x > 0 for x in react_map_nums), f"Invalid AtomMapNum: {react_map_nums}"
        self.react_idx_to_map = bidict({i: amap for i, amap in enumerate(set(react_map_nums))})

        # --- Validate atom map numbers in products
        prod_map_nums = [a.GetAtomMapNum() for a in self.prod_temp.GetAtoms() if a.GetAtomMapNum() > 0]
        # assert all(x > 0 for x in prod_map_nums), f"Invalid AtomMapNum: {prod_map_nums}"
        self.prod_idx_to_map = bidict({i: amap for i, amap in enumerate(set(prod_map_nums))})
        

        missing_maps = set(react_map_nums) - set(prod_map_nums)
        self.virtual_smirks = f"{reactant_smarts}>>{product_smarts}" + ("."+".".join(
            [f"[*:{m}]" for m in missing_maps]
        )) if len(missing_maps) > 0 else f""
        self.virtual_rxn = rdChemReactions.ReactionFromSmarts(self.virtual_smirks)
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', smirks='{self.smirks}', charge_mode='{self.charge_mode}')"
    
    def key(self, include_name: bool = False) -> Tuple:
        """
        Generate a unique key for hashing or equality checks.

        Args:
            include_name (bool): Whether to include the name field in comparison.

        Returns:
            tuple: Key representing the essential identity of the pattern.
        """
        if include_name:
            return (self.smirks, self.charge_mode, self.name)
        return (self.smirks, self.charge_mode)

    def __eq__(self, other: 'CleavagePattern') -> bool:
        """
        Default equality check for CleavagePattern objects.
        Compares SMIRKS and charge_mode by default (ignores name).

        To include name, call self.equals(other, include_name=True).
        """
        if not isinstance(other, CleavagePattern):
            return False
        return self.key(False) == other.key(False)
    
    def __hash__(self) -> int:
        """
        Default hash based on key excluding name.
        """
        return hash(self.key(False))

    def equals(self, other: 'CleavagePattern', include_name: bool = False) -> bool:
        """
        Extended equality check with optional name inclusion.

        Args:
            other (CleavagePattern): Another pattern to compare.
            include_name (bool): If True, also compare name.

        Returns:
            bool: True if equivalent under the given criteria.
        """
        if not isinstance(other, CleavagePattern):
            return False
        return self.key(include_name) == other.key(include_name)

    def __str__(self):
        sig = inspect.signature(self.__init__)
        arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]

        fields = [f"v={self.version}"]
        for name in arg_names:
            if hasattr(self, name):
                value = getattr(self, name)
                if isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)
                fields.append(f"{name}={value_str}")

        return f"({self.__class__.__name__};" + ";".join(fields) + ")"

    @property
    def num_reactant_atoms(self) -> int:
        """
        Get the number of atoms in the reactant template.
        Returns:
            int: Number of atoms in the reactant.
        """
        return len(self.react_idx_to_map)

    @property
    def num_product_atoms(self) -> int:
        """
        Get the number of atoms in the product template.
        Returns:
            int: Number of atoms in the product.
        """
        return len(self.prod_idx_to_map)

    def copy(self) -> 'CleavagePattern':
        """
        Create a copy of this CleavagePattern instance.
        Returns:
            CleavagePattern: A new instance with the same attributes.
        """
        sig = inspect.signature(self.__init__)
        init_args = [p.name for p in sig.parameters.values() if p.name != "self"]

        kwargs = {arg: getattr(self, arg) for arg in init_args if hasattr(self, arg)}

        return self.__class__(**kwargs)
    
    @classmethod
    def parse(cls, pattern_str: str) -> 'CleavagePattern':
        """
        Parse a CleavagePattern from its string representation.
        Args:
            pattern_str (str): String representation of the pattern.

        Returns:
            CleavagePattern: The parsed cleavage pattern.
        """
        # 1. Remove surrounding parentheses and class name
        if pattern_str.startswith("(") and pattern_str.endswith(")"):
            pattern_str = pattern_str[1:-1].strip()
        if pattern_str.startswith(cls.__name__):
            pattern_str = pattern_str[len(cls.__name__):].lstrip(";")

        # 2. Regex: match key=value pairs where value may be quoted and contain semicolons
        # Example match groups: key="smirks", value="[C:1]-[O:2];[P:3]>>[C:1]"
        pattern = re.compile(r'(\w+)=(".*?"|[^;]*)')
        matches = pattern.findall(pattern_str)

        # 3. Build dict of key-value pairs
        kwargs = {}
        for key, value in matches:
            value = value.strip()
            # Remove surrounding quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # Try numeric conversion
            if key in ("v", "version"):
                try:
                    value = float(value)
                except ValueError:
                    pass
            kwargs[key] = value

        # 4. Construct object from parsed values
        obj = cls(**{k: v for k, v in kwargs.items() if k in cls.__init__.__code__.co_varnames})
        obj.version = kwargs.get("v", kwargs.get("version", 1.0))
        return obj

    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Automatically convert only the constructor arguments (and version)
        to a serializable dictionary.
        """
        sig = inspect.signature(self.__init__)
        arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]

        data = {name: getattr(self, name) for name in arg_names if hasattr(self, name)}

        data["version"] = getattr(self, "version", 1.0)
        return data

    # -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleavagePattern":
        """
        Automatically construct an instance using only arguments
        that exist in the constructor.
        """
        sig = inspect.signature(cls.__init__)
        arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]

        init_kwargs = {k: v for k, v in data.items() if k in arg_names}
        obj = cls(**init_kwargs)

        if "version" in data:
            obj.version = data["version"]

        return obj

    def exists(self, compound: Compound) -> bool:
        """
        Check if the cleavage pattern exists in a molecule.
        Returns:
            bool: True if pattern is found, False otherwise.
        """
        if self.reactant_query is None:
            return False
        return compound.mol.HasSubstructMatch(self.reactant_query)
    
    def matches(self, compound: Compound) -> List[Tuple[int, ...]]:
        """
        Get all substructure matches of the cleavage pattern in a molecule.
        Returns:
            List[Tuple[int, ...]]: List of atom index tuples for each match.
        """
        if self.reactant_query is None:
            return []
        return compound.mol.GetSubstructMatches(self.reactant_query)

    def is_applicable(self, compound: Compound) -> bool:
        """
        Check if this cleavage pattern should be applied to the given compound.
        The decision is based on the compound's formal charge and charge_mode setting.
        """
        charge = compound.charge
        for cm in self.charge_mode.split("|"):
            if cm == "any":
                return True
            elif cm.startswith("positive"):
                if charge == int(cm.replace("positive", "").strip()):
                    return True
            elif cm.startswith("negative"):
                if charge == int(cm.replace("negative", "").strip()):
                    return True
            elif cm == "neutral":
                if charge == 0:
                    return True
            else:
                raise ValueError(f"Unsupported charge_mode: {cm}")
        return False

    # -------------------------------------------------------------------------
    def fragment(self, compound: Compound) -> CleavageResult:
        """
        Apply the SMIRKS cleavage pattern and return mapped fragment information.

        Returns:
            CleavageResult: Contains the original molecule SMILES, 
                            cleavage SMIRKS, and per-product fragment mapping.
        """
        if not self.exists(compound):
            return None
        
        mol = Chem.Mol(compound._mol)
        # Mapping: atom_map_number → atom_idx in the original molecule
        old_atom_map_to_idx = compound.atom_map_to_index
        # Mapping: atom_idx ↔ atom_map_number
        idx_to_atom_map = bidict({a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()})

        # Apply the SMIRKS pattern to the molecule
        product_sets = self.virtual_rxn.RunReactants((mol,))
        # Identify all substructure matches corresponding to the reactant query
        # reactant_matches = list(mol.GetSubstructMatches(self.reactant_query))

        cleavage_products:List[CleavageProduct] = []

        # Each product_set may contain multiple product molecules
        for product_group in product_sets:
            frag_info = {
                'react': [-1] * len(self.react_idx_to_map),
                'prod': [-1] * len(self.prod_idx_to_map),
            }
            atom_mapping_cache = {}
            for product_mol in product_group:

                # -----------------------------------------------------------------
                # (1) Extract RDKit's atom-level correspondence from reaction result
                for atom in product_mol.GetAtoms():
                    if atom.HasProp("react_atom_idx") and atom.HasProp("old_mapno"):
                        parent_idx = int(atom.GetProp("react_atom_idx"))
                        old_mapno = int(atom.GetProp("old_mapno"))
                        if parent_idx in idx_to_atom_map:
                            atom_map = idx_to_atom_map[parent_idx]
                            react_idx = self.react_idx_to_map.inverse[old_mapno]
                            if old_mapno in self.prod_idx_to_map.inverse:
                                prod_idx = self.prod_idx_to_map.inverse[old_mapno]
                            else:
                                prod_idx = -1
                            old_idx_in_canonical = old_atom_map_to_idx[atom_map]

                            atom_mapping_cache[atom_map] = {
                                'react_idx': react_idx,
                                'prod_idx': prod_idx,
                                'old_idx': old_idx_in_canonical
                            }
                            atom.SetAtomMapNum(idx_to_atom_map[parent_idx])

            # -----------------------------------------------------------------
            # (2) Build new compound and compute final index mapping
            try:
                new_compound = Compound(product_group[0])
            except Exception as e:
                continue
            
            if not self.is_applicable(new_compound):
                continue
            new_atom_map_to_idx = new_compound.atom_map_to_index
            mapped_pairs = []

            for atom_map, info in atom_mapping_cache.items():
                frag_info['react'][info['react_idx']] = info['old_idx']
                if info['prod_idx'] != -1:
                    new_idx = new_atom_map_to_idx[atom_map]
                    frag_info['prod'][info['prod_idx']] = new_idx
                    mapped_pairs.append((info['react_idx'], idx_to_atom_map.inverse[atom_map]))

            # -----------------------------------------------------------------
            # (3) Consistency check
            assert all(idx != -1 for idx in frag_info['react']), \
                "Not all reactant indices were mapped."
            assert all(idx != -1 for idx in frag_info['prod']), \
                "Not all product indices were mapped."

            # -----------------------------------------------------------------
            # (4) Append the result
            cleavage_products.append(
                CleavageProduct(
                    smiles=new_compound.smiles,
                    reactant_indices=tuple(frag_info['react']),
                    product_indices=tuple(frag_info['prod'])
                )
            )
        cleavage_result = CleavageResult(
            cleavage=self.copy(),
            reactant_smiles=compound.smiles,
            products=tuple(cleavage_products)
        )
        return cleavage_result