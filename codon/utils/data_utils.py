import gemmi
import numpy as np
from openfold.np import residue_constants
from openfold.np.residue_constants import restype_order_with_x


def parse_mmcif(mmcif_path):
    # This function is adapted from Jeremy Wohlwend https://github.com/jwohlwend
    # returns a list of dictionaries where each one corresponds to a protein chain
    block = gemmi.cif.read(str(mmcif_path))[0]

    structure = gemmi.make_structure_from_block(block)
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    entities = {}
    for entity in structure.entities:
        if entity.entity_type.name == "Water":
            continue
        for chain_id in entity.subchains:
            entities[chain_id] = entity

    model = structure[0]
    prots = []
    for chain in model:
        polymer = chain.get_polymer()
        subchain_id = polymer.subchain_id()
        entity = entities[subchain_id]
        polymer_type = entity.polymer_type.name
        if polymer_type not in {"PeptideL", "PeptideD"}:
            continue

        sequence = entity.full_sequence
        coords = np.zeros((len(sequence), 37, 3))
        mask = np.zeros((len(sequence), 37), dtype=bool)

        result = gemmi.align_sequence_to_polymer(
            sequence,
            polymer,
            gemmi.PolymerType.PeptideL,
            gemmi.AlignmentScoring(),
        )

        i = 0
        for j, match in enumerate(result.match_string):
            if match != "|":
                continue

            res = polymer[i]
            if res.name != sequence[j]:
                raise ValueError("Alignment mismatch!")

            # Load atoms
            for atom in res:
                if atom.name in residue_constants.atom_order:
                    coords[j, residue_constants.atom_order[atom.name]] = atom.pos.tolist()
                    mask[j, residue_constants.atom_order[atom.name]] = 1.0
                elif atom.name.upper() == "SE" and res.name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    coords[j, residue_constants.atom_order["SD"]] = atom.pos.tolist()
                    mask[j, residue_constants.atom_order["SD"]] = 1.0

            # Increment polymer index
            i += 1

        seq = gemmi.one_letter_code(sequence)
        seq = np.array([restype_order_with_x.get(r, restype_order_with_x["X"]) for r in seq], dtype=np.int32)
        prot = {
            'chain_id': polymer.subchain_id(),
            'seq': seq,
            'atom37': coords,
            'atom_mask': mask
        }
        prots.append(prot)
    return prots