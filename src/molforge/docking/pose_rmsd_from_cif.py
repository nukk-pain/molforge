"""Compute RMSD between a Boltz-predicted pose and a PoseBench reference pose.

Boltz emits structures in PDB (or CIF); the PoseBench reference is stored as
PDBQT. Both formats tag the ligand with HETATM records. We:

- extract heavy-atom coordinates from each file (skipping hydrogens)
- enforce atom count equality (same ligand, same molecule)
- delegate to `molforge.docking.posebench._aligned_rmsd` for Kabsch alignment
  + RMSD, matching the signature of the existing Phase 2 gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .posebench import _aligned_rmsd


ResidueKey = tuple[int, str]
Coord3D = tuple[float, float, float]
_STANDARD_RESIDUE_TO_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True, slots=True)
class CAResidue:
    key: ResidueKey
    resname: str
    coord: Coord3D


@dataclass(frozen=True, slots=True)
class SequenceFallbackResult:
    predicted_coords: list[Coord3D] | None
    crystal_coords: list[Coord3D] | None
    predicted_sequence_length: int
    crystal_sequence_length: int
    unique_hit_count: int
    matched_len: int
    reason: str | None


def extract_ligand_heavy_atom_coords(
    path: Path, *, chain_id: str | None = None
) -> list[tuple[float, float, float]]:
    """Read HETATM (or ATOM in ligand chain) heavy-atom coords from PDB/PDBQT.

    Respects MODEL/ENDMDL blocks when present (Vina multi-pose PDBQT):
    only atoms inside the first MODEL block are returned. When no MODEL
    record exists, every HETATM is collected. Matches
    `posebench._extract_pose_coordinates` semantics.

    `chain_id` restricts to a specific chain when Boltz puts the ligand on a
    known chain ID (e.g., "B"). When None, every HETATM heavy atom is taken.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    has_model_records = any(line.startswith("MODEL") for line in lines)

    coords: list[tuple[float, float, float]] = []
    in_model = not has_model_records
    for line in lines:
        if line.startswith("MODEL"):
            if coords:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            if coords:
                break
            in_model = False
            continue
        if not line.startswith(("HETATM", "ATOM")):
            continue
        if not in_model:
            continue
        if chain_id is not None and len(line) >= 22:
            line_chain = line[21].strip()
            # When `chain_id` is set, filter by it but still allow the ligand chain
            # if it matches exactly (PDBQT reference may not have chain id set).
            if line_chain != chain_id and line_chain != "":
                continue
        # Skip hydrogens (element column or trailing atom name heuristic).
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element == "H" or atom_name.startswith("H"):
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
            except ValueError:
                continue
        coords.append((x, y, z))
    return coords


def extract_ligand_heavy_atom_coords_sdf(
    sdf_path: Path,
) -> list[tuple[float, float, float]]:
    """Read heavy-atom coordinates from an RDKit-parseable SDF.

    DiffDock-L emits pose files in SDF rather than PDB. SDF's V2000 block
    puts coordinates in lines 4..(4+atom_count) with a "xx.xxxx  yy.yyyy
    zz.zzzz  Elem ..." layout. We parse via RDKit to avoid reimplementing
    the V2000 spec — it handles both V2000 and V3000.
    """
    from rdkit import Chem  # lazy

    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
    mol = next(iter(supplier), None)
    if mol is None:
        return []
    conf = mol.GetConformer()
    coords: list[tuple[float, float, float]] = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == "H":
            continue
        p = conf.GetAtomPosition(i)
        coords.append((p.x, p.y, p.z))
    return coords


def rmsd_against_reference(
    predicted_path: Path,
    reference_pdbqt: Path,
    *,
    predicted_chain_id: str | None = None,
) -> float:
    """Kabsch-aligned heavy-atom RMSD between predicted pose and reference.

    ⚠️ **Assumes atom ordering matches between predicted and reference.**
    For Boltz-2 output (PDB/CIF) compared to the same-stem PoseBench
    reference that was generated from the same HETATM block, atom order
    lines up and this function is correct.

    For DiffDock SDFs or anywhere the atom order is NOT guaranteed to
    match, use `rmsd_against_reference_symmetry_aware` instead — the
    naive ordering here can over-estimate RMSD by many Å on molecules
    with aromatic ring symmetries (observed 3ERT_OHT: 7.03Å naive vs
    0.33Å symmetry-aware).
    """
    if predicted_path.suffix.lower() == ".sdf":
        predicted = extract_ligand_heavy_atom_coords_sdf(predicted_path)
    else:
        predicted = extract_ligand_heavy_atom_coords(
            predicted_path, chain_id=predicted_chain_id
        )
    reference = extract_ligand_heavy_atom_coords(reference_pdbqt)
    if not predicted or not reference:
        raise ValueError(
            f"Empty ligand coord list — predicted={len(predicted)}, "
            f"reference={len(reference)} (paths: {predicted_path}, {reference_pdbqt})"
        )
    if len(predicted) != len(reference):
        raise ValueError(
            f"Ligand atom count mismatch: predicted has {len(predicted)} heavy "
            f"atoms, reference has {len(reference)}. Same molecule required for "
            f"RMSD; check ligand preparation / SMILES equivalence."
        )
    return _aligned_rmsd(reference, predicted)


def _load_pose_mol(
    path: Path,
    ligand_smiles: str,
    *,
    is_boltz_complex: bool,
):
    """Load a single RDKit Mol for the ligand at `path`, with bond orders
    assigned from the SMILES template."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from .boltz_pdb_split import split_boltz_pdb

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"invalid template SMILES: {ligand_smiles!r}")

    if path.suffix.lower() == ".sdf":
        mol = next(
            iter(Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)),
            None,
        )
    elif is_boltz_complex or (
        path.suffix.lower() in (".pdb", ".cif") and "boltz_out_" in str(path)
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            split = split_boltz_pdb(path, ligand_smiles, Path(tmp))
            mol = next(
                iter(
                    Chem.SDMolSupplier(
                        str(split.ligand_sdf), sanitize=True, removeHs=True
                    )
                ),
                None,
            )
    else:
        raw = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=True)
        if raw is None:
            mol = None
        else:
            mol = AllChem.AssignBondOrdersFromTemplate(template, raw)

    if mol is None:
        raise ValueError(f"could not load ligand from {path}")
    return mol


def rmsd_positional_symmetry_aware(
    predicted_path: Path,
    reference_path: Path,
    ligand_smiles: str,
    *,
    predicted_is_boltz_complex: bool = False,
    reference_is_boltz_complex: bool = False,
) -> float:
    """Positional heavy-atom RMSD with symmetry-aware atom matching.

    NO rigid-body alignment is performed — coordinates are used as-is
    from both files, and the minimum RMSD is taken over all automorphism
    mappings (symmetry equivalent atom re-labellings) discovered via
    substructure matching against the SMILES template.

    Correct metric for pose evaluation when both predicted and reference
    are already in the same coordinate frame (e.g. DiffDock receiving
    the crystal PDB as input). For Boltz-2 (which folds its own
    receptor), use `rmsd_receptor_aligned_symmetry_aware` so the
    predicted ligand is first transformed into the crystal protein's
    frame via Cα superposition.
    """
    from rdkit import Chem

    pred = _load_pose_mol(
        predicted_path, ligand_smiles, is_boltz_complex=predicted_is_boltz_complex
    )
    ref = _load_pose_mol(
        reference_path, ligand_smiles, is_boltz_complex=reference_is_boltz_complex
    )
    if pred.GetNumAtoms() != ref.GetNumAtoms():
        raise ValueError(
            f"heavy atom count mismatch: predicted={pred.GetNumAtoms()}, "
            f"reference={ref.GetNumAtoms()}"
        )

    # Enumerate all substructure matches of pred onto ref (gives every
    # symmetry-equivalent atom mapping), compute positional RMSD for each,
    # return the minimum.
    matches = ref.GetSubstructMatches(pred, uniquify=False)
    if not matches:
        matches = [tuple(range(pred.GetNumAtoms()))]  # identity fallback

    pred_conf = pred.GetConformer()
    ref_conf = ref.GetConformer()
    pred_coords = [pred_conf.GetAtomPosition(i) for i in range(pred.GetNumAtoms())]

    best = float("inf")
    for mapping in matches:
        total_sq = 0.0
        for i, j in enumerate(mapping):
            r = ref_conf.GetAtomPosition(j)
            p = pred_coords[i]
            dx = p.x - r.x
            dy = p.y - r.y
            dz = p.z - r.z
            total_sq += dx * dx + dy * dy + dz * dz
        rmsd = (total_sq / pred.GetNumAtoms()) ** 0.5
        if rmsd < best:
            best = rmsd
    return best


def _format_residue_key(key: ResidueKey) -> str:
    resid, insertion_code = key
    return f"{resid}{insertion_code}" if insertion_code else str(resid)


def _ordered_ca_residues_from_pdb_text(
    text: str, *, chain: str = "A"
) -> list[CAResidue]:
    """Ordered Cα residues for a chain, preserving file order."""
    residues: list[CAResidue] = []
    seen: set[ResidueKey] = set()
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip() if len(line) >= 27 else ""
        residue_key = (resid, insertion_code)
        if residue_key in seen:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        residues.append(
            CAResidue(
                key=residue_key,
                resname=line[17:20].strip().upper(),
                coord=(x, y, z),
            )
        )
        seen.add(residue_key)
    return residues


def _ca_coords_from_pdb_text(
    text: str, *, chain: str = "A"
) -> dict[ResidueKey, tuple[float, float, float]]:
    """Map of `(residue_seq_id, insertion_code)` → Cα (x, y, z) for a chain.

    Handles both raw crystal PDBs (RCSB download) and Boltz complex PDBs.
    Only altloc-default atoms and chain `chain` are included.
    """
    out: dict[ResidueKey, tuple[float, float, float]] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        if line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip() if len(line) >= 27 else ""
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        residue_key = (resid, insertion_code)
        # If the same residue id appears twice (rare), keep the first.
        out.setdefault(residue_key, (x, y, z))
    return out


def _residues_to_one_letter_sequence(residues: list[CAResidue]) -> str | None:
    letters: list[str] = []
    for residue in residues:
        letter = _STANDARD_RESIDUE_TO_ONE_LETTER.get(residue.resname)
        if letter is None:
            return None
        letters.append(letter)
    return "".join(letters)


def _unique_subsequence_hits(longer: str, shorter: str) -> list[int]:
    hits: list[int] = []
    start = longer.find(shorter)
    while start != -1:
        hits.append(start)
        start = longer.find(shorter, start + 1)
    return hits


def _contiguous_ca_blocks(residues: list[CAResidue]) -> list[list[CAResidue]]:
    blocks: list[list[CAResidue]] = []
    current: list[CAResidue] = []
    for residue in residues:
        if not current:
            current = [residue]
            continue
        prev = current[-1]
        prev_resid, prev_insertion = prev.key
        resid, insertion = residue.key
        is_contiguous = (
            prev_insertion == "" and insertion == "" and resid == prev_resid + 1
        )
        if is_contiguous:
            current.append(residue)
            continue
        blocks.append(current)
        current = [residue]
    if current:
        blocks.append(current)
    return blocks


def _find_unique_ordered_block_placement(
    longer_sequence: str,
    block_sequences: list[str],
) -> tuple[list[int] | None, int]:
    hit_options = [
        _unique_subsequence_hits(longer_sequence, block_sequence)
        for block_sequence in block_sequences
    ]
    if any(not hits for hits in hit_options):
        return None, 0

    placements: list[list[int]] = []

    def _search(block_index: int, next_min_start: int, chosen: list[int]) -> None:
        if len(placements) > 1:
            return
        if block_index == len(block_sequences):
            placements.append(chosen.copy())
            return
        block_length = len(block_sequences[block_index])
        for start in hit_options[block_index]:
            if start < next_min_start:
                continue
            chosen.append(start)
            _search(block_index + 1, start + block_length, chosen)
            chosen.pop()

    _search(0, 0, [])
    if len(placements) != 1:
        return None, len(placements)
    return placements[0], 1


def _find_unique_exact_ca_gapped_pairs(
    predicted_residues: list[CAResidue],
    crystal_residues: list[CAResidue],
    *,
    min_common: int = 10,
) -> SequenceFallbackResult:
    predicted_sequence = _residues_to_one_letter_sequence(predicted_residues)
    crystal_sequence = _residues_to_one_letter_sequence(crystal_residues)
    predicted_len = len(predicted_residues)
    crystal_len = len(crystal_residues)

    if predicted_sequence is None or crystal_sequence is None:
        return SequenceFallbackResult(
            predicted_coords=None,
            crystal_coords=None,
            predicted_sequence_length=predicted_len,
            crystal_sequence_length=crystal_len,
            unique_hit_count=0,
            matched_len=0,
            reason="non_standard_residue",
        )

    shorter_residues = predicted_residues
    longer_residues = crystal_residues
    shorter_sequence = predicted_sequence
    longer_sequence = crystal_sequence
    predicted_is_shorter = True
    if predicted_len > crystal_len:
        shorter_residues = crystal_residues
        longer_residues = predicted_residues
        shorter_sequence = crystal_sequence
        longer_sequence = predicted_sequence
        predicted_is_shorter = False

    if len(shorter_residues) < min_common:
        return SequenceFallbackResult(
            predicted_coords=None,
            crystal_coords=None,
            predicted_sequence_length=predicted_len,
            crystal_sequence_length=crystal_len,
            unique_hit_count=0,
            matched_len=len(shorter_residues),
            reason="sequence_too_short",
        )

    shorter_blocks = _contiguous_ca_blocks(shorter_residues)
    block_sequences = []
    for block in shorter_blocks:
        block_sequence = _residues_to_one_letter_sequence(block)
        if block_sequence is None:
            return SequenceFallbackResult(
                predicted_coords=None,
                crystal_coords=None,
                predicted_sequence_length=predicted_len,
                crystal_sequence_length=crystal_len,
                unique_hit_count=0,
                matched_len=0,
                reason="non_standard_residue",
            )
        block_sequences.append(block_sequence)

    placements, placement_count = _find_unique_ordered_block_placement(
        longer_sequence,
        block_sequences,
    )
    if placements is None:
        return SequenceFallbackResult(
            predicted_coords=None,
            crystal_coords=None,
            predicted_sequence_length=predicted_len,
            crystal_sequence_length=crystal_len,
            unique_hit_count=placement_count,
            matched_len=0,
            reason=(
                "ambiguous_exact_gapped_match"
                if placement_count > 1
                else "no_exact_gapped_match"
            ),
        )

    shorter_coords: list[Coord3D] = []
    longer_coords: list[Coord3D] = []
    for block, start in zip(shorter_blocks, placements):
        matched_block = longer_residues[start : start + len(block)]
        if len(matched_block) != len(block):
            return SequenceFallbackResult(
                predicted_coords=None,
                crystal_coords=None,
                predicted_sequence_length=predicted_len,
                crystal_sequence_length=crystal_len,
                unique_hit_count=0,
                matched_len=0,
                reason="no_exact_gapped_match",
            )
        for shorter_residue, longer_residue in zip(block, matched_block):
            if shorter_residue.resname != longer_residue.resname:
                return SequenceFallbackResult(
                    predicted_coords=None,
                    crystal_coords=None,
                    predicted_sequence_length=predicted_len,
                    crystal_sequence_length=crystal_len,
                    unique_hit_count=0,
                    matched_len=0,
                    reason="no_exact_gapped_match",
                )
            shorter_coords.append(shorter_residue.coord)
            longer_coords.append(longer_residue.coord)

    if predicted_is_shorter:
        predicted_coords = shorter_coords
        crystal_coords = longer_coords
    else:
        predicted_coords = longer_coords
        crystal_coords = shorter_coords

    return SequenceFallbackResult(
        predicted_coords=predicted_coords,
        crystal_coords=crystal_coords,
        predicted_sequence_length=predicted_len,
        crystal_sequence_length=crystal_len,
        unique_hit_count=1,
        matched_len=len(shorter_coords),
        reason=None,
    )


def _find_unique_exact_ca_subsequence_pairs(
    predicted_residues: list[CAResidue],
    crystal_residues: list[CAResidue],
    *,
    min_common: int = 10,
) -> SequenceFallbackResult:
    predicted_sequence = _residues_to_one_letter_sequence(predicted_residues)
    crystal_sequence = _residues_to_one_letter_sequence(crystal_residues)
    predicted_len = len(predicted_residues)
    crystal_len = len(crystal_residues)

    if predicted_sequence is None or crystal_sequence is None:
        return SequenceFallbackResult(
            predicted_coords=None,
            crystal_coords=None,
            predicted_sequence_length=predicted_len,
            crystal_sequence_length=crystal_len,
            unique_hit_count=0,
            matched_len=0,
            reason="non_standard_residue",
        )

    shorter_len = min(predicted_len, crystal_len)
    if shorter_len < min_common:
        return SequenceFallbackResult(
            predicted_coords=None,
            crystal_coords=None,
            predicted_sequence_length=predicted_len,
            crystal_sequence_length=crystal_len,
            unique_hit_count=0,
            matched_len=shorter_len,
            reason="sequence_too_short",
        )

    if predicted_len <= crystal_len:
        hit_positions = _unique_subsequence_hits(crystal_sequence, predicted_sequence)
        if len(hit_positions) != 1:
            return SequenceFallbackResult(
                predicted_coords=None,
                crystal_coords=None,
                predicted_sequence_length=predicted_len,
                crystal_sequence_length=crystal_len,
                unique_hit_count=len(hit_positions),
                matched_len=shorter_len,
                reason=(
                    "ambiguous_exact_subsequence"
                    if len(hit_positions) > 1
                    else "no_exact_subsequence"
                ),
            )
        predicted_slice = predicted_residues
        crystal_slice = crystal_residues[
            hit_positions[0] : hit_positions[0] + shorter_len
        ]
    else:
        hit_positions = _unique_subsequence_hits(predicted_sequence, crystal_sequence)
        if len(hit_positions) != 1:
            return SequenceFallbackResult(
                predicted_coords=None,
                crystal_coords=None,
                predicted_sequence_length=predicted_len,
                crystal_sequence_length=crystal_len,
                unique_hit_count=len(hit_positions),
                matched_len=shorter_len,
                reason=(
                    "ambiguous_exact_subsequence"
                    if len(hit_positions) > 1
                    else "no_exact_subsequence"
                ),
            )
        predicted_slice = predicted_residues[
            hit_positions[0] : hit_positions[0] + shorter_len
        ]
        crystal_slice = crystal_residues

    return SequenceFallbackResult(
        predicted_coords=[residue.coord for residue in predicted_slice],
        crystal_coords=[residue.coord for residue in crystal_slice],
        predicted_sequence_length=predicted_len,
        crystal_sequence_length=crystal_len,
        unique_hit_count=1,
        matched_len=shorter_len,
        reason=None,
    )


def _kabsch_rotation_translation(
    source: list[tuple[float, float, float]],
    target: list[tuple[float, float, float]],
):
    """Return (R, t) such that R @ source + t ≈ target, as plain lists.

    Uses NumPy internally (imported lazily) for SVD. Avoids adding NumPy
    as a hard dep of the top-level module by deferring the import.
    """
    import numpy as np

    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"source/target must be matching (N, 3) arrays (got {src.shape} vs {tgt.shape})"
        )
    if src.shape[0] < 3:
        raise ValueError(
            f"at least 3 paired points required for Kabsch (got {src.shape[0]})"
        )
    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    H = (src - src_centroid).T @ (tgt - tgt_centroid)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = tgt_centroid - R @ src_centroid
    return R, t, src_centroid, tgt_centroid


def rmsd_receptor_aligned_symmetry_aware(
    predicted_complex_pdb: Path,
    crystal_pdb_path: Path,
    ligand_code: str,
    ligand_smiles: str,
    *,
    predicted_chain: str = "A",
    crystal_chain: str = "A",
) -> tuple[float, float, int]:
    """Receptor-aligned, symmetry-aware positional RMSD for Boltz-style poses.

    Flow:
      1. Extract Cα coordinates from both predicted and crystal PDBs.
      2. Find common residue indices, Kabsch-align predicted onto crystal.
      3. Apply the transformation to the ligand atoms extracted from the
         predicted complex via `split_boltz_pdb`.
      4. Extract the crystal ligand (HETATM matching `ligand_code` on
         `crystal_chain`) into an RDKit Mol with bond orders from
         `ligand_smiles`.
      5. Return the symmetry-aware positional RMSD (minimum over
         substructure-match atom mappings) between the transformed
         predicted ligand and the crystal ligand. NO further alignment.

    Returns `(ligand_rmsd, ca_rmsd, n_common_ca)` so callers can report
    Cα alignment quality alongside the pose RMSD — a large Cα RMSD means
    the predicted protein frame differs substantially from crystal, and
    the ligand RMSD is correspondingly less meaningful.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from .boltz_pdb_split import split_boltz_pdb

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"invalid template SMILES: {ligand_smiles!r}")

    predicted_text = predicted_complex_pdb.read_text(encoding="utf-8")
    crystal_text = crystal_pdb_path.read_text(encoding="utf-8")

    predicted_ca = _ca_coords_from_pdb_text(predicted_text, chain=predicted_chain)
    crystal_ca = _ca_coords_from_pdb_text(crystal_text, chain=crystal_chain)
    common = sorted(set(predicted_ca) & set(crystal_ca))
    common_count = len(common)
    if len(common) < 10:
        predicted_ordered = _ordered_ca_residues_from_pdb_text(
            predicted_text, chain=predicted_chain
        )
        crystal_ordered = _ordered_ca_residues_from_pdb_text(
            crystal_text, chain=crystal_chain
        )
        fallback = _find_unique_exact_ca_subsequence_pairs(
            predicted_ordered,
            crystal_ordered,
            min_common=10,
        )
        if fallback.predicted_coords is None or fallback.crystal_coords is None:
            fallback = _find_unique_exact_ca_gapped_pairs(
                predicted_ordered,
                crystal_ordered,
                min_common=10,
            )
        if (
            fallback.predicted_coords is not None
            and fallback.crystal_coords is not None
        ):
            predicted_ca_pts = fallback.predicted_coords
            crystal_ca_pts = fallback.crystal_coords
            common_count = fallback.matched_len
        else:
            predicted_only = sorted(set(predicted_ca) - set(crystal_ca))
            crystal_only = sorted(set(crystal_ca) - set(predicted_ca))
            predicted_has_insertion_codes = any(key[1] for key in predicted_ca)
            crystal_has_insertion_codes = any(key[1] for key in crystal_ca)
            predicted_sample = [
                _format_residue_key(key) for key in list(predicted_only[:5])
            ]
            crystal_sample = [
                _format_residue_key(key) for key in list(crystal_only[:5])
            ]
            raise ValueError(
                f"too few common Cα (predicted chain {predicted_chain} ∩ crystal "
                f"chain {crystal_chain}): {len(common)} "
                f"(predicted={len(predicted_ca)}, crystal={len(crystal_ca)}). "
                f"Likely a residue-id mismatch. insertion_codes_present="
                f"predicted:{predicted_has_insertion_codes},"
                f"crystal:{crystal_has_insertion_codes}; "
                f"predicted_only_sample={predicted_sample}; "
                f"crystal_only_sample={crystal_sample}; "
                f"fallback_attempted=True; "
                f"fallback_reason={fallback.reason}; "
                f"fallback_predicted_sequence_length={fallback.predicted_sequence_length}; "
                f"fallback_crystal_sequence_length={fallback.crystal_sequence_length}; "
                f"fallback_unique_hit_count={fallback.unique_hit_count}."
            )
    else:
        predicted_ca_pts = [predicted_ca[i] for i in common]
        crystal_ca_pts = [crystal_ca[i] for i in common]
    R, t, src_c, tgt_c = _kabsch_rotation_translation(predicted_ca_pts, crystal_ca_pts)

    predicted_ca_np = np.asarray(predicted_ca_pts)
    crystal_ca_np = np.asarray(crystal_ca_pts)
    ca_transformed = (predicted_ca_np - src_c) @ R.T + tgt_c
    ca_rmsd = float(
        np.sqrt(np.mean(np.sum((ca_transformed - crystal_ca_np) ** 2, axis=1)))
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        split = split_boltz_pdb(predicted_complex_pdb, ligand_smiles, Path(tmp))
        predicted_lig = next(
            iter(
                Chem.SDMolSupplier(str(split.ligand_sdf), sanitize=True, removeHs=True)
            ),
            None,
        )
    if predicted_lig is None:
        raise ValueError(
            f"could not split ligand out of Boltz complex at {predicted_complex_pdb}"
        )

    conf = predicted_lig.GetConformer()
    pred_xyz = np.asarray(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(predicted_lig.GetNumAtoms())
        ]
    )
    pred_transformed = (pred_xyz - src_c) @ R.T + tgt_c

    crystal_lig_lines: list[str] = []
    for line in crystal_text.splitlines():
        if (
            line.startswith("HETATM")
            and len(line) >= 27
            and line[17:20].strip() == ligand_code
            and line[21] == crystal_chain
            and line[16] in (" ", "A")
        ):
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            name = line[12:16].strip().upper()
            if element == "H" or name.startswith("H"):
                continue
            crystal_lig_lines.append(line)
    if not crystal_lig_lines:
        raise ValueError(
            f"no crystal ligand HETATM for code {ligand_code!r} on chain "
            f"{crystal_chain} in {crystal_pdb_path}"
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write("\n".join(crystal_lig_lines) + "\nEND\n")
        ref_path = f.name
    raw = Chem.MolFromPDBFile(ref_path, sanitize=False, removeHs=True)
    if raw is None:
        raise ValueError(f"could not parse crystal ligand atoms for {ligand_code}")
    crystal_lig = AllChem.AssignBondOrdersFromTemplate(template, raw)
    ref_conf = crystal_lig.GetConformer()
    ref_xyz = np.asarray(
        [
            [
                ref_conf.GetAtomPosition(i).x,
                ref_conf.GetAtomPosition(i).y,
                ref_conf.GetAtomPosition(i).z,
            ]
            for i in range(crystal_lig.GetNumAtoms())
        ]
    )
    if predicted_lig.GetNumAtoms() != crystal_lig.GetNumAtoms():
        raise ValueError(
            f"heavy atom count mismatch: predicted={predicted_lig.GetNumAtoms()}, "
            f"crystal={crystal_lig.GetNumAtoms()}"
        )

    matches = crystal_lig.GetSubstructMatches(predicted_lig, uniquify=False)
    if not matches:
        matches = [tuple(range(predicted_lig.GetNumAtoms()))]
    best = float("inf")
    for mapping in matches:
        reordered = np.asarray([ref_xyz[j] for j in mapping])
        rmsd = float(
            np.sqrt(np.mean(np.sum((pred_transformed - reordered) ** 2, axis=1)))
        )
        if rmsd < best:
            best = rmsd
    return best, ca_rmsd, common_count


def _detect_crystal_chains(crystal_text: str) -> list[str]:
    """Return unique chain IDs from ATOM records in PDB text, in appearance order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in crystal_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 22:
            continue
        chain = line[21]
        if chain and chain != " " and chain not in seen_set:
            seen.append(chain)
            seen_set.add(chain)
    return seen


def _crystal_chain_has_ligand(crystal_text: str, ligand_code: str, chain: str) -> bool:
    """Return True if any HETATM with ligand_code exists on the given chain."""
    for line in crystal_text.splitlines():
        if (
            line.startswith("HETATM")
            and len(line) >= 27
            and line[17:20].strip() == ligand_code
            and line[21] == chain
        ):
            return True
    return False


def rmsd_chain_equivalent_symmetry_aware(
    predicted_complex_pdb: Path,
    crystal_pdb: Path,
    ligand_code: str,
    ligand_smiles: str,
    *,
    predicted_chain: str = "A",
    crystal_chains: tuple[str, ...] | None = None,
) -> tuple[float, float, int, str]:
    """Chain-equivalent receptor-aligned symmetry-aware RMSD.

    Coordinate frame: predicted and crystal may differ by rigid transformation
    per-chain; applies Cα Kabsch alignment per candidate chain then ligand
    transform so both ligands are compared in the same crystal coordinate frame.
    Symmetry-aware: RDKit substructure match enumerates aromatic ring
    automorphisms.

    For each crystal chain in `crystal_chains` (default: auto-detect all chains
    with ATOM records), computes rmsd_receptor_aligned_symmetry_aware for that
    chain and returns the minimum. Falls back to single-chain behavior if the
    crystal has only one chain with the target ligand.

    Chains that do not contain a HETATM record matching `ligand_code` are
    silently skipped. If all candidate chains fail, the last exception is
    re-raised so the caller gets a meaningful error.

    Returns ``(min_ligand_rmsd, best_ca_rmsd, best_n_common_ca, best_chain_id)``
    so callers have provenance: which crystal chain produced the minimum RMSD
    and how well the Cα alignment performed for that chain.

    Typical use — homomeric targets (e.g. 1IEP BCR-ABL homodimer):
    DiffDock may place its rank-1 pose in chain B while the crystal reference
    ligand is in chain A. Without a chain prior, naive single-chain RMSD
    inflates to 37 Å. This function pre-specifies "pick the minimum across all
    crystal chains" as the honest production metric.
    """
    crystal_text = crystal_pdb.read_text(encoding="utf-8")

    if crystal_chains is None:
        chains_to_try = _detect_crystal_chains(crystal_text)
    else:
        chains_to_try = list(crystal_chains)

    # Filter to chains that actually carry the ligand
    eligible = [
        ch for ch in chains_to_try
        if _crystal_chain_has_ligand(crystal_text, ligand_code, ch)
    ]

    # If none carry the ligand (or crystal_chains was explicitly provided
    # with an override), fall back to the full candidate list so the
    # underlying function produces its own descriptive error.
    if not eligible:
        eligible = chains_to_try if chains_to_try else ["A"]

    best_ligand_rmsd = float("inf")
    best_ca_rmsd = float("inf")
    best_n_ca = 0
    best_chain = eligible[0]
    last_exc: Exception | None = None

    for chain in eligible:
        try:
            lig_rmsd, ca_rmsd, n_ca = rmsd_receptor_aligned_symmetry_aware(
                predicted_complex_pdb,
                crystal_pdb,
                ligand_code,
                ligand_smiles,
                predicted_chain=predicted_chain,
                crystal_chain=chain,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

        if lig_rmsd < best_ligand_rmsd:
            best_ligand_rmsd = lig_rmsd
            best_ca_rmsd = ca_rmsd
            best_n_ca = n_ca
            best_chain = chain

    if best_ligand_rmsd == float("inf"):
        if last_exc is not None:
            raise last_exc
        raise ValueError(
            f"No eligible crystal chain found for ligand {ligand_code!r} "
            f"in {crystal_pdb} (tried: {eligible})"
        )

    return best_ligand_rmsd, best_ca_rmsd, best_n_ca, best_chain


def rmsd_against_reference_symmetry_aware(
    predicted_path: Path,
    reference_pdbqt: Path,
    ligand_smiles: str,
    *,
    predicted_is_boltz_complex: bool = False,
) -> float:
    """⚠️ DEPRECATED for pose correctness evaluation. Prefer
    `rmsd_positional_symmetry_aware` (or `rmsd_receptor_aligned_symmetry_aware`
    when the predicted protein frame differs from the crystal's).

    `GetBestRMS` performs internal rigid-body alignment on the ligand
    before taking RMSD, which collapses translationally-offset
    conformers to ~0 Å. For evaluating whether a predicted pose is
    positioned correctly in the binding pocket, that alignment is
    unwanted — positional RMSD (no alignment) is the PoseBench
    convention.

    Kept here so prior reports remain reproducible.

    Loads both poses via RDKit, assigns bond orders from `ligand_smiles`
    (so aromatic rings are recognised), and computes the minimum RMSD
    over all chemically-equivalent atom mappings AFTER rigid-body
    alignment.

    Input handling:
    - `.sdf` predicted → RDKit SDF parser (DiffDock output).
    - `.pdb`/`.cif` predicted + `predicted_is_boltz_complex=False` →
      tries to parse the whole file as a single ligand (only correct if
      the file is ligand-only).
    - `predicted_is_boltz_complex=True` → runs `split_boltz_pdb` first
      to extract the chain-B HETATM ligand, then reads that SDF.
    - Auto-detection: if the path lives under a `boltz_out_*` tree
      (molforge convention), boltz-complex handling is enabled
      implicitly.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    from .boltz_pdb_split import split_boltz_pdb

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"invalid template SMILES: {ligand_smiles!r}")

    auto_boltz = predicted_path.suffix.lower() in (
        ".pdb",
        ".cif",
    ) and "boltz_out_" in str(predicted_path)
    is_boltz = predicted_is_boltz_complex or auto_boltz

    predicted: Chem.Mol | None = None
    if predicted_path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(predicted_path), sanitize=True, removeHs=True)
        predicted = next(iter(supplier), None)
    elif is_boltz:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            split = split_boltz_pdb(predicted_path, ligand_smiles, Path(tmp))
            predicted = next(
                iter(
                    Chem.SDMolSupplier(
                        str(split.ligand_sdf), sanitize=True, removeHs=True
                    )
                ),
                None,
            )
    else:
        predicted = Chem.MolFromPDBFile(
            str(predicted_path), sanitize=False, removeHs=True
        )
        if predicted is not None:
            predicted = AllChem.AssignBondOrdersFromTemplate(template, predicted)

    if predicted is None:
        raise ValueError(f"could not load predicted pose from {predicted_path}")

    reference = Chem.MolFromPDBFile(str(reference_pdbqt), sanitize=False, removeHs=True)
    if reference is None:
        raise ValueError(f"could not load reference pose from {reference_pdbqt}")
    reference = AllChem.AssignBondOrdersFromTemplate(template, reference)

    if predicted.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError(
            f"heavy atom count mismatch: predicted={predicted.GetNumAtoms()}, "
            f"reference={reference.GetNumAtoms()}"
        )

    return rdMolAlign.GetBestRMS(predicted, reference)
