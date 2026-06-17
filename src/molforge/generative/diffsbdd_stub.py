from __future__ import annotations

from contracts.schema import BindingPocket, GeneratedMolecule


class DiffSBDDBackend:
    name: str = "diffsbdd"

    def generate(
        self,
        pocket: BindingPocket,
        n: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]:
        _ = (pocket, n, seed_smiles)
        raise NotImplementedError(
            "DiffSBDDBackend is intentionally stubbed in Phase 3. "
            + "The public backend slot is reserved for a later pocket-aware implementation."
        )
