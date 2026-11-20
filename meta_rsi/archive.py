"""DarwinX / DGM Lineage Archive and Gating Policies (naive, preserve, metamorphic)."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ArchiveEntry:
    gen: int
    parent_gen: int | None
    harness_path: str
    rationale: str
    dev_low_rate: float
    in_loop_rate: float
    dev_error_rate: float
    avg_llm_calls: float
    solved_ids: list[str]
    oracle_dev_rate: float | None = None
    oracle_dev_lucky: float | None = None
    oracle_test_seen_rate: float | None = None
    oracle_test_heldout_rate: float | None = None
    children: int = 0
    recombined_from: int | None = None


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    newly_solved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    meta_blocked: list[str] = field(default_factory=list)


def evaluate_gate(
    gate_mode: str,
    parent: ArchiveEntry,
    candidate_eval: dict,
    *,
    max_regress: int = 0,
    allow_stepping_stone: bool = True,
) -> GateDecision:
    """Decide whether candidate harness enters the archive using STRICTLY in-loop signals."""
    s = candidate_eval["summary"]
    records = candidate_eval["records"]

    parent_solved = set(parent.solved_ids)
    cand_solved = set(s["solved_ids_in_loop"])
    newly_solved = sorted(cand_solved - parent_solved)
    regressed = sorted(parent_solved - cand_solved)
    meta_blocked = sorted(
        r["task_id"] for r in records if r["dev_low_ok"] and r["meta_checked"] and not r["meta_consistent"]
    )

    if gate_mode == "naive":
        if s["dev_low_rate"] > parent.dev_low_rate:
            return GateDecision(
                accepted=True,
                reason=f"naive: dev_low_rate improved {parent.dev_low_rate:.3f} -> {s['dev_low_rate']:.3f} "
                       f"(+{len(newly_solved)} / -{len(regressed)})",
                newly_solved=newly_solved,
                regressed=regressed,
            )
        return GateDecision(
            accepted=False,
            reason=f"naive: dev_low_rate did not improve ({parent.dev_low_rate:.3f} -> {s['dev_low_rate']:.3f})",
            newly_solved=newly_solved,
            regressed=regressed,
        )

    if gate_mode in ("preserve", "metamorphic", "oracle"):
        tag = {
            "preserve": "preserve-and-extend",
            "metamorphic": "metamorphic-preserve-and-extend",
            "oracle": "oracle-preserve-and-extend",
        }[gate_mode]
        if len(regressed) > max_regress:
            # DarwinX/DGM scale-invariant multi-lineage stepping-stone admission:
            # If a major structural rewrite yields >=3x more newly verified tasks than regressions
            # (with regression rate <= 15% of dev set), admit as an alternative archive lineage
            # so recombination can later recover the few regressed tasks.
            n_tasks = max(int(s.get("n_tasks") or 1), 1)
            max_ss_regress = max(2, int(round(0.15 * n_tasks)))
            if (
                allow_stepping_stone
                and len(newly_solved) >= 2
                and len(regressed) <= max_ss_regress
                and len(newly_solved) >= 3 * len(regressed)
            ):
                return GateDecision(
                    accepted=True,
                    reason=f"{tag}[stepping-stone]: +{len(newly_solved)} solved, -{len(regressed)} regressed "
                           f"(in_loop {parent.in_loop_rate:.3f} -> {s['in_loop_rate']:.3f}, "
                           f"meta_blocked={len(meta_blocked)})",
                    newly_solved=newly_solved,
                    regressed=regressed,
                    meta_blocked=meta_blocked,
                )
            return GateDecision(
                accepted=False,
                reason=f"{tag}: rejected due to regression on {len(regressed)} tasks ({regressed[:3]}), "
                       f"max_regress={max_regress}; meta_blocked={len(meta_blocked)}",
                newly_solved=newly_solved,
                regressed=regressed,
                meta_blocked=meta_blocked,
            )
        if len(newly_solved) == 0:
            # Allow strictly cleaner/safer tie-break if error rate drops significantly with 0 regression
            # and (in metamorphic mode) no unverified lucky hacks introduced.
            if (
                len(regressed) == 0
                and (gate_mode != "metamorphic" or len(meta_blocked) == 0)
                and s["dev_error_rate"] + 0.05 < parent.dev_error_rate
            ):
                return GateDecision(
                    accepted=True,
                    reason=f"{tag}: accepted via stability gain (error {parent.dev_error_rate:.3f} -> "
                           f"{s['dev_error_rate']:.3f}, 0 regressions, meta_blocked={len(meta_blocked)})",
                    newly_solved=newly_solved,
                    regressed=regressed,
                    meta_blocked=meta_blocked,
                )
            return GateDecision(
                accepted=False,
                reason=f"{tag}: no new verified tasks solved (in_loop={s['in_loop_rate']:.3f}, "
                       f"dev_low={s['dev_low_rate']:.3f}, meta_blocked={len(meta_blocked)})",
                newly_solved=newly_solved,
                regressed=regressed,
                meta_blocked=meta_blocked,
            )
        return GateDecision(
            accepted=True,
            reason=f"{tag}: accepted (+{len(newly_solved)} solved, -{len(regressed)} regressed, "
                   f"in_loop {parent.in_loop_rate:.3f} -> {s['in_loop_rate']:.3f}, "
                   f"meta_blocked={len(meta_blocked)})",
            newly_solved=newly_solved,
            regressed=regressed,
            meta_blocked=meta_blocked,
        )

    raise ValueError(f"Unknown gate_mode: {gate_mode}")


class LineageArchive:
    def __init__(self, run_dir: Path, gate_mode: str):
        self.run_dir = run_dir
        self.gate_mode = gate_mode
        self.entries: list[ArchiveEntry] = []
        self.attempts: list[dict] = []

    def add(self, entry: ArchiveEntry) -> None:
        if entry.parent_gen is not None:
            for e in self.entries:
                if e.gen == entry.parent_gen:
                    e.children += 1
                    break
        self.entries.append(entry)

    def record_attempt(self, item: dict) -> None:
        self.attempts.append(item)

    def best(self) -> ArchiveEntry:
        return max(self.entries, key=lambda e: (e.in_loop_rate, -e.dev_error_rate, e.gen))

    def select_parent(self, rng: random.Random) -> tuple[ArchiveEntry, ArchiveEntry | None]:
        """DGM/DarwinX parent selection: balance in-loop score and novelty (fewer children).

        Also returns an optional complementary partner from the archive for recombination
        when another lineage solves tasks the selected parent misses.
        """
        if len(self.entries) == 1:
            return self.entries[0], None
        if self.gate_mode == "naive":
            return self.best(), None

        weights = []
        for e in self.entries:
            score_w = 0.25 + e.in_loop_rate
            novelty_w = 1.0 / (1.0 + e.children)
            weights.append(score_w * novelty_w)
        parent = rng.choices(self.entries, weights=weights, k=1)[0]

        # Check if any other archive entry solves tasks that `parent` does NOT solve.
        p_set = set(parent.solved_ids)
        partner = None
        best_gain = 0
        for cand in self.entries:
            if cand.gen == parent.gen:
                continue
            gain = len(set(cand.solved_ids) - p_set)
            if gain > best_gain:
                best_gain = gain
                partner = cand
        return parent, partner

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "gate_mode": self.gate_mode,
            "entries": [asdict(e) for e in self.entries],
            "attempts": self.attempts,
        }
        (self.run_dir / "archive.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "LineageArchive":
        data = json.loads((run_dir / "archive.json").read_text())
        arch = cls(run_dir, data["gate_mode"])
        arch.entries = [ArchiveEntry(**e) for e in data["entries"]]
        arch.attempts = data.get("attempts", [])
        return arch
