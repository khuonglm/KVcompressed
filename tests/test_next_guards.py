"""With kvsel_next present, kvsel's freeze records still accept the code (LLaVA-NeXT spec, unchanged guards)."""

import json
from pathlib import Path

import pytest

import kvsel_next  # noqa: F401  the package these guards are about must be importable
from kvsel.confirm import code_hash_all
from kvsel.confirm_analysis import CODE_AMENDMENTS, code_accepted
from kvsel.freeze import DEFAULT_RECORD, code_hash
from kvsel.replication.protocol import code_hash_tree

FREEZE = Path(__file__).parents[1] / "freeze"


def frozen(name):
    record = FREEZE / name
    if not record.exists():
        pytest.skip(f"no {name} in this checkout")
    return json.loads(record.read_text())["code_sha256"]


def test_stage1_screening_hash_matches_its_record():
    assert code_hash() == frozen(DEFAULT_RECORD.name)


def test_stage3_hash_is_accepted():
    assert code_accepted(frozen("comparative_gqa.json"), code_hash_all(), FREEZE / CODE_AMENDMENTS)


def test_replication_tree_hash_is_accepted():
    assert code_accepted(frozen("replication_gqa.json"), code_hash_tree(), FREEZE / CODE_AMENDMENTS)


def test_next_family_is_not_in_kvsels_table():
    from kvsel.families import FAMILIES

    assert set(FAMILIES) == {"llava"}
