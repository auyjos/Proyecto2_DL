"""Contrato del pipeline C1: secuencias, splits, features y artefactos."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.sequences import (
    ACCOUNT_COLUMNS,
    TRANSACTION_COLUMNS,
    build_artifacts,
    fit_preprocessor,
    get_sender,
    make_dataloaders,
    transform_sequence,
)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def transaction(
    timestamp: str,
    bank: str,
    account: str,
    destination: str,
    amount: float,
    label: int,
    *,
    currency: str = "US Dollar",
    payment_format: str = "Wire",
) -> list[object]:
    return [timestamp, bank, account, "99", destination, amount, currency, amount, currency, payment_format, label]


def make_dataset(root: Path, *, senders: int = 20, transactions_per_sender: int = 3) -> tuple[Path, Path]:
    trans_path = root / "HI-Small_Trans.csv"
    accounts_path = root / "HI-Small_accounts.csv"
    accounts = []
    transactions = []
    for sender in range(senders):
        account = f"A{sender:03d}"
        # Cada cinco cuentas comparten entidad: un split por cuenta filtraría identidad.
        entity = f"E{sender // 5:03d}"
        accounts.append(["Origin Bank", "1", account, entity, f"Entity {entity}"])
        for event in range(transactions_per_sender):
            label = int(sender % 4 == 0 and event == transactions_per_sender - 1)
            transactions.append(
                transaction(
                    f"2022/09/{event + 1:02d} {sender % 24:02d}:00",
                    "01",
                    account,
                    f"D{event}",
                    10 + sender + event,
                    label,
                    currency="Euro" if sender % 2 else "US Dollar",
                    payment_format="ACH" if event % 2 else "Wire",
                )
            )
    for event in range(transactions_per_sender):
        accounts.append(["Destination Bank", "99", f"D{event}", f"DEST{event}", f"Destination {event}"])
    # Desordenar el archivo de forma determinista para comprobar orden temporal real.
    write_csv(trans_path, TRANSACTION_COLUMNS, list(reversed(transactions)))
    write_csv(accounts_path, ACCOUNT_COLUMNS, accounts)
    return trans_path, accounts_path


def test_transform_sequence_is_causal_and_uses_expected_feature_order() -> None:
    prefix = [
        transaction("2022/09/01 01:00", "01", "A", "D1", 9, 0),
        transaction("2022/09/01 02:00", "01", "A", "D1", 99, 0),
    ]
    future = transaction("2022/09/03 04:00", "01", "A", "D2", 999, 0, currency="Euro", payment_format="ACH")
    preprocessor = fit_preprocessor([prefix])

    before, names = transform_sequence(prefix, preprocessor)
    after, _ = transform_sequence(prefix + [future], preprocessor)

    np.testing.assert_allclose(before, after[:2])
    assert names[:9] == [
        "log_amount_paid_z",
        "log_delta_hours_z",
        "log_previous_24h_z",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "cross_bank",
        "destination_seen_before",
    ]
    assert before[0, 8] == 0
    assert before[1, 8] == 1


def test_unknown_categories_and_zero_variance_remain_finite() -> None:
    normal = [transaction("2022/09/01 00:00", "01", "A", "D", 10, 0)]
    preprocessor = fit_preprocessor([normal, normal])
    unseen = [transaction("2022/09/01 00:00", "01", "B", "D", 10, 0, currency="Yen", payment_format="Cash")]

    features, names = transform_sequence(unseen, preprocessor)

    assert np.isfinite(features).all()
    assert features[0, names.index("currency=<UNK>")] == 1
    assert features[0, names.index("format=<UNK>")] == 1
    assert preprocessor["numeric_scale"] == [1.0, 1.0, 1.0]


def test_build_artifacts_creates_entity_disjoint_reproducible_contract(tmp_path: Path) -> None:
    trans_path, _ = make_dataset(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = build_artifacts(trans_path, first_dir, n_senders=20, max_len=3, seed=42)
    second = build_artifacts(trans_path, second_dir, n_senders=20, max_len=3, seed=42)

    assert first["contract_version"] == 1
    assert first["counts"]["source_senders"] == 20
    assert first["counts"]["selected_senders"] == 20
    assert first["counts"]["retained_transactions"] == 60
    assert set(first["runtime"]) >= {
        "scan_validate_select_seconds",
        "collect_truncate_seconds",
        "split_transform_persist_seconds",
        "checksum_seconds",
        "total_seconds",
    }
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["splits"] == second["splits"]
    assert first["feature_names"] == second["feature_names"]
    assert set(first["artifacts"]) == {"arrays", "transactions", "preprocessor", "splits", "manifest"}
    for filename in first["artifacts"].values():
        assert (first_dir / filename).is_file()
    with np.load(first_dir / "sequences.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["full_lengths"], np.full(20, 3))

    split_data = json.loads((first_dir / "splits.json").read_text(encoding="utf-8"))
    entities = [set(split_data[name]["entity_ids"]) for name in ("train", "validation", "test")]
    assert not (entities[0] & entities[1] or entities[0] & entities[2] or entities[1] & entities[2])
    assert sum(len(split_data[name]["sender_ids"]) for name in split_data) == 20


def test_truncation_relabels_sequence_and_reports_duplicate_rows(tmp_path: Path) -> None:
    trans_path = tmp_path / "HI-Small_Trans.csv"
    accounts_path = tmp_path / "HI-Small_accounts.csv"
    old_positive = transaction("2022/09/01 00:00", "01", "A", "D1", 5, 1)
    duplicate = transaction("2022/09/03 00:00", "01", "A", "D2", 7, 0)
    write_csv(trans_path, TRANSACTION_COLUMNS, [duplicate, old_positive, duplicate])
    write_csv(accounts_path, ACCOUNT_COLUMNS, [["Bank", "1", "A", "E", "Entity"], ["Dest", "99", "D1", "D1", "D1"], ["Dest", "99", "D2", "D2", "D2"]])

    manifest = build_artifacts(trans_path, tmp_path / "out", n_senders=1, max_len=2, seed=42)
    sender_id = manifest["splits"]["train"]["sender_ids"][0]
    sample = get_sender(sender_id, tmp_path / "out", split="train")

    assert sample["y"].item() == 0
    assert manifest["counts"]["positive_senders_before_truncation"] == 1
    assert manifest["counts"]["positive_senders_after_truncation"] == 0
    assert manifest["validation"]["duplicate_rows_in_selected_histories"] == 1
    assert [row["transaction_id"] for row in sample["transactions"]] == [0, 2]


def test_invalid_amount_stops_with_row_diagnostic(tmp_path: Path) -> None:
    trans_path, _ = make_dataset(tmp_path, senders=1, transactions_per_sender=1)
    rows = [transaction("2022/09/01 00:00", "01", "A000", "D0", -1, 0)]
    write_csv(trans_path, TRANSACTION_COLUMNS, rows)

    with pytest.raises(ValueError, match=r"fila 2.*importe"):
        build_artifacts(trans_path, tmp_path / "out", n_senders=1, max_len=2)


def test_dataloaders_match_contract_and_train_normal_has_no_positives(tmp_path: Path) -> None:
    trans_path, _ = make_dataset(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    manifest = build_artifacts(trans_path, artifact_dir, n_senders=20, max_len=3, seed=42)
    loaders = make_dataloaders(artifact_dir, batch_size=4, seed=42)

    assert set(loaders) == {"train", "train_normal", "validation", "test"}
    batch = next(iter(loaders["train"]))
    feature_count = len(manifest["feature_names"])
    assert batch["x"].shape == (4, 3, feature_count)
    assert batch["mask"].dtype == torch.bool
    assert batch["lengths"].dtype == torch.int64
    assert batch["transaction_ids"].shape == (4, 3)
    assert isinstance(batch["sender_id"], list)
    assert all(normal_batch["y"].eq(0).all() for normal_batch in loaders["train_normal"])


def test_training_loader_order_does_not_depend_on_iterating_other_loader(tmp_path: Path) -> None:
    trans_path, _ = make_dataset(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    build_artifacts(trans_path, artifact_dir, n_senders=20, max_len=3, seed=42)
    first = make_dataloaders(artifact_dir, batch_size=4, seed=77)
    second = make_dataloaders(artifact_dir, batch_size=4, seed=77)

    next(iter(first["train_normal"]))
    first_train_ids = next(iter(first["train"]))["sender_id"]
    second_train_ids = next(iter(second["train"]))["sender_id"]

    assert first_train_ids == second_train_ids


def test_get_sender_enforces_split_and_matches_raw_rows(tmp_path: Path) -> None:
    trans_path, _ = make_dataset(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    manifest = build_artifacts(trans_path, artifact_dir, n_senders=20, max_len=3, seed=42)
    sender_id = manifest["splits"]["test"]["sender_ids"][0]

    sample = get_sender(sender_id, artifact_dir, split="test")

    length = sample["lengths"].item()
    assert sample["x"].shape[0] == 1
    assert sample["mask"][0, :length].all()
    assert not sample["mask"][0, length:].any()
    assert [row["transaction_id"] for row in sample["transactions"]] == sample["transaction_ids"][0, :length].tolist()
    with pytest.raises(KeyError, match="no pertenece"):
        get_sender(sender_id, artifact_dir, split="train")
    with pytest.raises(KeyError, match="desconocido"):
        get_sender('["missing","missing"]', artifact_dir, split="test")
