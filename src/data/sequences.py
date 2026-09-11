"""Pipeline reproducible de secuencias por cuenta emisora para IBM AML.

Las funciones públicas son:

``build_artifacts``
    Lee el CSV, selecciona remitentes, crea splits por entidad, ajusta las
    transformaciones solo con entrenamiento y persiste el contrato.
``make_dataloaders``
    Carga los artefactos y devuelve DataLoaders para ambas etapas.
``get_sender``
    Recupera un remitente con sus tensores y filas originales para el MVP.

Los archivos ``Patterns.txt`` y las etiquetas nunca se incorporan a ``x``.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


CONTRACT_VERSION = 1
TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
ACCOUNT_COLUMNS = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
NUMERIC_FEATURES = ("log_amount_paid", "log_delta_hours", "log_previous_24h")
BASE_FEATURE_NAMES = [
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


def _normalize_bank(value: object) -> str:
    bank = str(value).strip()
    if not bank:
        raise ValueError("identificador de banco vacío")
    return (bank.lstrip("0") or "0") if bank.isascii() and bank.isdecimal() else bank


def _sender_key(bank: object, account: object) -> tuple[str, str]:
    bank_value = str(bank).strip()
    account_value = str(account).strip()
    if not bank_value or not account_value:
        raise ValueError("identificador de remitente vacío")
    # Se conserva el banco tal como aparece en Trans para que sender_id sea trazable.
    return bank_value, account_value


def _sender_id(key: tuple[str, str]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: object, row_number: int | None = None) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y/%m/%d %H:%M")
    except ValueError as exc:
        prefix = f"fila {row_number}: " if row_number is not None else ""
        raise ValueError(f"{prefix}timestamp inválido: {value!r}") from exc


def _finite_nonnegative(value: object, label: str, row_number: int | None = None) -> float:
    prefix = f"fila {row_number}: " if row_number is not None else ""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}{label} inválido: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{prefix}{label} debe ser finito y no negativo: {value!r}")
    return result


def _parse_transaction_row(row: Sequence[object], row_number: int, transaction_id: int) -> dict:
    if len(row) != len(TRANSACTION_COLUMNS):
        raise ValueError(f"fila {row_number}: se esperaban {len(TRANSACTION_COLUMNS)} columnas")
    timestamp = str(row[0])
    _parse_timestamp(timestamp, row_number)
    from_bank, from_account = _sender_key(row[1], row[2])
    to_bank, to_account = _sender_key(row[3], row[4])
    amount_received = _finite_nonnegative(row[5], "importe recibido", row_number)
    amount_paid = _finite_nonnegative(row[7], "importe pagado", row_number)
    receiving_currency = str(row[6]).strip()
    payment_currency = str(row[8]).strip()
    payment_format = str(row[9]).strip()
    if not receiving_currency or not payment_currency or not payment_format:
        raise ValueError(f"fila {row_number}: moneda o formato de pago vacío")
    if str(row[10]) not in {"0", "1"}:
        raise ValueError(f"fila {row_number}: etiqueta no binaria: {row[10]!r}")
    return {
        "transaction_id": int(transaction_id),
        "timestamp": timestamp,
        "from_bank": from_bank,
        "from_account": from_account,
        "to_bank": to_bank,
        "to_account": to_account,
        "amount_received": amount_received,
        "receiving_currency": receiving_currency,
        "amount_paid": amount_paid,
        "payment_currency": payment_currency,
        "payment_format": payment_format,
        "is_laundering": int(row[10]),
    }


def _coerce_transaction(value: Sequence[object] | dict, index: int) -> dict:
    if isinstance(value, dict):
        required = {
            "timestamp",
            "from_bank",
            "from_account",
            "to_bank",
            "to_account",
            "amount_paid",
            "payment_currency",
            "payment_format",
            "is_laundering",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"transacción {index}: faltan campos {sorted(missing)}")
        transaction = dict(value)
        _parse_timestamp(transaction["timestamp"])
        transaction["amount_paid"] = _finite_nonnegative(transaction["amount_paid"], "importe pagado")
        return transaction
    return _parse_transaction_row(value, index + 2, index)


def _raw_numeric_features(transactions: Sequence[Sequence[object] | dict]) -> tuple[np.ndarray, list[dict]]:
    parsed = [_coerce_transaction(value, index) for index, value in enumerate(transactions)]
    if not parsed:
        raise ValueError("una secuencia no puede estar vacía")
    timestamps = [_parse_timestamp(item["timestamp"]) for item in parsed]
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("las transacciones deben estar ordenadas cronológicamente")
    result = np.zeros((len(parsed), 3), dtype=np.float64)
    left = 0
    for index, (item, timestamp) in enumerate(zip(parsed, timestamps)):
        delta_hours = 0.0 if index == 0 else (timestamp - timestamps[index - 1]).total_seconds() / 3600
        while left < index and timestamp - timestamps[left] > timedelta(hours=24):
            left += 1
        result[index] = (
            math.log1p(item["amount_paid"]),
            math.log1p(delta_hours),
            math.log1p(index - left),
        )
    return result, parsed


def fit_preprocessor(
    normal_sequences: Iterable[Sequence[Sequence[object] | dict]],
    vocabulary_sequences: Iterable[Sequence[Sequence[object] | dict]] | None = None,
) -> dict:
    """Ajustar escalas con normalidad y vocabularios con todo entrenamiento."""
    normal_sequences = list(normal_sequences)
    if not normal_sequences:
        raise ValueError("se necesita al menos una secuencia normal de entrenamiento")
    numeric_blocks = []
    for sequence in normal_sequences:
        numeric, _ = _raw_numeric_features(sequence)
        numeric_blocks.append(numeric)
    numeric_values = np.concatenate(numeric_blocks, axis=0)
    means = numeric_values.mean(axis=0)
    scales = numeric_values.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)

    vocabulary_source = normal_sequences if vocabulary_sequences is None else list(vocabulary_sequences)
    if not vocabulary_source:
        raise ValueError("se necesita entrenamiento para ajustar vocabularios")
    currencies, formats = set(), set()
    for sequence in vocabulary_source:
        _, parsed = _raw_numeric_features(sequence)
        currencies.update(item["payment_currency"] for item in parsed)
        formats.update(item["payment_format"] for item in parsed)
    return {
        "version": CONTRACT_VERSION,
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_mean": means.tolist(),
        "numeric_scale": scales.tolist(),
        "currency_vocabulary": sorted(currencies),
        "format_vocabulary": sorted(formats),
    }


def _feature_names(preprocessor: dict) -> list[str]:
    return [
        *BASE_FEATURE_NAMES,
        "currency=<UNK>",
        *(f"currency={value}" for value in preprocessor["currency_vocabulary"]),
        "format=<UNK>",
        *(f"format={value}" for value in preprocessor["format_vocabulary"]),
    ]


def transform_sequence(
    transactions: Sequence[Sequence[object] | dict], preprocessor: dict
) -> tuple[np.ndarray, list[str]]:
    """Transformar una secuencia ordenada sin utilizar información futura."""
    if preprocessor.get("version") != CONTRACT_VERSION:
        raise ValueError("versión de preprocesador incompatible")
    numeric, parsed = _raw_numeric_features(transactions)
    means = np.asarray(preprocessor["numeric_mean"], dtype=np.float64)
    scales = np.asarray(preprocessor["numeric_scale"], dtype=np.float64)
    numeric = (numeric - means) / scales
    currency_vocabulary = list(preprocessor["currency_vocabulary"])
    format_vocabulary = list(preprocessor["format_vocabulary"])
    currency_index = {value: index + 1 for index, value in enumerate(currency_vocabulary)}
    format_index = {value: index + 1 for index, value in enumerate(format_vocabulary)}
    width = len(BASE_FEATURE_NAMES) + 1 + len(currency_vocabulary) + 1 + len(format_vocabulary)
    features = np.zeros((len(parsed), width), dtype=np.float32)
    features[:, :3] = numeric.astype(np.float32)
    seen_destinations = set()
    currency_offset = len(BASE_FEATURE_NAMES)
    format_offset = currency_offset + 1 + len(currency_vocabulary)
    for index, item in enumerate(parsed):
        timestamp = _parse_timestamp(item["timestamp"])
        hour = timestamp.hour + timestamp.minute / 60
        features[index, 3] = math.sin(2 * math.pi * hour / 24)
        features[index, 4] = math.cos(2 * math.pi * hour / 24)
        features[index, 5] = math.sin(2 * math.pi * timestamp.weekday() / 7)
        features[index, 6] = math.cos(2 * math.pi * timestamp.weekday() / 7)
        features[index, 7] = float(_normalize_bank(item["from_bank"]) != _normalize_bank(item["to_bank"]))
        destination = (_normalize_bank(item["to_bank"]), str(item["to_account"]))
        features[index, 8] = float(destination in seen_destinations)
        seen_destinations.add(destination)
        features[index, currency_offset + currency_index.get(item["payment_currency"], 0)] = 1
        features[index, format_offset + format_index.get(item["payment_format"], 0)] = 1
    if not np.isfinite(features).all():
        raise ValueError("las features contienen NaN o infinito")
    return features, _feature_names(preprocessor)


def _transaction_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if next(reader, None) != TRANSACTION_COLUMNS:
            raise ValueError(f"encabezado inesperado en {path}")
        for transaction_id, row in enumerate(reader):
            yield transaction_id, row


def _accounts_path(csv_path: Path) -> Path:
    suffix = "_Trans.csv"
    if not csv_path.name.endswith(suffix):
        raise ValueError("el CSV IBM debe terminar en _Trans.csv para localizar Accounts")
    path = csv_path.with_name(csv_path.name[: -len(suffix)] + "_accounts.csv")
    if not path.is_file():
        raise FileNotFoundError(f"no se encontró el catálogo Accounts: {path}")
    return path


def _load_entity_map(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if next(reader, None) != ACCOUNT_COLUMNS:
            raise ValueError(f"encabezado inesperado en {path}")
        for row_number, row in enumerate(reader, 2):
            if len(row) != len(ACCOUNT_COLUMNS) or not str(row[2]).strip() or not str(row[3]).strip():
                raise ValueError(f"fila {row_number}: registro Accounts inválido")
            key = (_normalize_bank(row[1]), str(row[2]).strip())
            entity_id = str(row[3]).strip()
            previous = result.get(key)
            if previous is not None and previous != entity_id:
                raise ValueError(f"fila {row_number}: cuenta asociada a entidades contradictorias")
            result[key] = entity_id
    return result


def _split_targets(total: int) -> list[int]:
    train = int(total * SPLIT_RATIOS[0])
    validation = int(total * SPLIT_RATIOS[1])
    return [train, validation, total - train - validation]


def _grouped_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """Asignar entidades completas aproximando 70/15/15 y prevalencia global.

    El problema es un bin packing discreto: si una entidad posee muchas cuentas,
    no siempre existen tamaños exactos. El manifiesto conserva objetivos y tamaños
    logrados. El algoritmo es determinista para una misma semilla y datos.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["entity_id"]].append(record)
    # Con una sola entidad, toda la información debe permanecer junta. Se
    # asigna a entrenamiento para que sea posible ajustar el preprocesador.
    if len(groups) == 1:
        return {"train": records, "validation": [], "test": []}
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: (len(item[1]), sum(r["y"] for r in item[1])), reverse=True)
    targets = _split_targets(len(records))
    total_positive = sum(record["y"] for record in records)
    positive_targets = [total_positive * ratio for ratio in SPLIT_RATIOS]
    assigned = [[] for _ in SPLIT_NAMES]
    sizes = [0, 0, 0]
    positives = [0, 0, 0]
    for index, (_, group) in enumerate(items):
        empty = [split for split in range(3) if not assigned[split]]
        remaining_groups = len(items) - index
        candidates = empty if empty and remaining_groups == len(empty) else list(range(3))
        group_size = len(group)
        group_positive = sum(record["y"] for record in group)

        def score(split: int) -> tuple[float, int]:
            projected_sizes = list(sizes)
            projected_positives = list(positives)
            projected_sizes[split] += group_size
            projected_positives[split] += group_positive
            size_error = sum(
                ((value - target) / max(target, 1)) ** 2
                for value, target in zip(projected_sizes, targets)
            )
            positive_error = sum(
                ((value - target) / max(target, 1.0)) ** 2
                for value, target in zip(projected_positives, positive_targets)
            )
            return size_error + 0.35 * positive_error, split

        chosen = min(candidates, key=score)
        assigned[chosen].extend(group)
        sizes[chosen] += group_size
        positives[chosen] += group_positive
    return {
        name: sorted(values, key=lambda item: item["sender_id"])
        for name, values in zip(SPLIT_NAMES, assigned)
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def build_artifacts(
    csv_path,
    output_dir,
    *,
    n_senders: int = 50000,
    max_len: int = 64,
    seed: int = 42,
) -> dict:
    """Construir y guardar el contrato C1 desde IBM ``*_Trans.csv``.

    La selección es uniforme por cuenta emisora y no consulta etiquetas. Los
    splits se hacen por Entity ID para impedir que cuentas hermanas crucen
    conjuntos. Los tamaños pueden diferir levemente de 70/15/15 por esa regla.
    """
    started = time.perf_counter()
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if n_senders < 1 or max_len < 1:
        raise ValueError("n_senders y max_len deben ser positivos")
    accounts_path = _accounts_path(csv_path)

    # Primera pasada: validar el archivo completo y descubrir remitentes sin
    # consultar etiquetas para el muestreo.
    pass_one_started = time.perf_counter()
    sender_keys = set()
    total_rows = 0
    for transaction_id, row in _transaction_rows(csv_path):
        parsed = _parse_transaction_row(row, transaction_id + 2, transaction_id)
        sender_keys.add((parsed["from_bank"], parsed["from_account"]))
        total_rows += 1
    source_sender_count = len(sender_keys)
    first_pass_seconds = time.perf_counter() - pass_one_started
    if n_senders > len(sender_keys):
        raise ValueError(f"se solicitaron {n_senders} remitentes, pero solo existen {len(sender_keys)}")
    selected = random.Random(seed).sample(sorted(sender_keys), n_senders)
    selected_set = set(selected)
    selection_sha256 = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    del sender_keys

    # Segunda pasada: conservar las últimas max_len operaciones con heap; el
    # índice original desempata timestamps iguales y permanece como transaction_id.
    histories: dict[tuple[str, str], list[tuple[str, int, dict]]] = {key: [] for key in selected}
    full_counts = Counter()
    positive_before = set()
    positive_transactions_before = 0
    seen_rows = set()
    duplicate_rows = 0
    collect_started = time.perf_counter()
    for transaction_id, row in _transaction_rows(csv_path):
        key = _sender_key(row[1], row[2])
        if key not in selected_set:
            continue
        parsed = _parse_transaction_row(row, transaction_id + 2, transaction_id)
        row_signature = tuple(str(value) for value in row)
        if row_signature in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(row_signature)
        full_counts[key] += 1
        if parsed["is_laundering"]:
            positive_before.add(key)
            positive_transactions_before += 1
        heap = histories[key]
        item = (parsed["timestamp"], transaction_id, parsed)
        if len(heap) < max_len:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    if any(full_counts[key] == 0 for key in selected):
        raise RuntimeError("un remitente seleccionado desapareció entre las dos pasadas")
    collect_truncate_seconds = time.perf_counter() - collect_started

    prepare_started = time.perf_counter()
    entity_map = _load_entity_map(accounts_path)
    records = []
    for key in selected:
        normalized = (_normalize_bank(key[0]), key[1])
        if normalized not in entity_map:
            raise ValueError(f"el remitente {_sender_id(key)} no aparece en Accounts")
        transactions = [item[2] for item in sorted(histories[key])]
        records.append(
            {
                "sender_id": _sender_id(key),
                "entity_id": entity_map[normalized],
                "transactions": transactions,
                "y": int(any(item["is_laundering"] for item in transactions)),
                "full_length": full_counts[key],
            }
        )
    splits = _grouped_split(records, seed)
    train_records = splits["train"]
    train_normal = [record["transactions"] for record in train_records if record["y"] == 0]
    if not train_normal:
        raise ValueError("el split de entrenamiento no contiene secuencias normales")
    preprocessor = fit_preprocessor(
        train_normal,
        vocabulary_sequences=[record["transactions"] for record in train_records],
    )
    feature_names = _feature_names(preprocessor)

    ordered_records = [record for name in SPLIT_NAMES for record in splits[name]]
    values, transaction_ids, transaction_labels, offsets = [], [], [], [0]
    labels, sender_ids, entity_ids, split_codes, raw_rows = [], [], [], [], []
    full_lengths = []
    split_lookup = {name: code for code, name in enumerate(SPLIT_NAMES)}
    for split_name in SPLIT_NAMES:
        for record in splits[split_name]:
            features, names = transform_sequence(record["transactions"], preprocessor)
            if names != feature_names:
                raise RuntimeError("el orden de features cambió durante la transformación")
            values.append(features)
            ids = [item["transaction_id"] for item in record["transactions"]]
            transaction_ids.extend(ids)
            transaction_labels.extend(item["is_laundering"] for item in record["transactions"])
            offsets.append(offsets[-1] + len(ids))
            labels.append(record["y"])
            full_lengths.append(record["full_length"])
            sender_ids.append(record["sender_id"])
            entity_ids.append(record["entity_id"])
            split_codes.append(split_lookup[split_name])
            for item in record["transactions"]:
                raw_rows.append({**item, "sender_id": record["sender_id"], "entity_id": record["entity_id"], "split": split_name})

    arrays_file = "sequences.npz"
    transactions_file = "transactions.parquet"
    preprocessor_file = "preprocessor.json"
    splits_file = "splits.json"
    manifest_file = "manifest.json"
    _atomic_npz(
        output_dir / arrays_file,
        x_values=np.concatenate(values).astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        y=np.asarray(labels, dtype=np.float32),
        full_lengths=np.asarray(full_lengths, dtype=np.int64),
        sender_ids=np.asarray(sender_ids, dtype=np.str_),
        entity_ids=np.asarray(entity_ids, dtype=np.str_),
        split_codes=np.asarray(split_codes, dtype=np.int8),
        transaction_ids=np.asarray(transaction_ids, dtype=np.int64),
        transaction_labels=np.asarray(transaction_labels, dtype=np.int8),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_temp = output_dir / (transactions_file + ".tmp")
    pd.DataFrame(raw_rows).to_parquet(parquet_temp, index=False)
    os.replace(parquet_temp, output_dir / transactions_file)
    _atomic_write_text(output_dir / preprocessor_file, json.dumps(preprocessor, ensure_ascii=False, indent=2) + "\n")

    split_manifest = {}
    targets = _split_targets(n_senders)
    for index, name in enumerate(SPLIT_NAMES):
        values_for_split = splits[name]
        split_manifest[name] = {
            "target_sender_count": targets[index],
            "sender_count": len(values_for_split),
            "positive_sender_count": sum(record["y"] for record in values_for_split),
            "sender_ids": [record["sender_id"] for record in values_for_split],
            "entity_ids": sorted({record["entity_id"] for record in values_for_split}),
        }
    split_entity_sets = [set(split_manifest[name]["entity_ids"]) for name in SPLIT_NAMES]
    entity_overlap_count = sum(
        len(split_entity_sets[left] & split_entity_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    if entity_overlap_count:
        raise RuntimeError("se detectaron entidades compartidas entre splits")
    _atomic_write_text(output_dir / splits_file, json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n")
    split_transform_persist_seconds = time.perf_counter() - prepare_started

    checksum_started = time.perf_counter()
    transaction_sha256 = _sha256(csv_path)
    accounts_sha256 = _sha256(accounts_path)
    checksum_seconds = time.perf_counter() - checksum_started

    positive_after = sum(record["y"] for record in ordered_records)
    retained_positive_transactions = sum(transaction_labels)
    negative_train = sum(record["y"] == 0 for record in train_records)
    positive_train = sum(record["y"] == 1 for record in train_records)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "transactions": str(csv_path.resolve()),
            "transactions_bytes": csv_path.stat().st_size,
            "transactions_sha256": transaction_sha256,
            "accounts": str(accounts_path.resolve()),
            "accounts_bytes": accounts_path.stat().st_size,
            "accounts_sha256": accounts_sha256,
        },
        "configuration": {
            "n_senders": n_senders,
            "max_len": max_len,
            "seed": seed,
            "split_ratios": list(SPLIT_RATIOS),
            "split_unit": "Entity ID",
            "sequence_unit": "(From Bank, Account origen)",
            "truncation": "últimas transacciones por (Timestamp, transaction_id)",
        },
        "selection_sha256": selection_sha256,
        "feature_names": feature_names,
        "counts": {
            "source_transactions": total_rows,
            "source_senders": source_sender_count,
            "selected_senders": n_senders,
            "selected_transactions_before_truncation": sum(full_counts.values()),
            "retained_transactions": len(transaction_ids),
            "truncated_senders": sum(count > max_len for count in full_counts.values()),
            "positive_senders_before_truncation": len(positive_before),
            "positive_senders_after_truncation": positive_after,
            "positive_transactions_before_truncation": positive_transactions_before,
            "positive_transactions_after_truncation": retained_positive_transactions,
            "normal_train_senders": negative_train,
            "positive_train_senders": positive_train,
            "pos_weight": negative_train / positive_train if positive_train else None,
        },
        "validation": {
            "duplicate_rows_in_selected_histories": duplicate_rows,
            "entity_overlap_between_splits": bool(entity_overlap_count),
            "entity_overlap_count": entity_overlap_count,
            "padding_value": 0.0,
            "transaction_id_padding": -1,
            "labels_excluded_from_features": True,
        },
        "splits": split_manifest,
        "artifacts": {
            "arrays": arrays_file,
            "transactions": transactions_file,
            "preprocessor": preprocessor_file,
            "splits": splits_file,
            "manifest": manifest_file,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scan_validate_select_seconds": round(first_pass_seconds, 3),
            "collect_truncate_seconds": round(collect_truncate_seconds, 3),
            "split_transform_persist_seconds": round(split_transform_persist_seconds, 3),
            "checksum_seconds": round(checksum_seconds, 3),
            "first_pass_seconds": round(first_pass_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    }
    _atomic_write_text(output_dir / manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _load_contract(artifact_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"contrato incompatible: {manifest.get('contract_version')}; se esperaba {CONTRACT_VERSION}"
        )
    arrays = np.load(artifact_dir / manifest["artifacts"]["arrays"], allow_pickle=False)
    sequence_count = len(arrays["y"])
    if len(arrays["offsets"]) != sequence_count + 1 or len(arrays["sender_ids"]) != sequence_count:
        arrays.close()
        raise ValueError("artefacto de secuencias inconsistente")
    return manifest, arrays


class _SequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        start, end = (int(value) for value in self.arrays["offsets"][index : index + 2])
        return {
            "x": self.arrays["x_values"][start:end],
            "y": float(self.arrays["y"][index]),
            "sender_id": str(self.arrays["sender_ids"][index]),
            "entity_id": str(self.arrays["entity_ids"][index]),
            "transaction_ids": self.arrays["transaction_ids"][start:end],
        }


def _collate(samples: list[dict], max_len: int, feature_count: int) -> dict:
    batch_size = len(samples)
    x = torch.zeros((batch_size, max_len, feature_count), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    transaction_ids = torch.full((batch_size, max_len), -1, dtype=torch.int64)
    lengths = torch.empty(batch_size, dtype=torch.int64)
    y = torch.empty(batch_size, dtype=torch.float32)
    for index, sample in enumerate(samples):
        length = len(sample["x"])
        x[index, :length] = torch.from_numpy(sample["x"])
        mask[index, :length] = True
        transaction_ids[index, :length] = torch.from_numpy(sample["transaction_ids"])
        lengths[index] = length
        y[index] = sample["y"]
    return {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "y": y,
        "sender_id": [sample["sender_id"] for sample in samples],
        "entity_id": [sample["entity_id"] for sample in samples],
        "transaction_ids": transaction_ids,
    }


def make_dataloaders(artifact_dir, *, batch_size: int = 128, seed: int = 42) -> dict:
    """Crear loaders deterministas; solo los dos loaders train barajan."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser positivo")
    artifact_dir = Path(artifact_dir)
    manifest, loaded = _load_contract(artifact_dir)
    # Copiar a memoria permite cerrar el archivo NPZ sin dejar handles abiertos.
    arrays = {name: loaded[name] for name in loaded.files}
    loaded.close()
    max_len = manifest["configuration"]["max_len"]
    feature_count = len(manifest["feature_names"])
    loaders = {}
    split_codes = arrays["split_codes"]
    definitions = {
        "train": (np.flatnonzero(split_codes == 0), True),
        "train_normal": (np.flatnonzero((split_codes == 0) & (arrays["y"] == 0)), True),
        "validation": (np.flatnonzero(split_codes == 1), False),
        "test": (np.flatnonzero(split_codes == 2), False),
    }
    for loader_index, (name, (indices, shuffle)) in enumerate(definitions.items()):
        dataset = _SequenceDataset(arrays, indices)
        generator = torch.Generator().manual_seed(seed + loader_index) if shuffle else None
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and len(dataset) > 0,
            generator=generator,
            num_workers=0,
            collate_fn=lambda samples, ml=max_len, fc=feature_count: _collate(samples, ml, fc),
        )
    return loaders


def get_sender(sender_id: str, artifact_dir, *, split: str = "test") -> dict:
    """Recuperar un remitente en formato batch de tamaño uno y filas del MVP."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split inválido: {split!r}")
    artifact_dir = Path(artifact_dir)
    manifest, arrays = _load_contract(artifact_dir)
    matches = np.flatnonzero(arrays["sender_ids"] == sender_id)
    if not len(matches):
        arrays.close()
        raise KeyError(f"sender_id desconocido: {sender_id}")
    index = int(matches[0])
    actual_split = SPLIT_NAMES[int(arrays["split_codes"][index])]
    if actual_split != split:
        arrays.close()
        raise KeyError(f"el remitente no pertenece al split {split}; pertenece a {actual_split}")
    start, end = (int(value) for value in arrays["offsets"][index : index + 2])
    sample = {
        "x": arrays["x_values"][start:end].copy(),
        "y": float(arrays["y"][index]),
        "sender_id": str(arrays["sender_ids"][index]),
        "entity_id": str(arrays["entity_ids"][index]),
        "transaction_ids": arrays["transaction_ids"][start:end].copy(),
    }
    arrays.close()
    batch = _collate([sample], manifest["configuration"]["max_len"], len(manifest["feature_names"]))
    frame = pd.read_parquet(
        artifact_dir / manifest["artifacts"]["transactions"],
        filters=[("sender_id", "==", sender_id)],
    )
    by_id = {int(row["transaction_id"]): row for row in frame.to_dict(orient="records")}
    ordered = []
    for transaction_id in sample["transaction_ids"]:
        value = by_id.get(int(transaction_id))
        if value is None:
            raise ValueError(f"falta transaction_id {transaction_id} en el artefacto Parquet")
        ordered.append(value)
    batch["transactions"] = ordered
    return batch
