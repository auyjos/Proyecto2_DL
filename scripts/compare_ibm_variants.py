"""Perfilado completo de seis variantes IBM y auditoría de sus catálogos Accounts.

No mezcla variantes ni crea splits. Procesa un par Trans/Accounts a la vez y
guarda resultados por variante. El notebook puede visualizar los JSON obtenidos
sin releer los aproximadamente 41 GB de CSV originales.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

if __package__:
    from .compare_datasets import file_fingerprint, length_summary, rows
else:
    from compare_datasets import file_fingerprint, length_summary, rows


VARIANTS = tuple(f"{level}-{size}" for size in ("Small", "Medium", "Large") for level in ("HI", "LI"))
ACCOUNT_HEADER = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
SCHEMA_VERSION = 1


def normalize_bank(bank):
    """Normalizar SOLO el banco para el cruce: '03208' -> '3208'.

    Las cuentas y entidades no se convierten a números ni pierden ceros.
    Se conserva la identidad original de los remitentes para comparar con C1.
    """
    bank = bank.strip()
    if not bank:
        raise ValueError("Banco vacío")
    return (bank.lstrip("0") or "0") if bank.isascii() and bank.isdecimal() else bank


def load_accounts(path):
    """Índice (banco normalizado, cuenta) -> (entidad, banco original).

    Si una clave normalizada apunta a dos entidades, queda ambigua (None).
    Esas claves se reportan y no se asignan a ninguna entidad durante el cruce.
    Los conteos de entidades se basan en cuentas únicas no ambiguas.
    """
    start = time.perf_counter()
    index = {}
    duplicates = 0
    total = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if next(reader, None) != ACCOUNT_HEADER:
            raise ValueError(f"Encabezado Accounts inesperado: {path}")
        for row_number, row in enumerate(reader, 2):
            if len(row) != 5 or not row[2] or not row[3]:
                raise ValueError(f"Registro Accounts inválido: fila {row_number}")
            total += 1
            key = (normalize_bank(row[1]), row[2])
            value = (row[3], row[1])
            if key in index:
                duplicates += 1
                if index[key] is None or index[key][0] != row[3]:
                    index[key] = None
            else:
                index[key] = value
    entities = Counter(value[0] for value in index.values() if value is not None)
    return index, {
        "rows": total,
        "unique_normalized_bank_accounts": len(index),
        "duplicate_normalized_rows": duplicates,
        "ambiguous_keys": sum(value is None for value in index.values()),
        "entities": len(entities),
        "entities_multiple_accounts": sum(count > 1 for count in entities.values()),
        "max_accounts_per_entity": max(entities.values(), default=0),
        "seconds": round(time.perf_counter() - start, 3),
    }


def audit_endpoint(counts, catalog, *, include_entities=False):
    """Cobertura por cuenta y por transacción; distinguir cruce literal/normalizado.

    El cruce se realiza una vez por cuenta observada, ponderando por su número
    de operaciones, para evitar cientos de millones de búsquedas redundantes.
    """
    matched = literal = matched_rows = literal_rows = ambiguous = 0
    missing_examples = []
    entity_accounts = Counter()
    seen_normalized = set() if include_entities else None
    aliases = 0
    for (bank, account), transactions in counts.items():
        key = (normalize_bank(bank), account)
        value = catalog.get(key)
        if value is None:
            if key in catalog:
                ambiguous += 1
            elif len(missing_examples) < 5:
                missing_examples.append([bank, account])
            continue
        matched += 1
        matched_rows += transactions
        if bank == value[1]:
            literal += 1
            literal_rows += transactions
        if include_entities:
            if key in seen_normalized:
                aliases += 1
            else:
                seen_normalized.add(key)
                entity_accounts[value[0]] += 1
    total_rows = sum(counts.values())
    result = {
        "observed_accounts": len(counts),
        "matched_accounts_literal": literal,
        "matched_accounts_normalized": matched,
        "matched_transactions_literal": literal_rows,
        "matched_transactions_normalized": matched_rows,
        "transaction_coverage_pct": 100 * matched_rows / total_rows if total_rows else 0,
        "account_coverage_pct": 100 * matched / len(counts) if counts else 0,
        "unmatched_accounts": len(counts) - matched,
        "ambiguous_observed_accounts": ambiguous,
        "missing_examples": missing_examples,
    }
    if include_entities:
        result.update({
            "active_entities": len(entity_accounts),
            "active_entities_multiple_sending_accounts": sum(n > 1 for n in entity_accounts.values()),
            "sending_accounts_in_multiaccount_entities": sum(n for n in entity_accounts.values() if n > 1),
            "raw_sender_aliases_after_bank_normalization": aliases,
        })
    return result


def profile_variant(trans_path, accounts_path, *, progress_every=10000000):
    """Leer Trans completo; conservar contadores, no las filas del CSV.

    Remitente positivo = alguna operación con Is Laundering=1 en TODO el archivo.
    Los conteos de operaciones retenibles con límite 64 no recalculan etiquetas:
    no deben interpretarse como positivos después del recorte.
    """
    start = time.perf_counter()
    senders, destinations = Counter(), Counter()
    positives = set()
    total = positive_rows = 0
    first = last = None
    variant = Path(trans_path).name.removesuffix("_Trans.csv")
    for _, row in rows(trans_path, "ibm"):
        origin, destination = (row[1], row[2]), (row[3], row[4])
        senders[origin] += 1
        destinations[destination] += 1
        total += 1
        if row[10] == "1":
            positive_rows += 1
            positives.add(origin)
        elif row[10] != "0":
            raise ValueError(f"Etiqueta inválida en {variant}, fila {total + 1}")
        timestamp = row[0]
        first = timestamp if first is None or timestamp < first else first
        last = timestamp if last is None or timestamp > last else last
        if progress_every and total % progress_every == 0:
            print(f"  {variant}: {total:,} transacciones ({time.perf_counter() - start:.0f}s)", flush=True)
    scan_seconds = time.perf_counter() - start
    summary = {
        "variant": variant,
        "rows": total,
        "senders": len(senders),
        "destinations": len(destinations),
        "positive_transactions": positive_rows,
        "positive_senders": len(positives),
        "positive_transaction_pct": 100 * positive_rows / total,
        "positive_sender_pct": 100 * len(positives) / len(senders),
        "positive_senders_ge_5": sum(senders[key] >= 5 for key in positives),
        "senders_over_64": sum(n > 64 for n in senders.values()),
        "transactions_retainable_at_64": sum(min(n, 64) for n in senders.values()),
        "time_min": first,
        "time_max": last,
        **length_summary(senders),
        "scan_seconds": round(scan_seconds, 3),
    }
    del positives
    print(f"  {variant}: cruzando catálogo Accounts...", flush=True)
    catalog, account_summary = load_accounts(accounts_path)
    summary["accounts"] = account_summary
    summary["origin_join"] = audit_endpoint(senders, catalog, include_entities=True)
    del senders
    summary["destination_join"] = audit_endpoint(destinations, catalog)
    summary["profile_and_join_seconds"] = round(time.perf_counter() - start, 3)
    return summary


def stat_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run_variants(ibm_dir, output_dir, *, variants=VARIANTS, refresh=False):
    """Cache explícita por versión de código, ruta, tamaño y mtime.

    Cada perfil nuevo guarda además SHA-256 de ambos CSV. Reusar no vuelve a
    verificar el hash completo; --refresh fuerza recalcular desde los originales.
    """
    ibm_dir, output_dir = Path(ibm_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # El resultado también depende del lector y cálculo de percentiles compartido.
    helper_hash = hashlib.sha256(Path(__file__).with_name("compare_datasets.py").read_bytes()).hexdigest()
    results = []
    for variant in variants:
        trans = ibm_dir / f"{variant}_Trans.csv"
        accounts = ibm_dir / f"{variant}_accounts.csv"
        signature = {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": code_hash,
            "helper_sha256": helper_hash,
            "python_version": platform.python_version(),
            "input_directory": str(ibm_dir.resolve()),
            "files": [stat_signature(trans), stat_signature(accounts)],
        }
        destination = output_dir / f"{variant}.json"
        previous = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else None
        if not refresh and previous and previous.get("cache_signature") == signature:
            print(f"Reutilizando perfil completo guardado: {variant}", flush=True)
            results.append(previous)
            continue
        print(f"Analizando {variant}: {trans.stat().st_size / 1e9:.2f} GB de transacciones", flush=True)
        start = time.perf_counter()
        result = profile_variant(trans, accounts)
        result["sources"] = [file_fingerprint(trans), file_fingerprint(accounts)]
        if signature["files"] != [stat_signature(trans), stat_signature(accounts)]:
            raise RuntimeError(f"Los archivos de {variant} cambiaron durante el análisis")
        result["total_seconds_including_hash"] = round(time.perf_counter() - start, 3)
        result["cache_signature"] = signature
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        results.append(result)
        print(f"Guardado {variant}: {result['rows']:,} operaciones, {result['total_seconds_including_hash']:.1f}s", flush=True)
        gc.collect()
    return results


def comparison_rows(results):
    return [{
        "variante": r["variant"],
        "transacciones": r["rows"],
        "remitentes": r["senders"],
        "positivos_transaccion_pct": r["positive_transaction_pct"],
        "positivos_remitente": r["positive_senders"],
        "p95_longitud": r["length_quantiles"]["0.95"],
        "remitentes_mas_64_pct": 100 * r["senders_over_64"] / r["senders"],
        "cuentas_catalogo": r["accounts"]["rows"],
        "entidades_catalogo": r["accounts"]["entities"],
        "entidades_con_varias_cuentas": r["accounts"]["entities_multiple_accounts"],
        "entidades_activas_con_varias_cuentas_emisoras": r["origin_join"]["active_entities_multiple_sending_accounts"],
        "cobertura_origen_pct": r["origin_join"]["transaction_coverage_pct"],
        "cobertura_destino_pct": r["destination_join"]["transaction_coverage_pct"],
        "trans_gb": r["sources"][0]["bytes"] / 1e9,
        "segundos_perfilado_y_cruce": r["profile_and_join_seconds"],
    } for r in results]


def variants_markdown(results):
    lines = ["| Variante | Transacciones | Remitentes | % trans. positivas | P95 longitud | Entidades multicuentas |", "|---|---:|---:|---:|---:|---:|"]
    for r in comparison_rows(results):
        lines.append(f"| {r['variante']} | {r['transacciones']:,} | {r['remitentes']:,} | {r['positivos_transaccion_pct']:.4f} | {r['p95_longitud']} | {r['entidades_con_varias_cuentas']:,} |")
    lines.append("\nEntidades multicuentas: catálogo completo, no solo cuentas emisoras. Variantes analizadas por separado; no son splits de entrenamiento/prueba.")
    return "\n".join(lines)


def create_variant_figures(results):
    import matplotlib.pyplot as plt

    names = [r["variant"] for r in results]
    colors = ["#2870a8" if name.startswith("HI") else "#c36724" for name in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    metrics = [
        ([r["rows"] / 1e6 for r in results], "Volumen", "Millones de transacciones", "%.2f"),
        ([r["positive_transaction_pct"] for r in results], "Desbalance", "% de transacciones de lavado", "%.3f"),
        ([r["length_quantiles"]["0.95"] for r in results], "Historial por remitente", "P95 de transacciones por remitente", "%.0f"),
        ([r["profile_and_join_seconds"] for r in results], "Costo local medido", "Segundos de perfilado y cruce (sin hash)", "%.1f"),
    ]
    for ax, (values, title, ylabel, fmt) in zip(axes.flat, metrics):
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values, default=1) * 1.18 or 1))
        ax.tick_params(axis="x", labelrotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("IBM AML · seis archivos Trans completos\nAzul: HI · Naranja: LI · los tiempos no son de entrenamiento", fontsize=14)

    entities, entity_axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    values = [100 * r["accounts"]["entities_multiple_accounts"] / r["accounts"]["entities"] for r in results]
    bars = entity_axes[0].bar(names, values, color=colors)
    entity_axes[0].bar_label(bars, fmt="%.1f", padding=3)
    entity_axes[0].set(title="Entidades con varias cuentas", ylabel="% de entidades del catálogo", ylim=(0, max(values) * 1.2))
    values = [100 * r["origin_join"]["sending_accounts_in_multiaccount_entities"] / r["origin_join"]["matched_accounts_normalized"] if r["origin_join"]["matched_accounts_normalized"] else 0 for r in results]
    bars = entity_axes[1].bar(names, values, color=colors)
    entity_axes[1].bar_label(bars, fmt="%.1f", padding=3)
    entity_axes[1].set(title="Cuentas emisoras vinculadas a otras emisoras", ylabel="% de cuentas emisoras con cruce válido", ylim=(0, max(values) * 1.2 or 1))
    for ax in entity_axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    entities.suptitle("Accounts permite auditar una separación por entidad\nEstas proporciones muestran vínculos; no miden fuga efectiva en un split", fontsize=14)
    return {"variantes_ibm": fig, "entidades_multicuentas": entities}


def save_variant_outputs(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_rows(results)
    with (output_dir / "resumen_variantes.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output_dir / "resumen_variantes.md").write_text(variants_markdown(results) + "\n", encoding="utf-8")
    import matplotlib.pyplot as plt
    for name, figure in create_variant_figures(results).items():
        for extension in ("png", "svg"):
            figure.savefig(output_dir / f"{name}.{extension}", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-dir", type=Path, default=Path("datasets/IBM"))
    parser.add_argument("--output", type=Path, default=Path("reports/ibm_variants"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--refresh", action="store_true", help="Releer archivos aunque exista un perfil compatible")
    args = parser.parse_args()
    results = run_variants(args.ibm_dir, args.output, variants=args.variants, refresh=args.refresh)
    save_variant_outputs(results, args.output)
    print(variants_markdown(results))


if __name__ == "__main__":
    main()
