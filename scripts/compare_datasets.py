"""Reproduce la comparación de PaySim, IBM HI-Small y la muestra propuesta.

El perfilado usa la biblioteca estándar y lee los CSV completos fila a fila.
La memoria depende del número de remitentes, no de todas las transacciones.
Matplotlib solo es necesario al generar las gráficas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path


HEADERS = {
    "paysim": "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud".split(","),
    "ibm": "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering".split(","),
}


def rows(path, kind):
    """Usar posiciones evita confundir las dos columnas Account de IBM.

    csv.reader conserva bancos y cuentas como cadenas, incluidos ceros iniciales.
    Los CSV originales se abren exclusivamente en modo lectura.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if next(reader, None) != HEADERS[kind]:
            raise ValueError(f"Encabezado inesperado en {path}")
        for index, row in enumerate(reader):
            if len(row) != len(HEADERS[kind]):
                raise ValueError(f"Número de columnas inválido: {path}, fila {index + 2}")
            yield index, row


def sender_key(row, kind):
    """PaySim: nameOrig. IBM: (banco de origen, cuenta de origen)."""
    return row[3] if kind == "paysim" else (row[1], row[2])


def positive_label(row, kind):
    value = row[9] if kind == "paysim" else row[10]
    if value not in {"0", "1"}:
        raise ValueError(f"Etiqueta no binaria: {value!r}")
    return int(value)


def length_summary(counts):
    """Percentil empírico: menor longitud cuya frecuencia acumulada alcanza q*N.

    No hay interpolación: es el método de rango más próximo, ceil(q*N).
    El histograma completo permite rehacer gráficas sin releer el CSV.
    """
    histogram = Counter(counts.values())
    total = sum(histogram.values())
    if not total:
        raise ValueError("No hay remitentes para analizar")
    quantiles = {}
    for q in (0.5, 0.9, 0.95, 0.99, 1.0):
        cumulative = 0
        for length, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= math.ceil(q * total):
                quantiles[str(q)] = length
                break
    return {
        "singleton_senders": histogram[1],
        "senders_ge_5": sum(n for length, n in histogram.items() if length >= 5),
        "length_quantiles": quantiles,
        "length_histogram": dict(sorted(histogram.items())),
    }


def profile_dataset(path, kind):
    """Un remitente positivo tiene al menos una operación positiva en todo el CSV.

    Esto es descriptivo: no implica que fraude y lavado sean equivalentes.
    Devuelve también conteos por remitente para reutilizarlos en el muestreo.
    """
    start = time.perf_counter()
    counts = Counter()
    positives = set()
    total = positive_rows = 0
    first = last = None
    for _, row in rows(path, kind):
        key = sender_key(row, kind)
        label = positive_label(row, kind)
        counts[key] += 1
        total += 1
        positive_rows += label
        if label:
            positives.add(key)
        # Los timestamps IBM tienen formato fijo YYYY/MM/DD HH:MM.
        # Orden lexicográfico equivale a cronológico para estos archivos.
        timestamp = int(row[0]) if kind == "paysim" else row[0]
        first = timestamp if first is None or timestamp < first else first
        last = timestamp if last is None or timestamp > last else last
    summary = {
        "dataset": "PaySim" if kind == "paysim" else "IBM HI-Small",
        "label_meaning": "Fraude (isFraud)" if kind == "paysim" else "Lavado (Is Laundering)",
        "rows": total,
        "senders": len(counts),
        "positive_transactions": positive_rows,
        "positive_senders": len(positives),
        "positive_transaction_pct": 100 * positive_rows / total if total else 0,
        "positive_sender_pct": 100 * len(positives) / len(counts) if counts else 0,
        "positive_senders_ge_5": sum(counts[key] >= 5 for key in positives),
        "time_min": first,
        "time_max": last,
        "time_unit": "step (hora simulada)" if kind == "paysim" else "Timestamp",
        **length_summary(counts),
        "seconds": round(time.perf_counter() - start, 3),
    }
    return summary, counts


def compare_ibm_sample(path, sender_counts, n_senders=50000, max_len=64, seed=42):
    """Selección uniforme de remitentes, SIN consultar etiquetas.

    Se ordenan claves antes de muestrear para no depender del orden de un set.
    Un min-heap por remitente conserva las últimas L operaciones por timestamp;
    el índice de fila desempata fechas iguales. Nunca se toma simplemente el
    último bloque de filas del archivo, porque no está totalmente ordenado.
    """
    if not 1 <= n_senders <= len(sender_counts) or max_len < 1:
        raise ValueError("Muestra o longitud inválida para el dataset disponible")
    selected = random.Random(seed).sample(sorted(sender_counts), n_senders)
    histories = {key: [] for key in selected}
    full_counts = Counter()
    full_positive = set()
    positive_before = 0
    for index, row in rows(path, "ibm"):
        key = sender_key(row, "ibm")
        if key not in histories:
            continue
        label = positive_label(row, "ibm")
        full_counts[key] += 1
        positive_before += label
        if label:
            full_positive.add(key)
        heap = histories[key]
        item = (row[0], index, label)
        if len(heap) < max_len:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if any(full_counts[key] != sender_counts[key] for key in selected):
        raise ValueError("Los conteos cambiaron entre perfilado y muestreo")
    retained_counts = {key: len(heap) for key, heap in histories.items()}
    retained_positive = sum(any(item[2] for item in heap) for heap in histories.values())
    truncated = sum(count > max_len for count in full_counts.values())
    return {
        "sample_senders": n_senders,
        "seed": seed,
        "max_len": max_len,
        "selection": "random.Random(seed).sample(sorted(sender_keys), n_senders)",
        "selected_sender_keys_sha256": hashlib.sha256(
            json.dumps(sorted(selected), separators=(",", ":")).encode()
        ).hexdigest(),
        "full_transactions": sum(full_counts.values()),
        "retained_transactions": sum(retained_counts.values()),
        "truncated_senders": truncated,
        "truncated_sender_pct": 100 * truncated / n_senders,
        "positive_senders_full": len(full_positive),
        "positive_senders_retained": retained_positive,
        "positive_transactions_full": positive_before,
        "positive_transactions_retained": sum(item[2] for h in histories.values() for item in h),
        "positive_sender_pct_retained": 100 * retained_positive / n_senders,
        "before": length_summary(full_counts),
        "after": length_summary(retained_counts),
    }


def file_fingerprint(path):
    """Identifica los bytes exactos utilizados; una pasada adicional de lectura."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def run_comparison(paysim_path, ibm_path, n_senders=50000, max_len=64, seed=42):
    summaries = []
    sources = []
    for kind, path in (("paysim", paysim_path), ("ibm", ibm_path)):
        print(f"Leyendo archivo completo: {Path(path).name}", flush=True)
        sources.append(file_fingerprint(path))
        summary, counts = profile_dataset(path, kind)
        summaries.append(summary)
        if kind == "ibm":
            print(f"Comparando muestra de {n_senders:,} remitentes...", flush=True)
            sample = compare_ibm_sample(path, counts, n_senders, max_len, seed)
        del counts  # Liberar millones de claves PaySim antes de analizar IBM.
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "sources": sources,
        "datasets": summaries,
        "sample": sample,
    }


def comparison_markdown(result):
    datasets = result["datasets"]
    table = ["| Medición | PaySim | IBM HI-Small |", "|---|---:|---:|"]
    metrics = [
        ("Transacciones", "rows"), ("Remitentes", "senders"),
        ("Con una transacción", "singleton_senders"),
        ("Con al menos 5 transacciones", "senders_ge_5"),
        ("Transacciones positivas", "positive_transactions"),
        ("Remitentes positivos", "positive_senders"),
    ]
    for label, key in metrics:
        table.append(f"| {label} | {datasets[0][key]:,} | {datasets[1][key]:,} |")
    for q, label in (("0.5", "Mediana"), ("0.95", "Percentil 95"), ("0.99", "Percentil 99"), ("1.0", "Máximo")):
        values = [d["length_quantiles"][q] for d in datasets]
        table.append(f"| {label} de longitud | {values[0]:,} | {values[1]:,} |")
    table.append("\nPositivo significa fraude en PaySim y lavado en IBM; no son etiquetas equivalentes.")
    return "\n".join(table)


def create_figures(result):
    """Gráficas descriptivas: denominadores explícitos y ejes lineales desde cero."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#2870a8", "#c36724")
    labels = [d["dataset"] for d in result["datasets"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    bins = [(1, 1), (2, 4), (5, 16), (17, 64), (65, math.inf)]
    for i, dataset in enumerate(result["datasets"]):
        hist = {int(k): v for k, v in dataset["length_histogram"].items()}
        values = [100 * sum(v for k, v in hist.items() if lo <= k <= hi) / dataset["senders"] for lo, hi in bins]
        positions = [x + (i - .5) * .38 for x in range(len(bins))]
        bars = axes[0].bar(positions, values, width=.38, color=colors[i], label=labels[i])
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        qs = [dataset["length_quantiles"][q] for q in ("0.5", "0.9", "0.95", "0.99")]
        bars = axes[1].bar([x + (i - .5) * .38 for x in range(4)], qs, width=.38, color=colors[i])
        axes[1].bar_label(bars, fontsize=9, padding=3)
        rates = [dataset["positive_transaction_pct"], dataset["positive_sender_pct"]]
        bars = axes[2].bar([x + (i - .5) * .38 for x in range(2)], rates, width=.38, color=colors[i])
        axes[2].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axes[0].set(xticks=range(5), xticklabels=["1", "2–4", "5–16", "17–64", "65+"], ylabel="% de remitentes", xlabel="Transacciones por remitente", title="Historial disponible", ylim=(0, 113))
    axes[0].legend(frameon=False)
    axes[1].set(xticks=range(4), xticklabels=["P50", "P90", "P95", "P99"], ylabel="Transacciones por remitente", title="Percentiles de longitud")
    axes[2].set(xticks=range(2), xticklabels=["Transacciones", "Remitentes"], ylabel="% positivo en cada población", title="Desbalance de etiquetas distintas")
    for ax in axes[1:]:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.suptitle("PaySim vs IBM HI-Small · archivos completos\nPositivo = fraude en PaySim; lavado en IBM", fontsize=14)

    sample = result["sample"]
    crop, crop_axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, title, keys in zip(crop_axes, ["Transacciones conservadas", "Remitentes positivos", "Percentil 99 de longitud"], [
        ("full_transactions", "retained_transactions"),
        ("positive_senders_full", "positive_senders_retained"),
        (None, None),
    ]):
        values = [sample[k] for k in keys] if keys[0] else [sample[stage]["length_quantiles"]["0.99"] for stage in ("before", "after")]
        bars = ax.bar(["Antes", "Después"], values, color=["#8996a4", colors[1]])
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4)
        ax.set(title=title, ylim=(0, max(values) * 1.2))
        ax.ticklabel_format(axis="y", style="plain")
    crop.suptitle(f"Muestra uniforme: {sample['sample_senders']:,} remitentes · semilla {sample['seed']}\nÚltimas {sample['max_len']} operaciones: {sample['truncated_sender_pct']:.2f}% de historiales recortados", fontsize=13)
    return {"comparacion_datasets": fig, "efecto_recorte": crop}


def save_results(result, output_dir, with_plots=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparacion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "resumen.md").write_text(comparison_markdown(result) + "\n", encoding="utf-8")
    fields = [key for key in result["datasets"][0] if key not in {"length_histogram", "length_quantiles"}]
    with (output / "resumen.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["datasets"])
    if with_plots:
        import matplotlib.pyplot as plt
        for name, figure in create_figures(result).items():
            figure.savefig(output / f"{name}.png", dpi=160)
            figure.savefig(output / f"{name}.svg")
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paysim", type=Path, default=Path("datasets/paysim.csv"))
    parser.add_argument("--ibm", type=Path, default=Path("datasets/IBM/HI-Small_Trans.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_comparison"))
    parser.add_argument("--n-senders", type=int, default=50000)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="Ejecutar sin dependencias externas")
    args = parser.parse_args()
    result = run_comparison(args.paysim, args.ibm, args.n_senders, args.max_len, args.seed)
    save_results(result, args.output, with_plots=not args.no_plots)
    print(comparison_markdown(result))
    print(f"\nResultados guardados en {args.output.resolve()}")


if __name__ == "__main__":
    main()
