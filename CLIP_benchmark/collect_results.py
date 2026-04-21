#!/usr/bin/env python3
"""
整理 CLIP Benchmark 实验结果。

用法:
    python collect_results.py \
        --result_dir result \
        --model_file benchmark/models.txt \
        --output_dir output

    # 可同时指定多个结果目录
    python collect_results.py \
        --result_dir result result/clip-refine \
        --model_file benchmark/models.txt benchmark/models_clip-refine.txt \
        --output_dir output

    # 额外加入 OpenAI baseline 对比
    python collect_results.py \
        --result_dir result \
        --model_file benchmark/models.txt \
        --output_dir output \
        --add_openai_baseline
"""

import argparse
import json
import os
import glob
import csv
from collections import OrderedDict


def parse_model_file(model_file):
    """解析 models.txt，返回 [(arch, pretrained_path), ...]"""
    models = []
    with open(model_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                models.append((parts[0].strip(), parts[1].strip()))
    return models


def load_results(result_dirs):
    """加载所有 JSON 结果文件"""
    results = []
    for d in result_dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    data["_file"] = f
                    results.append(data)
            except (json.JSONDecodeError, KeyError):
                print(f"[WARN] 跳过无法解析的文件: {f}")
    return results


def match_pretrained(result_pretrained, model_pretrained):
    """判断结果文件中的 pretrained 字段是否匹配模型配置"""
    if result_pretrained == model_pretrained:
        return True
    rp = os.path.basename(result_pretrained)
    mp = os.path.basename(model_pretrained)
    if rp == mp and rp != "":
        return True
    if model_pretrained.endswith(result_pretrained) or result_pretrained.endswith(model_pretrained):
        return True
    return False


def get_dataset_short(dataset):
    return dataset.replace("wds/", "").replace("vtab/", "")


def make_model_label(arch, pretrained):
    if pretrained == "openai":
        return f"{arch} (OpenAI)"
    basename = os.path.basename(pretrained)
    name = os.path.splitext(basename)[0]
    return f"{arch} ({name})"


def extract_acc(metrics):
    """从 metrics 中提取主要精度指标，返回 dict"""
    out = OrderedDict()
    for key in ["acc1", "acc5", "mean_per_class_recall",
                 "lp_acc1", "lp_acc5", "lp_mean_per_class_recall"]:
        if key in metrics and metrics[key] is not None:
            try:
                val = float(metrics[key])
                if val == val:  # filter NaN
                    out[key] = val
            except (TypeError, ValueError):
                pass
    return out


def collect_for_models(model_list, all_results, add_openai):
    """
    返回 {model_label: {dataset_short: {metric: value}}}
    """
    table = OrderedDict()

    for arch, pretrained in model_list:
        label = make_model_label(arch, pretrained)
        table[label] = {}
        for r in all_results:
            if r["model"] != arch:
                continue
            if not match_pretrained(r["pretrained"], pretrained):
                continue
            ds = get_dataset_short(r["dataset"])
            acc = extract_acc(r["metrics"])
            if acc:
                table[label][ds] = acc

    if add_openai:
        arches = set(arch for arch, _ in model_list)
        for arch in sorted(arches):
            label = f"{arch} (OpenAI)"
            if label in table:
                continue
            table[label] = {}
            for r in all_results:
                if r["model"] != arch or r["pretrained"] != "openai":
                    continue
                ds = get_dataset_short(r["dataset"])
                acc = extract_acc(r["metrics"])
                if acc:
                    table[label][ds] = acc

    return table


def primary_metric(metrics_dict):
    for key in ["acc1", "lp_acc1"]:
        if key in metrics_dict:
            return metrics_dict[key]
    return None


def write_markdown(table, output_path):
    all_datasets = sorted(set(ds for model_data in table.values() for ds in model_data))
    model_names = list(table.keys())

    if not model_names or not all_datasets:
        print("[WARN] 无数据可输出")
        return

    lines = []
    lines.append("# CLIP Benchmark 实验结果\n")

    # --- 主表 (acc1 / lp_acc1) ---
    lines.append("## Zero-shot / Linear Probe 精度 (acc1 %)\n")
    header = "| Dataset | " + " | ".join(model_names) + " |"
    sep = "|---| " + " | ".join(["---"] * len(model_names)) + " |"
    lines.append(header)
    lines.append(sep)

    avg_sums = {m: 0.0 for m in model_names}
    avg_counts = {m: 0 for m in model_names}

    for ds in all_datasets:
        row = f"| {ds}"
        for m in model_names:
            v = primary_metric(table[m].get(ds, {}))
            if v is not None:
                row += " | %.2f" % (v * 100)
                avg_sums[m] += v
                avg_counts[m] += 1
            else:
                row += " | -"
        row += " |"
        lines.append(row)

    row = "| **Average**"
    for m in model_names:
        if avg_counts[m] > 0:
            row += " | **%.2f**" % (avg_sums[m] / avg_counts[m] * 100)
        else:
            row += " | -"
    row += " |"
    lines.append(row)
    lines.append("")

    # --- Delta 表 (相对第一个模型) ---
    if len(model_names) > 1:
        base = model_names[0]
        lines.append("## 相对 %s 的精度变化 (Delta %%)\n" % base)
        compare_models = model_names[1:]
        header = "| Dataset | " + " | ".join(compare_models) + " |"
        sep = "|---| " + " | ".join(["---"] * len(compare_models)) + " |"
        lines.append(header)
        lines.append(sep)
        for ds in all_datasets:
            base_v = primary_metric(table[base].get(ds, {}))
            row = f"| {ds}"
            for m in compare_models:
                v = primary_metric(table[m].get(ds, {}))
                if v is not None and base_v is not None:
                    d = (v - base_v) * 100
                    row += " | %+.2f" % d
                else:
                    row += " | -"
            row += " |"
            lines.append(row)
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown 已写入: {output_path}")


def write_csv(table, output_path):
    all_datasets = sorted(set(ds for model_data in table.values() for ds in model_data))
    model_names = list(table.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + model_names)
        for ds in all_datasets:
            row = [ds]
            for m in model_names:
                v = primary_metric(table[m].get(ds, {}))
                row.append("%.4f" % v if v is not None else "")
            writer.writerow(row)

    print(f"[OK] CSV 已写入: {output_path}")


def write_json_detail(table, output_path):
    out = {}
    for model, datasets in table.items():
        out[model] = {}
        for ds, metrics in datasets.items():
            out[model][ds] = {k: round(v, 6) for k, v in metrics.items()}

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON 已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="整理 CLIP Benchmark 实验结果")
    parser.add_argument("--result_dir", nargs="+", required=True,
                        help="结果 JSON 文件所在目录，可指定多个")
    parser.add_argument("--model_file", nargs="+", required=True,
                        help="模型配置文件 (models.txt 格式)，可指定多个")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录")
    parser.add_argument("--add_openai_baseline", action="store_true",
                        help="自动加入 OpenAI baseline 结果作为对比")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_list = []
    for mf in args.model_file:
        model_list.extend(parse_model_file(mf))

    print(f"从模型文件中加载了 {len(model_list)} 个模型:")
    for arch, pt in model_list:
        print(f"  {arch}  ->  {pt}")

    all_results = load_results(args.result_dir)
    print(f"从结果目录中加载了 {len(all_results)} 个结果文件")

    table = collect_for_models(model_list, all_results, args.add_openai_baseline)

    total_entries = sum(len(v) for v in table.values())
    print(f"匹配到 {total_entries} 条结果，涉及 {len(table)} 个模型")

    if total_entries == 0:
        print("[WARN] 未匹配到任何结果，请检查 result_dir 和 model_file 是否对应")
        return

    write_markdown(table, os.path.join(args.output_dir, "results.md"))
    write_csv(table, os.path.join(args.output_dir, "results.csv"))
    write_json_detail(table, os.path.join(args.output_dir, "results.json"))

    print("\n完成！输出文件:")
    for fname in ["results.md", "results.csv", "results.json"]:
        print(f"  {os.path.join(args.output_dir, fname)}")


if __name__ == "__main__":
    main()
