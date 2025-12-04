import os
import json
import argparse
from typing import Dict, Any, List, Tuple

METRICS = ["AP_grape", "mAP", "matchious_mean", "matchious_max",
           "acc_vis", "acc_cause"]


def parse_one_log(path: str) -> Dict[str, Tuple[float, Dict[str, Any]]]:
    """
    从单个 log.json 中找出各个 metric 的最佳记录（只看 mode == 'val'）。
    返回: metric -> (best_value, record_dict)
    """
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if rec.get("mode") != "val":
                continue

            for m in METRICS:
                if m not in rec:
                    continue
                v = rec[m]
                cur = best.get(m)
                if cur is None or v > cur[0]:
                    best[m] = (v, rec)

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Scan mmdet3d JSON logs and find best epochs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="One or more *.log.json files"
    )
    args = parser.parse_args()

    all_files: List[str] = []

    # 支持传目录或单文件
    for p in args.logs:
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".log.json"):
                    all_files.append(os.path.join(p, fn))
        else:
            all_files.append(p)

    all_files = sorted(set(all_files))

    if not all_files:
        print("No log files found.")
        return

    # 每个文件自己的 best
    global_best_AP: Tuple[float, str, Dict[str, Any]] = (-1.0, "", {})
    global_best_mAP: Tuple[float, str, Dict[str, Any]] = (-1.0, "", {})

    print("=== Per-file best results ===")
    for path in all_files:
        best = parse_one_log(path)
        print("\n-----", os.path.basename(path), "-----")

        if not best:
            print("  (no val records found)")
            continue

        def fmt_rec(val, rec):
            ep = rec.get("epoch")
            it = rec.get("iter")
            msg = f"  value={val:.5f} @ epoch={ep}, iter={it}"
            extra = []
            for m in ["matchious_mean", "matchious_max", "acc_vis", "acc_cause"]:
                if m in rec:
                    extra.append(f"{m}={rec[m]:.5f}")
            if extra:
                msg += " | " + ", ".join(extra)
            return msg

        # 重点打印 AP_grape & mAP
        if "AP_grape" in best:
            v, rec = best["AP_grape"]
            print("[best AP_grape]")
            print(fmt_rec(v, rec))
            if v > global_best_AP[0]:
                global_best_AP = (v, path, rec)

        if "mAP" in best:
            v, rec = best["mAP"]
            print("[best mAP]")
            print(fmt_rec(v, rec))
            if v > global_best_mAP[0]:
                global_best_mAP = (v, path, rec)

        # 其它指标随缘打印
        for m in METRICS:
            if m in ("AP_grape", "mAP"):
                continue
            if m in best:
                v, rec = best[m]
                print(f"[best {m}]")
                print(fmt_rec(v, rec))

    print("\n=== Global best across all logs ===")

    if global_best_AP[0] >= 0:
        v, path, rec = global_best_AP
        print("\n>>> Global best AP_grape:")
        print(f"  file = {os.path.basename(path)}")
        print(f"  value = {v:.5f}")
        print(f"  epoch = {rec.get('epoch')}, iter = {rec.get('iter')}")
        for m in ["matchious_mean", "matchious_max", "acc_vis", "acc_cause"]:
            if m in rec:
                print(f"  {m} = {rec[m]:.5f}")

    if global_best_mAP[0] >= 0:
        v, path, rec = global_best_mAP
        print("\n>>> Global best mAP:")
        print(f"  file = {os.path.basename(path)}")
        print(f"  value = {v:.5f}")
        print(f"  epoch = {rec.get('epoch')}, iter = {rec.get('iter')}")
        for m in ["matchious_mean", "matchious_max", "acc_vis", "acc_cause"]:
            if m in rec:
                print(f"  {m} = {rec[m]:.5f}")


if __name__ == "__main__":
    main()
