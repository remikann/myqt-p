import sys, os, json, glob

root = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/sweeps_1126_0317"
runs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])


def read_json_lines(log_json):
    """读取日志里的所有 JSON 行，返回 list[dict]."""
    if not os.path.isfile(log_json):
        return []
    lines = []
    with open(log_json, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                try:
                    lines.append(json.loads(ln))
                except Exception:
                    pass
    return lines


def pick_log_file(run_dir):
    """在每个 run 目录下选择要解析的那个 json 日志文件。

    规则：
      1. 先找 *.log.json
      2. 如果没有，再找 *.json
      3. 如果有多个，按文件名排序，取最后一个（通常是时间戳最大的那个）
    """
    # 1) *.log.json
    cand = glob.glob(os.path.join(run_dir, "*.log.json"))
    # 2) 没有的话，再找 *.json
    if not cand:
        cand = glob.glob(os.path.join(run_dir, "*.json"))

    if not cand:
        return None

    # 你说现在每个目录只有一个 json，这里写稳一点：
    cand = sorted(cand)
    return cand[-1]


def pick_metrics(lines):
    """按 epoch 聚合训练 / 验证指标，并返回该 run 最优的 epoch 统计.

    优先级：
      1) val 的 mAP 最大
      2) 其次 bbox_mAP
      3) 再次 train_mIoU_mean
    """
    epoch_stats = {}  # epoch -> dict

    for x in lines:
        epoch = x.get("epoch", None)
        if epoch is None:
            continue
        mode = x.get("mode", None)
        stat = epoch_stats.setdefault(epoch, {"epoch": epoch})

        if mode == "train":
            mi = x.get("matched_ious", None)
            if mi is not None:
                stat.setdefault("_train_miou_list", []).append(mi)
            loss = x.get("loss", None)
            if loss is not None:
                stat.setdefault("_train_loss_list", []).append(loss)

        elif mode == "val":
            # 一般每个 epoch 只有一条 val 行
            if "mAP" in x:
                stat["mAP"] = x["mAP"]
            if "bbox_mAP" in x:
                stat["bbox_mAP"] = x["bbox_mAP"]
            if "NDS" in x:
                stat["NDS"] = x["NDS"]
            # 把所有 AP_xxx 一并拷过来，比如 AP_grape
            for k, v in x.items():
                if isinstance(k, str) and k.startswith("AP_"):
                    stat[k] = v

    # 汇总 train 的均值 / 最大值
    for e, stat in epoch_stats.items():
        mi_list = stat.pop("_train_miou_list", [])
        if mi_list:
            stat["train_mIoU_mean"] = float(sum(mi_list) / len(mi_list))
            stat["train_mIoU_max"] = float(max(mi_list))
        loss_list = stat.pop("_train_loss_list", [])
        if loss_list:
            stat["train_loss_mean"] = float(sum(loss_list) / len(loss_list))
            stat["train_loss_last"] = float(loss_list[-1])

    if not epoch_stats:
        return {}

    # 选出“最好的一轮”
    def score(s):
        if "mAP" in s:
            return s["mAP"]
        if "bbox_mAP" in s:
            return s["bbox_mAP"]
        if "train_mIoU_mean" in s:
            return s["train_mIoU_mean"]
        return -1.0

    best = max(epoch_stats.values(), key=score)
    if "mAP" in best or "bbox_mAP" in best:
        best["mode"] = "val"
    else:
        best["mode"] = "train"
    return best


def read_hparams_from_run_name(run_dir):
    """从 run 目录名里解析 sweep 的超参，例如:
       work_dirs/.../transfusion_K64_ov0.1_mr2_vw0.5
    """
    name = os.path.basename(run_dir)
    parts = name.split("_")
    hp = {}
    for p in parts:
        if p.startswith("K"):
            try:
                hp["K"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("ov"):
            try:
                hp["gaussian_overlap"] = float(p[2:])
            except Exception:
                pass
        elif p.startswith("mr"):
            try:
                hp["min_radius"] = int(p[2:])
            except Exception:
                pass
        elif p.startswith("vw"):
            try:
                hp["vis_cause_w"] = float(p[2:])
            except Exception:
                pass
    return hp


def fmt(v, fmt_str="{:.4f}"):
    return fmt_str.format(v) if isinstance(v, (int, float)) else "NA"


rows = []

for r in runs:
    logj = pick_log_file(r)
    if logj is None:
        continue

    lines = read_json_lines(logj)
    if not lines:
        continue

    best = pick_metrics(lines)
    if not best:
        continue

    hp = read_hparams_from_run_name(r)
    rows.append({"run": r, **hp, **best})

if not rows:
    print("No results parsed. Check logs exist and contain JSON lines.")
    sys.exit(0)

# 排序：先按 val mAP/bbox_mAP，再按 train_mIoU_mean，最后按 train_loss_mean
def sort_key(x):
    m = x.get("mAP", x.get("bbox_mAP", -1.0))
    mi = x.get("train_mIoU_mean", -1.0)
    loss_mean = x.get("train_loss_mean", 1e9)
    return (-m, -mi, loss_mean)


rows.sort(key=sort_key)

print("\n=== Top-10 by val mAP (desc), then train_mIoU_mean (desc), loss_mean (asc) ===")
for i, r in enumerate(rows[:10], 1):
    m = r.get("mAP", r.get("bbox_mAP", None))
    mi_mean = r.get("train_mIoU_mean", None)
    mi_max = r.get("train_mIoU_max", None)
    loss_mean = r.get("train_loss_mean", None)
    ap_grape = r.get("AP_grape", None)

    print(
        f"[{i:02d}] ep={r.get('epoch')}  "
        f"mAP={fmt(m)}  "
        f"AP_grape={fmt(ap_grape)}  "
        f"train_mIoU_mean={fmt(mi_mean)}  "
        f"train_mIoU_max={fmt(mi_max)}  "
        f"loss_mean={fmt(loss_mean, '{:.3f}')}  "
        f"K={r.get('K')} ov={r.get('gaussian_overlap')} "
        f"mr={r.get('min_radius')} vw={r.get('vis_cause_w')}  "
        f"{r['run']}"
    )
