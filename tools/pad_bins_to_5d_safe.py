# tools/pad_bins_to_5d_safe.py
import os
import json
import numpy as np

# 这里填你在服务器上的数据根目录
ROOT = "data/test"

# 可能存在的 version 目录名，自行按需增删
VERSIONS = ["v1.0-custom-trainval", "v1.0-custom"]

# 先 dry-run 看看效果，确认没问题后改成 False 就会真正写回文件
DRY_RUN = False


def find_sample_data_json(root, versions):
    for ver in versions:
        p = os.path.join(root, ver, "sample_data.json")
        if os.path.exists(p):
            return ver, p
    raise FileNotFoundError("sample_data.json not found under any of: "
                            + ", ".join(os.path.join(root, v) for v in versions))


def patch_one_bin(bin_path):
    """对单个 .bin 进行补 5 维处理，返回标签字符串用于统计。"""
    arr = np.fromfile(bin_path, dtype=np.float32)
    total = arr.size
    if total == 0:
        return "empty"

    div3 = (total % 3 == 0)
    div4 = (total % 4 == 0)
    div5 = (total % 5 == 0)

    # 1) 优先检查是否是真·5维（第5维全0）
    if div5:
        pts5 = arr.reshape(-1, 5)
        last = pts5[:, 4]
        max_abs_last = float(np.max(np.abs(last)))
        if np.allclose(last, 0.0, atol=1e-6):
            # 真正已经是 [x,y,z,i,0] 的5维：保持不动
            return "keep_5d_zero"

        # 否则，第5维非0，极大可能是“本来是4维，但因为 total%5==0 被误当成5维”
        if div4:
            # 按4维重解，然后 pad 成5维
            pts4 = arr.reshape(-1, 4)
            N = pts4.shape[0]
            pad = np.zeros((N, 1), dtype=np.float32)
            out = np.hstack([pts4, pad]).astype(np.float32)
            if not DRY_RUN:
                out.tofile(bin_path)
            return "fix_fake5_to5"
        else:
            # 真的存在非0第5维的5D数据（对你来说很少见，先打log不动）
            return f"weird_5d_nonzero(max_abs={max_abs_last:.3g})"

    # 2) 正常 4 维 [x,y,z,intensity] -> [x,y,z,intensity, 0]
    if div4:
        pts4 = arr.reshape(-1, 4)
        N = pts4.shape[0]
        pad = np.zeros((N, 1), dtype=np.float32)
        out = np.hstack([pts4, pad]).astype(np.float32)
        if not DRY_RUN:
            out.tofile(bin_path)
        return "pad_4_to5"

    # 3) 老的 3 维 [x,y,z] -> [x,y,z,1,0] （强行补一个强度=1 占位）
    if div3:
        pts3 = arr.reshape(-1, 3)
        N = pts3.shape[0]
        intensity = np.ones((N, 1), dtype=np.float32)
        pad = np.zeros((N, 1), dtype=np.float32)
        out = np.hstack([pts3, intensity, pad]).astype(np.float32)
        if not DRY_RUN:
            out.tofile(bin_path)
        return "pad_3_to5"

    # 4) 既不是3/4/5维，认为异常
    return f"bad_dim(total={total})"


def main():
    # 1) 找到 sample_data.json
    ver, sd_path = find_sample_data_json(ROOT, VERSIONS)
    print(f"[info] using version={ver}, sample_data.json={sd_path}")

    with open(sd_path, encoding="utf-8") as f:
        sample_data = json.load(f)

    n_total = 0
    stats = {}  # tag -> count
    show_first = 10  # 前几条打印出来看看

    for rec in sample_data:
        fn = rec.get("filename", "")
        chan = rec.get("channel", "")

        # 只处理 LiDAR_TOP 的 .bin
        if "LIDAR_TOP" not in fn:
            continue
        if not fn.lower().endswith(".bin"):
            continue

        # nuScenes 风格路径 -> 真实路径
        fpath = os.path.join(ROOT, fn.replace("\\", "/"))
        if not os.path.exists(fpath):
            print(f"[miss] {fpath}")
            continue

        tag = patch_one_bin(fpath)
        stats[tag] = stats.get(tag, 0) + 1
        n_total += 1

        if n_total <= show_first:
            print(f"[{n_total:03d}] {fpath} -> {tag}")

    print("\n[summary] processed LiDAR_TOP bin files:", n_total)
    for k, v in sorted(stats.items(), key=lambda x: x[0]):
        print(f"  {k}: {v}")

    if DRY_RUN:
        print("\n[warning] DRY_RUN=True，本次没有真正修改任何文件。"
              "确认统计结果合理后可将 DRY_RUN 改为 False 再运行一次。")


if __name__ == "__main__":
    main()
