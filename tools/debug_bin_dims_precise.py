# tools/debug_bin_dims_precise.py
import os
import numpy as np

# 确认这是你 create_data 用的 root-path
ROOT = 'data/test'

def iter_samples_bin_files(root):
    """只遍历 data/test/<scene>/samples/LIDAR_TOP/*.bin，不扫 sweeps。"""
    for scene_name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene_name)
        lidar_dir = os.path.join(scene_dir, 'samples', 'LIDAR_TOP')
        if not os.path.isdir(lidar_dir):
            continue
        for fname in sorted(os.listdir(lidar_dir)):
            if fname.endswith('.bin'):
                yield os.path.join(lidar_dir, fname)

def main():
    bins_4 = []            # 推断为原生 4 维
    bins_5_padded = []     # 推断为“补过的 5 维”，第 5 维全 0
    bins_5_weird = []      # 5 维但第 5 维不全 0（非常可疑）
    bins_weird = []        # 既不是 4 维也不是 5 维（比如 3 维）

    for bin_path in sorted(iter_samples_bin_files(ROOT)):
        arr = np.fromfile(bin_path, dtype=np.float32)
        total = arr.size
        if total == 0:
            bins_weird.append((bin_path, total))
            continue

        div4 = (total % 4 == 0)
        div5 = (total % 5 == 0)

        # 我们现在只认 4 / 5 维，其他一律丢到 weird
        if div5:
            # 尝试按 5 维理解，检查第 5 维是不是全 0
            pts5 = arr.reshape(-1, 5)
            last = pts5[:, 4]
            max_abs = float(np.max(np.abs(last)))
            if np.allclose(last, 0.0, atol=1e-6):
                # 补过的 5 维：第 5 维几乎全 0
                bins_5_padded.append(bin_path)
            else:
                # 真正意义上的 5 维（第 5 维有值），对你来说很可疑
                bins_5_weird.append((bin_path, total, max_abs))
        elif div4:
            # 不能被 5 整除但能被 4 整除 -> 原生 4 维
            bins_4.append(bin_path)
        else:
            # 既不是 4 维也不是 5 维（大概率是 3 维或损坏）
            bins_weird.append((bin_path, total))

    print('=== 统计结果（只看 samples/LIDAR_TOP） ===')
    print(f'推断为原生 4 维的 bin 数量: {len(bins_4)}')
    print(f'推断为补过的 5 维 bin 数量(第5维全0): {len(bins_5_padded)}')
    print(f'解析成5维但第5维不全0的“可疑5维”数量: {len(bins_5_weird)}')
    print(f'既不能被4也不能被5整除的“异常bin”(含3维)数量: {len(bins_weird)}')

    # 写出列表，方便你对照具体是哪些文件
    with open('bins_4dim.txt', 'w', encoding='utf-8') as f:
        for p in bins_4:
            f.write(p + '\n')

    with open('bins_5dim_padded.txt', 'w', encoding='utf-8') as f:
        for p in bins_5_padded:
            f.write(p + '\n')

    if bins_5_weird:
        with open('bins_5dim_weird.txt', 'w', encoding='utf-8') as f:
            for p, total, max_abs in bins_5_weird:
                f.write(f'{p}  total={total}  max_abs_5th={max_abs}\n')

    if bins_weird:
        with open('bins_weird.txt', 'w', encoding='utf-8') as f:
            for p, total in bins_weird:
                f.write(f'{p}  total={total}\n')

if __name__ == '__main__':
    main()
