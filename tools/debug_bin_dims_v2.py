import os
import numpy as np

ROOT = 'data/test'          # 本地就改成 F:/Dataset/cupcup 那个根
POSSIBLE_DIMS = (3, 4, 5)   # 如果你确定只有 4/5 维，可以改成 (4, 5)

def iter_bin_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # 只看 LIDAR_TOP 目录，避免扫到别的 bin
        if os.path.basename(dirpath) != 'LIDAR_TOP':
            continue
        for fname in filenames:
            if fname.endswith('.bin'):
                yield os.path.join(dirpath, fname)

def main():
    stats = {}          # dim -> {'count':..., 'N':[...] }
    ambiguous = {}      # tuple(cands) -> [ (path, total, {d:N_d}) ]
    weird = []          # 无法被 POSSIBLE_DIMS 整除

    for bin_path in sorted(iter_bin_files(ROOT)):
        arr = np.fromfile(bin_path, dtype=np.float32)
        total = arr.size

        # 所有可能的维度候选
        cands = [d for d in POSSIBLE_DIMS if total % d == 0]
        if not cands:
            weird.append((bin_path, total))
            continue

        if len(cands) == 1:
            # 唯一候选：认为 dim 就是它
            d = cands[0]
            N = total // d
            if d not in stats:
                stats[d] = {'count': 0, 'N': []}
            stats[d]['count'] += 1
            stats[d]['N'].append(N)
        else:
            # 多个维度都能整除：记录成“歧义文件”
            Ns = {d: total // d for d in cands}
            key = tuple(cands)
            ambiguous.setdefault(key, []).append((bin_path, total, Ns))

    print('=== 唯一维度推断的文件统计（不含歧义文件） ===')
    for d in sorted(stats.keys()):
        s = stats[d]
        N_arr = np.array(s['N'], dtype=np.int64)
        print(f'[dim={d}] files={s["count"]}, '
              f'N_min={N_arr.min()}, N_max={N_arr.max()}, N_mean={N_arr.mean():.1f}')

    if ambiguous:
        print('\n=== 存在「多个维度都能整除」的歧义文件 ===')
        for key, lst in ambiguous.items():
            print(f'候选维度 {key}: 共 {len(lst)} 个文件（只展示前 10 个）')
            for path, total, Ns in lst[:10]:
                Ns_str = ', '.join([f'd={d}: N={Ns[d]}' for d in key])
                print(f'  {path}')
                print(f'    total_floats={total}  ->  {Ns_str}')

    if weird:
        print('\n=== 无法被 POSSIBLE_DIMS 整除的奇怪 bin ===')
        for path, total in weird[:20]:
            print(f'  {path}  total_floats={total}')
        if len(weird) > 20:
            print(f'  ... 还有 {len(weird) - 20} 个省略')

if __name__ == "__main__":
    main()