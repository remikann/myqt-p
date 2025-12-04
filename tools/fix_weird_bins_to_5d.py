# tools/fix_weird_bins_to_5d.py
import os
import numpy as np

LIST_FILE = 'bins_5dim_weird.txt'  # debug_bin_dims_precise.py 生成的

def main():
    if not os.path.exists(LIST_FILE):
        print(f'{LIST_FILE} 不存在，请先运行 debug_bin_dims_precise.py')
        return

    paths = []
    with open(LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行格式大概是：
            # data/test/.../xxx.bin  total=XXXXX  max_abs_5th=...
            path = line.split()[0]
            paths.append(path)

    print(f'将尝试修复 {len(paths)} 个“假 5 维” bin 文件')
    fixed = 0
    failed = 0

    for p in paths:
        if not os.path.exists(p):
            print(f'[miss] {p}')
            failed += 1
            continue

        arr = np.fromfile(p, dtype=np.float32)
        total = arr.size

        # 根据 extractor 的设定，这些其实都是 4 维写出来的：
        if total % 4 != 0:
            print(f'[skip] {p}: total={total} 不能被 4 整除，怀疑不是 4 维原始数据')
            failed += 1
            continue

        pts4 = arr.reshape(-1, 4)  # [x,y,z,intensity]
        N = pts4.shape[0]
        pad = np.zeros((N, 1), dtype=np.float32)  # 新增一列全 0，作为 ring/占位
        pts5 = np.hstack([pts4, pad]).astype(np.float32)

        pts5.tofile(p)
        fixed += 1
        print(f'[fixed] {p}: total={total} -> 形状 {pts5.shape} (4D -> 5D)')

    print(f'\n完成。成功修复 {fixed} 个，失败 {failed} 个。')

if __name__ == '__main__':
    main()
