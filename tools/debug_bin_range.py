import numpy as np

bin_path = "data/test/HD1080_SN31574219_16-11-44/samples/LIDAR_TOP/1757578305105070.bin"

pts = np.fromfile(bin_path, dtype=np.float32)
print("总 float 数：", pts.size)

for load_dim in [4, 5, 6]:
    if pts.size % load_dim == 0:
        xyz = pts.reshape(-1, load_dim)[:, :3]
        mn = xyz.min(0)
        mx = xyz.max(0)
        print(f"[load_dim={load_dim}] N={xyz.shape[0]}, "
              f"x:[{mn[0]:.2f},{mx[0]:.2f}] "
              f"y:[{mn[1]:.2f},{mx[1]:.2f}] "
              f"z:[{mn[2]:.2f},{mx[2]:.2f}]")
