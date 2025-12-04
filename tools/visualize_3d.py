import open3d as o3d
import numpy as np
import pickle
import os
import sys
import time

# ==================== 环境配置 ====================
os.environ["OPEN3D_CPU_RENDERING"] = "true"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

def create_bbox_lineset(box_params, color=[1, 0, 0]):
    """将 3D 框参数转换为 Open3D 线框"""
    center = np.array([box_params[0], box_params[1], box_params[2]])
    size = np.array([box_params[3], box_params[4], box_params[5]])
    yaw = box_params[6]

    # 计算旋转矩阵
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 计算顶点
    corners = np.array([
        [-size[0]/2, -size[1]/2, -size[2]/2],
        [size[0]/2, -size[1]/2, -size[2]/2],
        [size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, size[1]/2, size[2]/2],
        [-size[0]/2, size[1]/2, size[2]/2]
    ])

    # 应用旋转和平移
    corners = np.dot(corners, rot.T) + center
    print(f"线框顶点范围 - Min: {np.min(corners, axis=0)}, Max: {np.max(corners, axis=0)}")
    # 定义边
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

    # 创建线框
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.paint_uniform_color(color)
    return lineset

# ==================== 离屏渲染核心逻辑 ====================
def visualize_pkl_headless(pkl_path, output_image="result.png"):
    """无头模式渲染并保存为图片"""
    # 加载数据
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"数据键名: {data.keys()}")
        print(f"预测框数量: {len(data.get('pred_boxes', []))}")
        print(f"真实框数量: {len(data.get('gt_boxes', []))}")
    except Exception as e:
        print(f"加载 PKL 文件失败: {e}")
        return

     # 生成几何体
    geometries = []
    for box in data.get('pred_boxes', []):
        geometries.append(create_bbox_lineset(box, [1, 0, 0]))
    for box in data.get('gt_boxes', []):
        geometries.append(create_bbox_lineset(box, [0, 1, 0]))

    # 初始化渲染器
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(visible=False)
    except Exception as e:
        print(f"创建窗口失败: {e}")
        return

    # 添加几何体
    for geom in geometries:
        vis.add_geometry(geom)

    # 强制更新渲染
    for _ in range(10):
        vis.update_geometry(None)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

     # ------------------------- 动态计算视角 -------------------------
    if len(geometries) == 0:
        print("警告：无几何体可渲染")
        return

    # 合并所有线框的顶点
    all_points = []
    for geom in geometries:
        if isinstance(geom, o3d.geometry.LineSet):
            all_points.append(np.asarray(geom.points))
    if not all_points:
        print("警告：无有效顶点数据")
        return
    points = np.concatenate(all_points)

    # 计算场景包围盒
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    max_size = max(size)

    # 设置相机参数
    view_ctl = vis.get_view_control()
    if view_ctl:
        # 根据场景大小动态调整缩放
        zoom = 0.6 * max_size  # 缩放系数与场景大小成反比
        view_ctl.set_zoom(zoom)

        # 设置相机位置和视角
        camera_pos = center + np.array([0, -max_size, max_size])  # 右前上方
        view_ctl.set_lookat(center)
        view_ctl.set_front(camera_pos - center)
        view_ctl.set_up([0, 0, 1])

    # ------------------------- 渲染优化 -------------------------
    # 设置白色背景和高对比度线框
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.line_width = 5.0  # 加粗线宽

    # 强制多次渲染更新
    for _ in range(20):
        vis.update_geometry(None)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    # 保存结果
    # 在 capture_screen_image 前添加以下代码
    depth_image = vis.capture_depth_float_buffer()
    depth_array = np.asarray(depth_image)

    if np.all(depth_array == 1.0):  # Open3D深度图默认无物体时为1.0
        print("错误：深度图全白，说明物体未被渲染")
    else:
        # 保存深度图供调试
        import matplotlib.pyplot as plt
        plt.imsave("/wangjinhai1/zYu/QTNet/training_results/img/debug_depth.png", depth_array, cmap='gray')
        print("深度图已保存到 debug_depth.png")
    vis.capture_screen_image(output_image)
    vis.destroy_window()
    print(f"渲染结果已保存至: {output_image}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_3d.py <pkl文件路径> [输出图片路径]")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    output_img = sys.argv[2] if len(sys.argv) > 2 else "result.png"
    visualize_pkl_headless(pkl_file, output_img)