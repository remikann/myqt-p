import open3d as o3d
import numpy as np
import pickle
import os

def visualize_pkl_results_web(pkl_path):
    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 创建所有线框的集合
    geometries = []
    
    # 添加预测框（红色）
    for box in data['pred_boxes']:
        lineset = create_bbox_lineset(box, color=[1, 0, 0])
        geometries.append(lineset)
    
    # 添加真实框（绿色）
    for box in data['gt_boxes']:
        lineset = create_bbox_lineset(box, color=[0, 1, 0])
        geometries.append(lineset)

    # 导出为 HTML
    o3d.visualization.draw(geometries, output_path="result.html")
    print("可视化结果已保存到 result.html，请下载并在浏览器中打开")
def visualize_pkl_results(pkl_path):
    # 从 pkl 文件加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 提取预测框和真实框
    pred_boxes = data['pred_boxes']  # 形状: [N, 7] (x, y, z, dx, dy, dz, yaw)
    gt_boxes = data['gt_boxes']      # 形状: [M, 7]

    # 创建 Open3D 可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800)

    # ------------------------- 可视化预测框 -------------------------
    for box in pred_boxes:
        # 将 3D 框参数转换为线框
        lineset = create_bbox_lineset(box, color=[1, 0, 0])  # 红色为预测框
        vis.add_geometry(lineset)

    # ------------------------- 可视化真实框 -------------------------
    for box in gt_boxes:
        lineset = create_bbox_lineset(box, color=[0, 1, 0])  # 绿色为真实框
        vis.add_geometry(lineset)

    # ------------------------- 设置视角和渲染 -------------------------
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, -1, 0])  # 视角朝向 y 轴负方向
    view_ctl.set_up([0, 0, 1])      # Z 轴为垂直方向
    view_ctl.set_zoom(0.5)          # 缩放系数

    vis.run()
    vis.destroy_window()

def create_bbox_lineset(box_params, color=[1, 0, 0]):
    """
    将 3D 框参数转换为 Open3D 线框
    Args:
        box_params: [x, y, z, dx, dy, dz, yaw] (中心坐标、尺寸、偏航角)
        color: 线框颜色，默认红色
    Returns:
        lineset: Open3D LineSet 对象
    """
    # 解析参数
    center = np.array([box_params[0], box_params[1], box_params[2]])
    size = np.array([box_params[3], box_params[4], box_params[5]])
    yaw = box_params[6]

    # 计算旋转矩阵
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 计算 8 个顶点
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

    # 定义 12 条边（立方体结构）
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # 创建 LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.paint_uniform_color(color)

    return lineset

# 使用示例
pkl_file = "/wangjinhai1/zYu/QTNet/training_results/result_batch0_step0_20250313_145011.pkl"  # 替换为实际路径
visualize_pkl_results_web(pkl_file)
