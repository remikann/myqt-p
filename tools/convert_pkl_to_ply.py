import pickle
import numpy as np

def create_bbox_vertices(box_params):
    """
    根据3D框参数生成立方体的8个顶点坐标
    Args:
        box_params: [x, y, z, dx, dy, dz, yaw, ...]（至少7个值）
    Returns:
        vertices: 8个顶点的坐标 (8x3)
    """
    # 只提取前7个值
    x, y, z, dx, dy, dz, yaw = box_params[:7]
    center = np.array([x, y, z])
    size = np.array([dx, dy, dz])
    
    # 生成局部坐标系下的顶点
    corners = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2]
    ])
    
    # 计算旋转矩阵（绕Z轴旋转yaw）
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    # 应用旋转和平移
    rotated_corners = np.dot(corners, rot.T) + center
    return rotated_corners

def save_as_ply(pkl_path, output_ply="detection_boxes.ply"):
    """
    将PKL文件中的3D框保存为PLY文件（包含颜色信息）
    """
    # 加载PKL数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查数据格式
    print(f"pred_boxes shape: {data['pred_boxes'].shape}")
    print(f"gt_boxes shape: {data['gt_boxes'].shape}")
    
    # 确保数据是二维数组（N x M）
    if len(data['pred_boxes'].shape) != 2 or len(data['gt_boxes'].shape) != 2:
        raise ValueError("数据格式错误：pred_boxes 和 gt_boxes 应为二维数组")
    
    # 合并所有顶点、边和颜色
    all_vertices = []
    all_edges = []
    all_colors = []  # 存储顶点颜色
    vertex_count = 0
    
    # 处理预测框（红色）
    for box in data.get('pred_boxes', []):
        vertices = create_bbox_vertices(box)
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面
            [0,4], [1,5], [2,6], [3,7]   # 侧面
        ]
        # 调整顶点索引（全局）
        adjusted_edges = [[v + vertex_count for v in edge] for edge in edges]
        all_edges.extend(adjusted_edges)
        all_vertices.extend(vertices)
        
        # 为每个顶点分配红色
        for _ in range(len(vertices)):
            all_colors.append([255, 0, 0])  # 红色 (R, G, B)
        
        vertex_count += len(vertices)
    
    # 处理真实框（绿色）
    for box in data.get('gt_boxes', []):
        vertices = create_bbox_vertices(box)
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        adjusted_edges = [[v + vertex_count for v in edge] for edge in edges]
        all_edges.extend(adjusted_edges)
        all_vertices.extend(vertices)
        
        # 为每个顶点分配绿色
        for _ in range(len(vertices)):
            all_colors.append([0, 255, 0])  # 绿色 (R, G, B)
        
        vertex_count += len(vertices)
    
    # 写入PLY文件
    with open(output_ply, 'w') as f:
        # 头信息
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")  # 颜色属性
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(all_edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # 写入顶点和颜色
        for vertex, color in zip(all_vertices, all_colors):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        
        # 写入边
        for edge in all_edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"PLY文件已保存至: {output_ply}")

# 使用示例
pkl_file = "/wangjinhai1/zYu/QTNet/training_results/result_batch1_step400_20250313_145856.pkl"
save_as_ply(pkl_file, "/wangjinhai1/zYu/QTNet/training_results/img/output_boxes1.ply")