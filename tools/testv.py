import open3d as o3d

# 测试 Open3D 是否支持离屏渲染
def test_offscreen():
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(visible=False)
        print("Open3D 支持离屏渲染！")
        vis.destroy_window()
    except Exception as e:
        print(f"Open3D 不支持离屏渲染: {e}")

test_offscreen()