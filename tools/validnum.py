import mmcv  
import pickle  
  
# 加载训练集annotation文件  
ann_file = '/wangjinhai1/zYu/QTNet/data/nuscenes/nuscenes_infos_train.pkl'  
data = mmcv.load(ann_file)  
train_infos = data['infos']  
  
print(f"总样本数量: {len(train_infos)}")  
  
# 检查有效样本数量（有ground truth的样本）  
valid_samples = 0  
empty_gt_samples = 0  
  
for i, info in enumerate(train_infos):  
    if 'gt_names' in info and len(info['gt_names']) > 0:  
        # 检查是否有有效的ground truth标签  
        valid_labels = [name for name in info['gt_names'] if name != '']  
        if len(valid_labels) > 0:  
            valid_samples += 1  
        else:  
            empty_gt_samples += 1  
    else:  
        empty_gt_samples += 1  
  
print(f"有效样本数量: {valid_samples}")  
print(f"空GT样本数量: {empty_gt_samples}")