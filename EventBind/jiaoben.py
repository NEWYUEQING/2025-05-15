import os
import torch
from PIL import Image
from torchvision import transforms
from model.EventCLIP import load_clip_to_cpu
from model.ImageEncoder import ImageEncoder

def get_feature_path(frame_path, feature_root, root_dir):
    """
    根据原始帧路径生成特征存储路径，保持目录结构。

    参数：
        frame_path (str): 原始帧文件路径，例如 '/home/username001/nyq/Data_HAIHE/train/407abnormal1/event_frame_1741068918936329.png'
        feature_root (str): 特征存储根目录，例如 '/home/username001/nyq/features'
        root_dir (str): 数据集根目录，例如 '/home/username001/nyq/Data_HAIHE'

    返回：
        str: 特征存储路径，例如 '/home/username001/nyq/features/train/407abnormal1/event_frame_1741068918936329.pt'
    """
    relative_path = os.path.relpath(frame_path, start=root_dir)
    feature_path = os.path.join(feature_root, relative_path)
    feature_path = os.path.splitext(feature_path)[0] + '.pt'
    return feature_path

def read_yaml(yaml_path):
    """
    读取 YAML 配置文件。

    参数：
        yaml_path (str): 配置文件路径

    返回：
        dict: 配置内容
    """
    import yaml
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result

def extract_and_save_features(root_dir, feature_root, cfg):
    """
    遍历 root_dir 下的所有视频目录，提取视频帧特征并保存到 feature_root 下的对应目录。

    参数：
        root_dir (str): 数据集根目录，例如 '/home/username001/nyq/Data_HAIHE'
        feature_root (str): 特征存储根目录，例如 '/home/username001/nyq/features'
        cfg (dict): 配置文件内容
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载 CLIP 模型
    clip_model = load_clip_to_cpu(cfg)
    print(f"加载 CLIP 模型: {cfg['MODEL']['BACKBONE']['Name']}")

    # 实例化 ImageEncoder
    image_encoder = ImageEncoder(cfg, clip_model).to(device)
    image_encoder.eval()  # 设置为评估模式

    # 定义图像预处理，与训练时一致
    preprocess = transforms.Compose([
        transforms.Resize((cfg['Dataset']['resize_size'][0], cfg['Dataset']['resize_size'][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 遍历 root_dir 下的所有子目录
    for video_dir, _, files in os.walk(root_dir):
        if not files:
            continue  # 跳过没有文件的目录

        # 过滤出图像文件，处理 event_frame 开头的文件
        image_files = [f for f in files if f.endswith(('.jpg', '.png')) and f.startswith('image_frame')]
        if not image_files:
            continue  # 跳过没有图像文件的目录

        image_files.sort()  # 确保帧按顺序处理

        for file in image_files:
            frame_path = os.path.join(video_dir, file)

            # 加载并预处理图像
            image = Image.open(frame_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).half().to(device)  # [1, C, H, W]

            # 提取特征
            with torch.no_grad():
                feature = image_encoder(image_tensor)
                if isinstance(feature, tuple):
                    feature = feature[0]  # 只取主特征，忽略低层次特征
                feature = feature.cpu()  # 转移到 CPU 保存

            # 生成特征存储路径
            feature_path = get_feature_path(frame_path, feature_root, root_dir)
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)

            # 保存特征
            torch.save(feature, feature_path)
            print(f"特征已保存至: {feature_path}")

if __name__ == '__main__':
    # 配置参数
    cfg_path = '/home/username001/nyq/EventBind-master/Configs/HAIHE.yaml'
    root_dir = '/home/username001/nyq/Data_HAIHE'  # 修改为整个数据集根目录
    feature_root = '/home/username001/nyq/features'  # 特征存储根目录

    # 读取配置文件
    cfg = read_yaml(cfg_path)

    # 执行特征提取和保存
    extract_and_save_features(root_dir, feature_root, cfg)