import zarr
import numpy as np
import os
# from loguru import logger
import time

class DataRecorder:
    def __init__(self, 
                 save_dir: str,
                 data_types: list = None,
                 camera_img_shape=(240, 320, 3),
                 depth_img_shape=(240, 320, 1),  # 添加深度图形状
                 tactile_img_shape=(240, 320, 3),
                 mask_shape=(240, 320, 4)):
        """
        初始化数据记录器
        Args:
            save_dir: 保存目录
            data_types: 需要记录的数据类型列表，如果为None则记录所有支持的类型
            camera_img_shape: 彩色图像形状
            depth_img_shape: 深度图像形状
            tactile_img_shape: 触觉图像形状
            mask_shape: mask图像形状
        """
        self.save_dir = save_dir
        
        # 定义所有支持的数据类型及其配置
        self.supported_data_config = {
            'timestamp': {'shape': (1,), 'dtype': 'float32'},
            'action': {'shape': (5,), 'dtype': 'float32'},
            'wrist_camera_img': {'shape': camera_img_shape, 'dtype': 'uint8'},
            'head_camera_img': {'shape': camera_img_shape, 'dtype': 'uint8'},
            'wrist_depth_img': {'shape': depth_img_shape, 'dtype': 'float32'},  # 添加深度图配置
            'head_depth_img': {'shape': depth_img_shape, 'dtype': 'float32'},  # 添加深度图配置
            'tactile_img': {'shape': tactile_img_shape, 'dtype': 'uint8'},
            'joint_state': {'shape': (7,), 'dtype': 'float32'},
            'tcp_poses': {'shape': (6,), 'dtype': 'float32'},
            'gripper_width': {'shape': (1,), 'dtype': 'float32'},
            'tactile_marker': {'shape': (15,), 'dtype': 'float32'},
            'mask': {'shape': mask_shape, 'dtype': 'uint8'}
        }
        
        # 设置要记录的数据类型
        self.data_types = data_types if data_types is not None else list(self.supported_data_config.keys())
        
        # 初始化数据存储
        self.data = {key: [] for key in self.data_types}
        self.episode_ends = []
        self.total_steps = 0
        self.current_episode = 0
        
        # logger.info(f"将记录以下数据类型: {self.data_types}")

    def add_step_data(self, **kwargs):
        """
        添加一个步骤的数据
        只记录在data_types中指定的数据类型
        """
        # 检查并记录提供的数据
        for key, value in kwargs.items():
            if key not in self.data_types:
                continue
                
            # 验证数据形状
            expected_shape = self.supported_data_config[key]['shape']
            assert value.shape == expected_shape, \
                f"{key}维度应为{expected_shape}，但得到{value.shape}"
            
            self.data[key].append(value)
        
        self.total_steps += 1

    def end_episode(self):
        """标记当前episode结束"""
        self.episode_ends.append(self.total_steps)
        self.current_episode += 1

    def save_to_zarr(self):
        """将数据保存为zarr格式"""
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, 'replay_buffer.zarr')
        
        root = zarr.group(save_path)
        data = root.create_group('data')
        meta = root.create_group('meta')
        
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        # 保存实际收集的数据
        for key in self.data_types:
            if not self.data[key]:  # 跳过空列表
                continue
                
            array_data = np.array(self.data[key], 
                                dtype=self.supported_data_config[key]['dtype'])
            
            # 根据数据类型设置不同的chunk大小
            if len(array_data.shape) > 2:  # 图像类数据
                chunks = (1000, *array_data.shape[1:])
            else:  # 其他数据类型
                chunks = (10000, *array_data.shape[1:])
            
            data.create_dataset(
                key, 
                data=array_data,
                chunks=chunks,
                dtype=self.supported_data_config[key]['dtype'],
                compressor=compressor
            )

        # 保存episode信息
        if self.episode_ends:
            meta.create_dataset(
                'episode_ends',
                data=np.array(self.episode_ends, dtype=np.int64),
                chunks=(10000,),
                dtype='int64',
                compressor=compressor
            )

        # logger.info(f"数据已保存至 {save_path}")
        # logger.info(f"总步数: {self.total_steps}")
        # logger.info(f"总episode数: {self.current_episode}")
        # logger.info("数据结构:")
        # logger.info(data.tree())
    def clear_data(self):
        """清空当前存储的数据"""
        self.data = {key: [] for key in self.data_types}
        self.episode_ends = []
        self.total_steps = 0
        self.current_episode = 0

    def check_data_lengths(self):
        """检查各类数据长度是否一致"""
        lengths = {}
        for key in self.data.keys():
            if isinstance(self.data[key], list):
                lengths[key] = len(self.data[key])
        return lengths
    
if __name__ == "__main__":
    # 示例：记录包含深度图的数据
    recorder = DataRecorder(
        save_dir="./DP/dataset/demo_dataset_zarr_test",
        data_types=[
            'timestamp', 
            'action', 
            'camera_img',
            'depth_img',  # 添加深度图数据
            'tcp_poses', 
            'gripper_width',
            # 'tactile_img',  # 可选的触觉数据
            # 'mask'          # 可选的mask数据
        ],
        camera_img_shape=(140, 140, 3),
        depth_img_shape=(140, 140, 1),  # 设置深度图形状
        tactile_img_shape=(128, 128, 3),
        mask_shape=(140, 140, 4)
    )
    
    # 模拟数据收集
    for episode in range(2):
        for step in range(100):
            # 准备基本数据
            data = {
                'timestamp': np.array([time.time()]),
                'action': np.random.rand(4),
                'camera_img': np.random.randint(0, 255, (140, 140, 3), dtype=np.uint8),
                'depth_img': np.random.rand(140, 140, 1).astype(np.float32),  # 添加深度图数据
                'tcp_pose': np.random.rand(6),
                'gripper_width': np.random.rand(1)
            }
            
            # 可选：添加触觉数据（如果有的话）
            if 'tactile_img' in recorder.data_types:
                data['tactile_img'] = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                
            # 可选：添加mask数据（如果有的话）
            if 'mask' in recorder.data_types:
                data['mask'] = np.random.randint(0, 255, (140, 140, 4), dtype=np.uint8)
            
            # 记录数据
            recorder.add_step_data(**data)
            
        recorder.end_episode()
    
    recorder.save_to_zarr()