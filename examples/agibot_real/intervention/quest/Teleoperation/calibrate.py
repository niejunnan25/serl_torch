import numpy as np
import time
import threading
from oculus_reader import OculusReader
import logging
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import signal
import sys

class VRTeleoperationController:
    def __init__(self, scaling_factor=0.8, motion_mode='xyz', control_freq=50, enable_visualization=True, controller_mode='right'):
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("VRTeleoperation")
        
        # 初始化VR控制器
        self.vr = OculusReader()
        self.scaling_factor = scaling_factor
        self.rotation_scaling = scaling_factor
        
        # 手柄控制模式: 'left', 'right', 'both'
        self.controller_mode = controller_mode
        self._validate_controller_mode()
        
        # 状态变量
        self.is_running = False
        
        # 左右手柄状态（统一字典结构）
        self.controller_state = {
            'right': {
                'current_position': np.zeros(3),
                'current_rotation': np.eye(3),
                'is_button_pressed': False,
                'last_vr_position': None,
                'last_vr_rotation': None,
                'relative_motion_enabled': False,
                'initial_vr_position': None,
                'initial_vr_rotation': None,
                'initial_position': np.zeros(3),
                'initial_rotation': np.eye(3),
                'last_end_position': np.zeros(3),
                'last_end_rotation': np.eye(3)
            },
            'left': {
                'current_position': np.zeros(3),
                'current_rotation': np.eye(3),
                'is_button_pressed': False,
                'last_vr_position': None,
                'last_vr_rotation': None,
                'relative_motion_enabled': False,
                'initial_vr_position': None,
                'initial_vr_rotation': None,
                'initial_position': np.zeros(3),
                'initial_rotation': np.eye(3),
                'last_end_position': np.zeros(3),
                'last_end_rotation': np.eye(3)
            }
        }

        # 添加运动限制
        self.max_linear_speed = 5.0  # 最大线速度 (m/s)
        self.max_angular_speed = 5.0  # 最大角速度 (rad/s)
        
        # 运动模式和频率控制
        self.motion_mode = motion_mode
        self._validate_motion_mode()
        self.control_freq = control_freq
        self.control_period = 1.0 / control_freq
        
        # 修改可视化相关的初始化
        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            # 不在初始化时创建可视化窗口，而是在主线程中创建
            self.fig = None
            self.ax = None
            self.point = None  # 右手柄点
            self.left_point = None  # 左手柄点
            self.local_axes = {}  # 右手柄坐标轴
            self.left_local_axes = {}  # 左手柄坐标轴
        
        # 增量控制信号
        self.delta_control_signals = {
            'position_delta': np.zeros(3),
            'rotation_delta': np.zeros(3),
            'button_states': {},
            'joystick': (0.0, 0.0),
            'trigger': 0.0,
            'grip': 0.0
        }
        
        # 位置更新锁
        self.position_lock = threading.Lock()
        
        # 添加可视化数据队列
        self.vis_data = {
            'position': np.zeros(3),
            'rotation': np.eye(3),
            'left_position': np.zeros(3),
            'left_rotation': np.eye(3),
            'need_update': False
        }
        self.vis_lock = threading.Lock()

        # 添加坐标变换矩阵
        self.coord_transform_matrix = np.eye(3)  # 默认为单位矩阵
        self.set_coordinate_mapping('default')    # 默认映射

        self.last_set_calibrate_rot_time = 0.0
        #Transform from handle coordinate  to  robot's coordinate, to facilitate human-machine interaction
        self.init_transfer_right_hand = np.array([[-0.15647381, -0.98612506, -0.05543753],
       [-0.69193056,  0.06939324,  0.71862138],
       [-0.70480356,  0.15080434, -0.69318828]])
        self.init_transfer_left_hand = np.array([[ 0.31834174, -0.9471677 ,  0.03913927],
       [-0.71898245, -0.21432842,  0.66115623],
       [-0.61783717, -0.23861408, -0.74922664]])
        
     

        



    def set_coordinate_mapping(self, mapping_type='default'):
        """设置坐标轴映射关系"""
        mapping_dict = {
            'default': np.eye(3),
            'ur5_sim': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            'sim': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        }
        
        if mapping_type in mapping_dict:
            self.coord_transform_matrix = mapping_dict[mapping_type]
        else:
            self.logger.warning(f"未知映射类型: {mapping_type}, 使用默认")
            self.coord_transform_matrix = np.eye(3)
            

    def transform_coordinates(self, position_delta):
        """转换坐标系
        Args:
            position_delta: 原始位置增量 (3,)
        Returns:
            转换后的位置增量 (3,)
        """
        return self.coord_transform_matrix @ position_delta

    def setup_visualization(self):
        """设置matplotlib可视化环境"""
        if not self.enable_visualization:
            return
            
        # 在主线程中创建和设置可视化
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([-0.5, 0.5])
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 根据控制模式创建控制点
        if self.controller_mode in ['right', 'both']:
            # 创建右手柄控制点(红色球体)
            self.point = self.ax.scatter([0], [0], [0], color='red', s=100, label='Right')
        else:
            self.point = None
        
        if self.controller_mode in ['left', 'both']:
            # 创建左手柄控制点(蓝色球体)
            self.left_point = self.ax.scatter([0], [0], [0], color='blue', s=100, label='Left')
        else:
            self.left_point = None
        
        # 根据控制模式创建局部坐标轴
        axis_length = 0.1  # 坐标轴长度
        if self.controller_mode in ['right', 'both']:
            # 创建右手柄局部坐标轴
            self.local_axes = {
                'x': self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', alpha=0.5),
                'y': self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', alpha=0.5),
                'z': self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', alpha=0.5)
            }
        else:
            self.local_axes = {}
        
        if self.controller_mode in ['left', 'both']:
            # 创建左手柄局部坐标轴
            self.left_local_axes = {
                'x': self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', alpha=0.3),
                'y': self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', alpha=0.3),
                'z': self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', alpha=0.3)
            }
        else:
            self.left_local_axes = {}
        
        # 添加全局参考坐标轴
        self.ax.plot([0, 0.2], [0, 0], [0, 0], 'r-', alpha=0.3)
        self.ax.plot([0, 0], [0, 0.2], [0, 0], 'g-', alpha=0.3)
        self.ax.plot([0, 0], [0, 0], [0, 0.2], 'b-', alpha=0.3)
        
        # 添加图例
        self.ax.legend(loc='upper right')

    def update_visualization(self):
        """更新可视化数据（不直接更新图形）"""
        if not self.enable_visualization:
            return
            
        with self.position_lock:
            with self.vis_lock:
                self.vis_data['position'] = self.controller_state['right']['current_position'].copy()
                self.vis_data['rotation'] = self.controller_state['right']['current_rotation'].copy()
                self.vis_data['left_position'] = self.controller_state['left']['current_position'].copy()
                self.vis_data['left_rotation'] = self.controller_state['left']['current_rotation'].copy()
                self.vis_data['need_update'] = True

    def render_visualization(self):
        """在主线程中渲染可视化"""
        if not self.enable_visualization or self.fig is None:
            return
            
        with self.vis_lock:
            if not self.vis_data['need_update']:
                return
                
            try:
                axis_length = 0.1
                
                # 更新手柄的辅助函数
                def update_controller(point, axes_dict, position_key, rotation_key, alpha):
                    if point is None:
                        return
                    # 更新点位置
                    pos = self.vis_data[position_key]
                    point._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                    
                    # 更新局部坐标轴
                    for i, axis in enumerate(['x', 'y', 'z']):
                        direction = self.vis_data[rotation_key][:, i] * axis_length
                        try:
                            if axis in axes_dict:
                                axes_dict[axis].remove()
                        except:
                            pass
                        axes_dict[axis] = self.ax.quiver(
                            pos[0], pos[1], pos[2],
                            direction[0], direction[1], direction[2],
                            color=['r', 'g', 'b'][i], alpha=alpha
                        )
                
                # 根据模式更新手柄
                if self.controller_mode in ['right', 'both']:
                    update_controller(self.point, self.local_axes, 'position', 'rotation', 0.5)
                if self.controller_mode in ['left', 'both']:
                    update_controller(self.left_point, self.left_local_axes, 'left_position', 'left_rotation', 0.3)
                
                # 根据模式更新标题
                if self.controller_mode == 'right':
                    title = f'Right: [{self.vis_data["position"][0]:.3f}, {self.vis_data["position"][1]:.3f}, {self.vis_data["position"][2]:.3f}]'
                elif self.controller_mode == 'left':
                    title = f'Left: [{self.vis_data["left_position"][0]:.3f}, {self.vis_data["left_position"][1]:.3f}, {self.vis_data["left_position"][2]:.3f}]'
                else:  # both
                    title = (f'Right: [{self.vis_data["position"][0]:.3f}, {self.vis_data["position"][1]:.3f}, {self.vis_data["position"][2]:.3f}] | '
                            f'Left: [{self.vis_data["left_position"][0]:.3f}, {self.vis_data["left_position"][1]:.3f}, {self.vis_data["left_position"][2]:.3f}]')
                self.ax.set_title(title)
                
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                
                self.vis_data['need_update'] = False
                
            except Exception as e:
                self.logger.warning(f"可视化渲染失败: {e}")

    def _validate_motion_mode(self):
        """验证运动模式的有效性"""
        valid_modes = ['xyz', 'xyzrz', 'xyzrxryrz']
        if self.motion_mode not in valid_modes:
            raise ValueError(f"无效的运动模式: {self.motion_mode}. 有效模式: {valid_modes}")
    
    def _validate_controller_mode(self):
        """验证手柄控制模式的有效性"""
        valid_modes = ['left', 'right', 'both']
        if self.controller_mode not in valid_modes:
            raise ValueError(f"无效的手柄模式: {self.controller_mode}. 有效模式: {valid_modes}")
        self.logger.info(f"手柄控制模式: {self.controller_mode}")
            
    def _filter_motion(self, pos_diff, rot_diff):
        """根据运动模式过滤运动
        Args:
            pos_diff: 位置差异向量 (3,)
            rot_diff: 旋转差异向量 (3,)
        Returns:
            tuple: (过滤后的位置差异, 过滤后的旋转差异)
        """
        # 位置控制
        if self.motion_mode == 'xyz':
            return pos_diff, np.zeros(3)
            
        # 位置和Z轴旋转
        elif self.motion_mode == 'xyzrz':
            filtered_rot = np.array([0, 0, rot_diff[2]])
            return pos_diff, filtered_rot
            
        # 完整6自由度控制
        else:  # 'xyzrxryrz'
            return pos_diff, rot_diff
            
    def _process_vr_data(self, transforms, buttons):
        """处理VR数据
        Args:
            transforms: 包含左右手柄变换矩阵的字典
            buttons: 按钮状态字典
        Returns:
            tuple: (right_position, right_rot_vec, right_buttons, left_position, left_rot_vec, left_buttons)
        """
        # 默认值设置
        default_position = np.zeros(3)
        default_rot_vec = np.zeros(3)
        default_right_buttons = {
            'A': False,        
            'B': False,        
            'RThU': False,     
            'RJ': False,      
            'RG': False,      
            'RTr': False,    
            'rightJS': (0.0, 0.0),    
            'rightTrig': (0.0,),    
            'rightGrip': (0.0,)     
        }
        
        default_left_buttons = {
            'X': False,
            'Y': False,
            'LThU': False,
            'LJ': False,
            'LG': False,
            'LTr': False,
            'leftJS': (0.0, 0.0),
            'leftTrig': (0.0,),
            'leftGrip': (0.0,)
        }

        try:
            # 处理右手柄
            if not transforms or 'r' not in transforms:
                self.logger.warning("未检测到右手柄数据，使用默认值")
                right_position, right_rot_vec, right_buttons = default_position, default_rot_vec, default_right_buttons
            else:
               
                right_transform = transforms['r']
                try:
                    right_position = right_transform[:3, 3] 
                    rotation_matrix = right_transform[:3, :3]
                    rotation_matrix = rotation_matrix @ self.init_transfer_right_hand
                    rotation = Rotation.from_matrix(rotation_matrix)
                    right_rot_vec = rotation.as_rotvec()
                except Exception as e:
                    self.logger.warning(f"右手柄位姿数据处理错误，使用默认值: {e}")
                    right_position, right_rot_vec = default_position, default_rot_vec

                # 获取右手柄所有按钮状态
                right_buttons = {
                    'A': buttons.get('A', False),
                    'B': buttons.get('B', False),
                    'RThU': buttons.get('RThU', False),
                    'RJ': buttons.get('RJ', False),
                    'RG': buttons.get('RG', False),
                    'RTr': buttons.get('RTr', False),
                    'rightJS': buttons.get('rightJS', (0.0, 0.0)),
                    'rightTrig': buttons.get('rightTrig', (0.0,)),
                    'rightGrip': buttons.get('rightGrip', (0.0,))
                }
                if right_buttons['A']:
                     cur_time = time.time()
                     if cur_time - self.last_set_calibrate_rot_time > 0.5:
                         self.last_set_calibrate_rot_time = cur_time

                         matrix = np.array(rotation_matrix @ (self.init_transfer_right_hand.T))
                         print("rotation_matrix = np.array(")
                         print("[")
                         for row in matrix:
                            print(f"    [{', '.join(map(str, row))}],")
                         print("])")
                         
                        
            
            # 处理左手柄
            if not transforms or 'l' not in transforms:
                self.logger.warning("未检测到左手柄数据，使用默认值")
                left_position, left_rot_vec, left_buttons = default_position, default_rot_vec, default_left_buttons
            else:
                left_transform = transforms['l']
                try:
                    left_position = left_transform[:3, 3]
                    rotation_matrix = left_transform[:3, :3] @ self.init_transfer_left_hand
                    rotation = Rotation.from_matrix(rotation_matrix)
                    left_rot_vec = rotation.as_rotvec()
                except Exception as e:
                    self.logger.warning(f"左手柄位姿数据处理错误，使用默认值: {e}")
                    left_position, left_rot_vec = default_position, default_rot_vec

                # 获取左手柄所有按钮状态
                left_buttons = {
                    'X': buttons.get('X', False),
                    'Y': buttons.get('Y', False),
                    'LThU': buttons.get('LThU', False),
                    'LJ': buttons.get('LJ', False),
                    'LG': buttons.get('LG', False),
                    'LTr': buttons.get('LTr', False),
                    'leftJS': buttons.get('leftJS', (0.0, 0.0)),
                    'leftTrig': buttons.get('leftTrig', (0.0,)),
                    'leftGrip': buttons.get('leftGrip', (0.0,))
                }

                if left_buttons['X']:
                     cur_time = time.time()
                     if cur_time - self.last_set_calibrate_rot_time > 0.5:
                         self.last_set_calibrate_rot_time = cur_time

                         matrix = np.array(rotation_matrix @ (self.init_transfer_left_hand.T))
                         print("rotation_matrix = np.array(")
                         print("[")
                         for row in matrix:
                            print(f"    [{', '.join(map(str, row))}],")
                         print("])")

            return right_position, right_rot_vec, right_buttons, left_position, left_rot_vec, left_buttons

        except Exception as e:
            self.logger.warning(f"VR数据处理错误,使用默认值: {e}")
            return default_position, default_rot_vec, default_right_buttons, default_position, default_rot_vec, default_left_buttons
        
    def _calculate_relative_motion(self, current_vr_pose, hand='right'):
        """计算相对于初始按下时刻的运动
        Args:
            current_vr_pose: 当前VR控制器位姿 (位置和旋转向量)
            hand: 'right' 或 'left'
        Returns:
            tuple: (相对位移, 相对旋转矢量)
        """
        state = self.controller_state[hand]
        
        if not state['relative_motion_enabled'] or state['last_vr_position'] is None:
            return np.zeros(3), np.zeros(3)
            
        # 提取当前位置和旋转
        current_position, current_rotvec = current_vr_pose
        
        # 计算相对于初始位置的位移
        position_diff = (current_position - state['initial_vr_position']) * self.scaling_factor
        
        # 应用坐标变换
        position_diff = self.transform_coordinates(position_diff)
        
        # 计算相对于初始旋转的旋转差异
        initial_rot = Rotation.from_rotvec(state['initial_vr_rotation'])
        curr_rot = Rotation.from_rotvec(current_rotvec)
        
        rot_diff = (initial_rot.inv()  * curr_rot).as_rotvec() * self.rotation_scaling

        
        # 根据运动模式过滤运动
        position_diff, rot_diff = self._filter_motion(position_diff, rot_diff)
        
        # 限制运动速度
        position_diff = self._limit_velocity(position_diff, self.max_linear_speed)
        rot_diff = self._limit_velocity(rot_diff, self.max_angular_speed)
        
        # 更新上一次位置和旋转（用于状态跟踪）
        state['last_vr_position'] = current_position
        state['last_vr_rotation'] = current_rotvec
        
        return position_diff, rot_diff
    
    def _process_controller_input(self, hand, position, rotvec, button_states):
        """处理单个手柄的输入
        Args:
            hand: 'right' 或 'left'
            position: 手柄位置
            rotvec: 手柄旋转向量
            button_states: 按钮状态
        Returns:
            tuple: (pos_diff, rot_diff, grip_value)
        """
        state = self.controller_state[hand]
        grip_key = 'RG' if hand == 'right' else 'LG'
        joystick_key = 'rightJS' if hand == 'right' else 'leftJS'
        
        pos_diff = np.zeros(3)
        rot_diff = np.zeros(3)
        grip_value = 0.0

        
                 

        
        # 检测按钮按下
        if button_states[grip_key] and not state['is_button_pressed']:
            state['is_button_pressed'] = True
            state['relative_motion_enabled'] = True
            state['initial_vr_position'] = position.copy()
            state['initial_vr_rotation'] = rotvec.copy()

            
            
            state['initial_position'] = state['last_end_position'].copy()

            state['initial_rotation'] = state['last_end_rotation'].copy()

         
            state['last_vr_position'] = position.copy()
            state['last_vr_rotation'] = rotvec.copy()
            
        elif not button_states[grip_key] and state['is_button_pressed']:
            state['is_button_pressed'] = False
            state['relative_motion_enabled'] = False
            with self.position_lock:
                state['last_end_position'] = state['current_position'].copy()
                state['last_end_rotation'] = state['current_rotation'].copy()
        




        # 计算运动
        if state['relative_motion_enabled']:
            pos_diff, rot_diff = self._calculate_relative_motion((position, rotvec), hand)
            print("rot_diff:",rot_diff)

            
            with self.position_lock:
                state['current_position'] = state['initial_position'] + pos_diff
                rot_matrix = Rotation.from_rotvec(rot_diff).as_matrix()
                state['current_rotation'] =  rot_matrix@ state['initial_rotation'] 
            
            joystick_x = button_states.get(joystick_key, (0.0, 0.0))[0]
            grip_value = np.clip(joystick_x, -1.0, 1.0)
        
        return pos_diff, rot_diff, grip_value
    
    def _limit_velocity(self, vector, max_speed):
        """限制向量的速度大小
        Args:
            vector: 速度向量
        Returns:
            限制后的速度向量
        """
        magnitude = np.linalg.norm(vector)
        if magnitude > max_speed:
            return vector * (max_speed / magnitude)
        return vector
        
    def _control_loop(self):
        """控制循环"""
        last_control_time = time.time()
        
        while self.is_running:
            try:
                # 控制频率
                current_time = time.time()
                elapsed_time = current_time - last_control_time
                if elapsed_time < self.control_period:
                    time.sleep(self.control_period - elapsed_time)
                    continue
                last_control_time = current_time
                
                # 获取VR数据
                transforms, buttons = self.vr.get_transformations_and_buttons()
                right_position, right_rotvec, right_button_states, left_position, left_rotvec, left_button_states = self._process_vr_data(transforms, buttons)
                
                # 检查B键退出（仅在右手柄模式或both模式下）
                if self.controller_mode in ['right', 'both'] and right_button_states.get('B', False):
                    self.logger.info("检测到B键按下，程序即将退出...")
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                
                # 检查Y键退出（仅在左手柄模式或both模式下）
                if self.controller_mode in ['left', 'both'] and left_button_states.get('Y', False):
                    self.logger.info("检测到Y键按下，程序即将退出...")
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                
                # 处理右手柄控制逻辑
                pos_diff, rot_diff, grip_value = np.zeros(3), np.zeros(3), 0.0
                if self.controller_mode in ['right', 'both']:
                    pos_diff, rot_diff, grip_value = self._process_controller_input('right', right_position, right_rotvec, right_button_states)

       

                # 处理左手柄控制逻辑
                left_pos_diff, left_rot_diff, left_grip_value = np.zeros(3), np.zeros(3), 0.0
                if self.controller_mode in ['left', 'both']:
                    left_pos_diff, left_rot_diff, left_grip_value = self._process_controller_input('left', left_position, left_rotvec, left_button_states)
                with self.position_lock:
                    self.delta_control_signals.update({
                        'position_delta': pos_diff.copy(),
                        'rotation_delta': rot_diff.copy(),
                        'button_states': right_button_states,
                        'joystick': right_button_states.get('rightJS', (0.0, 0.0)),
                        'trigger': right_button_states.get('rightTrig', (0.0,))[0],
                        'grip': grip_value,
                        # 添加左手柄数据
                        'left_position_delta': left_pos_diff.copy(),
                        'left_rotation_delta': left_rot_diff.copy(),
                        'left_button_states': left_button_states,
                        'left_joystick': left_button_states.get('leftJS', (0.0, 0.0)),
                        'left_trigger': left_button_states.get('leftTrig', (0.0,))[0],
                        'left_grip': left_grip_value
                    })
                    
                    # print(f"右手柄 - 位置增量: {pos_diff}, 旋转增量: {rot_diff}, 夹爪: {grip_value}")
                    # print(f"左手柄 - 位置增量: {left_pos_diff}, 旋转增量: {left_rot_diff}, 夹爪: {left_grip_value}")
                
                # 更新可视化数据（不直接渲染）
                if self.enable_visualization:
                    self.update_visualization()
                    
            except Exception as e:
                self.logger.error(f"控制错误: {e}")
                time.sleep(0.3)

    def get_control_signals(self):
        """获取当前控制增量信号
        Returns:
            dict: 包含左右手柄位置增量、旋转增量和按钮状态的字典
        """
        with self.position_lock:
            return {
                # 右手柄数据
                'position_delta': self.delta_control_signals['position_delta'].copy(),
                'rotation_delta': self.delta_control_signals['rotation_delta'].copy(),
                'button_states': self.delta_control_signals['button_states'].copy(),
                'joystick': self.delta_control_signals['joystick'],
                'trigger': self.delta_control_signals['trigger'],
                'grip': self.delta_control_signals['grip'],
                # 左手柄数据
                'left_position_delta': self.delta_control_signals.get('left_position_delta', np.zeros(3)).copy(),
                'left_rotation_delta': self.delta_control_signals.get('left_rotation_delta', np.zeros(3)).copy(),
                'left_button_states': self.delta_control_signals.get('left_button_states', {}).copy(),
                'left_joystick': self.delta_control_signals.get('left_joystick', (0.0, 0.0)),
                'left_trigger': self.delta_control_signals.get('left_trigger', 0.0),
                'left_grip': self.delta_control_signals.get('left_grip', 0.0)
            }
    
    def start(self):
        """启动遥操作控制"""
        try:
            self.is_running = True
            # self.robot.connect()
            # self.vr.start()
            
            # 启动控制线程
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.start()
            
            self.logger.info("遥操作控制已启动")
            
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            self.stop()
            
    def stop(self):
        """停止控制"""
        self.is_running = False
        if hasattr(self, 'control_thread'):
            self.control_thread.join()
        if self.enable_visualization and self.fig is not None:
            plt.close(self.fig)
        self.logger.info("控制已停止")

def main():
    def signal_handler(sig, frame):
        print("\n检测到中断信号,正在停止...")
        controller.stop()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建控制器实例
    # controller_mode 可选值: 'left' (仅左手柄), 'right' (仅右手柄), 'both' (双手柄)
    controller = VRTeleoperationController(
        scaling_factor=1.0,
        motion_mode='xyzrxryrz',
        control_freq=30,
        enable_visualization=True,
        controller_mode='both'  # 修改此参数来选择手柄模式
    )
    # 修改映射关系示例
    controller.set_coordinate_mapping('sim')  # 手柄x轴映射到目标y轴，以此类推
    try:
        # 在主线程中设置可视化
        if controller.enable_visualization:
            # 确保在主线程中初始化matplotlib
            plt.ion()
            controller.setup_visualization()
        
        # 启动控制器
        controller.start()
        print("VR控制器已启动，按Ctrl+C退出...")
        
        # 主循环（在主线程中处理可视化）
        while controller.is_running:
            if controller.enable_visualization:
                try:
                    controller.render_visualization()
                except Exception as e:
                    print(f"可视化渲染错误: {e}")
            plt.pause(0.01)  # 在主线程中使用 plt.pause
                    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        controller.stop()
        if controller.enable_visualization:
            plt.close('all')

if __name__ == "__main__":
    main()