import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, 
                 enable_depth=True, enable_color=True, 
                 serial_number=None,
                 resize_width=None, resize_height=None):
        """初始化RealSense相机
        Args:
            width: 原始图像宽度
            height: 原始图像高度
            fps: 帧率
            enable_depth: 是否启用深度图
            enable_color: 是否启用彩色图
            serial_number: 相机序列号
            resize_width: 调整后的宽度(可选)
            resize_height: 调整后的高度(可选)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.serial_number = serial_number
        self.resize_width = resize_width
        self.resize_height = resize_height
        
        # 状态标志
        self.is_running = False
        self.frame_count = 0
        
        # 日志设置
        self.logger = logging.getLogger(f"RealSense_{serial_number}")
        self.logger.setLevel(logging.INFO)
        
        # 初始化RealSense管线和配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 如果指定了序列号，设置对应设备
        if serial_number:
            self.config.enable_device(serial_number)
            
        # 配置数据流
        if enable_color:
            self.config.enable_stream(
                rs.stream.color,
                width, height,
                rs.format.bgr8,
                fps
            )
        
        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                width, height,
                rs.format.z16,
                fps
            )
            
        # 创建对齐对象
        self.align = rs.align(rs.stream.color)

    def start(self):
        """启动相机"""
        try:
            # 检查相机是否已连接
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self.logger.error("未检测到RealSense相机")
                return False
                
            if self.serial_number:
                device_found = False
                for dev in devices:
                    if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                        device_found = True
                        break
                if not device_found:
                    self.logger.error(f"未找到序列号为 {self.serial_number} 的相机")
                    return False
            
            # 启动管线
            self.profile = self.pipeline.start(self.config)
            # device = self.profile.get_device()
            # device.hardware_reset()
            self.is_running = True
            
            # 等待相机稳定
            time.sleep(1.0)
            
            # 获取相机信息
            device = self.profile.get_device()
            self.logger.info(f"相机已启动: {device.get_info(rs.camera_info.name)}")
            self.logger.info(f"序列号: {device.get_info(rs.camera_info.serial_number)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动相机失败: {e}")
            return False

    def get_frames(self):
        """获取彩色图和深度图
        Returns:
            tuple: (color_frame, depth_frame) 或在错误时返回 (None, None)
        """
        if not self.is_running:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_image = None
            depth_image = None
            
            if self.enable_color:
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    if self.resize_width and self.resize_height:
                        color_image = cv2.resize(color_image, 
                                               (self.resize_width, self.resize_height))
            
            if self.enable_depth:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    if self.resize_width and self.resize_height:
                        depth_image = cv2.resize(depth_image, 
                                               (self.resize_width, self.resize_height))
                    
            self.frame_count += 1
            return color_image, depth_image
            
        except Exception as e:
            self.logger.error(f"获取帧错误: {e}")
            return None, None

    def stop(self):
        """停止相机"""
        if self.is_running:
            try:
                self.pipeline.stop()
                self.is_running = False
                self.logger.info(f"相机已停止，共捕获 {self.frame_count} 帧")
            except Exception as e:
                self.logger.error(f"停止相机错误: {e}")

    def __del__(self):
        """析构函数"""
        self.stop()

def test_camera(serial_number=None):
    """测试相机功能"""
    camera = RealSenseCamera(
        width=640,
        height=480,
        fps=30,
        enable_depth=True,
        enable_color=True,
        serial_number=serial_number,
        # resize_width=140,    # 设置调整后的尺寸
        # resize_height=140
    )
    
    if not camera.start():
        print("相机启动失败")
        return
    
    try:
        while True:
            color_img, depth_img = camera.get_frames()
            
            if color_img is not None:
                cv2.imshow('Color', color_img)
                
            if depth_img is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth', depth_colormap)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 可以通过传入序列号来测试特定相机
    test_camera('211622069098')