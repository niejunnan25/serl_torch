# Data collection details
+ Step1: 数据收集主程序: python speed_control.py
```
按下‘RG’键开始控制机械臂移动，左右推动遥杆控制机械夹爪的闭合
按下‘trigger’键记录数据，按下‘A’键机械臂复位
按下‘B'退出程序
```

+ Step2: 数据拼接： python data_process/data_process.py

+ Step3: 数据mask处理： python VLM/data_mask_process.py

## other files
```
数据格式检查： data_check.py
数据图片查看: data_show.py
数据复现: action_test.py
```
