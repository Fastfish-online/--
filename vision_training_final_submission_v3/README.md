# 第二次培训：OpenCV C++ 图像处理基础——三任务完整工程

本工程将培训要求的任务 1、任务 2、任务 3 统一到同一个 CMake 工程中，保留各任务独立源码、输入资源和结果，并提供一次性构建/运行脚本。

## 1. 工程结构

```text
.
├── CMakeLists.txt                 # 根 CMake，统一生成三个可执行程序
├── README.md
├── run_all.sh                     # Linux/macOS 一键构建并运行
├── run_all.ps1                    # Windows PowerShell 一键构建并运行
├── docs/
│   └── 第二次培训.pdf
├── resources/
│   ├── test_image.jpg             # 任务1
│   ├── task_2.mp4                 # 任务2
│   ├── task_3.mp4                 # 任务3：小能量机关
│   └── task_4.mp4                 # 任务3：大能量机关
├── config/                      # 三个任务的参数记录
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   └── README.md
├── src/
│   ├── task1_image/main.cpp
│   ├── task2_fit/main.cpp
│   └── task3_tracking/task3_tracker.cpp
└── result/
    ├── task1_images/              # 16 张图片
    ├── task2_fit/                 # 拟合图、跟踪视频、CSV
    ├── task2_fit_result.md
    ├── task3_windmill/            # task_3/task_4 跟踪结果
    └── task4_tracking/            # task_4.mp4 跟踪结果
```

## 2. 配置目录

`config/` 按培训讲义要求保留，用于记录本次提交对应的关键处理参数。源码使用固定编译参数，因此无需额外 YAML 依赖；配置文件作为结果复现记录，与源码中的实际参数保持一致。

## 2. 三个任务对应的程序

| 培训任务 | 可执行程序 | 主要内容 |
|---|---|---|
| 任务1 | `task1_image` | 灰度/HSV、均值/高斯/中值滤波、红色分割、形态学、轮廓、绘制、旋转、裁剪 |
| 任务2 | `task2_fit` | 青色目标识别、角度展开、角速度模型参数拟合、误差分析、标注视频 |
| 任务3 | `task3_tracker` | 小/大能量机关识别、中心跟踪、目标 ID 保持、lost/reselected 状态 |

任务3的一个程序分别运行 `task_3.mp4` 和 `task_4.mp4`，产生两个完整结果目录。

## 3. 任务1完成情况

- 灰度图：`gray.png`
- HSV H/S/V 单通道：`hsv_h.png`、`hsv_s.png`、`hsv_v.png`
- 三种滤波：`mean_filter.png`、`gaussian_filter.png`、`median_filter.png`
- HSV 双区间红色掩膜：`red_mask.png`
- 腐蚀/膨胀/开/闭：`erode.png`、`dilate.png`、`open.png`、`close.png`
- 轮廓及外接框：`contours_boxes.png`
- 圆/矩形/文字：`drawing.png`
- 35°旋转：`rotated_35deg.png`
- 左上角 1/4：`crop_top_left.png`

采用讲义中的主要参数：5×5 滤波核、Gaussian sigma=1.5、红色 H 双区间 [0,10] 与 [170,179]、S/V [100,255]、5×5 形态学核、轮廓面积阈值 500 px²、外接框长宽比 0.2~5.0。

## 4. 任务2完成情况

输入视频实际读取参数：682×512、30 FPS、720 帧、24 s。程序根据实际视频尺寸计算画面中心。

拟合模型：

`ω(t) = b + A sin(Ωt + φ)`

角度积分模型：

`θ(t)=θ0+b*t+(A/Ω)*(cos(φ)-cos(Ωt+φ))`

本次已得到：

| 参数 | 数值 | 单位 |
|---|---:|---|
| θ0 | 0.350113 | rad |
| b | 1.350008 | rad/s |
| A | 0.550110 | rad/s |
| Ω | 1.650018 | rad/s |
| φ | 0.701248 | rad |
| 角度 RMSE | 0.002427 | rad |
| 角速度诊断 RMSE | 0.022480 | rad/s |
| 速度变化周期 | 3.807949 | s |

详细参数和方法见 `result/task2_fit_result.md`。

## 5. 任务3完成情况

两个真实录像分别完成识别与稳定跟踪：

- `result/task3_windmill/task_3/recognition_overlay.mp4`：`task_3.mp4`，小能量机关
- `result/task3_windmill/task_4/recognition_overlay.mp4`：`task_4.mp4`，大能量机关

跟踪输出包括完整标注视频和逐帧 CSV；规则与结果分析见 `result/task3_tracking_result.md`。算法采用 HSV 橙红色提取、形态学去噪、Hough 圆候选、三点圆模型拟合以及基于上一帧位置/速度的关联；短时漏检保持身份并标记 `lost`，持续丢失超过 60 帧后才允许重新选择并递增 ID。

## 6. 一次性构建与运行

### Linux / macOS

```bash
./run_all.sh
```

或手动：

```bash
cmake -S . -B build
cmake --build build -j4
./build/task1_image resources/test_image.jpg result/task1_images/
./build/task2_fit resources/task_2.mp4 result/task2_fit
./build/task3_tracker resources/task_3.mp4 result/task3_windmill/task_3/recognition_overlay.mp4 result/task3_windmill/task_3/tracking.csv
./build/task3_tracker resources/task_4.mp4 result/task3_windmill/task_4/recognition_overlay.mp4 result/task3_windmill/task_4/tracking.csv
```

### Windows PowerShell

```powershell
.\run_all.ps1
```

依赖：C++17、CMake 3.16+、OpenCV、Eigen3（任务2）。

## 7. 已生成的结果

本提交包已经包含三项任务的现成处理结果，因此无需先重新运行程序即可查看 `result/` 中的图片、CSV、Markdown 和 MP4。重新运行脚本会覆盖相应结果。

> 注：任务3属于真实录像跟踪任务，培训要求强调不使用播放时间进行运动参数拟合，因此本工程没有对任务3做任务2式参数拟合。

## 统一头文件目录
`include/` 按任务与公共功能组织：
- `include/common/vision_common.hpp`：统一目录创建、通用数值工具。
- `include/task1_image/task1_config.hpp`：任务1参数。
- `include/task2_fit/task2_model.hpp`：任务2模型与输入几何参数。
- `include/task3_windmill/task3_config.hpp`：任务3检测、Hough 与丢失容忍参数。
