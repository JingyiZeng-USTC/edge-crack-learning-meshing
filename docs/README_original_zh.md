# 最终版全流程代码

本目录包含学习型绝对尺寸场、加速Line Graph-GraphSAGE连接、结构化解码、20°质量调整、裂纹面自由度分离、顶边均布线载荷下的逐网格FCNN-SM计算，以及有限元应变/应力恢复。

已打包最终尺寸场模型、结构化GNN模型、GNN初始模型和原200张直裂纹参考网格。训练/验证/测试仍为120/40/40。无需访问原D:\itl\eg\3项目的代码或模型。Python环境不复制，继续使用现有conda环境。

## 1. 一键完整计算

在本目录打开终端，激活原conda环境后运行：

```bat
run_full_pipeline.bat --case-name straight_demo
```

默认几何：左侧入口y=1.07 m，长度0.32 m，角度11°；节点采样种子74032。网格k=20，最小角目标20°；顶边线载荷5e5 N/m，E=2.1e11 Pa，泊松比0.3，厚度0.01 m，平面应力。FCNN使用2560个无标签载荷幅值样本、2000轮优化、16个低刚度模态，双精度，采用Ritz势能、残差和合力混合目标，无PCG或直接解后校正。

自定义直裂纹：

```bat
run_full_pipeline.bat --edge left --origin 1.0 --length 0.30 --angle-deg 0 --case-name my_straight
```

弯曲裂纹：

```bat
run_full_pipeline.bat --geometry-json examples\curved_left_mid_up.json --seed 76001 --case-name curved_demo
```

只生成和导出网格，不训练FCNN：

```bat
run_full_pipeline.bat --case-name mesh_demo --mesh-only
```

程序拒绝覆盖已存在的非空算例目录。再次运行请改case-name或output。新裂纹需保持至少0.20 m的顶边净距。

## 2. 原12个直裂纹和4个弯曲裂纹

```bat
run_full_pipeline.bat --suite
run_full_pipeline.bat --suite --mesh-only --output results_mesh_only
```

套件使用打包的最终模型、原几何和种子。新脚本调用加速网格入口，随后调用已有FCNN套件。全套可耗时数分钟以上，每张网格的FCNN均独立训练；不表示已经完成多步裂纹扩展。

## 3. 可选离线重新训练

当前模型已训练，可以直接推理。仅需要重新训练时执行：

```bat
python -B run_training.py --stage size --size-epochs 300
python -B run_training.py --stage gnn --gnn-epochs 80
```

首次执行自动从打包的原200张网格准备数据。为与已打包GNN的历史训练一致，默认训练候选k=16，在线推理k=20；可以显式指定--candidate-k 20做新训练，但那不再是原训练配置。生成线图可能占用数GB磁盘，请使用D盘。

重新训练的模型另存retrained_size_field.pt和retrained_strict_model.pt，不覆盖打包模型。单算例推理通过--size-model和--gnn-model显式选择新模型。训练数据不会先由尺寸场重新采样。

## 4. 速度对照

```bat
cd gnn
run_mesh_speed_benchmark.bat --suite --repeats 3 --output artifacts\new_speed_benchmark
```

用独立子进程对原版与加速版公平计时。原项目最终48组对照的记录附在evidence\mesh_speed_benchmark中：完整运行平均77.215 s降至16.920 s，最终网格完全一致。该数字是原环境测量，移动目录或更换硬件后需重测。阶段父项与子项的inclusive时间不能重复相加。

## 5. 输出

- results\算例名\meshes：几何网格NPZ。
- candidates：候选边、特征、线图和GNN概率。
- figures：网格图片。
- timings：各阶段计时JSON/CSV。
- fcnn_meshes：裂纹面节点分离后的力学网格NPZ/VTU。
- fcnn_results：训练记录、位移、应变、应力、模型及同网格FEM对照指标。

在线网格生成不调用CDT、Delaunay或Triangle作连接回退。包内个别兼容文件保留旧通用接口，但主入口不调用其中传统三角化/旧坐标FCNN路径。直接FEM参照只用于训练后评价，不作为FCNN位移或应力训练标签。

## 6. 环境与文件

入口优先使用激活的CONDA_PREFIX；也可设置PIPELINE_PYTHON。未激活时默认现有D:\itl\eg\pinn_gpu\python.exe。requirements.txt为最低依赖范围，environment_versions.json记录打包时实际版本。已有CUDA PyTorch可用时不要直接重装torch。

verify_bundle.py执行语法、依赖、模型与200样本划分检查，不训练网络。PACKAGE_MANIFEST.json记录源文件位置与SHA256，便于确认复制完整性。

整理时已通过79个Python文件的语法检查、实际依赖导入、模型结构和200个样本完整性检查。results\_smoke_check是已运行的短程全流程测试：最终网格、裂纹面节点分离、FCNN训练及应力恢复接口均跑通，但只训练2轮、使用16个幅值样本，其力学误差不作为论文结果。正式运行使用默认2000轮和2560个样本。

目录中的module43仅作为当前module44调用的结果保存/物理辅助兼容依赖；FCNN主入口是module44_load_surrogate_fcnn_sm.train_solve。
