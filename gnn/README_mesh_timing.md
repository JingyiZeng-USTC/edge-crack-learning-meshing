# 网格生成分阶段计时与等价加速

本入口不训练尺寸场或GNN，不改变检查阈值、k=20候选集、概率排序、裂纹约束或20°质量目标。原10号脚本及原实验结果不覆盖。加速仅在新入口的当前进程内启用。

## 快速生成一个网格

激活原conda环境，在D:\itl\eg\3\gnn中运行：

```bat
run_fast_mesh_generation.bat --edge left --origin 1.07 --length 0.32 --angle-deg 11 --seed 74032 --case-name fast_straight
```

网格和图保存在artifacts\mesh_fast；latest_timing.json及csv记录这一次的各阶段耗时（后续调用会更新latest文件）。弯曲裂纹使用原入口相同的--geometry-json参数。可加--no-figure关闭绘图，论文比较时必须保持两组绘图选项一致。

## 原版/加速版公平对照

```bat
run_mesh_speed_benchmark.bat
run_mesh_speed_benchmark.bat --suite --repeats 3
```

默认比较一个直裂纹和一个弯曲裂纹。--suite使用原12+4几何、种子和模型；每个模式都使用新进程。--repeats控制重复次数，奇偶重复交替执行顺序。运行时仅有同一对照模式执行，避免GPU资源竞争。完整16例多次对照可能需要较长时间。

默认结果：artifacts\mesh_speed_benchmark\summary.json和stage_timings.csv。每个算例/重复/模式目录中另存timing.json、timing.csv、run.log、候选图、网格和图。节点、候选边、特征、最终三角形和选边必须逐元素一致。GPU图聚合可能产生微小浮点差异，因此概率另外报告最大绝对差与逐元素相等状态，允许绝对差不超过5e-6；最终网格仍要求完全一致。几何不一致或概率超过容差均停止。

--resume可复用参数和模式一致的已完成记录，供中断后续跑；代码、环境或模型变化后不要复用旧目录，指定新的--output目录。

## 计时口径

- script_wall_seconds：从加载原入口到网格输出结束，包括导入库、模型加载、计算和输出；不含Python解释器启动及计时报告本身的保存。
- process_wall_seconds：对照驱动器外部测量的完整子进程时间，含解释器启动和计时结果保存；与历史实验对子进程计时的口径更接近。
- imports_seconds：原入口模块加载。
- size_field_setup：尺寸场模型加载、几何特征、网络前向、网格连续化及复杂度恢复。子项size_model_load、size_features、size_network_forward、complexity_rescale分别报告。
- adaptive_sampling：布点，包含采样期间反复查询已预测的连续尺寸场。
- candidate_edges、edge_features、line_graph：候选边、边特征、线图。
- gnn_model_load、gnn_forward：GNN加载及网络前向。
- structured_decode_and_checks：连接合法性过滤、三角形恢复、候选对角线质量改善和恢复合法性检查。子项edge_legality_filter、triangle_recovery、candidate_diagonal_quality、decoder_topology_checks分别报告。
- node_quality_smoothing：保持约束的节点质量平滑。
- final_angle_quality、final_topology_checks：最终网格验证。
- compressed_data_export、plot_export：数据输出和绘图。
- other_setup_and_reporting_seconds：其余参数解析、尺寸批量查询、JSON生成/保存等小额开销。

inclusive_seconds含子项，不能把父项和子项重复相加；exclusive_seconds扣除已计时子项。CUDA网络阶段进行同步，避免只测到异步提交时间。所有耗时都是当前运行环境实测值，不应把一两个代表算例的时间写成16例平均。

## 加速原理

common\fast_inference.py用保持同样公式和容差的标量计算替代热点里的小数组创建：

1. 单点尺寸查询仍使用原二维网格、四项双线性插值及h_min/h_max裁剪。
2. 线段方向、相交与共线重叠保持原判定规则，用二维标量叉积替代每次np.cross、concatenate和argmax调用。

不减少候选边，不删检查，不关闭质量改善，不引入CDT或三角化回退。原入口仍可直接运行作回退。--call-profile可生成.prof用于进一步函数级诊断，但会增加测量开销，不用于正式速度对照。
