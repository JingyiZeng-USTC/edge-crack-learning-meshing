# Gmsh局部加密与学习型尺寸场对照

在已有项目和conda环境中运行，不训练模型。两组都使用Gmsh算法6、相同零厚度裂纹处理、CST平面应力单元与稀疏直接FEM求解。

## 运行

```bat
conda activate D:\itl\eg\pinn_gpu
cd /d D:\itl\eg\3\gnn\gmsh_local_refinement_comparison
run_comparison.bat
```

默认结果在 `D:\itl\eg\3\gnn\artifacts\gmsh_local_vs_learned`。重复实验请换 `--output`，已有summary时不会覆盖。

## 对照方案

- Gmsh组：原生Distance分别计算到裂尖点及裂纹线距离；各自接Threshold线性过渡，通过Min组合。
- 裂尖尺寸0.004 m，距裂尖0.008 m以内保持该尺寸，在0.08 m处增至0.030 m。
- 裂纹线尺寸0.008 m，距裂纹0.004 m以内保持该尺寸，在0.05 m处增至0.030 m。
- 学习组：读取现有 `single_straight_crack_local_v4.pt`，查询预测尺寸，基础范围0.004至0.030 m。
- 全局倍率同时缩放各组目标尺寸，保留空间加密范围。横轴是实测节点数，不假定同倍率节点数相等。
- 默认7个倍率2.4、2、1.6、1.3、1、0.8、0.65。无需Poisson、GNN或FCNN参与此尺寸场消融试验。
- 加载为顶边5e5 N/m均布线载荷，厚度0.01 m，E=210 GPa，nu=0.3；底边uy=0并固定一个ux，与原项目一致。

## 参数

```bat
run_comparison.bat --origin 1.07 --crack-length 0.32 --angle-deg 11 --tip-inner 0.008 --tip-outer 0.08 --crack-inner 0.004 --crack-outer 0.05 --crack-size 0.008 --output D:\itl\eg\3\gnn\artifacts\local_case2
```

基线参数是预设工程规则，不是Gmsh自动识别裂纹后的默认行为，也不代表最优Gmsh策略。建议在独立验证裂纹上选择加密范围与裂纹尺寸，然后冻结参数用于未见裂纹；不要看测试结果后单独调弱基线。两个组均不需要预先求解待测网格的应力来指定尺寸。

## 参考与指标

两组共用一份参考解。参考尺寸取两组基础尺寸的逐点最小值，再乘0.4和0.28生成两级参考网格，使用较细一级的应变能。参考中的线距离采用精确线段距离；Gmsh基线曲线Distance使用1000采样点。

报告 `abs(U-Uref)/abs(Uref)*100%`，以及最小角、残差、节点数和耗时。参考两级能量变化默认须小于0.10%；未通过时仍保存原始曲线但不报告节点节省率。该检查只反映稳定性，不保证绝对误差界。可减小 `--reference-scale` 继续验证。

目标1%交点只在相邻实际数据间线性插值，不做单调包络或外推。最粗级已小于目标或最细级仍达不到目标时，交点留空，应扩充 `--scales`。允许学习组更差，节省率可以为负；不预设原均匀基线的77.6%结论。

## 输出

- `results.csv`：每个真实网格的全部原始指标（节点数包括裂纹面复制节点，同时报告几何节点数）。
- `summary.json`：两级参考检查、交点状态和有效时的节点节省率。
- `config.json`：全部设置、Gmsh版本及checkpoint SHA256。
- `meshes/*.npz`：参考与各组网格，nodes与triangles数组。
- `Figure_size_field_node_efficiency.png/pdf/svg/tiff`：两组曲线及目标误差线。

依赖沿用已有conda环境：gmsh、numpy、scipy、torch、matplotlib、Pillow以及项目中的module10_fem和module30_learned_size_field。无需重新安装PyTorch。`common_pipeline.py`复用既有几何、裂纹面分离、FEM和出图实现。

Gmsh尺寸场说明：https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
