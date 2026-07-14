# Stage 2：Neural Physics Degradation Bank

如果目标机器目前只有第一阶段代码，请先阅读 [Stage 2 服务器 Agent 交接说明](Stage2_Server_Agent_Handoff.md)，其中包含文件同步清单、GPU smoke、正式训练和验收步骤。

## 1. 用途与边界

Stage 2 从配对水下数据中学习两类表示，并把它们压缩成可离线复用的物理退化 Bank：

- Key Encoder 只接收 `[T, B, 1-T]`，输出描述局部物理退化状态的归一化 key；
- Value Encoder 接收 `[J, J-I]`，输出与恢复相关、默认不归一化的 value；
- 临时恢复网络通过 value 恢复 `J`，迫使 value 携带有效恢复信息；
- 投影头只用于训练期间的 Key–Value 软关系对齐。

本阶段**不是最终水下图像恢复网络**。临时恢复网络只服务于 Stage 2 预训练；最终推理不需要 Value Encoder 或临时恢复网络，只需由预测的 `T/B` 生成查询 key，再使用 Bank 中的 `keys` 和 `values` 检索恢复先验。

完整流程为：

```text
训练 Stage 2 -> 提取训练集 key/value -> 仅对 keys 聚类 -> 按 key 簇聚合 values -> 验证 Bank
```

## 2. 数据布局与划分

配置项 `data.root` 默认指向 `basicsr/data/DATA`。加载器只扫描四个显式子目录，不把 `_logs`、`_work` 或 `manifest.csv` 当作样本来源：

```text
basicsr/data/DATA/
  I/00001.png   # 水下降质图像
  J/00001.png   # 清晰/参考图像
  T/00001.png   # 三通道透射图
  B/00001.png   # 三通道后向散射图
```

四路图像必须具有相同 basename、空间尺寸和 RGB 对齐关系。当前加载器要求 8-bit 图像，读取后转换为 RGB、CHW、float32 `[0,1]`。期望物理关系为：

```text
I ~= J * T + B
```

当前工作区数据实测为四路各 9,710 张、RGB 512×512 PNG。默认 `val_ratio: 0.1` 按排序后的样本 ID 确定性划分：

- 训练集 8,739 张，当前边界为 `00001` 至 `09250`；
- 验证集 971 张，当前边界为 `09251` 至 `10255`。

ID 中存在缺号，因此边界表示排序位置，不表示连续整数范围。训练和恢复 checkpoint 会记录 `train_ids`、`val_ids` 及文件指纹；恢复训练和提取 embedding 时若数据或划分变化，会拒绝继续。

Bank embedding **只遍历训练集**。验证集仅用于 Stage 2 训练期间选取 checkpoint，不能参与 embedding 提取或 Bank 构建。`train_limit` 和 `val_limit` 仅建议用于 smoke test，正式运行应保持为 `null`。

## 3. Stage 1 T/B 预测器

在 `configs/bank_stage2.yaml` 中配置已有的 `tb_prediction` checkpoint：

```yaml
stage1:
  checkpoint: experiments/tb_prediction/best.pth
  base_channels: null
```

- 完整训练 checkpoint 会自动读取其中的 `model_config`；
- 只有加载纯 `state_dict` 且宽度不是默认值时，才需要填写 `base_channels`；
- Stage 1 在 Stage 2 中始终处于 `eval` 和冻结状态，不进入优化器；
- 配置为 `null` 时，程序会明确警告并将 predicted-query consistency loss `L_query` 置零，其余 Stage 2 训练仍可进行；
- 若配置了路径但文件不存在，程序会直接报错。

缺少 Stage 1 时可以验证主流程，但不会训练“预测 T/B 查询”和 GT 查询之间的一致性。用于最终系统的正式 Bank 建议配置已训练好的 Stage 1 checkpoint。

## 4. 配置

默认配置为 `configs/bank_stage2.yaml`，包含数据、模型维度、各项 loss 权重、训练、提取、聚类和检索参数。常用设置包括：

```yaml
data:
  patch_size: 64
  patches_per_image: 8
model:
  key_dim: 64
  value_dim: 128
  projection_dim: 64
training:
  batch_size: 32
  epochs: 60
  warmup_epochs: 10
bank:
  num_prototypes: 64
  trim_fraction: 0.2
retrieval:
  top_k: 4
  temperature: 0.1
```

配置加载器会拒绝未知字段。所有命令应从仓库根目录执行。

`data.fingerprint_mode` 默认为 `stat`，使用样本 ID、文件名、大小和纳秒修改时间快速检测数据变化；需要字节级强校验时可设为 `content`，此模式会读取四个目录中的全部文件并计算 SHA-256，因此在当前约 7.3 GiB 数据上启动会明显变慢。

## 5. 训练与恢复

开始训练：

```bash
python scripts/train_bank.py --config configs/bank_stage2.yaml
```

可用 `--device cuda` 或 `--device cpu` 覆盖配置中的设备。训练输出默认位于 `outputs/bank_stage2/`：

```text
last.pt                  # 每个 epoch 更新，可用于恢复
best_warmup.pt           # warm-up 阶段最佳验证 checkpoint
best.pt                  # joint 对齐阶段最佳 checkpoint，后续提取应使用它
config_snapshot.yaml
config_resume_snapshot.yaml   # 恢复运行时保存
tensorboard/
```

从 `last.pt` 恢复：

```bash
python scripts/train_bank.py \
  --config configs/bank_stage2.yaml \
  --resume outputs/bank_stage2/last.pt
```

恢复会校验模型、训练日程、数据划分及数据指纹，并恢复优化器、scheduler 和随机状态；AMP 模式一致时也会恢复 scaler，模式变化时会警告并使用新的 scaler。不要用 `best_warmup.pt` 构建最终 Bank，因为其 Key–Value 对齐尚未完成。

## 6. 提取训练集 embedding

使用 joint 阶段的 `best.pt`：

```bash
python scripts/extract_bank_embeddings.py \
  --config configs/bank_stage2.yaml \
  --checkpoint outputs/bank_stage2/best.pt \
  --output-dir artifacts/bank_embeddings
```

输出目录已有提取文件时，程序默认拒绝覆盖；确认需要重建后添加 `--overwrite`。提取会重新校验 checkpoint 的模型配置、训练/验证 ID 和数据指纹，并固定遍历训练 split。

## 7. 构建 Neural Physics Bank

```bash
python scripts/build_neural_bank.py \
  --config configs/bank_stage2.yaml \
  --embeddings artifacts/bank_embeddings \
  --checkpoint outputs/bank_stage2/best.pt \
  --output artifacts/neural_physics_bank_v0.pt \
  --backend auto
```

`--checkpoint` 可省略；脚本会优先读取提取配置中记录的 checkpoint，再回退到配置输出目录下的 `best.pt`。显式传入可避免路径歧义。

构建过程严格遵循以下规则：

1. L2 归一化 neural keys；
2. **仅对 keys 执行 K-means，不拼接 keys 与 values**；
3. 每簇丢弃距中心最远的 20%（由 `trim_fraction` 配置）；
4. 对保留 keys 求均值并再次归一化；
5. 在相同成员上聚合未归一化 values。

`--backend auto` 优先使用 scikit-learn `MiniBatchKMeans`；未安装 scikit-learn 时会警告并回退到确定性的 PyTorch K-means。也可明确指定 `--backend sklearn` 或 `--backend torch`。

## 8. 验证 Bank

```bash
python scripts/validate_neural_bank.py \
  --bank artifacts/neural_physics_bank_v0.pt \
  --output-dir artifacts/bank_validation
```

脚本在终端打印 JSON 报告，并在输出目录写入 `bank_validation_report.json`。报告包含原型数、簇大小、空簇、簇方差、key norm 和 value norm 等统计。

安装 matplotlib 后还会生成：

```text
cluster_size_histogram.png
cluster_variance_histogram.png
key_tsne_or_umap.png
```

没有 matplotlib 时只跳过绘图，数值验证仍会完成。key 可视化在 scikit-learn 可用时尝试 t-SNE，否则使用 SVD 降维回退。

## 9. 文件格式

### 9.1 Stage 2 训练 checkpoint

`last.pt`、`best_warmup.pt` 和 `best.pt` 是 PyTorch mapping，主要字段为：

```python
{
    "epoch": int,
    "global_step": int,
    "key_encoder": state_dict,
    "key_decoder": state_dict,
    "value_encoder": state_dict,
    "temporary_restorer": state_dict,
    "key_projector": state_dict,
    "value_projector": state_dict,
    "optimizer": state_dict,
    "scheduler": state_dict,
    "scaler": state_dict,
    "config": dict,
    "data_manifest": {
        "train_ids": list,
        "val_ids": list,
        "train_fingerprint": str,
        "val_fingerprint": str,
        "training_split_only": True,
    },
    "metrics": dict,
    "rng_state": dict,
    "amp_enabled": bool,
}
```

### 9.2 Embedding bundle

embedding 采用增量分片，避免把全部样本同时放入 RAM：

```text
artifacts/bank_embeddings/
  keys_00000.npy       # float32 [N_i, key_dim]，已 L2 归一化
  values_00000.npy     # float32 [N_i, value_dim]，不做 L2 归一化
  ...
  metadata.jsonl
  manifest.json
  extraction_config.yaml
```

`manifest.json` 记录总样本数、维度、dtype、chunk size 和各分片文件；只有完整提取结束后才会生成。`metadata.jsonl` 每行对应一对 key/value，包含 `sample_id`、图像路径、patch 序号与坐标、几何变换、物理重建误差及 `mean_T/mean_B`。`extraction_config.yaml` 明确记录 `training_split_only: true` 和 Stage 2 checkpoint 路径。

提取器还会对 Key Encoder、Value Encoder 和两个投影头计算规范化 SHA-256 指纹。构建 Bank 时该指纹必须与传入的 Stage 2 checkpoint 一致，从而避免把一轮模型产生的 prototypes 与另一轮编码器权重混用。

### 9.3 最终 Bank checkpoint

`neural_physics_bank_v0.pt` 的核心字段为：

```python
{
    "keys": Tensor[num_prototypes, key_dim],
    "values": Tensor[num_prototypes, value_dim],
    "cluster_count": Tensor[num_prototypes],
    "cluster_variance": Tensor[num_prototypes],
    "key_encoder": state_dict,
    "value_encoder": state_dict,
    "key_projector": state_dict,
    "value_projector": state_dict,
    "temporary_restorer": state_dict,       # 当前构建流程会保留
    "configuration": {
        "patch_size": int,
        "key_dim": int,
        "value_dim": int,
        "projection_dim": int,
        "num_prototypes": int,
        "top_k": int,
        "retrieval_temperature": float,
        "key_input": "[T, B, 1-T]",
        "value_input": "[J, J-I]",
        "stage2_checkpoint": str,
        "training_split_only": True,
    },
}
```

checkpoint 还可包含 `prototype_metadata`、`clustering_backend` 和 `trim_fraction`。最终恢复推理的必要部分是 `key_encoder`、`keys` 和 `values`；Value Encoder、投影头和临时恢复网络主要为复现、诊断或重建 Bank 保留。

## 10. 常见失败模式

- **缺少 Stage 1**：`checkpoint: null` 会警告并关闭 `L_query`；配置了不存在的路径则报错。正式 Bank 建议补充已训练 Stage 1。
- **I/J/T/B 缺失或不对齐**：任一 basename 缺失、尺寸/通道不一致、非 8-bit、NaN/Inf 或超出范围都会报错，不会静默夹紧。
- **物理误差偏高**：程序计算 `mean(abs(I-(J*T+B)))`。当前导出的 8-bit 数据包含量化和模型残差，默认在 0.10 以上警告、`physical_error_fail: null` 不硬失败，并限制每个 worker 的警告数。不要在不了解误差分布时设置过低的失败阈值。
- **临时恢复网络忽略 value**：诊断会比较 correct、跨样本 shuffled 和 zero value 的 PSNR。若 correct 不优，程序会警告；应检查 Value Encoder/FiLM 梯度、训练时长与 loss，并可按实验开启 `lambda_rank`，不应直接把该 checkpoint 用于最终 Bank。
- **样本数不足**：embedding 数必须不少于 `bank.num_prototypes`；smoke test 使用少量样本时，应同步调小原型数。
- **embedding 目录已存在或提取中断**：使用新目录，或确认后加 `--overwrite`。缺少最终 `manifest.json` 表示提取未完整结束，应重新提取。
- **恢复或提取配置不兼容**：模型维度、patch size、训练/验证 ID 或数据指纹不一致时会拒绝继续；应使用训练 checkpoint 对应的数据和配置，而不是绕过校验。
- **RAM cache 过大**：默认 `cache_data: none` 配合有界解码缓存。当前全量 float32 RAM cache 约需 113.8 GiB，超过 `ram_cache_max_gib` 会提前报错；除非明确估算过内存，否则不要启用 `ram`。
- **Bank 构建内存**：embedding 提取本身是增量分片的，但带 20% robust trimming 的标准 K-means 构建会把 keys、values 与 metadata 载入内存。工具会预估并警告大任务；超出内存时应减少提取 patch 数，或基于 `EmbeddingReader.iter_shards()` 实现多遍流式构建。
- **可选聚类/绘图依赖缺失**：scikit-learn 缺失时 `--backend auto` 回退 PyTorch；指定 `--backend sklearn` 则必须安装 scikit-learn。matplotlib 缺失只影响验证图，不影响 JSON 报告和 Bank 有效性检查。
- **空簇或非法 Bank**：构建器会尝试修复空簇；若仍存在空簇、NaN/Inf、非单位 key 或配置与张量维度不一致，构建/验证会明确失败或在报告中列出问题。
