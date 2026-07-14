# Stage 2 服务器 Agent 交接说明

更新时间：2026-07-14

## 1. 这份文档解决什么问题

有显卡的服务器目前只有第一阶段 `tb_prediction` 代码。本地工作区已经完成第二阶段 Neural Physics Degradation Bank 的代码实现，但没有进行正式 60 epoch 训练，也没有生成生产级 Bank。

这份说明是独立交接文档。服务器 Agent 不需要本机的 `task.MD` 或 `AGENT.md`，只要取得本节列出的代码文件、配置和测试，就能理解第二阶段已经实现了什么，以及接下来应如何训练和验收。

第二阶段的目标不是训练最终恢复主干，而是：

```text
训练 Key/Value 表示
    -> 从训练集 patch 提取 neural (q, v)
    -> 只对归一化 q 做 K-means
    -> 按 key 簇稳健聚合 value
    -> 保存 neural_physics_bank_v0.pt
```

最终恢复模型将在后续阶段使用预测的 `T/B` 查询该 Bank。本阶段的 Temporary Restorer 仅用于迫使 value 携带有效恢复信息，不是最终恢复网络。

## 2. 服务器需要同步的文件

必须同步以下新增文件：

```text
stage2/
  __init__.py
  config.py
  data.py
  models.py
  losses.py
  stage1.py
  runtime.py
  trainer.py
  extraction.py
  bank.py
  retrieval.py
  utils.py

configs/
  bank_stage2.yaml

scripts/
  train_bank.py
  extract_bank_embeddings.py
  build_neural_bank.py
  validate_neural_bank.py

tests/
  test_bank_shapes.py
  test_bank_build.py
  test_stage2_data_extraction.py
  test_stage2_trainer.py

docs/
  Stage2_Neural_Physics_Bank.md
  Stage2_Server_Agent_Handoff.md
```

还要同步这些修改：

```text
requirements.txt   # 增加 scikit-learn
.gitignore         # 忽略数据、训练输出和本地 Agent 文件
readme.md          # 增加 Stage 2 入口
```

不要通过 Git 同步以下内容：

```text
basicsr/data/DATA/
outputs/
artifacts/
*.pt / *.pth
AGENT.md / task.MD
```

数据和 checkpoint 应通过服务器已有存储、对象存储或人工拷贝放置。

## 3. 已发现的数据格式

本地数据结构为：

```text
basicsr/data/DATA/
  I/00001.png   # 水下降质图像
  J/00001.png   # 清晰/参考图像
  T/00001.png   # 三通道 transmission map
  B/00001.png   # 三通道 backscatter map
```

数据统计：

- I/J/T/B 各 9,710 张；
- basename 完全配对；
- RGB、512×512、8-bit PNG；
- 默认 `val_ratio: 0.1` 时，排序后前 8,739 张用于训练，后 971 张用于验证；
- Bank embedding 只允许从前述训练 split 提取；验证集不能进入 Bank。

物理关系按以下公式检查：

```text
I ~= J * T + B
physical_error = mean(abs(I - (J*T + B)))
```

实际导出的 8-bit 数据包含量化和第一阶段模型残差，抽样平均误差约 0.044。默认只在误差超过 0.10 时警告，不会静默 clamp；shape、通道数、NaN/Inf、越界等基础错误会直接失败。

数据加载器的关键行为：

- 四路使用相同 crop、水平/垂直翻转和 90 度旋转；
- 不对 I/J/T/B 独立做颜色增强；
- 每张图默认提取 8 个 patch；
- grouped sampler 让同图 patch 相邻，以便 worker 内有界 LRU 只解码一次原图；
- checkpoint 保存训练/验证 ID 和数据指纹，resume 与 extraction 会严格核对；
- `fingerprint_mode: stat` 默认使用名称、大小和 mtime，若需要逐字节 SHA-256，可改为 `content`，但会完整读取约 7.3 GiB 数据。

## 4. Stage 1 如何接入

第一阶段模型是：

```python
tb_prediction.model.TBResUNet
```

第二阶段通过以下已有接口加载：

```python
from tb_prediction.infer import load_model
```

完整第一阶段 checkpoint 的关键字段是：

```python
{
    "model": state_dict,
    "model_config": {...},
    "epoch": ...,
    "optimizer": ...,
    "scheduler": ...,
}
```

在 `configs/bank_stage2.yaml` 中填写服务器上的实际路径：

```yaml
stage1:
  checkpoint: /absolute/path/to/tb_prediction/best.pth
  base_channels: null
```

加载后 Stage 1 会被强制：

```text
eval()
requires_grad_(False)
no_grad/inference_mode forward
```

它只产生 `T_pred/B_pred`，供 predicted-query consistency loss 使用，不进入第二阶段优化器。

如果 `checkpoint: null`，第二阶段仍能训练，但会明确警告并把 `L_query` 置零。本地烟雾测试使用了该模式；服务器正式训练应优先配置已训练好的 Stage 1 checkpoint。

## 5. 模型已经实现了什么

### 5.1 Key 分支

输入严格是：

```text
P_gt = concat(T, B, 1-T)   # [B, 9, H, W]
```

Key Encoder 不接收 I 或 J。它输出：

```text
q_raw: [B, 64]
q:     [B, 64]，L2 normalized
```

Key Decoder 从 `q_raw` 重建 9 通道物理图，确保 key 保留 transmission、backscatter、通道衰减和空间变化信息。

### 5.2 Value 分支

输入严格是：

```text
V_input = concat(J, J-I)   # [B, 6, H, W]
```

Value Encoder 输出默认不归一化的：

```text
v: [B, 128]
```

Bank 不保存 clean patch，只保存聚合后的 value embedding。

### 5.3 Temporary Restorer

输入为 `I` 和 `v`，通过瓶颈及解码阶段的多处 FiLM 注入 value，预测 residual：

```text
J_temp = clamp(I + residual, 0, 1)
```

它的作用是防止 Value Encoder 学成无用向量。正式 Bank 构建后，最终推理不需要 Temporary Restorer，也不需要 Value Encoder。

### 5.4 投影头

```text
Key Projector:   64 -> 64
Value Projector: 128 -> 64
```

投影头只用于 Key/Value soft relation alignment；Bank 中保存的是未投影的 main value。

### 5.5 默认参数量

| 模块 | 参数量 |
|---|---:|
| Key Encoder | 3,475,168 |
| Key Decoder | 1,376,201 |
| Value Encoder | 3,482,560 |
| Temporary Restorer | 1,243,619 |
| Key Projector | 8,320 |
| Value Projector | 24,768 |
| Stage 2 总计 | 9,610,636 |

所有低层图像编码器均未使用 BatchNorm。

## 6. 损失与训练日程

已经实现：

```text
L_key_rec = L1 + lambda_grad * gradient loss
L_restore = Charbonnier + lambda_ssim * SSIM + lambda_fft * FFT
L_query   = 1 - cosine(q_pred, stop_gradient(q_gt))
L_align   = 双向 KL(soft physical relation || projected embedding relation)
L_inv     = value content invariance
L_rank    = optional correct-vs-wrong value ranking
```

Value invariance 的增强流程严格保持物理一致：

```text
J_aug = moderate_content_augmentation(J)
I_aug = J_aug * T + B
```

T/B 不随内容增强改变。

默认总损失权重：

```yaml
lambda_restore: 1.0
lambda_key: 0.5
lambda_align: 0.1
lambda_query: 0.2
lambda_inv: 0.1
lambda_rank: 0.0
```

训练日程：

- epoch 1–10：warm-up，关闭 alignment；
- epoch 11–60：joint，打开 alignment；
- `best_warmup.pt` 只记录 warm-up 最优模型；
- `best.pt` 只从 joint 阶段选择，后续 extraction 必须使用它；
- `last.pt` 用于 resume。

这样避免把投影头尚未训练的 warm-up checkpoint 错当成最终 Bank 编码器。

## 7. 训练和恢复已经具备的能力

`stage2/trainer.py` 已实现：

- CPU/CUDA；
- CUDA AMP；
- deterministic seed；
- gradient clipping；
- AdamW + cosine scheduler；
- TensorBoard 和逐项 loss/LR 日志；
- 定期 validation；
- temporary restorer correct/shuffled/zero value 诊断；
- checkpoint save/resume；
- Python/NumPy/Torch/CUDA、sampler 和 worker generator RNG 状态保存；
- split ID、数据指纹和配置兼容检查；
- AMP 模式变化时安全重建 scaler；
- Stage 1 缺失时清晰降级。

Resume 会拒绝悄悄改变以下训练语义的配置：模型维度、patch 数、增强方式、损失、epoch 日程、batch size、学习率、worker 数和 Stage 1 配置等。

## 8. Embedding 提取与 Bank 构建

Embedding 提取：

```text
q_i = normalize(KeyEncoder([T_i, B_i, 1-T_i]))
v_i = ValueEncoder([J_i, J_i-I_i])
```

输出采用增量分片：

```text
artifacts/bank_embeddings/
  keys_00000.npy
  values_00000.npy
  ...
  metadata.jsonl
  manifest.json
  extraction_config.yaml
```

metadata 包含 sample ID、输入逻辑路径、crop 坐标、几何变换、物理误差、mean T 和 mean B。

提取器对 Key Encoder、Value Encoder 和两个 projector 的 state tensor 计算规范化 SHA-256。构建 Bank 时必须使用同一 checkpoint；换成另一轮同维度 checkpoint 也会因 fingerprint 不匹配而被拒绝。

Bank 构建规则：

1. 归一化 neural keys；
2. 只对 keys 做 K-means，绝不拼接 value；
3. 默认 64 个原型；
4. 每簇丢弃距中心最远的 20%；
5. retained keys 求均值并再次归一化；
6. retained values 在相同成员上求稳健均值；
7. 记录原始 cluster count、variance、空簇修复和物理 metadata 聚合。

`backend: auto` 优先使用 scikit-learn MiniBatchKMeans；环境缺少时会回退到确定性的 PyTorch K-means。

Bank 验证会检查：

- 原型数和维度；
- cluster min/max/mean；
- empty/singleton cluster；
- cluster variance；
- key/value norm；
- duplicate/near-duplicate prototype；
- minimum inter-prototype cosine distance；
- encoder state 和配置完整性；
- `training_split_only=True`。

## 9. 最终 Bank checkpoint 格式

目标文件：

```text
artifacts/neural_physics_bank_v0.pt
```

主要字段：

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
    "temporary_restorer": state_dict,
    "configuration": {
        "patch_size": 64,
        "key_dim": 64,
        "value_dim": 128,
        "projection_dim": 64,
        "num_prototypes": 64,
        "top_k": 4,
        "retrieval_temperature": 0.1,
        "key_input": "[T, B, 1-T]",
        "value_input": "[J, J-I]",
        "stage2_checkpoint": ".../best.pt",
        "stage2_checkpoint_fingerprint": "...",
        "training_split_only": True,
    },
}
```

最终恢复模型的必要部分是：

```text
key_encoder + keys + values
```

Value Encoder、projectors 和 Temporary Restorer 保留用于复现、诊断和重建 Bank。

## 10. 服务器上的第一步：同步后自检

服务器 Agent 应先执行：

```bash
git status --short
python -m pip install -r requirements.txt
python -m compileall -q stage2 scripts tests
python -m unittest discover -s tests -p "test_*.py" -v
```

预期：全部 7 个测试通过。测试覆盖：

- shape；
- full backward；
- physical consistency；
- tiny extraction；
- 8-prototype Bank；
- global/token-wise retrieval；
- warm-up -> joint -> resume；
- checkpoint fingerprint mismatch rejection。

然后检查第一阶段 checkpoint：

```bash
python -c "from pathlib import Path; p=Path('/absolute/path/to/best.pth'); print(p, p.is_file(), p.stat().st_size if p.is_file() else None)"
```

不要在确认路径和格式前直接启动 60 epoch 训练。

## 11. 先跑 GPU smoke test

从正式配置复制一份 smoke 配置，不要直接改坏生产配置：

```bash
cp configs/bank_stage2.yaml configs/bank_stage2_smoke.yaml
```

建议修改：

```yaml
data:
  train_limit: 8
  val_limit: 2
  patches_per_image: 2

model:
  base_channels: 8

training:
  output_dir: outputs/bank_stage2_smoke
  batch_size: 4
  epochs: 2
  warmup_epochs: 1
  num_workers: 0
  use_amp: true
  diagnostic_frequency: 0

extraction:
  output_dir: artifacts/bank_embeddings_smoke
  patches_per_image: 2
  batch_size: 4
  num_workers: 0

bank:
  output: artifacts/neural_physics_bank_smoke.pt
  num_prototypes: 4

retrieval:
  top_k: 4
```

运行：

```bash
python scripts/train_bank.py \
  --config configs/bank_stage2_smoke.yaml \
  --device cuda
```

必须确认：

- 日志显示 `device=cuda`、AMP enabled；
- Stage 1 checkpoint 成功加载，没有 missing warning；
- 所有 loss、PSNR、SSIM、query cosine 有限；
- epoch 1 生成 `best_warmup.pt`；
- epoch 2 生成 joint `best.pt`；
- `L_query` 非零且 query cosine 被记录；
- 没有 NaN/Inf 或 shape 错误。

继续 smoke extraction/build/validate：

```bash
python scripts/extract_bank_embeddings.py \
  --config configs/bank_stage2_smoke.yaml \
  --checkpoint outputs/bank_stage2_smoke/best.pt \
  --output-dir artifacts/bank_embeddings_smoke

python scripts/build_neural_bank.py \
  --config configs/bank_stage2_smoke.yaml \
  --embeddings artifacts/bank_embeddings_smoke \
  --checkpoint outputs/bank_stage2_smoke/best.pt \
  --output artifacts/neural_physics_bank_smoke.pt

python scripts/validate_neural_bank.py \
  --bank artifacts/neural_physics_bank_smoke.pt \
  --output-dir artifacts/bank_validation_smoke
```

smoke Bank 要求：原型数正确、无空簇、所有 tensor finite、key norm 接近 1、retrieval weight 和为 1。少量样本若导致 duplicate prototype 或 singleton 占比过高，验证可能判 invalid；此时应增加 smoke extraction patch 数，而不是关闭验证。

## 12. 正式训练步骤

Smoke 通过后，恢复正式配置：

```yaml
data:
  patch_size: 64
  patches_per_image: 8
  train_limit: null
  val_limit: null

model:
  key_dim: 64
  value_dim: 128
  projection_dim: 64
  base_channels: 32

training:
  output_dir: outputs/bank_stage2
  batch_size: 32
  epochs: 60
  warmup_epochs: 10
  learning_rate: 0.0002
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  use_amp: true
  num_workers: 8
```

启动：

```bash
python scripts/train_bank.py \
  --config configs/bank_stage2.yaml \
  --device cuda
```

如果显存不足，优先减小 batch size。修改 batch size 后不要用旧 checkpoint 继续 resume，因为配置兼容检查会拒绝改变训练语义。应为新实验使用新的 output directory。

恢复中断训练：

```bash
python scripts/train_bank.py \
  --config configs/bank_stage2.yaml \
  --resume outputs/bank_stage2/last.pt \
  --device cuda
```

训练完成后必须使用 joint `best.pt`：

```bash
python scripts/extract_bank_embeddings.py \
  --config configs/bank_stage2.yaml \
  --checkpoint outputs/bank_stage2/best.pt

python scripts/build_neural_bank.py \
  --config configs/bank_stage2.yaml \
  --embeddings artifacts/bank_embeddings \
  --output artifacts/neural_physics_bank_v0.pt

python scripts/validate_neural_bank.py \
  --bank artifacts/neural_physics_bank_v0.pt \
  --output-dir artifacts/bank_validation
```

## 13. 正式验收标准

服务器 Agent 完成正式训练后，应在报告中给出：

1. Stage 1 checkpoint 的路径、epoch、模型配置和加载结果；
2. Stage 2 `best.pt` 的 epoch 与全部 validation metrics；
3. correct/shuffled/zero value restoration PSNR；
4. embedding 总数及训练 split 证明；
5. Bank 原型数、cluster size min/max/mean；
6. empty、singleton、duplicate prototype 数；
7. cluster variance 和 key/value norm 统计；
8. checkpoint fingerprint 校验结果；
9. retrieval shape、index 范围和 weight-sum 检查；
10. 最终文件路径与 SHA-256。

关键质量条件：

```text
correct value restoration > shuffled value restoration
correct value restoration > zero value restoration
empty clusters == 0
duplicate prototype pairs == 0
all key norms ~= 1
training_split_only == True
```

若 correct/shuffled/zero 几乎相同，说明 Temporary Restorer 可能忽略 value。此时不要发布 Bank；应检查 FiLM 梯度、Value Encoder 梯度、训练时长，并按实验开启 `lambda_rank`。

## 14. 已完成验证与尚未完成事项

本地已验证：

- 7/7 自动测试通过；
- 默认维度 forward/backward 通过；
- CPU 两阶段训练和 resume 通过；
- RTX 5060 Ti CUDA AMP 单步通过；
- 真实 I/J/T/B 的缩小流程完成 2 epoch、提取 16 个 embedding、构建并验证 2 原型 Bank；
- checkpoint mismatch 和 collapsed embedding 会被拒绝；
- global 与 token-wise retrieval 通过。

尚未完成：

- 本地没有训练好的 Stage 1 checkpoint，因此没有用真实 `T_pred/B_pred` 完成正式 query consistency 训练；
- 没有执行默认 60 epoch；
- 没有生成生产级 `outputs/bank_stage2/best.pt`；
- 没有生成生产级 `artifacts/neural_physics_bank_v0.pt`；
- 收敛后的 correct/shuffled/zero 诊断尚未验证。

服务器下一项真正需要完成的工作，就是使用已有 Stage 1 checkpoint 和 GPU 执行第 11–13 节，并保存完整训练及 Bank 验收报告。

## 15. 常见故障

### Stage 1 checkpoint missing

确认 YAML 使用服务器绝对路径，且 checkpoint 包含 `model` 和 `model_config`。只有纯 state dict 时，非默认宽度需要填写 `base_channels`。

### Resume configuration incompatible

不要修改原实验的模型、数据 split、增强、batch size、epoch 总数或 Stage 1 配置。要修改时新建实验目录并重新训练。

### Embedding checkpoint fingerprint mismatch

提取 embedding 与最终 Bank 必须使用同一个 `best.pt`。不要用另一次训练的同维模型替换，也不要使用 `best_warmup.pt`。

### Too few samples for prototypes

保证 embedding 数不少于 prototype 数。Smoke 配置应降低 prototype 数；正式配置应有约 `8739 * 8 = 69,912` 个 patch embedding。

### Bank invalid because of duplicate/singleton clusters

先增加 extraction patch 数或检查 Key Encoder 是否塌缩，不要通过修改验证器绕过。

### CUDA OOM

先减小 batch size，再考虑减小 `base_channels`。正式实验改变结构后必须重新训练和提取，不能复用原 Bank。

### Bank build memory warning

Embedding writer 是增量分片，但带 20% trimming 的标准构建会把 embeddings 和 metadata 读入内存。默认约 69,912 条通常可控；若大幅增加 patch 数，应根据预估警告改用更大内存或实现基于 `EmbeddingReader.iter_shards()` 的多遍构建。

## 16. 给服务器 Agent 的简短执行指令

```text
先阅读 docs/Stage2_Server_Agent_Handoff.md 和
docs/Stage2_Neural_Physics_Bank.md。

确认 Stage 2 文件已同步，保留服务器现有 Stage 1 代码和 checkpoint。
不要提交或覆盖数据集、checkpoint、outputs、artifacts。

先运行 compileall 与 7 个 unittest；然后配置真实 Stage 1 checkpoint，
完成 2 epoch GPU smoke 的 train -> extract -> build -> validate。
Smoke 全部通过后，使用默认 60 epoch 配置正式训练。

最终只使用 joint best.pt 提取训练集 embedding；只对 keys 聚类；
校验 checkpoint fingerprint、training-only provenance、cluster coverage、
correct/shuffled/zero value 诊断与 retrieval 权重。

不要把 smoke Bank 当作正式产物，也不要在缺少真实 Stage 1 checkpoint
或 value diagnostic 未通过时宣布第二阶段训练完成。
```
