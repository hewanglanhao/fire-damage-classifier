# FireDamageClassification 实验汇总

> 说明：本文件基于 `results/*` 中的 `model_info.txt`、`results.json` 和 `test_metrics.json` 整理，重点总结各实验的模型架构与测试集表现。共同超参数（若未特殊说明）：`batch_size=32`，`image_size=224`，`epochs=70`，`lr=1e-4`，`weight_decay=1e-3`，`dropout=0.2`，`drop_path_rate=0.1`，`lambda_cls=1.0`，`lambda_vae=0.1`，`lambda_align=0.1`。

---

## 1. 1_baseline

- 模型类型：`BaselineViT`（`model.type=baseline`），`backbone=vit_base_patch16_224`，仅图像输入，层次化分类头（binary + severity），无文本分支。
- 主要训练参数：同上默认训练参数；`model.use_fine=False`（只做单层标签）。
- 复杂度：参数量 `108,883,207`，FLOPs `16,870,809,856`。
- 测试集结果：
  - `accuracy=0.6542`，`macro_f1=0.4062`，`weighted_f1=0.6686`。
  - 每类准确率约：[0.53, 0.10, 0.17, 0.92, 0.37]。
- 验证集最佳：`best_val_accuracy=0.7257`。

## 2. 1_baseline_cnn_resnet

- 模型类型：`BaselineCNN`（`model.type=baseline_cnn`），`backbone=resnet101`，仅图像输入，层次化分类头。
- 主要训练参数：同默认训练参数。
- 复杂度：参数量 `105,447,495`，FLOPs `7,927,324,672`（比 ViT 基线更轻量）。
- 测试集结果：
  - `accuracy=0.7273`，`macro_f1=0.4637`，`weighted_f1=0.7260`。
  - 每类准确率约：[0.68, 0.19, 0.06, 0.94, 0.43]。
- 验证集最佳：`best_val_accuracy=0.8176`。

## 3. 2_method_a_vae

- 模型架构：`FireDamageClassifier`，`method_option=vae`（方法 A：图像 VAE），使用：
  - 文本分支：`TextVAE`（默认 coarse+fine 配置，但 `config` 中 `use_fine=False`，只启用 coarse 文本 VAE）。
  - 图像分支：`ImageVAE`（ViT 编码 + 图像解码器，用图像重构 + KL）。
  - 分类头：Transformer 编码 + `HierarchicalHead`。
- 训练目标：分类 + 文本重构（coarse）+ 图像重构（VAE），对齐项由 VAE 代替。
- 复杂度：参数量 `126,120,882`，FLOPs `20,588,554,752`。
- 测试集结果：
  - `accuracy=0.7077`，`macro_f1=0.4379`，`weighted_f1=0.7126`。
- 验证集最佳：`best_val_accuracy=0.7842`。

## 4. 2_method_b_alignment

- 模型架构：`FireDamageClassifier`，`method_option=alignment`（方法 B：CLIP 式对齐），使用：
  - 文本分支：coarse 文本 `TextVAE`（`use_fine=False`）。
  - 图像分支：`ImageAlignment`（ViT 编码 + 线性层，无图像重构）。
  - 对齐损失：`1 - cos(z_img, z_text)`，`z_text` 优先为 `z_fine`，此实验中实际为 `z_coarse`。
  - 分类头：Transformer + `HierarchicalHead`。
- 复杂度：参数量 `110,069,711`，FLOPs `18,854,078,976`。
- 测试集结果：
  - `accuracy=0.7398`，`macro_f1=0.4458`，`weighted_f1=0.7349`。
- 验证集最佳：`best_val_accuracy=0.8047`。

## 5. 3_only_coarse

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，消融设置：
  - `use_coarse=True`，`use_fine=False`，`use_image=True`。
  - coarse 文本 VAE + 图像 alignment 分支。
  - 融合：默认 transformer token 融合 CLS + coarse latent + 图像 latent。
- 复杂度：参数量 `110,069,711`，FLOPs `18,854,078,976`。
- 测试集结果：
  - `accuracy=0.6996`，`macro_f1=0.4160`，`weighted_f1=0.7066`。
- 验证集最佳：`best_val_accuracy=0.8009`。

## 6. 3_only_fine

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，消融设置：
  - `use_coarse=False`，`use_fine=True`，`use_image=True`。
  - fine 文本 VAE + 图像 alignment 分支（fine 文本为主要对齐目标）。
- 复杂度：参数量 `184,953,999`，FLOPs `33,878,552,064`（比仅 coarse 情况大幅增加）。
- 测试集结果：
  - `accuracy=0.6640`，`macro_f1=0.4135`，`weighted_f1=0.6766`。
- 验证集最佳：`best_val_accuracy=0.7340`。

## 7. 4_encoder_cnn

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，编码器消融：
  - 文本侧：`coarse_encoder_type='cnn'`，`fine_encoder_type='cnn'`（文本 VAE 使用 CNNEncoder）。
  - 图像侧：仍为 ViTEncoder 的 alignment 分支。
  - `config` 中 `use_fine=False`，主要使用 coarse 文本 VAE。
- 复杂度：参数量 `110,069,711`，FLOPs `18,854,078,976`。
- 测试集结果：
  - `accuracy=0.7023`，`macro_f1=0.4349`，`weighted_f1=0.7067`。
- 验证集最佳：`best_val_accuracy=0.8047`。

## 8. 4_encoder_vit

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，编码器消融：
  - 文本侧：`coarse_encoder_type='vit'`，`fine_encoder_type='vit'`（文本 VAE 使用 ViTEncoder）。
  - `config` 中 `use_fine=False`（仅 coarse 文本分支激活）。
- 复杂度：参数量 `184,953,999`，FLOPs `33,878,552,064`。
- 测试集结果：
  - `accuracy=0.6506`，`macro_f1=0.4155`，`weighted_f1=0.6679`。
- 验证集最佳：`best_val_accuracy=0.7067`。

## 9. 5_align_fusion_img_only

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，融合消融：
  - `use_coarse=True`，`use_fine=False`。
  - `fusion_include_image=True`，`fusion_include_coarse=False`，`fusion_include_fine=False`：
    - coarse 文本 VAE 仍训练（用于对齐损失），**但分类时只用图像 latent（z_img）** 作为 transformer token。
- 复杂度：参数量 `110,069,711`，FLOPs `18,849,876,480`。
- 测试集结果：
  - `accuracy=0.6586`，`macro_f1=0.4040`，`weighted_f1=0.6676`。
- 验证集最佳：`best_val_accuracy=0.7082`。

## 10. 5_align_fusion_img_coarse

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，融合消融：
  - `use_coarse=True`，`use_fine=False`。
  - `fusion_include_image=True`，`fusion_include_coarse=True`：
    - 分类时同时使用 z_img 和 z_coarse 两种 latent（图像 + coarse 文本语义）。
- 复杂度：参数量 `110,069,711`，FLOPs `18,854,078,976`。
- 测试集结果：
  - `accuracy=0.6925`，`macro_f1=0.4525`，`weighted_f1=0.7022`。
- 验证集最佳：`best_val_accuracy=0.7910`。

## 11. 6_gated_fusion

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，创新：门控融合（无显式 coarse/fine 开关覆盖）：
  - `use_gated_fusion=True`，`use_fine=False`（配置中未显式设定 coarse，默认 `use_coarse=True`）。
  - 文本侧 latent（coarse）与 z_img 通过 `GatedFusion` 动态加权融合，作为 transformer 输入。
- 复杂度：参数量 `110,595,024`，FLOPs `18,850,401,280`。
- 测试集结果：
  - `accuracy=0.6970`，`macro_f1=0.4253`，`weighted_f1=0.7056`。
- 验证集最佳：`best_val_accuracy=0.7895`。

## 12. 6_gated_fusion_coarse

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，门控 + coarse：
  - `use_gated_fusion=True`，`use_coarse=True`，`use_fine=False`。
  - z_coarse 与 z_img 经门控融合作为分类输入。
- 复杂度：参数量 `110,595,024`，FLOPs `18,850,401,280`。
- 测试集结果：
  - `accuracy=0.6943`，`macro_f1=0.4141`，`weighted_f1=0.7033`。
- 验证集最佳：`best_val_accuracy=0.7872`。

## 13. 6_gated_fusion_fine

- 模型架构：`FireDamageClassifier`，`method_option=alignment`，门控 + fine：
  - `use_gated_fusion=True`，`use_coarse=False`，`use_fine=True`。
  - z_fine 与 z_img 经门控融合作为分类输入。
- 复杂度：参数量 `185,479,312`，FLOPs `33,874,874,368`。
- 测试集结果：
  - `accuracy=0.6622`，`macro_f1=0.3981`，`weighted_f1=0.6605`。
- 验证集最佳：`best_val_accuracy=0.7074`。

---

## 总体对比简要结论

- 纯图像基线中，`BaselineCNN (resnet101)` 在测试集上表现最好（`acc≈0.73`），且 FLOPs 最低。
- 在方法 A vs B 对比中，`alignment`（2_method_b_alignment）优于 `vae`（2_method_a_vae），说明图像-文本对齐比图像重构更利于分类。
- 使用 coarse 文本辅助通常比仅图像略有提升（如 5_align_fusion_img_coarse vs 5_align_fusion_img_only），但引入 fine 文本和 ViT 文本 encoder 会显著增加参数和 FLOPs，收益有限或不稳定。
- 门控融合在测试集上带来小幅增益（与简单融合相比差异不大），但提供了更灵活的图像/文本信息权衡机制。

