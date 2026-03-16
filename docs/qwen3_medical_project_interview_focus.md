# Qwen3-4B × MedicalGPT 秋招项目：可执行全链路方案（含 A/B/C/F 三轮 SFT）

> 目标：在 `2×4090(24G)` 条件下，做出一个**真实可跑通**、可用于秋招面试的医疗大模型项目。方案按“能落地优先”，不做不切实际的 SOTA 承诺。

---

## 1. 项目总目标（你最终对外可讲）

1. 基于 `Qwen3-4B` 完成医疗 SFT 主线。
2. 在 SFT 上做“知识内化增强”（A/B/C 配比二轮 + badcase 三轮回灌）。
3. 做 RAG，对比“训练增益 vs 检索增益”。
4. 做 ORPO（主）+ DPO（小规模对照）。
5. 输出 API + Demo + 实验报告 + 面试 Q&A。

### 保守效果预期（真实口径）
- C-Eval 医疗子集：SFT 相比 Base 提升 `+2 ~ +7`。
- SFT+RAG 相比 SFT：事实型问题再提升 `+1 ~ +4`。
- ORPO 相比 SFT：高风险拒答率 `+8% ~ +18%`，且尽量将可答率损失控制在 `<=5%`。

---

## 2. 数据资源盘点与取舍（结合你给的数据）

## 2.1 可用大数据源

你给出的主数据：`train_zh_0.json`，约 195 万条，来自：
1. `Toyhom/Chinese-medical-dialogue-data` 六科室问诊数据（约 79 万）
2. `huatuo_encyclopedia_qa`（约 36 万）
3. `huatuo_knowledge_graph_qa`（约 79 万）

另外可用：`FreedomIntelligence/HuatuoGPT-sft-data-v1`。

## 2.2 资源受限下的现实取舍

你明确说“没有充足大模型数据改写/生成资源”，因此方案改成：
- **不依赖大规模 LLM 改写生成**。
- 主要依赖：规则抽取 + 模板重写 + 小规模人工抽检。
- 训练样本首轮控制在 `60k~90k`，保证你能稳定跑通并做消融。

---

## 3. SFT 数据工程（你指定的 A/B/C/F 池方案）

## 3.1 池定义总览

- A池：通用医疗问答基础池（学习“会答”）
- B池：知识单元内化池（学习“答得准”）
- C池：安全边界池（学习“答得稳、安全”）
- F池：badcase 回灌池（学习“修复具体错误”）

## 3.2 第一轮 SFT：A池主训

### 目标
先把模型从“能聊”拉到“能做医疗问答基础回答”。

### 配比
- `A:C = 9:1`

### A池格式（MedicalGPT原生）
```json
{"conversations":[
  {"from":"human","value":"高血压的诊断标准是什么？"},
  {"from":"gpt","value":"高血压诊断标准为非同日3次测量，收缩压≥140mmHg和/或舒张压≥90mmHg。"}
]}
```

### A池增强格式（建议采用，保留 system + meta）
```json
{
  "system": "你是一名谨慎的中文医疗助手。你可以提供医学常识，但不能替代医生诊断。",
  "conversations": [
    {"from": "human", "value": "布洛芬和对乙酰氨基酚有什么区别？"},
    {"from": "gpt", "value": "两者都可用于退热止痛，但作用特点、适用人群和注意事项不同。对乙酰氨基酚更常用于发热和轻中度疼痛；布洛芬属于非甾体抗炎药，还具有抗炎作用，但胃肠道刺激风险相对更高。具体用药应结合年龄、肝肾功能、既往病史和说明书。"}
  ],
  "meta": {
    "source": "medical_qa",
    "subject": "药学",
    "sample_type": "drug_compare",
    "risk_level": "medium"
  }
}
```

### 目录规范
```text
project_data/
  sft/
    round1_a_c_train.jsonl
    round1_valid.jsonl
  logs/
    cleaning_report.md
```

## 3.3 第二轮 SFT：知识内化增强（A:B:C = 2:7:1）

> 你的想法可实现，而且这是本项目最有“算法味”的改进点。

### 核心思想
不只是学习泛化回答，而是把高置信知识点“压入参数”。

### B池构造（知识单元 -> 重写回答）
你问的关键点是“到底用哪个数据集 + 怎么构造三类样本”，这里给可直接执行版本。

#### 使用数据集与来源字段（不依赖大模型改写）

1) `train_zh_0.json`（主来源）
- 子来源 a：`huatuo_knowledge_graph_qa`（优先做 B1/B3）
- 子来源 b：`huatuo_encyclopedia_qa`（优先做 B1/B2）
- 子来源 c：`Toyhom/Chinese-medical-dialogue-data`（优先做 B2/B3）

2) `FreedomIntelligence/HuatuoGPT-sft-data-v1`（补充来源）
- 用于补齐药学注意事项与场景化描述不足的条目。

#### 三类样本如何构造

- B1 定义/阈值类（8k）
  - 主要从 `knowledge_graph_qa` 和 `encyclopedia_qa` 抽取。
  - 规则关键词：`定义/标准/分期/阈值/正常范围/诊断标准/适应症/禁忌症`。
  - 过滤条件：答案中至少出现一个“数值或边界条件”（如 `>=`, `mg`, `岁`, `次/日`）。

- B2 症状-原因-行动类（8k）
  - 主要从 `Toyhom` 问诊对话抽取，辅以 `encyclopedia_qa`。
  - 结构要求：答案可切分为“症状描述 -> 可能原因 -> 建议行动/就医建议”三段中的至少两段。
  - 过滤条件：包含行动动词（如“观察/就医/复查/急诊”）。

- B3 药学注意事项类（8k）
  - 主要从 `knowledge_graph_qa`（药物关系）和 `HuatuoGPT-sft-data-v1` 抽取。
  - 规则关键词：`用法用量/不良反应/相互作用/禁忌/慎用/儿童/孕妇/肝肾功能`。
  - 过滤条件：答案含至少一条“风险提示或禁忌信息”。

合计 B池建议：`24k`。

#### 可执行抽样配额（建议）
- B1：`knowledge_graph_qa 5k + encyclopedia_qa 3k`
- B2：`Toyhom 6k + encyclopedia_qa 2k`
- B3：`knowledge_graph_qa 4k + HuatuoGPT-sft-data-v1 4k`

### B池重写样式（无需大模型大规模改写）
用规则模板生成 4 种回答结构（同一条知识最多重写 2 个版本，避免过拟合模板）：

1. 定义式（适合 B1）
- 模板：`[概念] 的定义/标准是 [核心条件]。若 [边界条件]，通常归为 [类别]。`

2. 阈值式（适合 B1/B3）
- 模板：`关键阈值：1) [阈值A]；2) [阈值B]。超过阈值时建议 [行动]。`

3. 场景式（适合 B2）
- 模板：`若出现 [症状场景]，常见原因包括 [原因列表]。建议先 [家庭处理]，若 [危险信号] 请及时就医。`

4. 对比式（适合 B3）
- 模板：`[药物A] 与 [药物B] 均可用于 [适应症]；区别在于 [机制/风险]。对 [特殊人群] 应优先考虑 [策略]。`

#### 重写后的质量门槛（自动+人工）
- 自动规则：
  - 长度 40~220 中文字；
  - 不得出现“100%治愈/绝对安全”等绝对化词；
  - 至少包含 1 个医学实体（疾病/药物/指标）。
- 人工抽检：每 1000 条抽 50 条，目标合格率 `>=92%`。

### 第二轮总量建议（可跑通）
- A：`8k`
- B：`28k`
- C：`4k`
- 合计：`40k`（按 `2:7:1`）

## 3.4 第三轮 SFT：badcase 回灌（F池 + 少量C + 少量B）

### 你的设想是否可行？
可行，而且非常适合面试讲“闭环优化”。

### 建议配比
- `F:C:B = 7:2:1`

### 样本规模（轻量）
- F池：`3.5k`
- C池：`1k`
- B池：`0.5k`
- 合计：`5k`

### F池来源
从以下评测中采集失败样本：
- C-Eval 医疗子集错题
- 自建开放问答集 badcase
- RAG 误检索导致的错误答案

记录字段建议：
- `error_type`（幻觉/答非所问/过度诊断/漏关键信息）
- `fix_strategy`（补知识点/改模板/加边界）

---

## 4. 数据处理细节（不依赖重型数据生成）

## 4.1 清洗规则（适用于 195万大池）

1. 去重：exact + simhash
2. 低质过滤：空答、极短答、乱码
3. 风险过滤：鼓励危险行为、明确替代医生诊断
4. 冲突处理：同问题冲突答案按“指南/权威表达优先”

## 4.2 子集抽样策略（首轮建议）

从 195万中抽样：
- Round1（A+C）：`60k`
- Round2（A+B+C）：`40k`
- Round3（F+C+B）：`5k`

总训练样本（去重后）约 `95k ~ 105k`。

## 4.3 验证集

- 固定验证集：`2k`（全程不变，便于横向比较）
- 每轮训练只替换 train，不替换 valid。

---

## 5. 训练参数（2×4090 可跑）

## 5.1 第一轮 SFT（A:C=9:1）

- `model_max_length=2048`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-5`
- `num_train_epochs=2`
- `qlora=True, load_in_4bit=True`
- `bf16=True`
- `target_modules=q_proj,v_proj`
- `lora_rank=16, lora_alpha=32, lora_dropout=0.05`

预估：
- 显存：`18~23GB/GPU`
- 时长：`6~12h`

## 5.2 第二轮 SFT（A:B:C=2:7:1）

- 在第一轮最佳 checkpoint 上继续训练
- `learning_rate`降到 `1e-5`
- `epochs=1~1.5`
- 其余参数不变

预估：
- 显存：`18~23GB`
- 时长：`4~8h`

## 5.3 第三轮 SFT（F回灌）

- `learning_rate=8e-6`
- `epochs=1`
- 小数据集，防止灾难性遗忘

预估：
- 显存：`16~22GB`
- 时长：`1~3h`

---

## 6. 对齐训练（ORPO主线，DPO对照）

## 6.1 ORPO（主）

- 偏好对：`6k~10k`
- 长度：`max_source=1024, max_target=512`
- beta：`0.05/0.1/0.3` 消融

预估：
- 显存：`16~22GB`
- 时长：`4~8h`

## 6.2 DPO（小规模对照）

- 偏好对：`2k~4k`
- 只做“可回答面试追问”的对照，不追最优

预估：
- 显存：`20~24GB`（更紧张）
- 时长：`3~6h`

---

## 7. RAG 完整落地方案

## 7.1 知识库数据

来源：
- `shibing624/medical`
- `HuatuoGPT-sft-data-v1`

目标：每行一个知识点，生成 `kb_facts.jsonl`。

字段：
```json
{"knowledge_id":"km_002","subject":"内科学","fact_text":"...","source":"..."}
```

规模建议：`80k~150k` facts。

## 7.2 检索配置

- Embedding：bge 中文向量模型
- FAISS：IndexFlatIP（首版）
- Top-k：`5`（消融 3/5/10）
- rerank：第二阶段再加（若时间紧可先不做）

## 7.3 防泄题（必须）

1. C-Eval 题干正则过滤
2. 与 C-Eval 题干 n-gram 重合过滤（4-gram > 0.25 剔除）

## 7.4 RAG推理输出结构

```json
{
  "answer":"...",
  "evidence":["..."],
  "knowledge_id":["km_002"],
  "risk_flag":"medium"
}
```

---

## 8. 评测设计（真实且可执行）

## 8.1 四组关键实验（固定命名）

1. `only_inference_base`
2. `inference_with_rag`
3. `inference_after_sft`
4. `inference_after_sft_with_rag`

解释：
- (2)-(1)：纯检索增益
- (3)-(1)：纯训练增益
- (4)-(3)：训练后检索增益
- (3) > (2)：出现知识内化迹象

## 8.2 C-Eval（主评测）

医疗子集：
- `basic_medicine`
- `clinical_medicine`
- `traditional_chinese_medicine`

通用对照：任选2科。

指标：Accuracy。

## 8.3 开放问答安全评测集（辅）

- 自建 `200~500` 条
- 类型：常识/用药/诊断/急症/分诊

指标：
- `RefusalRate_highrisk`
- `AnswerableRate_normal`
- `HallucinationRate`
- `NotFound@k`
- `EvidenceHit@k`

---

## 9. 消融实验（你面试最加分）

1. LoRA rank：`8/16/32`
2. target_modules：`q,v` vs `q,k,v,o`
3. ORPO/DPO beta：`0.05/0.1/0.3`
4. RAG top-k：`3/5/10`
5. SFT轮次对比：
   - 仅一轮A池
   - 两轮(A+B+C)
   - 三轮(+F回灌)

---

## 10. 部署方案（可演示）

## 10.1 架构
- 模型层：Qwen3-4B + LoRA adapter/merge
- 检索层：FAISS + embedding
- API层：`fastapi_server_demo.py` 或 `openai_api.py`
- 前端：`gradio_demo.py`

## 10.2 最小可交付接口
- `/chat`
- `/chat_rag`

返回：answer + evidence + knowledge_id + risk_flag。

---

## 11. 进度计划（4周）

### Week1
- 195万数据清洗 + 子集抽样
- Round1 训练 + 两组基线推理

### Week2
- Round2（A:B:C=2:7:1）
- C-Eval 主评测

### Week3
- Round3（F回灌）+ ORPO
- RAG 联调 + 四组实验跑齐

### Week4
- 消融整理 + 部署 + 报告 + 面试材料

---

## 12. 时间来不及时的“保底版”

如果只剩 7~10 天：
1. 必做：Round1 + 四组实验中的前三组
2. 次优：Round2 缩小到 15k
3. ORPO 缩到 2k 偏好对
4. 部署保留 FastAPI 单接口

这样仍可产出：
- 可验证增益
- 数据处理闭环
- RAG 实证
- 可演示系统

---

## 13. 面试表达要点（按你项目定制）

1. **为什么这样分池？**
- A 保会答，B 保准确内化，C 保安全边界，F 保迭代修复。

2. **为什么三轮SFT不是一轮？**
- 分阶段把“能力、知识、安全、稳定性”拆开训练，更可控且便于做消融归因。

3. **没有大规模改写资源怎么办？**
- 用规则模板与抽检替代重型生成，保证可执行性。

4. **如何证明你真做了实验？**
- 四组主实验 + 五组消融 + badcase 回灌记录。

---

## 14. 可直接写进简历的描述（短版）

“基于 Qwen3-4B 在 2×RTX4090 上完成医疗大模型全链路训练与部署：设计 A/B/C/F 分池三轮 SFT（A:C=9:1，A:B:C=2:7:1，F回灌），结合 ORPO 与 RAG 完成四组对照实验，评测覆盖 C-Eval 医疗子集与自建安全集，最终实现 FastAPI+Gradio 可演示系统，并通过消融与 badcase 复盘验证训练增益、检索增益与安全边界改进。”

