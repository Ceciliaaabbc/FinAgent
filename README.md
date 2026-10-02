# FinAgent：大模型驱动的 A 股投研与因子挖掘系统

> 用微调后的大模型分析财经文本、生成情绪因子；让 LLM 自动挖掘量化因子并回测；
> 同时提供一个能查数据、读研报、写分析报告的投研 Agent。

| JD 要求 | 项目中对应的模块 |
|---|---|
| 模型训练（微调、对齐） | 阶段 2：LoRA 微调金融情绪模型 + DPO 对齐 |
| 大模型量化（因子、信号、交易） | 阶段 3–5：情绪因子 + LLM 自动挖掘因子 + LightGBM + 回测 |
| 技术探索（RAG、Agent、OpenClaw） | 阶段 6：研报 RAG + 投研 Agent + OpenClaw 推送 |
| 工程开发（Python/Rust） | 阶段 7：FastAPI + Rust 加速模块 + 前端 + Docker |

## 系统架构

```
数据层：  行情（新浪/东方财富）/ 公告（巨潮资讯）/ 新闻（东方财富）/ 研报 PDF
             │
模型层：  Qwen 小模型 + LoRA 微调 → 文本情绪打分
             │
量化层：  情绪因子 + LLM 生成的因子 → LightGBM → 回测 → 交易信号
             │
应用层：  RAG 研报问答 + LangGraph 投研 Agent（MCP 工具）
             │
服务层：  FastAPI 后端 + Rust 加速模块 + 网页前端 + OpenClaw 每日推送
```

## 进度

- [x] 阶段 0：环境准备
- [x] 阶段 1：数据管道
- [x] 阶段 2：金融情绪模型（代码完成；标注、训练需 API Key 和 GPU）
- [ ] 阶段 3：情绪因子与因子检验
- [ ] 阶段 4：LLM 自动挖掘因子
- [ ] 阶段 5：LightGBM 合成与策略回测
- [ ] 阶段 6：RAG + 投研 Agent
- [ ] 阶段 7：工程化与包装

## 快速开始

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 先用 5 只股票测试整个流程（约 3 分钟）
python -m src.data.run_stage1 --limit 5

# 全量运行（首次约 2–3 小时，主要时间花在下载公告；中断后重新运行会断点续传）
python -m src.data.run_stage1
```

> 注意：AkShare 与 pandas 3 不兼容，`requirements.txt` 已限定 `pandas<3`。

## 项目结构

```
FinAgent/
├── data/                 # 数据（不上传 GitHub，运行脚本生成）
│   ├── raw/              # 原始下载（按股票分文件，支持断点续传）
│   └── processed/        # 合并清洗后的数据和质量报告
├── src/
│   ├── data/             # 阶段 1：数据采集
│   │   ├── config.py               # 路径、时间范围、股票池等配置
│   │   ├── utils.py                # 带重试的请求、交易日历
│   │   ├── fetch_universe.py       # 1.1 股票池
│   │   ├── fetch_prices.py         # 1.2 日线行情
│   │   ├── fetch_index.py          # 1.3 基准指数
│   │   ├── fetch_announcements.py  # 1.4 历史公告（方案 B）
│   │   ├── fetch_news.py           # 1.4 每日新闻（方案 C）
│   │   ├── align_events.py         # 1.5 时间对齐（防未来函数）
│   │   ├── check_data.py           # 1.6 数据质量检查
│   │   └── run_stage1.py           # 一键运行阶段 1
│   ├── sentiment/        # 阶段 2：情绪模型
│   │   ├── config.py               # 路径、标签、模型接口（读取 .env）
│   │   ├── prompts.py              # 2.1 标注规则、老师/学生提示词、输出解析
│   │   ├── keyword_baseline.py     # 关键词词典基线（对照组 / 无 API 时跑通流程）
│   │   ├── llm_client.py           # OpenAI 兼容客户端 + 并发批处理（断点续传）
│   │   ├── events.py               # 读取阶段 1 的文本事件
│   │   ├── sample_events.py        # 2.2 抽样待标注数据
│   │   ├── label_with_api.py       # 2.2 老师模型批量标注
│   │   ├── review.py               # 2.3 人工抽检
│   │   ├── build_dataset.py        # 2.4–2.5 按时间划分 + 转微调格式
│   │   ├── evaluate.py             # 2.7 评估与对比表
│   │   ├── build_dpo.py            # 2.8 构造 DPO 偏好数据
│   │   └── score_events.py         # 2.9 全量打分 → 阶段 3
│   ├── factors/          # 阶段 3–4：因子计算与挖掘
│   ├── backtest/         # 阶段 5：回测
│   ├── agent/            # 阶段 6：RAG + Agent
│   └── api/              # 阶段 7：后端接口
├── notebooks/            # 实验和分析
├── configs/
│   ├── sentiment_lora_sft.yaml     # 2.6 LoRA 微调配置（LLaMA-Factory）
│   └── sentiment_dpo.yaml          # 2.8 DPO 配置
├── outputs/              # 模型权重、评估结果（不上传 GitHub）
├── .env.example          # API 配置模板，复制为 .env 后填写
├── requirements.txt      # 本地依赖
└── requirements-gpu.txt  # GPU 服务器依赖（LLaMA-Factory、vLLM）
```

### 阶段 1 输出的数据

| 文件 | 内容 |
|---|---|
| `data/processed/universe.csv` | 沪深 300 成分股列表 |
| `data/processed/prices.parquet` | 日线行情（前复权）：date, code, open, high, low, close, volume, amount, turnover, pct_chg |
| `data/processed/benchmark.parquet` | 沪深 300 指数日线 |
| `data/processed/announcements_raw.parquet` | 历史公告原始数据 |
| `data/processed/news_raw.parquet` | 每日采集的新闻（持续积累） |
| `data/processed/events.parquet` | 对齐后的文本事件：code, trade_date, publish_time, title, content, source, url |
| `data/processed/data_report.md` | 数据质量报告 |

## 阶段 2 运行方法

整体流程：**本地**抽样、标注、建数据集 → **GPU 服务器**微调、部署 → 评估 → 全量打分。

### 第 1 步：配置老师模型（本地）

推荐先用**本地 Ollama**（免费，不用注册）。`.env.example` 默认就是这个方案：

```bash
ollama pull qwen3.5:9b        # 约 6.6GB，18GB 内存的 Mac 可以流畅运行
cp .env.example .env          # 默认配置即可，无需修改
```

也可以改用 DeepSeek 等云端 API，按 `.env.example` 中方案 B / C 的说明修改即可。

> Qwen3.5 等推理模型默认会先“思考”再回答，既慢又会导致输出为空。`.env` 中的 `TEACHER_REASONING_EFFORT=none` 用于关闭思考。

### 第 2 步：抽样、标注、抽检、建数据集（本地）

```bash
python -m src.sentiment.sample_events --n 5000        # 抽样 5000 条
python -m src.sentiment.label_with_api --limit 50 --workers 4   # 先标 50 条，检查效果
caffeinate -i python -m src.sentiment.label_with_api --workers 4 # 标注全部（防休眠，断点续传）
python -m src.sentiment.review --export --n 200       # 导出抽检表，在 Excel 中填写 human_label
python -m src.sentiment.review --score                # 计算老师模型与人工的一致率
python -m src.sentiment.build_dataset                 # 按时间划分 + 转成微调格式
```

> 没有 API Key 时，可以给 `label_with_api` 加 `--provider keyword`，用关键词基线先跑通流程（标签质量很差，只用于测试）。

### 第 3 步：LoRA 微调（GPU 服务器）

在 AutoDL 等平台租一张 24G 显卡（RTX 4090 / 3090）。项目和模型放在数据盘 `/root/autodl-tmp`（系统盘很小）：

```bash
cd /root/autodl-tmp
git clone https://github.com/Ceciliaaabbc/FinAgent.git && cd FinAgent
# 把本地的 data/llm/sentiment/ 整个目录上传到服务器同一位置（scp 或 JupyterLab 上传）

# LLaMA-Factory 和 vLLM 依赖的 PyTorch 版本不同，分两个环境安装，避免冲突
conda create -n train python=3.11 -y && conda activate train
pip install llamafactory

# 国内服务器下载模型慢时，二选一：
export HF_ENDPOINT=https://hf-mirror.com      # HuggingFace 镜像
# export USE_MODELSCOPE_HUB=1                  # 或改用 ModelScope
export HF_HOME=/root/autodl-tmp/hf_cache      # 模型缓存放在数据盘

llamafactory-cli train configs/sentiment_lora_sft.yaml
# 完成后得到 outputs/qwen2.5-3b-sentiment-lora/（LoRA 补丁 + training_loss.png）
```

### 第 4 步：部署与评估

```bash
# 在服务器上新建部署环境（与训练环境分开）
conda create -n serve python=3.11 -y && conda activate serve
pip install vllm

# 在 tmux 中启动 vLLM，SSH 断开后服务不会停：同时提供原始模型和微调后的模型（LoRA 模块名为 sentiment）
tmux new -s vllm
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 \
    --enable-lora --max-lora-rank 8 \
    --lora-modules sentiment=outputs/qwen2.5-3b-sentiment-lora

# 在本地 Mac 上评估：先建立 SSH 隧道（端口和地址换成 AutoDL 的登录指令），保持窗口不关
#   ssh -CNg -L 8000:127.0.0.1:8000 -p 端口 root@服务器地址
# 再另开一个终端运行：
python -m src.sentiment.evaluate --name keyword --provider keyword
python -m src.sentiment.evaluate --name teacher_api --target teacher
python -m src.sentiment.evaluate --name qwen3b_base --target student --model Qwen/Qwen2.5-3B-Instruct --prompt teacher
python -m src.sentiment.evaluate --name qwen3b_sft  --target student --model sentiment --prompt student
python -m src.sentiment.evaluate --summary            # 生成 outputs/eval/summary.md 对比表
```

**AWQ 4-bit 量化实验（可选）**：把基座换成 `Qwen/Qwen2.5-3B-Instruct-AWQ`，加载同一个 LoRA 补丁重新评估，对比准确率、显存和速度。补丁是在非量化模型上训练的，准确率可能略降，这正是要观察的权衡。

### 第 5 步（加分项）：DPO

```bash
python -m src.sentiment.evaluate --name qwen3b_sft --target student --model sentiment --split train
python -m src.sentiment.build_dpo --preds outputs/eval/preds_qwen3b_sft_train.jsonl
llamafactory-cli train configs/sentiment_dpo.yaml
# 用 --lora-modules sentiment_dpo=outputs/qwen2.5-3b-sentiment-dpo 部署后再评估一次
```

### 第 6 步：全量打分（输出给阶段 3）

```bash
python -m src.sentiment.score_events                      # 用微调模型
python -m src.sentiment.score_events --provider keyword   # 模型还没训练好时，先用关键词基线
# 输出 data/processed/sentiment_scores.parquet
```

### 阶段 2 的两个注意点

- **测试集的标准答案是老师模型的标签**。评估衡量的是“学生学到了老师几成”；老师本身的可靠性看人工抽检的一致率。
- **例行公告比例**：抽样时把例行公告（股东大会、付息、法律意见书等）控制在 30%，训练集中“中性”下采样到最多 50%；验证集和测试集保持原始分布，评估才真实。

---

# 详细实施计划

整个项目分 **8 个阶段**，每一步说明：**做什么 / 为什么 / 怎么做 / 完成标准**。建议严格按顺序做，后面的步骤依赖前面的产出。

```
阶段0 环境 → 阶段1 数据 → 阶段2 情绪模型 → 阶段3 情绪因子
→ 阶段4 LLM挖因子 → 阶段5 LightGBM+回测 → 阶段6 RAG+Agent → 阶段7 工程化
```

## 阶段 0：环境准备（1–2 天）

### 步骤 0.1：建立项目结构
- **做什么**：创建 GitHub 仓库，按功能划分目录（见上方“项目结构”）。
- **为什么**：结构清晰，面试官打开仓库就能看懂；自己后面也不会乱。

### 步骤 0.2：配置 Python 环境
- **做什么**：创建虚拟环境，安装依赖（见“快速开始”）。
- **完成标准**：`import akshare` 不报错。

### 步骤 0.3：准备算力和 API
- **大模型 API**：注册 DeepSeek 或其他模型 API，用来标注数据，成本大约几十元。
- **GPU**：在 AutoDL 等平台按小时租 24G 显卡（RTX 4090 / 3090），只有阶段 2 需要；其他阶段在自己电脑上就能做。

## 阶段 1：数据管道（第 1 周）✅

### 步骤 1.1：获取股票池
- **做什么**：获取沪深 300 成分股列表。
- **为什么**：全市场 5000 只股票数据量太大、新闻稀疏；沪深 300 是大公司，文本多、数据质量好。
- **局限**：用“当前”成分股回测过去，存在**幸存者偏差**（过去被剔除的公司不在名单里），要在报告中注明。

### 步骤 1.2：下载日线行情
- **做什么**：下载每只股票 2020 年至今的开高低收、成交量、成交额、换手率。
- **为什么**：这是计算因子和收益的基础。
- **怎么做**：优先用新浪接口，失败时自动改用东方财富；两者统一成相同的列。使用**前复权**，消除分红送股造成的价格跳空，否则程序会误以为股价暴跌。每只股票单独存文件，支持断点续传。

### 步骤 1.3：下载基准指数
- **做什么**：下载沪深 300 指数日线。
- **为什么**：后面计算“超额收益”时，要拿它当比较对象。

### 步骤 1.4：采集文本数据（整个项目最难的一步）
免费新闻接口**只返回每只股票最近约 10 条新闻**，拿不到多年历史。解决办法：

| 方案 | 做法 | 优缺点 |
|---|---|---|
| A. 公开数据集 | 在 HuggingFace、Kaggle、GitHub 上搜“中文金融新闻”数据集 | 历史长，但需要整理格式 |
| **B. 公告代替新闻（已实现）** | 从巨潮资讯下载上市公司公告（业绩预告、增减持、中标等） | 历史完整、来源权威；只有标题 |
| **C. 每日采集（已实现）** | 每天定时抓取新闻，逐步积累 | 数据干净，但需要时间积累 |

当前用 B 做历史回测，同时用 C 积累新闻。每日采集的定时任务设置方法见 `src/data/fetch_news.py` 文件开头的说明。

### 步骤 1.5：处理时间（防止“未来函数”）
- **做什么**：给每条文本分配一个“可交易日期”。
- **为什么**：A 股 15:00 收盘，收盘后发布的消息当天已经无法交易。如果当天就用，回测相当于“提前知道了消息”，结果会严重虚高。
- **规则**：
  - 交易日 15:00 前发布 → 当天可用
  - 15:00 及之后发布，或在周末、节假日发布 → 下一个交易日可用
  - 只有日期、没有具体时间 → 保守起见，视为盘后发布，下一个交易日可用

### 步骤 1.6：数据检查
- **做什么**：检查缺失值、重复、停牌、价格异常，统计文本覆盖情况，画样本股价格曲线。
- **完成标准**：生成 `data_report.md`，并确认“可交易日早于发布日”的条数为 0。

## 阶段 2：训练金融情绪模型（第 2–3 周）

### 步骤 2.1：设计标注规则
- **做什么**：明确定义什么算利好、利空。
- **为什么**：规则不清晰，标注就会混乱，模型学到的也是混乱的。
```
标签：利好 / 中性 / 利空      强度：1–5 分
"公司中标 20 亿元订单"   → 利好，4
"公司召开股东大会"       → 中性，1
"公司实控人被立案调查"   → 利空，5
```

### 步骤 2.2：用大模型 API 批量标注
- **做什么**：随机抽约 5000 条，让强模型按规则打标签，只输出 JSON。
- **为什么**：人工标注太慢。用强模型标注、再训练小模型，这种做法叫**知识蒸馏**，是业界常见做法。
```python
prompt = f"""你是金融分析师。判断下面信息对该股票的影响。
只输出 JSON：{{"label": "利好/中性/利空", "score": 1-5, "reason": "一句话理由"}}
内容：{title} {content[:500]}"""
```

### 步骤 2.3：人工抽检
- **做什么**：随机抽 200 条，自己判断标注是否正确。
- **为什么**：量化数据质量（如“API 标注准确率 91%”），体现对数据质量的重视。

### 步骤 2.4：划分数据集
- **做什么**：训练集 80%、验证集 10%、测试集 10%，**按时间顺序**划分。
- **为什么**：随机划分会让模型提前看到“未来”的同类文本，测试分数偏高。

### 步骤 2.5：转换成微调格式
```json
{"instruction": "判断这条信息对股票的影响",
 "input": "公司中标20亿元订单……",
 "output": "{\"label\": \"利好\", \"score\": 4}"}
```

### 步骤 2.6：LoRA 微调
- **做什么**：用 LLaMA-Factory 对 Qwen（先用 1.5B/3B，跑通后再试 7B）做 LoRA 微调。
- **为什么**：让小模型学会金融文本判断，部署成本远低于一直调用 API。
```yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
finetuning_type: lora
lora_rank: 8            # 补丁大小，越大能力越强，但越容易过拟合
lora_target: all        # 给哪些层加补丁
learning_rate: 1e-4
num_train_epochs: 3     # 把数据训练 3 遍
```
- **完成标准**：训练 loss 平稳下降，保存得到几十 MB 的 LoRA 补丁文件。

### 步骤 2.7：评估
- **做什么**：在测试集上比较原始 Qwen、微调后 Qwen、大模型 API 三者。

| 模型 | 准确率 | F1 | 每千条成本 | 速度 |
|---|---|---|---|---|
| 原始 Qwen-3B | ? | ? | ~0 | 快 |
| 微调 Qwen-3B | ? | ? | ~0 | 快 |
| 大模型 API | 基准 | 基准 | ¥x | 慢 |

### 步骤 2.8（加分项）：DPO 对齐
- **做什么**：构造“好回答 vs 差回答”偏好对（如有理由的判断 vs 理由和结论矛盾的判断），训练一轮 DPO。
- **为什么**：对应 JD 里的“对齐”。时间紧可以先跳过。

### 步骤 2.9：批量打分
- **做什么**：用 vLLM 部署微调模型，给全部文本打分；顺便试 AWQ 4-bit 量化，记录速度和显存变化。
- **完成标准**：得到表 `code, trade_date, label, score`。

## 阶段 3：情绪因子与因子检验（第 4 周）

### 步骤 3.1：计算情绪因子
- **做什么**：把每只股票每天的多条文本合成一个数字。
- **怎么做**：利好 = +score，利空 = −score，中性 = 0；当日情绪 = 当日分数之和；情绪因子 = 过去 5 天情绪的指数衰减加权和。没有文本的日子记为 0。

### 步骤 3.2：计算未来收益（标签）
```python
df["ret_5d"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1
```
- **为什么**：因子要预测未来，所以要拿“今天的因子”对比“未来的收益”。

### 步骤 3.3：计算 IC
```python
ic = df.groupby("trade_date").apply(
        lambda x: x["factor"].corr(x["ret_5d"], method="spearman"))
print("IC均值:", ic.mean(), " ICIR:", ic.mean() / ic.std())
```
- **判断标准**：RankIC > 0.03 说明有用；ICIR > 0.3 说明比较稳定。

### 步骤 3.4：分层回测
- **做什么**：每天按因子值把股票分成 5 组，画出每组的累计收益曲线。
- **为什么**：好因子的 5 条线应该**整齐地分开**。

### 步骤 3.5：写因子报告
- **完成标准**：notebook 包含 IC 时间序列图、分层收益图和结论。效果一般也要如实记录并分析原因。

## 阶段 4：LLM 自动挖掘因子（第 5 周）

### 步骤 4.1：写因子计算引擎
- **做什么**：输入因子表达式字符串，输出每只股票每天的因子值。
```
ts_mean(x, n)  过去 n 天均值      ts_std(x, n)  过去 n 天标准差
ts_delta(x, n) 与 n 天前的差值    rank(x)       当天横截面排名
示例："rank(ts_delta(close, 5) / ts_std(close, 20))"
```
- **安全提示**：不要用 `eval` 执行 LLM 生成的任意代码，只允许白名单里的算子和字段。

### 步骤 4.2：设计挖掘 prompt
- **做什么**：告诉 LLM 可用的字段和算子，请它提出因子，并说明金融逻辑。
- **为什么**：要求它解释逻辑，可以减少无意义的“凑数”公式。

### 步骤 4.3：实现自动循环
```
for 第 i 轮 in range(50):
    1. LLM 提出 5 个新因子（附带历史最优因子及其 IC 作为参考）
    2. 引擎计算因子值 → 在【训练期】计算 IC
    3. 把“哪些好、哪些差”反馈给 LLM
```

### 步骤 4.4：筛选因子
- **做什么**：保留训练期 IC 高的因子，去掉彼此高度相关（> 0.7）的，最后留约 10 个。

### 步骤 4.5：样本外验证（关键）
- **做什么**：在**从没用过的测试期**重新计算 IC。
- **为什么**：试了几百个公式，总能碰巧找到几个训练期表现好的（**数据挖掘偏差**）。只有测试期依然有效才算真的有用。
- **完成标准**：一张表，列出每个因子的公式、逻辑、训练期 IC 和测试期 IC。

## 阶段 5：LightGBM 合成与策略回测（第 6 周）

### 步骤 5.1：组装特征表
```
trade_date | code | 情绪因子 | LLM因子1..10 | 动量 | 波动率 | ret_5d(标签)
```
每天对因子做标准化（去极值，再转成排名或 z-score），因为不同因子的数值范围差别很大。

### 步骤 5.2：按时间划分
```
训练期：2020–2023    验证期：2024    测试期：2025–2026
```

### 步骤 5.3：训练 LightGBM
```python
import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                          num_leaves=31, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
          callbacks=[lgb.early_stopping(50)])   # 验证集不再提升时自动停止，防止过拟合
```

### 步骤 5.4：查看特征重要性
- **为什么**：回答面试问题“你的情绪因子到底有没有用”。

### 步骤 5.5：生成预测分数
- **做什么**：用模型给测试期内每天、每只股票打分。

### 步骤 5.6：编写回测引擎
```
每周第一个交易日：
  1. 按模型分数排序，选前 K=30 只
  2. 等权重分配资金
  3. 卖出不在名单里的，买入新进入名单的
  4. 扣除成本：买入 0.03% + 卖出 0.08% + 滑点 0.05%
  5. 记录每天的账户净值
```
A 股特有规则：涨停买不进、跌停卖不出；停牌股票无法交易。

### 步骤 5.7：计算评价指标
```python
年化收益 = (最终净值 / 初始净值) ** (252 / 交易天数) - 1
夏普比率 = 日收益均值 / 日收益标准差 * sqrt(252)
最大回撤 = (净值 / 净值的历史最高点 - 1).min()
超额收益 = 策略日收益 - 沪深300日收益，累加得到曲线
```

### 步骤 5.8：做对比实验

| 实验 | 目的 |
|---|---|
| 只用传统因子 | 基准 |
| 传统因子 + 情绪因子 | 看情绪因子有没有增加价值 |
| 传统因子 + 情绪因子 + LLM 因子 | 完整版 |
| 不扣成本 vs 扣成本 | 展示成本的影响 |
| K = 10 / 30 / 50 | 参数敏感性 |

### 步骤 5.9：写回测报告
- **完成标准**：净值曲线、超额收益曲线、指标对比表和结论分析。

## 阶段 6：研报 RAG + 投研 Agent（第 7 周）

### 步骤 6.1：收集研报 PDF
- 下载 30–50 份目标股票的研报或年报 PDF。

### 步骤 6.2：解析 PDF
- 用 MinerU 把 PDF 转成文本和表格，保留页码。研报表格多，普通文本提取会把表格弄乱。

### 步骤 6.3：切分和向量化
- 按段落切成约 500 字的小块，用 bge-m3 转成向量，存入 Chroma。

### 步骤 6.4：检索和重排
- 关键词检索（BM25）+ 向量检索各取 20 段，合并后用 bge-reranker 重排，取前 5 段。两种检索互补，重排进一步提升准确度。

### 步骤 6.5：生成带引用的回答
- 要求 LLM 标注出处（如“[研报3, 第12页]”），保证可追溯，减少编造。

### 步骤 6.6：评估 RAG
- 自建 50 个问题和标准答案，比较“只用向量检索”和“混合检索 + 重排”的准确率。

### 步骤 6.7：封装 Agent 工具（MCP）
```
get_price(code, days)      → 查行情
get_sentiment(code)        → 最近文本情绪（阶段 2）
get_model_score(code)      → LightGBM 评分和排名（阶段 5）
search_reports(question)   → 研报检索（本阶段）
```

### 步骤 6.8：用 LangGraph 构建 Agent
```
用户："分析一下 600519"
Agent：查行情 → 查情绪 → 查模型评分 → 检索研报 → 综合输出报告
```

### 步骤 6.9：接入 OpenClaw
- 写一个 OpenClaw skill，每天 16:00 自动找出情绪变化最大的 5 只股票，生成简报并推送到聊天软件。

## 阶段 7：工程化与包装（第 8 周）

### 步骤 7.1：FastAPI 后端
- 提供 `/sentiment`、`/factors`、`/backtest`、`/chat` 等接口，把脚本变成“系统”。

### 步骤 7.2：Rust 加速模块
- 用 Rust + PyO3 重写一个计算密集的函数（如滚动标准差因子），给出与 pandas 版本的速度对比。

### 步骤 7.3：前端页面
- Streamlit 做三个页面：因子看板、回测结果、Agent 对话。

### 步骤 7.4：Docker 打包
- `docker-compose.yml` 一条命令启动整个系统。

### 步骤 7.5：README 和演示
- 架构图、各阶段核心结果、运行方法、数据局限说明，外加 3 分钟演示视频。

## 进度检查表

| 周 | 阶段 | 完成标志 |
|---|---|---|
| 1 | 0 + 1 | 行情和文本数据入库 |
| 2–3 | 2 | 微调模型的评估对比表 |
| 4 | 3 | 情绪因子的 IC 和分层报告 |
| 5 | 4 | LLM 因子表（含测试期 IC） |
| 6 | 5 | 回测净值曲线 + 对比实验 |
| 7 | 6 | Agent 能回答问题，OpenClaw 能推送 |
| 8 | 7 | GitHub 仓库 + README + 演示视频 |

**时间紧的最短路线（约 3 周）**：阶段 1 → 阶段 2（只做 2.1–2.7）→ 阶段 3 → 简化版阶段 5。

## 三条贯穿全程的原则

1. **防止未来函数**：每一步都问自己，“在那一天，我真的能知道这个信息吗？”
2. **样本外验证**：所有结论都要在没参与训练和挑选的数据上验证。
3. **如实记录**：效果不好也要记录并分析原因。研究过程是否严谨，比收益数字更重要。
