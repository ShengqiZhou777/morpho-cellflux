## Context

仓库经历 5 层"考古地层"，中间无任何 commit 边界：

```
最后 commit (06-28, CRISPR) ──────────────▶ 现在 (07-11)
  L1 CRISPR/Diet (已删, 归档 archive/legacy_20260705/)
  L2 07-04/05  microalgae 迁移 + CLEANUP_PLAN/DATA/LEGACY
  L3 07-07     恒等映射调试 (ACTION_PLAN/EXECUTION_LOG/MONITORING_GUIDE)
  L4 07-08/09  Stage2 组学条件 (62d config, distribution_eval, morphology)
  L5 07-10/11  信号验证 scratch (histo.py, synthetic_validation ~10k 图)
```

当前状态：`git diff --stat` = 68 files, -8757 行；untracked = 10053（10004 在 data/）；`data/synthetic_validation/` 不在 `.gitignore`。维度真相漂移：CLAUDE.md=4 / proposal=61/92 / config=62 / 文件名=61d。memory 显示 L4-L5 是"活的"科学工作（62 维组学条件解坍缩），L1-L3 大多是已完成/过时残渣。

## Goals / Non-Goals

**Goals:**
- 建立一个已提交的、反映 microalgae 时代的干净 baseline，使后续 `git diff` 可用。
- 生成数据（synthetic_validation、png、临时 json）确定性地排除出版本控制。
- 条件维度真相在 config / 文档 / 数据文件名三处一致（= 62）。
- 收敛到"一条明显路径"：`README → docs/DATA → configs/README → scripts/train.sh`。
- 历史与 scratch material 隔离到 `archive/` 或 `docs/`，根目录只剩标准工程文件。

**Non-Goals:**
- 不修复恒等映射坍缩问题本身（科学方法属独立 change）。
- 不修改 `phenoflux/` 的训练/推理/评估运行时逻辑。
- 不删除 `archive/legacy_20260705/` 的 provenance material。
- 不重写训练历史（不 rebase 06-28 之前的 commit）。

## Decisions

**D1: 分逻辑块提交，而非单个巨型 commit**
- 选择：按语义拆 4 个 commit（① 删除 CRISPR/Diet/baselines ② microalgae 核心 phenoflux/models+training ③ eval + configs ④ scripts + docs）。
- 理由：-8757 行删除墙 + 万级新增若挤进一个 commit 无法 review、无法回滚单一维度。
- 备选（否决）：`git add -A && commit`——会连带 10k 生成图，且历史不可读。

**D2: gitignore 先行，再 add**
- 先补 `.gitignore`（`data/synthetic_validation/`、根 `*.png`、`args.json`、`histo.py` 若判定为 scratch），再任何 `git add`。
- 理由：避免 10k 生成物在 add 阶段误入索引；先修则 `git status` 立即变干净可读。

**D3: 脚本三分类基于引用关系**
- keep（当前路径引用）/ archive（历史 campaign，移 `archive/legacy_scripts_2026_07/`）/ delete（纯 scratch，无引用）。
- 分类前对每个脚本 `grep -r` 其被引用情况，避免误删活脚本。

**D4: 维度真相源 = 62**
- `embedding_61d.csv` → `embedding_62d.csv`，生成脚本输出名 + config `embedding_path` 同一 commit 内同步改。
- 理由：61 是 ACTION_PLAN 阶段的旧目标值，实际实现为 62（2+1+1+29+29）；文件名是唯一还带 61 的残留。

**D5: 根目录 scratch 归位规则**
- 叙事型（ACTION_PLAN/EXECUTION_LOG/MONITORING_GUIDE）→ `docs/experiments/collapse_campaign_2026_07/`（保留 provenance）。
- 数据/图/临时 json（histo.py/args.json/areaRatio.png）→ 删除或移 `data/reports/`（`data/reports/*.json` 已被忽略）。

## Risks / Trade-offs

- **误删还活着的脚本** → 分类前 `grep -r <script>` 全仓确认无引用再动；archive 优先于 delete。
- **重命名 embedding 破坏 config 路径** → 重命名与 config 修改放同一 commit，之后跑 `smoke_validate.py` 确认路径没断。
- **提交生成物** → D2 gitignore 先行 + 每个 commit 前 `git status` 人工核对文件清单。
- **histo.py 归属不明**（内容是温度/CN 实验，疑似跨项目 scratch）→ Phase 3 判定：若确无 phenoflux 引用则删除，保留判断记录。
- **过度归档丢失当前工作** → L4-L5 是活的，triage 时默认保留，只归档明确属 L1-L3 的产物。

## Migration Plan

1. 按 tasks.md Phase 顺序执行（1 止血 → 2 维度 → 3 根目录 → 4 脚本 → 5 配置 → 6 收尾）。
2. 每个 commit 前 `git status --short` 核对，确认无生成物。
3. Phase 2 后跑 `python scripts/smoke_validate.py` 确认 config 路径完整。
4. 回滚策略：每 Phase 独立 commit，出错 `git reset --soft HEAD~1` 退回，不影响已提交的前序 Phase。
5. 结束跑 `openspec validate converge-microalgae-repo --strict`。

## Open Questions

- **field lane 去留**：`microalgae_field.yaml` + field 脚本是否保留为第二活跃路径，还是归档为"未来实验"？（影响 Phase 5 config 集与 Phase 4 脚本分类）
- **synthetic_validation 归属**：保留在 `data/`（忽略）还是移到 `outputs/`（已忽略）作为评估产物？
- **commit 拆分粒度**：4 个逻辑 commit 是否够细，还是删除块需再拆（CRISPR vs Diet vs baselines）？
