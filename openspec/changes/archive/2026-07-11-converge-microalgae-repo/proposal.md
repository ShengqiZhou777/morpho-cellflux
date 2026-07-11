## Why

microalgae 时代的全部工作（2026-07-04 至今）从未提交，最后一次 commit `28fa0c5`（06-28）仍停在 CRISPR 时代。工作区里叠了 5 波 campaign 的产物：68 个 tracked 改动（-8757 行删除）+ 10053 个 untracked 文件，其中 10004 个在 `data/`（`data/synthetic_validation/` 约 1 万张生成图**未被 gitignore**，`git add -A` 会误提交）。同时条件维度真相在四处漂移（CLAUDE.md=4、proposal=61/92、config=62、embedding 文件名=61d）。在建立干净 baseline 之前，任何增量清理都无 diff 可依。

本 change 先**止血**（修 gitignore + 分块提交迁移，建立 microalgae baseline），再**收敛**（散落文件归位、脚本/配置分类、维度真相统一、过时规划文档退役）。纯仓库整理，不改训练/推理行为，不触碰坍缩问题的科学方法本身。

## What Changes

- **止血（Phase 1）**
  - 修 `.gitignore`：排除 `data/synthetic_validation/` 及根目录生成物（`*.png`、`args.json` 等），确保生成数据不入库。
  - 将 microalgae 迁移**分逻辑块提交**（删除 CRISPR/Diet → microalgae 核心 → eval+configs → scripts+docs），使 HEAD 反映 microalgae 时代。
- **维度真相统一（Phase 2）**：以 62 维为唯一真相源；`embedding_61d.csv` → `embedding_62d.csv`（含生成脚本输出名 + config 路径同步）；CLAUDE.md `base_condition_dim` 4 → 62。
- **根目录归位（Phase 3）**：`ACTION_PLAN.md`/`EXECUTION_LOG.md`/`MONITORING_GUIDE.md` 移入 `docs/experiments/`；`histo.py`/`args.json`/`areaRatio.png` 删除或移入 `data/reports/`。
- **脚本分类（Phase 4）**：28 个脚本按引用关系三分类（keep / archive / delete）。
- **配置收敛（Phase 5）**：8 个 config 收敛到活跃集，其余归档；`configs/README.md` 更新。
- **收尾（Phase 6）**：退役 `CLEANUP_PLAN.md`/`ACTION_PLAN.md` 等过时规划文档；CLAUDE.md 结构章节更新为当前真相。
- **BREAKING**: 无。不修改 `phenoflux/` 运行时行为、训练/推理逻辑、数据契约。

## Capabilities

### New Capabilities
- `repo-structure`: 定义仓库的收敛后契约——单一活跃训练路径、生成数据排除版本控制、已提交的 microalgae baseline、维度真相一致性、历史material隔离。

### Modified Capabilities
<!-- 无：openspec/specs/ 下暂无既有 spec，本 change 不修改既有 requirement。 -->

## Impact

- **配置/文档**：`.gitignore`、根目录散落文件、`scripts/`、`configs/`、`docs/`、`CLAUDE.md`。
- **数据**：`embedding_61d.csv` 重命名（Phase 2）；`data/synthetic_validation/` 转为忽略。
- **git 历史**：新增若干逻辑 commit，建立 microalgae baseline。
- **不受影响**：`phenoflux/` 训练/推理/评估的运行时行为；坍缩问题的科学方法（属另一 change）。
