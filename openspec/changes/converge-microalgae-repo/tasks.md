## 1. 止血 — 修 gitignore + 提交 microalgae baseline (a)

- [x] 1.1 补 `.gitignore`：新增 `data/synthetic_validation/`、根级 `*.png`、`args.json`；判定 `histo.py` 后决定是否加入
- [x] 1.2 `git status --short` 核对：确认 untracked 从 ~10053 降到可读量级，无生成图待入库
- [x] 1.3 提交删除块（commit ①）：`baselines/`、`configs/phenoflux_{crispr,diet}*`、`docs/` 旧文档、`phenoflux/models/{msa,pcd}.py`、`phenoflux/eval/{aggregate,figures,moa}.py`、旧 `scripts/`
- [x] 1.4 提交 microalgae 核心（commit ②）：`phenoflux/args.py`、`models/{__init__,configs,unet}.py`、`training/*`、`train.py`
- [x] 1.5 提交 eval + configs（commit ③）：`phenoflux/eval/{__init__,fid,morphology,distribution_eval,aggregate_microalgae}.py`、`configs/microalgae_*.yaml`、`configs/README.md`
- [x] 1.6 提交 scripts + docs（commit ④）：活跃 `scripts/*`、`docs/{DATA,LEGACY}.md`、`data/{README,processed/MANIFEST}.md`、`README.md`、`pyproject.toml`
- [x] 1.7 verify：`git log --oneline -5` 干净可读；`python scripts/smoke_validate.py` 通过；`git status` 无迁移文件残留

## 2. 维度真相统一到 62

- [x] 2.1 重命名 `data/processed/microalgae_v1/views/timepoint_512/embedding_61d.csv` → `embedding_62d.csv`
- [x] 2.2 同步生成脚本输出名（`interpolate_omics_to_timepoints.py` / `build_timegroup_data.py` 中的 `embedding_61d` 字面量）
- [x] 2.3 同步 config `embedding_path`（`microalgae_timepoint_512_62d.yaml` 及引用该 embedding 的其他 config）
- [x] 2.4 更新 `CLAUDE.md`：`base_condition_dim` 4 → 62，删除"待扩展"标注，写明 62 = 2+1+1+29+29
- [x] 2.5 标注/更新 `openspec/changes/microalgae-image-generation-workflow-analysis/proposal.md` 中 61/92 旧说法
- [x] 2.6 verify：`grep -rn '61d\|base_condition_dim.*4\|61维\|92维' configs/ phenoflux/ CLAUDE.md` 无当前活跃路径命中

## 3. 根目录 scratch 归位

- [x] 3.1 建 `docs/experiments/collapse_campaign_2026_07/`，移入 `ACTION_PLAN.md`、`EXECUTION_LOG.md`、`MONITORING_GUIDE.md`
- [x] 3.2 判定 `histo.py`（温度/CN 实验，无 phenoflux 引用）：非删除，移入 campaign 文件夹保持 untracked，待用户定夺归属
- [x] 3.3 处理 `args.json`、`areaRatio.png`：移入 `data/reports/`（gitignored）
- [x] 3.4 verify：仓库根仅剩标准工程文件（README/CLAUDE/LICENSE/Makefile/pyproject/environment.yml/CITATION 等）

## 4. scripts/ 三分类

- [x] 4.1 对 28 个脚本逐一 `grep -rn <script>` 全仓（含 CLAUDE.md/README/其他脚本），产出 keep/archive/delete 清单
- [x] 4.2 archive：13 个 campaign 脚本（monitor/watch/generate/convergence/ablation/stage2/quick+subset/processed wrapper）移入 `archive/legacy_scripts_2026_07/`
- [x] 4.3 保留决策：field lane 3 脚本 + `sample_microalgae_checkpoint.py` + `verify_signal_strength.py` 留在 scripts/（无纯删除项）
- [x] 4.4 更新残留引用：活跃文件（README/DATA/configs/README/train.sh/quick_validate.sh）无悬空引用，无需改动
- [x] 4.5 verify：`scripts/` 只剩 14 个活跃/复用脚本，`bash scripts/quick_validate.sh` 引用不断

## 5. configs/ 收敛

- [ ] 5.1 判定活跃 config 集（至少 `microalgae_timepoint_512_62d` + `microalgae_smoke`；field lane 去留见 design Open Questions）
- [ ] 5.2 归档/删除冗余 config（`microalgae_timepoint`、`_512`、`_quick` 等重叠项）
- [ ] 5.3 更新 `configs/README.md`：每个保留 config 标注 active/archived + 用途
- [ ] 5.4 verify：`configs/` 每个文件在 README 中有条目

## 6. 收尾验证

- [ ] 6.1 退役过时规划文档：`docs/CLEANUP_PLAN.md`、`ACTION_PLAN.md` 顶部加"RETIRED / superseded by converge-microalgae-repo"标注或移入 experiments 目录
- [ ] 6.2 更新 `CLAUDE.md` 项目结构章节，使之反映收敛后的真实布局
- [ ] 6.3 `openspec validate converge-microalgae-repo --strict` 通过
- [ ] 6.4 最终验证：`git status` 干净；`README → docs/DATA → configs/README → scripts/train.sh` 路径链完整可走通
