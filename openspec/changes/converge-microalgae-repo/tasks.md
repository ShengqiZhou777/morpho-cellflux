## 1. 止血 — 修 gitignore + 提交 microalgae baseline (a)

- [ ] 1.1 补 `.gitignore`：新增 `data/synthetic_validation/`、根级 `*.png`、`args.json`；判定 `histo.py` 后决定是否加入
- [ ] 1.2 `git status --short` 核对：确认 untracked 从 ~10053 降到可读量级，无生成图待入库
- [ ] 1.3 提交删除块（commit ①）：`baselines/`、`configs/phenoflux_{crispr,diet}*`、`docs/` 旧文档、`phenoflux/models/{msa,pcd}.py`、`phenoflux/eval/{aggregate,figures,moa}.py`、旧 `scripts/`
- [ ] 1.4 提交 microalgae 核心（commit ②）：`phenoflux/args.py`、`models/{__init__,configs,unet}.py`、`training/*`、`train.py`
- [ ] 1.5 提交 eval + configs（commit ③）：`phenoflux/eval/{__init__,fid,morphology,distribution_eval,aggregate_microalgae}.py`、`configs/microalgae_*.yaml`、`configs/README.md`
- [ ] 1.6 提交 scripts + docs（commit ④）：活跃 `scripts/*`、`docs/{DATA,LEGACY}.md`、`data/{README,processed/MANIFEST}.md`、`README.md`、`pyproject.toml`
- [ ] 1.7 verify：`git log --oneline -5` 干净可读；`python scripts/smoke_validate.py` 通过；`git status` 无迁移文件残留

## 2. 维度真相统一到 62

- [ ] 2.1 重命名 `data/processed/microalgae_v1/views/timepoint_512/embedding_61d.csv` → `embedding_62d.csv`
- [ ] 2.2 同步生成脚本输出名（`interpolate_omics_to_timepoints.py` / `build_timegroup_data.py` 中的 `embedding_61d` 字面量）
- [ ] 2.3 同步 config `embedding_path`（`microalgae_timepoint_512_62d.yaml` 及引用该 embedding 的其他 config）
- [ ] 2.4 更新 `CLAUDE.md`：`base_condition_dim` 4 → 62，删除"待扩展"标注，写明 62 = 2+1+1+29+29
- [ ] 2.5 标注/更新 `openspec/changes/microalgae-image-generation-workflow-analysis/proposal.md` 中 61/92 旧说法
- [ ] 2.6 verify：`grep -rn '61d\|base_condition_dim.*4\|61维\|92维' configs/ phenoflux/ CLAUDE.md` 无当前活跃路径命中

## 3. 根目录 scratch 归位

- [ ] 3.1 建 `docs/experiments/collapse_campaign_2026_07/`，移入 `ACTION_PLAN.md`、`EXECUTION_LOG.md`、`MONITORING_GUIDE.md`
- [ ] 3.2 判定 `histo.py`（内容为温度/CN 实验）：`grep -rn 'histo' .` 确认无引用后删除，保留判定记录到 design Open Questions
- [ ] 3.3 处理 `args.json`、`areaRatio.png`：删除或移入 `data/reports/`
- [ ] 3.4 verify：仓库根仅剩标准工程文件（README/CLAUDE/LICENSE/Makefile/pyproject/environment.yml/CITATION 等）

## 4. scripts/ 三分类

- [ ] 4.1 对 28 个脚本逐一 `grep -rn <script>` 全仓（含 CLAUDE.md/README/其他脚本），产出 keep/archive/delete 清单
- [ ] 4.2 archive：历史 campaign 脚本（convergence/monitor/watch/stage2/ablate 等）移入 `archive/legacy_scripts_2026_07/`
- [ ] 4.3 delete：确认无引用的纯 scratch 脚本
- [ ] 4.4 更新残留引用（`scripts/train.sh`、`quick_validate.sh` 等指向已移动脚本的路径）
- [ ] 4.5 verify：`scripts/` 只剩当前路径需要的脚本，`bash scripts/quick_validate.sh` 引用不断

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
