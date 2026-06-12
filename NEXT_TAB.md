# NEXT_TAB — 인계 (이어서 작업할 새 탭용)

> 워크플로우: 로컬(이 repo)서 코드 작성→git push, 서버(`explaintab311` env, py3.11)서 pull→실행.
> 서버엔 Claude 없음 → 로컬서 완성해 push. 최신 커밋 = `git log --oneline -1`.
> 읽는 순서: (배경부터면 OVERVIEW.md) → 이 파일 → **PLAN_V2.md(현행 계획)** → **PREREG_V2.md(결정규칙)**
> → RESULTS.md → FINDINGS.md. (PLAN_RESCUE/Q2B_PROPOSAL은 역사 문서.)

## 한 문단 요약 (2026-06-12 대전환)
**외부 감사로 Q2b "구조 ≤ 피처" 음성의 *해석*이 무효화됨**: (i) linear value 훅이 집계 하에
"Δt 피처 1개"로 붕괴(stale-label 보정 표현 불가 — 가설을 검정한 적 없음), (ii) 검색 기판 sub-TabR
(온도·key-proj 없음, train/eval context 불일치, knob 미튜닝), (iii) time_tabr만 직접 시간 피처 미보유.
또한 Claim A의 측정 프레임은 DISDE/WhyShift와 겹쳐 재포지셔닝 필요(인용+적응; 신규성은 "시간축
실증 + support 붕괴로 인한 측정불가" 발견에 있음 — 웹 검증 필요). **계획 전면 갱신 = PLAN_V2.md**:
R0(코드 수정, 완료) → R1(25시드 재검정, 서버) → R2(Claim A 재포지셔닝·Cai&Ye 판결) → R3(정렬·학회).
유효 자산: Phase 0 재현, Q1 PASS, within-overlap concept(+0.132), elec2 val→test 붕괴 발견, 진단 도구.

## V2 인프라 (R0 — 구현 완료, 이 커밋)
- `src/models/tabr.py`: **value_hook {mlp(주)|gate|linear(레거시)}** — mlp/gate는 집계에서 살아남는
  라벨×Δτ 상호작용(zero-init 등가 보존); **learnable τ + √d 스케일링**, **key projection**,
  metric 훅을 **top-k 이전**으로, **arch 5종 {mlp_t, tabr, tabr_t, time_tabr, time_tabr_t}**
  (`_t` = predictor에 τ(t) 직접 concat — 피처 보유 통제).
- `src/training/tabr_trainer.py`: **train ctx = batch+sampled-4096**(배치 제외)/inbatch(레거시),
  **eval ctx = full train**/fixed(레거시, 전용 RNG로 arm 공유), n_classes train∪val, 배치-skip 공통화,
  eval 시 context 1회 인코딩.
- `src/utils/stats.py`: **hedges_g_paired**(d_z·J) 추가 — 시드-페어 비교의 정직한 효과크기.
- `scripts/run_elec2_q2.py`: 5-arm, **val-fair+oracle 병기**, PAIR_PRIORITY(주 대비
  `time_tabr_t−tabr_t`), `--legacy`(pre-V2 정확 재현), `--topk` 등 knob CLI.
- `scripts/run_anchors.py` (신규): lgbm±t / knn±t / **no-change**(persistence) — 외부 보정.
- `scripts/smoke_test_tabr.py`: **선형-붕괴 표현력 가드**(hook이 문서화된 표현력과 다르면 FAIL) +
  init 등가성 테스트. `smoke_test_tabr_trainer.py`: 5-arm + 레거시 경로.

## ★ 다음 행동 — 서버 (순서대로)
```bash
conda activate explaintab311 && cd ~/explainableTabular && git pull
# 0) 배선 검증 (CPU/GPU 무관, 수 분)
python scripts/smoke_test_tabr.py
python scripts/smoke_test_tabr_trainer.py
python scripts/smoke_test_insects.py
# 1) 앵커 (외부 보정; lightgbm 없으면 pip install lightgbm)
python scripts/run_anchors.py --dataset elec2 --split temporal
python scripts/run_anchors.py --dataset insects --insects-variant incremental_balanced
# 2) R1.1 튜닝 단계 (3시드, val로 topk 선택 — PREREG_V2 §2.1; 변종별 1회씩)
for K in 8 32 128; do python scripts/run_elec2_q2.py --dataset insects --report-grid \
  --n-seeds 3 --splits temporal --bases trend --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk $K; done
# 3) R1.2 본 실행 (25시드; PREREG_V2 §2.2 — topk는 2)에서 val로 고른 값)
python scripts/run_elec2_q2.py --dataset insects --insects-variant incremental_balanced \
  --report-grid --n-seeds 25 --splits temporal --bases trend --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk <선택값>
# + abrupt_balanced, incremental_abrupt_balanced 동일. elec2는 보조(동일 커맨드, --dataset elec2).
# 결과: results/phase1/<dataset>_q2/diagnostics.jsonl 를 commit해 로컬로 환류.
```
**판정은 PREREG_V2 §4에 기계적으로** (주 대비 time_tabr_t−tabr_t, val-fair, clean ≥2/3).

## 병렬 작업 (로컬, R2)
- **R2.1 문헌 원문 검증 (웹 권한 필요, 최우선)**: DISDE(arXiv:2303.02011)/WhyShift(NeurIPS'23)/
  Webb'16/Cai&Ye ICML·NeurIPS'25/Drift-Resilient TabPFN + "TabReD에 X·Y|X 분해 적용한 2025-26 선행
  유무". → REFERENCES.md 갱신.
- R2.2 DISDE 퇴화 실험(시점분류기 재사용), R2.3 **Cai&Ye 판결**(LAMDA 코드, concept≈0인
  cooking/maps에서 변조 이득 = X-side 증명), R2.4 합성 2×2 + INSECTS 변종으로 도구킷 검증,
  R2.5 Q1 큰-회전. 상세 = PLAN_V2 §R2.

## 잠긴 설계 포인트 (V2)
- **주 대비 = time_tabr_t − tabr_t** (직접 시간 피처를 양쪽에 — 구조×주입위치 교락 제거). PREREG_V2 §4.
- value_hook 주 분석 = **mlp**, gate는 ablation. linear는 레거시 재현 전용(`--legacy`).
- 판정 = **val-fair** (oracle은 강한형 보조). 통계 = paired CI + Wilcoxon + **hedges_g_paired**.
- 헤드라인 문장에 (시드 수, CI, 데이터셋 수) 병기 — "잠금/확정" 단독 표현 금지.
- pre-V2 결과는 은폐하지 않고 "무효화된 선행 시도"로 부록 보고. `--legacy`로 재현 가능.
- Q1 지표·within-overlap concept 정의 등 기존 잠금(8라운드)은 유지.

## 미커밋 상태
없음(이 커밋에 모두 포함). 다음 탭/서버 = 위 "다음 행동" 그대로.
