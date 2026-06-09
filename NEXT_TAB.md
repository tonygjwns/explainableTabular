# NEXT_TAB — 인계 (이어서 작업할 새 탭용)

> 워크플로우: 로컬(이 repo)서 코드 작성→git push, 서버(`explaintab311` env, py3.11)서 pull→실행.
> 서버엔 Claude 없음 → 로컬서 완성해 push. 최신 커밋 = `git log --oneline -1` (작성 시점 `40d3d13`).
> 읽는 순서: 이 파일 → RESULTS.md → FINDINGS.md → PLAN_RESCUE.md → Q2B_PROPOSAL.md → REVIEW.md.

## 한 문단 요약
TabM 재현(Phase0) 8/8 PASS. 제안 메커니즘(시간-인덱싱 메모리+검색)은 **TabReD에서 성능 이득 0**
(합성 positive control로 *구현은 정확*함을 입증 → 음성은 신뢰 가능). 진단: TabReD는 **강한 covariate
드리프트**라 concept이 *측정 불가/착취 불가*. **그러나 Elec2는 within-overlap에서 *측정·확정된 concept
(+0.132 AUC, OOF p, p-층 [+0.12,+0.17] 안정)***. 학습-프로토타입 메모리는 §4.2(학습 V_k=정보없음)로
시간-피처를 못 넘음(프록시 확인). → **유일 레버 = 인스턴스+라벨 검색(TabR식)**. 현재 **Q2b 인프라 빌드 중**.

## 게이트 상태 (사전등록 PLAN_RESCUE §E)
- **Q1 충실성 PASS**: `run_q1_faithfulness` recovery 0.991 (10/10) vs ceiling 0.990/floor 0.894.
  (게이트 OK; 헤드라인용이면 큰-회전 robustness 1회 남음.)
- **F3/within-overlap**: Elec2만 measurable+concept(+0.132). 고-covariate 5개 측정불가, cooking/maps concept~0.
- → Q2b OPEN(정당). Q2 = "**구조(time-TabR)가 시간-피처를 측정된 concept(elec2)에서 넘는가**".

## 지금까지 빌드된 Q2b 인프라 (커밋됨)
- `src/models/tabr.py`:
  - `TimeTabR`(검색 코어): top-k, value=이웃 **실라벨**, **(t_q,t_i,y_i) 전부 노출**,
    `time_mode∈{none,metric,value,both}`, 변조항 **zero-init**(시작 ≈ TabR).
  - `TimeTabRModel`(3-arm): 공유 MLP 인코더 + `arch∈{mlp_t, tabr, time_tabr}`.
    caller가 context set(features,t,y) 공급 — 학습=in-batch(exclude_self), eval=고정 sample.
- `scripts/smoke_test_tabr.py`: 서버 통과(shape/grad/time-mod grad/exclude_self).

## ★ 다음 행동 — **서버에서 Q2b 요인설계 실행** (코딩은 완료, push됨)
인프라/러너 코딩 완료(아래 ✅). 다음 탭/서버는 **실행 → 결과 해석**.
- ✅ **`src/training/tabr_trainer.py`** (`train_timetabr(data, cfg)` + `TabRConfig`):
  - 피처 = `_prep_numeric`(quantile X_num+X_bin) + **cat one-hot**(글로벌 cardinality) → flat 인코더 입력.
  - **학습** = in-batch retrieval(context=같은 배치, `exclude_self=arange`, <2면 skip). loss=CE/MSE, L_smooth/LB 없음.
  - **eval** = 고정 context(train에서 `eval_context_size`=4096 sample, 매 evaluate마다 no-grad 재인코딩), 쿼리 배치별 검색.
  - 조기종료/시드/best_state = `phase1_trainer` 패턴.
- ✅ **`scripts/run_elec2_q2.py`**: 요인설계 `{mlp_t,tabr,time_tabr}×{trend,fourier}×{temporal,random}×seed`,
  공유 인코더+기저, 사전등록 대비 = **time_tabr vs mlp_t**(기저정합 구조 vs 피처) & **time_tabr vs tabr**(시간 훅 이득),
  paired Wilcoxon+Hedges' g(`positive`=Δ>0∧p<.05∧g≥.5). 결과 `results/phase1/elec2_q2/q2_<mode>.json`.
- ✅ **`scripts/smoke_test_tabr_trainer.py`**: 합성(시간에 따라 라벨 flip) CPU에서 3-arm 학습 배선 검증.

**서버 실행(권장 순서)**:
```
python scripts/smoke_test_tabr_trainer.py                              # 러너 배선(빠름)
python scripts/run_elec2_q2.py --config configs/phase1.yaml --n-seeds 25
```
빠른 사전점검은 `--n-seeds 3 --splits temporal --bases trend`.

**F2 변조 최종화는 정렬 후**: 기본 `time_mode='value'`(value-side 라벨-drift 보정 권고).
`--time-mode value`/`metric`/`both` 각각 끄는 **2단계 ablation**으로 어느 항이 이득을 나르는지(=concept 착취 vs covariate 적응).

## 사전등록된 해석 규칙 (결과에 안 휘둘리게)
- 구조 이득은 **temporal split**에 나타나야(early→late stale-label 존재), **random엔 ~0**
  (train이 양 기간 봄). random에서만 이득 → 적신호(concept 착취 아님).
- **이득 국소화**: 구조 이득이 concept 큰 영역(late-overlap)에 집중 → 기전 증거.
- **결정규칙**: time-TabR(trend, value) > 기저정합 time-feature, **≥2 데이터셋** temporal, 잠근(g,p) →
  구조 우위(좁은 성능 부활). 아니면 → "측정된 concept 위 음성"(강한 (다)). ≥2 둘째 = elec2 후 Insects(약화 금지).
- **EV/헤드라인**: 인스턴스도 in-dist redundancy 벽 → 완전 성공도 *좁은* 양성. 헤드라인은 **§6(다)**.

## 서버 검증 명령 (재현/확인)
```
conda activate explaintab311 && cd ~/explainableTabular && git pull
python scripts/smoke_test_tabr.py                                   # 인프라 배선
python scripts/run_q1_faithfulness.py                               # Q1 게이트 (PASS 재현)
python scripts/run_concept_overlap.py --config configs/phase1.yaml --all --elec2   # elec2 +0.132 재현
```
(전처리는 `tabred` env + `tabred-env.yaml`; SETUP.md §4 / CLAUDE.md gotchas.)

## 잠긴 설계 포인트 (리뷰 8라운드 — 바꾸지 말 것)
- Q1 지표 = **게이지-고정 per-t cos(ŵ,w), Procrustes 금지**; θ₁ 상대정의; CI는 **시드-간**.
- concept = **고정 late-overlap-test 위 전이격차**(난이도 아님), **OOF p** 영역선택, **p-층 안정**.
- F2 = 시간을 **value-side(라벨 drift 보정)**에 (metric-side만은 covariate 적응=피처와 중복).
- 프로토타입 메모리 경로는 **죽음**(학습 V_k). Insects는 **elec2 결과 후**(river+multiclass).
- 정렬-우선: Q2b는 최대 빌드 + F2가 핵심 novelty 분기 → 지도교수 결정 받고 변조 최종화.

## 미커밋 상태
없음(모두 push됨). Q2b 코딩 완료 → **다음 탭/서버는 `run_elec2_q2.py` 실행·해석**.
