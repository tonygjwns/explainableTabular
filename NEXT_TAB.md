# NEXT_TAB — 인계 (이어서 작업할 새 탭용)

> 워크플로우: 로컬(이 repo)서 코드 작성→git push, 서버(`explaintab311` env, py3.11)서 pull→실행.
> 서버엔 Claude 없음 → 로컬서 완성해 push. 최신 커밋 = `git log --oneline -1` (작성 시점 `40d3d13`).
> 읽는 순서: (배경부터면 OVERVIEW.md) → 이 파일 → RESULTS.md → FINDINGS.md → PLAN_RESCUE.md → Q2B_PROPOSAL.md → REVIEW.md.

## 한 문단 요약 (2026-06-09 갱신)
TabM 재현 8/8. TabReD=covariate(concept 측정불가). Elec2=within-overlap 측정·확정 concept(+0.132).
**Q2b 인스턴스 time-TabR 실행·판정 완료 → 구조 청구 = 견고한 음성**: elec2(temporal, 10시드, 정규화+min_epochs로
메커니즘 engage 보장)에서 **mlp_t(시간=피처) 0.9027 > tabr 0.8955 > time_tabr 0.8848**. 시간은 도움(mlp_t−tabr +0.007)
이나 **구조가 피처를 못 넘고(−0.018) 불안정(std .05~.075)**. ①train_loss 감소+temporal조기/random늦은 peak로 **drift 확정(버그 아님)**.
→ §6(다) "측정된 concept 위 음성". **≥2 데이터셋 충족: Insects(designed drift, multiclass)도 구조 음성**
(time_tabr−mlp_t=−0.011; mlp_t−tabr=+0.038로 시간은 매우 중요·구조만 못 넘음; val→test ρ=0.87로 elec2 val붕괴는 elec2 고유).
→ **사전등록 "구조 우위" 미충족 = §6(다) 음성/분석 paper 확정, A' method 경로 닫힘. 남은 건 지도교수 정렬.**
(RESULTS §10·§11, FINDINGS "Q2b ANSWERED" 참조.)

## 게이트 상태 (사전등록 PLAN_RESCUE §E)
- **Q1 충실성 PASS**: `run_q1_faithfulness` recovery 0.991 (10/10) vs ceiling 0.990/floor 0.894.
- **F3/within-overlap**: Elec2만 measurable+concept(+0.132). 고-covariate 5개 측정불가, cooking/maps concept~0.
- **Q2b(elec2) = 음성 확정**: 구조(time-TabR) ≤ 시간-피처. "메커니즘 미engage" 위협 제거(min_epochs로 학습시켜도 음성 유지).
- → 결정규칙 "≥2 데이터셋" 미충족 → **Insects가 결정타**(비-trivial concept서도 음성이면 §6(다) 확정 / 양성이면 elec2의 trivial-착취 한정).

## 지금까지 빌드된 Q2b 인프라 (커밋됨)
- `src/models/tabr.py`:
  - `TimeTabR`(검색 코어): top-k, value=이웃 **실라벨**, **(t_q,t_i,y_i) 전부 노출**,
    `time_mode∈{none,metric,value,both}`, 변조항 **zero-init**(시작 ≈ TabR).
  - `TimeTabRModel`(3-arm): 공유 MLP 인코더 + `arch∈{mlp_t, tabr, time_tabr}`.
    caller가 context set(features,t,y) 공급 — 학습=in-batch(exclude_self), eval=고정 sample.
- `scripts/smoke_test_tabr.py`: 서버 통과(shape/grad/time-mod grad/exclude_self).

## ★ 다음 행동 — **(0) 타깃 학회 합의 → 정렬 → 범위 작업** (비평 반영, 2026-06-10)
Q2b 판정 완료(구조 음성). **단 정밀 재진술 필수**(아래). 먼저 **어느 학회를 노릴지** 지도교수와 합의 — 그게 남은 작업량 결정.

**0. 타깃 결정 (정렬 1순위)**: 토대(엄밀성·정직성·진단)는 탑티어급, 막는 건 *범위·프레이밍*(재작업 아닌 추가작업).
- **(i) 워크숍/단편 = 지금 충분** (distribution-shift/tabular). 낮은 노력, 피드백.
- **(ii) NeurIPS D&B 트랙 = 현 자산 최적핏** (벤치·측정·분석·음성 가치; 적당 범위확장으로 현실권).
- **(iii) 메인트랙 분석 = 수개월**: 표 시간 **15~20개**로 Claim A 일반화 + **여러 시간-인지 방법**을 같은 렌즈로
  ("내 방법 실패"→"분야 현상") + 도구킷 1급화 + redundancy를 가설로 신중 전개.

**1. 정렬 전 정밀 재진술 (지도교수가 찌를 자리 — 선제)**:
- **elec2 = consistent-but-uninformative**(paired −0.005, CI[−.013,+.003], val→test .07). 음성은 **Insects가 나름**(paired −0.011, CI[−.023,+.001], borderline, val→test .87). "두 벤치 모두" 헤드라인 금지.
- **구조 vs 기판 분해**: time_tabr−tabr=+0.028(시간-구조는 작동) but tabr−mlp_t=−0.038(검색 substrate ≪ parametric) → 지는 건 substrate. "time_tabr>tabr인데 구조 되는 거 아니냐" 선제 차단.
- **통계 = paired**(시드 공유 → diff-SE ≪ marginal std). 러너에 `paired_contrasts` 박힘(`--report-grid` 출력).
- **리드 = Claim A**(robust·신규), **Claim B는 보조**(분리해 A 약화 방지). oracle 음성은 강한형이나 elec2 배포주장 불가 명시.

**2. 음성 견고화 (권장, 싼 것만 — B를 n=1-clean→n=3-clean)**:
```
python scripts/run_elec2_q2.py --dataset insects --insects-variant abrupt_balanced --report-grid \
    --n-seeds 10 --splits temporal --bases trend --lr-grid 1e-3 5e-4 2e-4 --dropout 0.1 --weight-decay 1e-4 --min-epochs 20
# + incremental_abrupt_balanced 등. (elec2-abrupt는 우선순위↓: val→test .07 노이즈 상속)
```
모든 진단 `results/phase1/insects_q2/diagnostics.jsonl` 누적(record에 dataset/variant·paired_contrasts 포함).

**3. Q1 큰-회전 robustness (헤드라인 논거용 미결)**: floor 0.894 좁은 동적범위 → 큰-회전(≥180°) 1회로 넓혀야
"메커니즘은 충실"이 헤드라인 논거로 robust. (게이트론 이미 충분.)

**4. paper 골격 (타깃 정해지면)**: A(covariate≫concept+측정프레임) 리드 → Q1 충실성 → B(구조≤피처, 분해·paired) 보조 → 도구킷 1급화.

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
