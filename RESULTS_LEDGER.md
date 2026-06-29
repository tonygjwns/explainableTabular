# RESULTS_LEDGER — 결과 출처·오염 원장 (공개 대비)

> 목적: 각 결과 artifact가 **어느 스크립트·어떤 시간기저**에서 나왔고, 2026-06-29에 발견한
> **Fourier 임베딩 버그**에 오염됐는지를 1:1로 추적한다. 논문/코드 공개 시 재현성 근거.
> 규율: superseded/tainted 결과도 **숨기지 않고** 기록한다("무효화된 선행 시도").

## 0. 발견된 버그 (2026-06-29)
**Fourier 시간 임베딩의 외삽 결함 두 가지** (`src/models/temporal_embedding.py`):
1. **앨리어싱**: 정규화 t를 train-max로 [0,1] 스케일 → test(미래) t>1. period가 데이터 범위 내 한
   주기를 완성하면(예: period=1.0) **φ(미래 t) = φ(t mod 1)** — 미래가 과거로 매핑(수치 거리 ~1e-16
   확인). 단조 드리프트에서 외삽이 *과거 규칙으로 되돌아감*.
2. **기본 주기 버그**: `tabr.py`/`temporal_modulation.py`가 `FourierTimeEmbedding`을 `periods` 없이
   생성 → 기본값 `(1.0, 1/12, 1/52, 1/365)×6 harmonics` → 정규화 [0,1]에서 ~2190 사이클 = 학습
   구간 내에서도 완전 앨리어싱.

**영향 정량화** (numpy 재현, train[0,1]→미래[1,1.5] 외삽 MSE):
- 선형 드리프트: 영향 0 (붙은 raw t 채널이 외삽). 
- 포화형: Fourier(P=1.0) 0.244 vs headroom 0.023 (10×). 
- 비단조: Fourier(P=1.0) 2.054 vs headroom 0.0068 (300×). trend(다항)는 비단조 0.78(headroom 무효).

**해결(설계 확정, 미구현)**: headroom 정규화(분모=train_span×2, [0,1] clamp) + period≥2×범위 + harmonic 1개.

## 1. 헤드라인 — CLEAN (Fourier 임베딩 미사용; HGB/sklearn 또는 trend 기저)
| artifact | 스크립트 | 기저/엔진 | 상태 |
|---|---|---|---|
| deployment_decay_summary.json | run_deployment_decay.py | HGB rolling-origin | CLEAN (현행) |
| gap_hygiene_summary.json / _summary2 / p0_ / 0626_ | run_gap_hygiene.py | HGB | CLEAN |
| representation_summary.json / _synth / p0_ | run_representation.py | HGB | CLEAN |
| gap_controls_summary.json / _synth | run_gap_controls.py | HGB (placebo) | CLEAN |
| disde_degeneration_summary.json | run_disde_degeneration.py | HGB | CLEAN |
| toolkit_validation_summary.json | run_toolkit_validation.py | HGB synth GT | CLEAN |
| toolkit_adversarial_summary.json | run_toolkit_adversarial.py | HGB | CLEAN |
| c1_ranking_summary.json | run_c1_ranking.py | HGB | CLEAN |
| whyshift_summary.json | run_whyshift.py | HGB (ACS) | CLEAN |
| anchors_summary.json | run_anchors.py | lgbm/knn/no-change | CLEAN |
| adversarial_probe_summary.json | run_adversarial_probe.py | HGB | CLEAN |
| correct_assumption*_summary.json (0626/0627/full) | run_correct_assumption.py | HGB + river | CLEAN |
| q1_verdict.json | run_q1_faithfulness.py | **trend** 기저 (메인) | CLEAN |
| grid_report.json | run_elec2_q2.py | B arms **trend** 기저 (R1 주음성) | CLEAN (단 trend 외삽 한계=설계 이슈) |

## 2. TAINTED — Fourier 앨리어싱 오염 (전부 부차/이미-은퇴)
| artifact | 스크립트 | 오염 원인 | 비고 |
|---|---|---|---|
| summary_fourier.json | run_modulation_adjudication.py | temporal_modulation 기본 periods(1/365) | Cai&Ye 변조; 이미 "inconclusive" |
| summary_fourier_tuned.json | run_modulation_adjudication.py | 〃 | 〃 (lr-tuned) |
| learned_retrieval_summary.json | run_learned_retrieval.py | time_basis="fourier" + tabr.py 기본 periods | V3.5-C niche; **이미 2차 적대리뷰로 은퇴** |
| q1_verdict_a6.28_fourier.json | run_q1_faithfulness.py | fourier 변종(period=1.0 앨리어싱) | 부분 오염; 메인 q1는 trend=CLEAN |

**조치**: 작업트리에서 제거(이전 커밋 git history에 원본 보존) + 로컬 `results/_tainted_fourier_bug/`에
사본(이 폴더는 .gitignore 대상이라 미추적 — 추적·근거는 본 원장이 담당). 재실행 필요시 고친 기저로.
변조(summary_fourier*)는 결론(inconclusive)이 안 바뀌면 caveat만. niche/q1-variant는 은퇴/부차라 재실행 불요.

## 3. 검증 완료
| artifact | 결과 |
|---|---|
| retrieval_vs_recency_summary.json | run_retrieval_vs_recency.py = model-light(sklearn kNN+HGB), Fourier 미사용 → **CLEAN** (단 niche 결론은 은퇴) |

## 3b. deployment-decay 도구 검증 (staleness_harm이 concept을 진짜 재나)
synth ground-truth + **적대 대조**로 staleness_harm의 교란 면역을 검증(`run_deployment_decay.py --synth`):
- concept(규칙회전)→CONCEPT / covariate(공변량이동·규칙고정)→DECAY-COVARIATE / stable→STABLE (PASS).
- **adversarial covariate_mc** (multiclass·prior이동·규칙고정, accuracy 지표): stale −0.003(CI<0) →
  DECAY-COVARIATE, **CONCEPT 거짓발화 안 함.** → insects=CONCEPT(stale+0.129)이 prior-이동/용량-타협
  artifact가 아님을 입증. binclass(AUC)는 prior 불변이라 애초 면역.
- 결과(2026-06-29 sweep): TabReD 8/8 stale≤0(concept 없음) / elec2 autocorr-risk 플래그 / insects=CONCEPT.
  두 렌즈(within-overlap+deployment) 데이터셋별 일치.

## 4. 코드 신뢰성 — 박을 테스트 (미구현)
- 시간 임베딩 ground-truth: 배포 범위서 앨리어싱 0, 매끈 드리프트 외삽 허용오차 내.
- headroom 정규화: 변환 후 t≤1, train→[0,0.5].
- 알려진-무해 경고 침묵(`np.nanstd` all-NaN 컬럼 → "DOF<=0", 결과는 옳음) + `-W error` 1회 스윕으로 미지 경고 분류.
- 기존 smoke 재실행: smoke_test_tabr(표현력/init), _trainer, _insects.
