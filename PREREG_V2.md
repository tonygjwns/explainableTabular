# PREREG_V2 — Q2b 재검정 사전등록 (단일 현행 문서)

> **작성일 2026-06-12. 이 문서가 포함된 커밋이 잠금 시점이다** (`git log --follow PREREG_V2.md`).
> 사전등록 문서가 PRE_REGISTRATION → PLAN_RESCUE → Q2B_PROPOSAL로 표류했던 문제를 해소하기 위해,
> **이후 모든 기준 변경은 이 문서에 새 섹션 + 커밋 해시 + 사유로만 누적**한다 (PLAN_V2 메타규칙).
> 결과를 본 후 기준 변경 금지. 모호하면 보수적(음성) 판정.

## 0. 왜 재검정인가 (배경 1줄)
pre-V2 Q2b의 "구조 ≤ 피처" 음성은 **검정이 가설과 불일치**해 증거로 무효: (i) linear value 훅이
집계 하에서 `Σw·Linear(Δτ)=Linear(Σw·Δτ)`로 붕괴(=Δt 피처 1개, stale-label 보정 표현 불가),
(ii) 검색 기판이 sub-TabR(온도/key-proj 없음, train 255 vs eval 4096 context 불일치, knob 미튜닝),
(iii) time_tabr만 직접 시간 피처를 못 받음(구조×주입위치 교락). (외부 감사 2026-06-12; PLAN_V2 §0.)

## 1. 가설과 arm (V2)
**H**: 시간-인덱싱된 검색(시간 훅)은, *직접 시간 피처가 양쪽 모두에 주어진 상태에서*,
검색+시간-피처 모델 대비 측정된 concept drift 위에서 예측 성능을 더한다.

| arm | 구성 | 역할 |
|---|---|---|
| `mlp_t` | 인코더 + τ(t) 피처 | 피처 baseline |
| `tabr` | 검색 기판(시간 전무) | 기판 격리 |
| `tabr_t` | 검색 + τ(t) 피처 | **대조군** (피처 보유 통제) |
| `time_tabr_t` | 검색 + 시간 훅(value) + τ(t) 피처 | **후보** |

공유: 인코더, basis(trend), V2 기판(learnable τ·key-proj·sampled-4096 train ctx·full eval ctx).
`value_hook=mlp`(라벨×Δτ 상호작용)이 주 분석, `gate`는 ablation. `time_mode=value` 주, metric/both ablation.

## 2. 프로토콜
1. **튜닝 단계** (판정에 미사용): 3시드, **val 기준**으로 데이터셋별 `topk ∈ {8,32,128}` 선택 후 고정.
   (context 방식·τ·key-proj는 V2 기본으로 고정 — 튜닝 축 아님.)
2. **본 실행**: **25시드**(0..24), lr-grid {1e-3, 5e-4, 2e-4} per-arm, dropout 0.1, wd 1e-4,
   min_epochs 20, split=temporal(주)+random(대조), INSECTS는 **full stream**(`--max-samples` 금지).
3. **선택 프로토콜 병기**: val-fair(주 판정) + oracle(강한형 보조; "oracle을 줘도"용). 판정은 **val-fair**로.
4. **앵커**: `run_anchors.py` (lgbm±t, knn±t, no-change) 동일 split에서 1회 — 외부 보정용(판정엔 미사용,
   단 §4 red flag에 사용).

## 3. 데이터셋
- **clean 셋 = INSECTS 3변종**: `incremental_balanced`, `abrupt_balanced`, `incremental_abrupt_balanced`.
  단 변종별 **val→test Spearman ≥ 0.3** 사전 확인(미달 변종은 clean에서 제외하고 그 사실 보고 —
  elec2 ρ=0.07 선례). 제외로 clean<2가 되면 `incremental_reoccurring_balanced`로 보충(이 순서 고정).
- **elec2 = 보조**(ρ=0.07 → 판정 비가담, 일관성 보고만).

## 4. 결정규칙 (사전 commit)
**주 대비** = `time_tabr_t − tabr_t`, temporal, paired per-seed (25시드), val-fair lr.

| 판정 | 조건 |
|---|---|
| **구조 우위** (좁은 양성 부활) | clean 변종 **≥2/3**에서: paired 95% CI > 0 **AND** Wilcoxon p<0.05 **AND** paired Hedges' g_z ≥ 0.5 |
| **음성 확정** (Claim B, V2-유효) | 위 미충족. 진술: "비퇴화 훅·정품화 기판·per-arm 선택을 줘도 시간-인덱싱 검색이 피처 위에 가치를 더하지 못함" |
| 모호 (1/3 양성 등) | 음성 쪽 보수 판정 + 변종별 이질성 자체를 보고 |

**보조 대비**(보고 의무, 판정 비가담): `tabr_t − mlp_t`(기판 가치), `time_tabr_t − mlp_t`(결합),
`tabr − mlp_t`(기판 적자 — pre-V2 −0.038이 V2 기판 정품화로 얼마나 줄었는지 명시).

**Red flags (사전 명시)**:
- random split에서만 이득 → concept 착취 아님(자기상관 누수) — 양성 청구 금지.
- elec2에서 `no_change` ≥ 모든 arm → elec2 결과는 "trivial baseline 이하" 라벨 필수.
- time_tabr_t의 시드-간 std가 mlp_t의 3배 초과 → 불안정성을 결과와 동급으로 보고.

**검정력**: n=25 paired, α=.05 양측, 80% power → 검출가능 d_z ≈ 0.58. 잠근 임계 g_z ≥ 0.5와 정합
(이보다 작은 효과는 "검출 못 함"으로 보고하지 "없음"으로 단정하지 않음).

## 5. 변경 금지 / 허용
- **금지**(결과 후): 주 대비 정의, 판정 임계(CI/p/g_z), clean 셋 정의·보충 순서, 시드 수, val-fair 주 판정.
- **허용**(사전 명시 범위): ablation 추가(gate 훅, metric/both, basis=fourier, 백본 TabM 스팟체크 —
  보강용, 판정 비가담), 시드 증설(25→40, 감소 불가), 데이터셋 *추가*(기존 기준 유지).

## 6. 결과 귀속 (어느 쪽이든 논문行)
- 구조 우위 → "측정된 concept 위에서 시간-인덱싱 검색이 가치를 더한다"(좁은 양성, Claim B 역전) +
  분해(기판 vs 시간-구조) 보고.
- 음성 → V2-유효 음성으로 Claim B 완성(분석 논문의 보조 기여). pre-V2 결과는 "무효화된 선행 시도"로
  부록 보고(은폐 금지).
