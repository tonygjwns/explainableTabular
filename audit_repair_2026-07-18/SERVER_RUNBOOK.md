# SERVER_RUNBOOK — 서버 큐 실행 명령 (PREREG §14 + §15 순서)

전제: 서버 repo 루트에서 실행 (아티팩트 경로로 추정컨대 `/home/tonyhuh/explainableTabular`).
`$REPO`는 repo 루트. 실행 순서는 규범적이다 — **0 → 1이 통과해야 2·3 진행 가능** (§4·§15a).

---

## 0. strict-shadow 패치 반영 (§15b) — GitHub 경유

로컬에서 `strict-shadow-fix` 브랜치가 origin(github.com/tonygjwns/explainableTabular)에
푸시되어 있다. 내용: 패치된 scripts/run_deployment_decay.py + PREREG §15 +
audit_repair_2026-07-18/ (배터리 재실행 아티팩트 2건, REVISION_NOTES, 이 런북).
§15b의 [D] 예측이 실행 전 커밋 타임스탬프를 얻는 것이 이 순서의 목적이다.

```bash
# (서버에서)
cd $REPO
git fetch origin
git diff main origin/strict-shadow-fix -- scripts/run_deployment_decay.py  # 델타 3곳 확인:
#   1) _injection_recovers: CI 반환 추가
#   2) 주입 실행 조건 확장(verdict 또는 verdict_strict) + strict 승격(rule B: CI하한>floor)
#   3) 출력 blob에 injected_staleness_ci / injection_recovered_strict 추가
git checkout strict-shadow-fix   # 이 브랜치에서 1~4를 실행; 통과 후 main에 merge
```

## 1. 배터리 게이트 — 서버 머신에서 --synth (§15a 봉인 + 패치 검증)

```bash
python scripts/run_deployment_decay.py --synth 2>&1 | tee logs/synth_server_$(date -u +%Y%m%dT%H%M%S).log
```

**통과 기준**: `GROUND-TRUTH PASS` + 14셀 판정이 커밋본(01ae6ae PASS)과 일치.
클린룸 대조 기준치(sklearn 1.9.0 venv, repair_20260718/): reg_early_noisy den +0.0045@gate 3.53,
reg_xdep_noise den +0.0063@3.72 근방이면 정상.

```bash
# PASS 아티팩트를 버전드 이름으로 커밋 (synth_summary.json은 덮어써지므로)
cp results/phase1/deployment_decay/synth_summary.json \
   results/phase1/deployment_decay/synth_battery_v3_PASS_server_$(git rev-parse --short HEAD).json
git add results/phase1/deployment_decay/synth_battery_v3_PASS_server_*.json
git commit -m "server-machine battery PASS under patched instrument (PREREG §15a seal)"
```

**FAIL 시**: §4 규칙 — 어떤 실데이터 실행도 금지. 원인 수정 후 PREREG 새 섹션으로 기록.

## 2. [D] injection-family sweep (§14, 탐색 시드 0–9)

인증서 셀 6(cooking/delivery/elec2/ecom/homecredit/weather) + insects 양성대조 × 3 패밀리.
패밀리별로 순차(또는 3개 병렬 — 원 드라이버처럼 phase-parallel 가능):

```bash
mkdir -p logs
for F in lowvar interaction subpop; do
  nohup sh -c "
    python scripts/run_deployment_decay.py \
      --tabred cooking_time delivery_eta ecom_offers homecredit_default weather \
      --config configs/phase1.yaml --n-seeds 10 --inj-family $F &&
    python scripts/run_deployment_decay.py --elec2 --n-seeds 10 --inj-family $F &&
    python scripts/run_deployment_decay.py --insects --n-seeds 10 --inj-family $F
  " > logs/injfam_${F}.log 2>&1 &
done
wait
```

(`--insects`의 기본 variant는 incremental_balanced = 지도의 유일 실데이터 CONCEPT 셀 —
§14가 요구하는 양성대조.)

**판독 (§14에 사전 커밋된 양방향 규정 그대로)**:
- cooking/delivery: 각 summary JSON의 해당 행에서 `injection_learnable`·`injected_staleness`·
  **`injection_recovered_strict`** 확인. 학습가능 패밀리 전반에서 회복 유지 → §7 한계(6)
  실측 방어로 격상, 부록 B.4 표. **아울러 `injection_recovered_strict=true`면 §15b의
  strict-확정 인증서 완결** → 원고 Table 4/§5.2의 caveat(REVISION_NOTES R4 a·b)를 확정
  라벨로 교체.
- 어떤 학습가능 패밀리에서 미회복 → 인증서를 family-상대적으로 재표기 (§14 규정),
  확증 시드 100–109 재실행 통과 필요.
- insects가 학습가능 패밀리에서 미회복 → 계기 감도 한계로 §6 보고 (은폐 금지).
  ※ 이것이 독립 판정이 명시한 최상위 뒤집힘 조건 — 결과와 무관하게 그대로 보고.
- 학습불능 패밀리는 어느 방향으로도 사용 금지 (vacuous 규율).

## 3. [C] full-span 감사 (§14, 탐색 시드 0–9)

```bash
nohup python scripts/run_deployment_decay.py \
  --tabred sberbank_housing homesite_insurance ecom_offers homecredit_default \
           cooking_time delivery_eta maps_routing weather \
  --config configs/phase1.yaml --n-seeds 10 --tabred-span full \
  > logs/fullspan.log 2>&1 &
```

**판독 (§14 규정)**: 판정 전 셀 유지 → 부록 B.3. 어떤 산업 셀이든 DEPLOYMENT-CONCEPT 발화
→ 4단계 확증 통과 시에만 인정, 인정되면 0/8 헤드라인을 "train 구간"으로 재스코프.

## 4. 확증 재실행 (판정이 기존 지도와 달라진 셀만, §14/§5 규율)

2·3에서 판정이 달라진 셀이 있으면 **동일 명령 + `--seed-base 100`** 재실행. 두 실행 판정
불일치 → unstable, 채택 금지.

```bash
# 예: full-span에서 weather가 발화했다면
python scripts/run_deployment_decay.py --tabred weather \
  --config configs/phase1.yaml --n-seeds 10 --tabred-span full --seed-base 100
```

## 5. 마감

```bash
git add results/ logs/
git commit -m "execute [C]/[D] (PREREG §14) under patched instrument; strict certificates emitted"
# 제출물에 git 이력 포함 (§8의 약속; 독립 판정의 뒤집힘 조건 3번 소거)
```

원고 반영: REVISION_NOTES.md R4의 (a)(b)를 [D] 결과에 따라 확정 라벨 또는 family-상대 표기로
교체, B.3/B.4 부록 추가, §8에 서버 배터리 봉인 문장 반영.

## 예상 소요

원 인프라 기준(§8: 전체 페이즈 ~17h phase-parallel): [D]는 셀-런 21개(패밀리당 7) ≈
tabred 배치가 지배, 패밀리 병렬 시 반나절 내. [C]는 8셋 1런 ≈ 기존 phase2 1회분. 배터리는
클린룸 재현에서 1.3~2.8h (서버 코어수면 더 짧을 것).
