# Q2b 제안서 — 인스턴스 time-TabR (지도교수/리뷰어 정렬용, 1p)

## 한 줄
elec2가 *측정·확정된* concept drift(+0.132 AUC, OOF·층안정)를 가짐을 확인했다.
**검색 *구조*(인스턴스+라벨)가 단순 *시간-피처*를 측정된 concept 위에서 넘는가?** 를 검정.

## 왜 지금 (들고 가는 근거)
- **Q1 PASS**: 재설계 메커니즘이 합성 drift를 충실 복원(recovery 0.991 vs ceiling 0.990, 10/10).
- **elec2 concept 확정**: 고정 late-test 전이격차 +0.132(난이도 아님), OOF p, p-층 [+0.12,+0.17] 안정(잔여 covariate 아님).
- **프로토타입 죽음**: 학습 V_k는 trend·load-balance로도 시간-피처 못 넘음(§4.2). → **인스턴스+라벨이 유일 레버.**

## 메커니즘 (F2 — 핵심 분기, 결정 요청)
인스턴스 검색(TabR식): 쿼리 x_q ← 학습 이웃 x_i, **value = 이웃 라벨 y_i의 함수**(파라미터에 없던 정보).
시간을 **두 경로**에 넣고 *어느 쪽이 이득을 나르는지* ablate:
- **value-side (핵심)**: 이웃 라벨 기여를 **(t_i→t_q) drift로 보정**(trend). concept=P(y|x) 변화는
  *라벨*에 사니, stale label 보정이 본질. → 이게 이득 나르면 **concept 착취 증거**.
- **metric-side**: 유사도를 trend(t_q)로 변조(어느 이웃 고를지). → 이것만 나르면 **covariate 적응**
  (시간-피처와 중복 → MLP+t로 붕괴). *이 구분 자체가 발견.*
- ✗ recency 커널(|t_q−t_i|): 외삽서 붕괴. ≠ 관계-drift 보정.

## 실험 설계
- **요인설계**: 구조 {MLP+t, TabR(무시간), time-TabR} × 기저 {fourier, trend} × λ;
  **time-feature baseline의 t-인코딩을 같은 기저로 정합**(구조 vs 기저 분리).
- **within-model 시간 on/off 2단계 ablation**(metric항·value항 각각) — 같은 파라미터/아키텍처,
  시간만 토글 → 아키텍처 confound 0 (측정가능-concept 데이터서 한 올바른 Test 1).
- **용량 정합**: time-TabR ≯ MLP+t 파라미터(승리가 용량 탓 배제).
- **검정력**: 시드 25~30 또는 임계 g≥0.5(n=10은 g≈0.6+만 탐지).

## 사전등록 — 예상 패턴 (결과에 안 휘둘리게)
- **temporal split**: 구조 이득은 *여기* 나타나야(early→late stale-label 문제 존재).
- **random split**: train이 양 기간을 봐 stale-label 없음 → 구조 이득 **~0이어야**.
  (random에서만 이득 → 적신호: concept 착취 아님.)
- **이득 국소화**: 구조 이득이 concept 큰 영역(late-overlap)에 집중 → 기전 증거.

## 결정규칙 (사전 commit)
time-TabR(trend, value-side)이 기저정합 time-feature를 **≥2 데이터셋의 temporal split**에서
잠근 (g,p,검정력)으로 이김 → **구조 우위(좁은 성능 부활)**. 아니면 → **"측정된 concept 위에서도
구조 ≤ 피처"**(=「측정 못함」보다 강한 음성).
- **전제 미확정**: ≥2 데이터셋이 elec2+**Insects**를 요구 → **Insects가 measurable-concept인지
  F3로 *먼저* 확인**(싼 체크). 통과 못 하면 단일-데이터셋 결과만 → 빌드 값매김 재고.

## EV / 헤드라인 (정직)
인스턴스 검색도 *in-dist*엔 redundancy 벽 → 완전 성공조차 "elec2(±Insects) temporal서 구조>피처"라는
**좁은 양성**. 빌드 정당화 = **그라디언트 완성(합성→설계→실)** + §6(다) 음성 강화. **헤드라인은 여전히 (다).**
어느 결과든(구조 승=부활 / 패=강한 음성) 논문이 강해짐.

## 착수 전 체크리스트
1. **F2 분기 결정**: value-side drift 보정 중심(권고) — 지도교수 승인.
2. **F3-on-Insects**(싼, F2-무관) — 결정규칙 전제 확인.
3. (병렬) F2-무관 TabR 인프라(효율 검색·context set·eval 하니스·요인설계 골격) 코딩 — 폐기위험 0.
4. **Q1 큰-회전 robustness 1회**(Q1을 paper probe로 쓸 거면 — "충실한 메커니즘조차 못 돕는다"의 충실 논거).

## 포인터
RESULTS.md(결과 ledger) · FINDINGS.md(증거) · PLAN_RESCUE.md(사전등록 프로토콜·결정규칙) · REVIEW.md(전체 비판).
