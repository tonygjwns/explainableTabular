# OVERVIEW_V2 — 현행 상태 반영판 (간이 논문, 2026-06-14)

> 기존 OVERVIEW.md는 V2 이전(Q2b 음성 무효화·재포지셔닝 이전) 서술이라 갱신 필요.
> 이 문서가 R1 완료 + R2.1–2.4 완료 시점의 현행 요약. 숫자 근거 = RESULTS.md §12–14,
> 증거사슬 = FINDINGS.md, 문헌 = REFERENCES.md §0, 계획 = PLAN_V2.md, 결정규칙 = PREREG_V2.md.

---

## 0. 요약 (Abstract)

표 데이터는 시간이 지나며 분포가 변한다(temporal distribution shift). 우리는 "시간에 따라
진화하는 프로토타입 메모리 + 검색"으로 이 변화를 해석 가능하게 다루려 했으나, 통제된 실험 끝에
**기여의 장르가 방법(method) 논문에서 측정·분석 논문으로 이동**했다. 핵심 발견 셋:

**(A, 리드)** 현실 표 시간데이터의 분포 변화는 **압도적으로 covariate(P(x))**이고, 강한 covariate가
과거/미래 공통 support를 무너뜨려 **concept(P(y|x)) 변화를 표준 조건부/재가중 렌즈로 측정조차
불가능**하게 만든다. within-overlap model-transfer 프레임으로 support 존재하는 곳의 concept을
복원한다(10 데이터셋, ground-truth 검증 완료).

**(B, 보조)** 측정 가능한 곳에서도, 시간을 *구조적으로 인덱싱한 검색*은 시간을 *입력 피처로 넣는 것*을
못 넘는다 — 교락 제거 공정 비교에서 **유의하게 못 넘는다**(25시드, paired CI<0). 지는 이유는
"외삽 장치가 아니라 in-distribution 장치"이기 때문.

**(판결)** 시간-인지 방법이 TabReD서 이긴다는 최근 결과(Cai & Ye)의 이득은, 그들 변조가 라벨에
의존 않는 피처-분포 변환임을 코드로 확인함으로써 **covariate 적응(X-side)**임을 보인다 — 그들이
"concept drift"라 부르는 것은 X-side이며 Claim A에 포섭된다.

---

## 1. 무슨 일을 한 것인가 (문제와 여정)

**원래 베팅**: TabM 백본 위에 `P_k(t) = P_k^base + drift_k(Fourier(t))` 시간-인덱싱 프로토타입
메모리를 얹고, 입력이 "자기 시점의 어느 프로토타입에 가까운지" 검색해 예측. 동기 = "Cai의 변조 ≠
검색 / TabR의 검색엔 시간 좌표 없음 → 시간-인덱싱 검색의 빈 교차점."

방향이 두 번 바뀐 검증 순서:
1. **토대**: TabM을 TabReD 8개서 공개치 ±1% 재현.
2. **메커니즘 작동 확인**: 순수 concept 합성서 시간-인덱싱 0.13 vs 시간무시 1.03 RMSE(+87%). 이후 음성 = 버그 아님 보장.
3. **실데이터 null + 진단**: TabReD 이득 0, 메모리 장식. 드리프트 해부 → 압도적 covariate(AUC≈1.0 pervasive), 시간↔정답 약함.
4. **측정 문제 발견(전환점 1)**: "concept 없음" ≠ "측정 못 함". covariate 심하면 과거/미래 입력영역 안 겹쳐 질문 자체 불가 → within-overlap 프레임. elec2 concept +0.132.
5. **충실성 게이트(Q1)**: 메커니즘이 합성 drift 복원 0.991(10/10) PASS → 음성은 "거짓말해서"가 아님.
6. **구조 vs 피처(Q2b)**: time-TabR + 시간 훅이 시간-피처 넘나? 초기 음성 →
7. **외부 감사로 음성 무효화(전환점 2)**: value 훅 선형 붕괴(`Σw·Lin(Δτ)=Lin(Σw·Δτ)`)=라벨무관 Δt피처 1개 → 가설 미검정. 기판 sub-TabR, 교락.
8. **V2 재검정**: 비퇴화 훅+정품 기판+피처 양쪽 고정+25시드 → 주 대비 두 clean 변종 CI<0 유의 음성. pre-V2보다 깨끗·강함.
9. **재포지셔닝(R2)**: 문헌으로 Claim A 미선점 확인, DISDE-퇴화 표로 3분법 확정, Cai&Ye 판결 코드 증명, 도구킷 ground-truth 검증.

요컨대 **"내 방법이 왜 안 되나"를 끝까지 추적하다 그 추적 자체가 측정 방법론·분야 현상 발견이 됨.**

---

## 2. 사전 연구 (R2.1 웹 원문 검증)

| 선행 | 무엇 | 우리와의 거리 |
|---|---|---|
| **DISDE** (Cai·Namkoong·Yadlowsky, OR 2025, arXiv:2303.02011) | 성능하락 3항 분해(seen내 어려움 / **Y\|X 변화** / 미관측영역), 공유분포 S+density-ratio | 측정 프레임 직접 선조. **ACS·위성만, TabReD/시간축 없음** → 인용+시간축·model-transfer 적응 차별화 |
| **WhyShift** (Liu et al., NeurIPS 2023) | 5 표 데이터(주로 공간) X vs Y\|X, **Y\|X 지배** | 반대 축(공간=Y\|X ↔ 시간=X). 우군화 |
| **TabReD** (Rubachev et al., ICLR 2025) | 산업 표+시간분할, 검색·DL 붕괴/GBDT 생존 | **X vs Y\|X 분해 안 함**(앙상블-std만) → 분해는 우리 고유 |
| **Cai & Ye ICML 2025** (arXiv:2502.20260) | TabReD 실패=프로토콜 결함+Fourier 임베딩 | 시간-피처 baseline 정당화 |
| **Cai & Ye NeurIPS 2025** (arXiv:2512.03678) | 피처 통계 시간 변조로 TabReD 능가, "concept" 처리 주장 | 최대 위협+기회 → R2.3 판결 |
| **Drift-Resilient TabPFN** (NeurIPS 2024) | 시간-인지 도움(SCM-shift prior) | 반례 → Claim B를 검색 *구조*로 한정 |
| FISH(~2011)·SAM-kNN(2016)·Žliobaitė Elec2 비판(2013) | 시간-인지 인스턴스 검색·Elec2 자기상관 | "빈 교차점"을 "현대 딥 표 검색 내"로 한정 |

**신규성**: TabReD에 X/Y\|X 분해 / covariate 지배→측정불가 선행 **미발견** → Claim A 미선점(프레임축만 DISDE와 PARTIAL).

---

## 3. 우리가 한 것 (검증된 기여)

### 3.1 Claim A — 측정불가 + 복원 (RESULTS §13–14)
10 데이터셋 도구킷(cov AUC + DISDE식 재가중 퇴화 + within-overlap gap) → 3분법:
- 고-covariate 5(sberbank/homesite/ecom/homecredit/weather): cov_AUC 1.0, overlap 0.000 → **측정불가**(disjoint)
- 저-covariate 2(cooking/maps): 측정가능, **concept≈0**(−0.005/−0.003)
- concept 벤치 2(elec2/insects): **실concept**(+0.132/+0.144), elec2는 DISDE 붕괴(ESS 0.55%)나 within-overlap 복원

DISDE 두 퇴화 모드(disjoint vs heavy-tail) 명시 + **도구킷 ground-truth 검증**(covariate×concept 4×4, 4/4 PASS:
복원 ρ=+1.0, concept 없으면 0, support 없으면 abstain). **ESS%=2.33(DISDE 사망)서도 within-overlap 동일 복원** = elec2 합성 증명.

### 3.2 Claim B — 구조 ≤ 피처, 교락 없는 유의 음성 (RESULTS §12)
V2 재검정(25시드, val-fair, 주 대비 `time_tabr_t − tabr_t`):
- incremental_balanced: −0.0067 [−0.012,−0.001] p=.006
- incremental_abrupt: −0.0205 [−0.034,−0.008] p<.001
둘 다 CI<0. 분해: ①기판 경쟁력 회복(−0.038 적자 소멸) ②시간 도움이나 피처가 더 나름
③**in-dist(random) 훅 도움 vs 외삽(temporal) 훅 해 = 외삽 장치 아님**(redundancy 뒷받침)
④trend기저 비단조 drift서 외삽붕괴.

### 3.3 판결 — Cai & Ye 이득은 X-side (REFERENCES §0.1)
`temporal_modulation.py` = `γ·YeoJohnson(x,λ)+β`, **라벨 y 미의존** → 시간-인덱싱 covariate 정규화,
P(y\|x) 착취 원리적 불가. 정의적 절반 완결, 경험적 절반(concept≈0 cooking/maps서도 이득>0) 서버 대기.

### 3.4 방법론 산출물
충실성 게이트(Q1), 사전등록(PREREG_V2), paired 통계, 합성 양성대조, 재사용 측정 도구킷(+ground-truth 검증).

---

## 4. 정직한 한계
- Claim B 범위 좁음(clean 2 + elec2 보조). class 음성엔 다중 벤치·방법 필요.
- 측정 프레임이 DISDE와 겹침 → "적응+확장"으로 정밀 포지셔닝.
- R2.3 경험·R2.5(Q1 큰-회전) 미실행.
- redundancy는 가설(외삽 뒤집힘 경험적 일관일 뿐, 증명 아님).

---

## 5. 현재 위치와 다음
기여 등급: "방법 실패"(논문 아님) → **"분야에서 시간-인지가 왜 안 통하는지를 측정 가능하게"**(논문).
워크숍 지금 충분, **NeurIPS D&B 주 타깃**. 남은 것: R2.3 경험(서버), R2.5 Q1 큰-회전, 지도교수 정렬.

---

## 부록 A — 문서 지도
- 현행 계획 `PLAN_V2.md` / 결정규칙 `PREREG_V2.md` / 숫자 ledger `RESULTS.md`(§12–14가 V2) /
  증거 `FINDINGS.md`("V2 RE-TEST VERDICT") / 문헌 `REFERENCES.md`(§0 R2.1 검증) / 인계 `NEXT_TAB.md`.
- 역사 문서(V2 이전): `OVERVIEW.md`, `PLAN_RESCUE.md`, `Q2B_PROPOSAL.md`, `REVIEW.md`.

---

## 부록 B — 한 줄 단위 해설 (Primer)

> 위 §0–§5는 압축 요약. 이 부록은 그 각 줄에 로드된 용어·수식·숫자를 입문자(또는 미래의
> 본인/공저자)가 잡을 수 있게 한 줄 단위로 푼 주석판이다.

### B.0 Abstract 해설

**"표 데이터는 시간이 지나며 분포가 변한다(temporal distribution shift)."**
- 표 데이터: 행=샘플, 열=피처(숫자/범주). 딥러닝이 항상 이기지 않는 영역(트리 앙상블이 자주 이김).
- 분포 `P(x,y)` = "입력 x와 정답 y의 어떤 조합이 얼마나 자주 나오나."
- 시간이 지나며 변한다: 과거로 학습→미래 예측인데 시장·계절·행동이 변해 과거에 맞던 게 미래엔 틀린다.

**"시간에 따라 진화하는 프로토타입 메모리 + 검색"**
- 프로토타입: 데이터 공간의 대표점(군집 중심). 메모리: 그 대표점을 모델 안에 저장. 검색(retrieval):
  새 입력이 오면 가까운 프로토타입/이웃을 찾아 예측에 씀. "시간에 따라 진화"=프로토타입을 시간 t의
  함수 `P_k(t)`로 만들어 움직이게 함.

**"기여 장르가 방법→측정·분석으로 이동"**
- 방법 논문=새 모델이 더 잘한다(성능이 기여). 측정·분석 논문=새 모델 아니라 "현상을 어떻게 재나/왜
  일어나나"가 기여. 우리 구조가 성능서 안 이겨(B), 기여를 측정+현상설명(A)으로 재정의.

**"분포 변화는 압도적으로 covariate(P(x))"**
- `P(x,y)=P(x)·P(y|x)`. covariate drift=P(x) 변화(입력 생김새 변함, 규칙은 그대로). concept drift=
  P(y|x) 변화(같은 입력 다른 정답=규칙 변함). "압도적 covariate"=실데이터 변화 거의 다 P(x), 규칙은 거의 불변.

**"강한 covariate가 공통 support를 무너뜨려 concept 측정조차 불가능"**
- support=데이터가 실제 존재하는 입력영역. 공통 support=과거·미래 둘 다 있는 겹침. covariate 심하면
  과거/미래 입력이 거의 안 겹침→공통영역 없음→"같은 입력에서 정답 변했나(concept)"를 물을 *같은 입력*이
  없음→질문 자체 불가. (concept이 없는 게 아니라 잴 수가 없음 — Claim A의 핵심 구분.)

**"within-overlap model-transfer 프레임으로 복원"**
- within-overlap=겹침 영역만으로 한정 측정. model-transfer=과거-학습 모델과 미래-학습 모델을 *같은
  미래-겹침 테스트*에 둘 다 적용, 성능차로 규칙변화 측정(난이도 통제). elec2서 concept +0.132 실측.

**Claim B 문단**
- "구조적으로 인덱싱한 검색"(time_tabr)=이웃 검색하되 시간차를 메커니즘에 명시. "입력 피처로"(mlp_t)=
  t를 그냥 열 하나로 추가. "못 넘는다"=복잡 구조가 단순 피처를 성능서 못 이김.
- "교락 제거 공정 비교": 두 arm 모두에 직접 시간 피처를 넣어 그 축 고정→차이는 "시간 훅(구조) 유무"뿐.
- "25시드 paired CI<0": 같은 시드 25쌍의 *차이*를 봄. CI 통째로 0 아래=구조가 유의하게 나쁨
  (−0.0067[−0.012,−0.001], −0.0205[−0.034,−0.008]).
- "외삽 장치 아니라 in-dist 장치": random split(분포내)서 훅 도움(+0.005,+0.021)/temporal(외삽)서
  해(−0.007,−0.021). 학습분포 안 미세조정엔 쓸모, 미래 외삽엔 무력→drift 무대는 외삽이라 진다.

**판결 문단**
- Cai&Ye 변조=`γ(t)·YeoJohnson(x,λ(t))+β(t)` (γ=스케일,β=이동,YeoJohnson=왜도 멱변환). 식에 정답 y
  안 들어감(label-free)→y 안 보고 x 분포만 바꿈=정의상 P(x) 적응(X-side), P(y|x) 착취 불가.
- 용어혼동: 그들 "concept drift"=피처 분포 변화=실은 covariate. 포섭=그들 양성결과가 A를 반박 않고
  "이기긴 하나 이유는 covariate 적응, 진짜 concept은 측정불가" 서사 안에 들어옴.

### B.1 여정 9단계 (핵심: 두 전환점)

1. **토대**: 파이프라인 신뢰 위해 TabM을 TabReD 8개서 공개치 ±1% 재현.
2. **양성대조**: 순수 concept 합성(`y=x·w(t)`, w 회전)서 시간메모리 0.13 vs 시간무시 1.03 RMSE(+87%).
   목적=이후 실데이터 음성이 "버그"가 아니라 "착취할 게 없어서"임을 보증.
3. **실데이터 null+진단**: TabReD 이득 0. `mem_gap≈0`(메모리 꺼도 손실 불변)=장식. 시점분류기 AUC≈1.0
   (drop-top5 후도 높음=pervasive)=강한 covariate. Spearman(t,y)≤0.13=시간↔정답 약함.
4. **★전환점1 — 측정 불가능성**: "concept 약함"이 *없는* 건지 *못 재는* 건지 불명. covariate 심하면
   과거/미래 입력영역 안 겹쳐 "같은 입력" 부재→질문 ill-posed. within-overlap 도입, elec2 +0.132. →
   문제 프레임이 "내 방법 실패"→"측정 자체가 어렵다"(Claim A 씨앗)로 바뀜.
5. **충실성 게이트(Q1)**: 합성서 진짜 w(t) 아니까 메커니즘 추정 ŵ(t)와 일치도 측정. 복원 0.991(10/10)
   vs 천장0.990/바닥0.894=PASS. 음성이 "거짓말/고장"이 아님을 봉인(2단계=작동하나, 5단계=정직한가).
6. **구조 vs 피처(Q2b)**: 프로토타입(학습 V_k=정보없음)→인스턴스 검색(실라벨)으로 전환. 3-arm
   mlp_t/tabr/time_tabr. 시간 훅 value-side(이웃 라벨을 시간차로 보정)가 본질. 초기=음성.
7. **★전환점2 — 붕괴 발견(무효화)**: value 훅이 선형→집계서 붕괴.
   - `Σ_k w_k·Linear(Δτ_k) = Linear(Σ_k w_k·Δτ_k)` (선형성+`Σw=1`). 왼쪽=이웃별 보정(의도), 오른쪽=
     "가중평균 시간차" 스칼라 1개를 변환한 피처 1개로 환원. 이웃 라벨 y_k 안 들어감→stale-label 보정
     표현 불가. → 실제 검정된 건 "검색가중 Δt 피처 vs 직접 t 피처"이지 구조 vs 피처가 아님.
   - 추가: 기판 sub-TabR(온도/key-proj 없음), time_tabr만 직접 시간피처 미보유(교락).
8. **V2 재검정**: 비퇴화 훅(`MLP([라벨,Δτ])`, zero-init)+정품 기판(τ·√d·key-proj)+4번째 arm `tabr_t`
   (검색+직접 시간피처)→주 대비 `time_tabr_t−tabr_t`로 교락 제거+25시드 val-fair. 결과=두 clean 변종
   CI<0 유의 음성. 분해: 기판 이제 MLP 대등(−0.038 적자 소멸), in-dist/외삽 뒤집힘.
9. **재포지셔닝(R2)**: R2.1 문헌(A 미선점), R2.2 DISDE 퇴화표(3분법), R2.3 판결(코드 증명), R2.4 도구킷 검증.

### B.2 사전연구 (겹침 vs 갈라짐)
- **DISDE**(OR 2025): 성능하락 3항 분해 — ①seen내 어려움(X) ②**Y|X 변화(concept)** ③미관측영역(X),
  공유분포 S+density-ratio. 겹침=overlap내 Y|X 측정 발상 자체. 갈라짐=실증조사 없음/퇴화 안 봄/TabReD
  미적용/측정불가 발견 없음. → "발명 아닌 적응+확장+실증".
- **WhyShift**(NeurIPS'23): 공간 표서 **Y|X 지배**. 반대 축 우군: 공간=Y|X ↔ 시간=X 대조 헤드라인.
- **TabReD**(ICLR'25): 검색·DL 붕괴/GBDT 생존 보고, **X vs Y|X 분해 안 함**(앙상블-std만)→분해는 우리 고유.
- **Cai&Ye**(ICML'25 프로토콜+Fourier / NeurIPS'25 변조). 변조=판결 대상(X-side).
- **Drift-Resilient TabPFN**(NeurIPS'24): 시간-인지 도움=반례→B를 "검색 *구조*"로 한정.
- **FISH/SAM-kNN/Žliobaitė**: 시간-인지 검색은 스트리밍에 *이미 있음*→"빈 교차점"을 "현대 딥 표 검색
  내"로 한정. Elec2 no-change ~85% 비판→앵커로 보고(elec2 no_change AUC 0.845).
- 린치핀: TabReD에 분해/측정불가 주장한 선행 없음→A 코어 미선점(프레임축만 PARTIAL).

### B.3 핵심 숫자 4개
- **within-overlap transfer gap (elec2 +0.132)**: ①시점분류기 P(late|x)∈[0.1,0.9]만 골라 공통영역 한정
  (OOF p). ②고정 late-overlap 테스트에 early-학습(AUC 0.716) vs late-학습(0.848) 둘 다 적용, gap=
  late−early=+0.132. 같은 테스트/입력영역이라 난이도 통제→차이는 순수 규칙변화. ③p-3등분 안정[+0.12,+0.17]
  =잔여 covariate 아님.
- **ESS / overlap_mass (DISDE 퇴화)**: density-ratio `w(x)=P(late|x)/P(early|x)`. ESS=`(Σw)²/Σw²`=
  "쏠림 감안 실질 표본수", ESS%=ESS/N. elec2 0.55%=재가중 죽음(heavy-tail). 단 완전분리면 w 균일→ESS 착시
  →overlap_mass(P(late|x)∈[.1,.9] 비율)가 진짜 신호: 고-covariate 5개 0.000=disjoint-support. ESS=분산,
  overlap_mass=편향. 두 퇴화모드 상보.
- **paired CI / Hedges' g_z (B 음성)**: 두 arm 같은 시드 공유→시드별 차이 `d_s=a_s−b_s` 직접 봄(공통
  변동 상쇄). m=mean(d), SE=std(d)/√n, CI=m±t·SE. CI<0=유의 음성. p=paired Wilcoxon(.006). g_z=
  mean(d)/std(d)·J=표준화 효과(paired로 계산해야 정직, ~−0.5~−0.6=중간). 25시드=검정력(g_z≈0.5 탐지).
- **도구킷 4×4 검증**: 생성기가 covariate(mu, 규칙-무관 차원 이동)와 concept(theta, 규칙 회전)을 직교
  통제. 검증: ①복원 Spearman(θ,gap)=+1.0 ②거짓양성 없음(θ=0서 max|gap|=0.002) ③퇴화단조(mu↑→cov_AUC↑
  ρ+1.0/overlap↓ρ−1.0) ④실패모드(mu=3.0 overlap0.002→4/4 abstain). ★mu=0.70 행: ESS%=2.33(DISDE 죽음)
  인데 gap이 mu=0 행과 동일 복원=elec2 현상(DISDE 붕괴/within-overlap +0.132)의 합성 증명.

### B.4 한계 (각=리뷰어 공격+방어)
- **B 범위 좁음(clean 2+elec2)**: 음성 2개로 class 일반화 약함. 방어=B는 보조(A가 10개 리드)+교락없는
  유의음성+깨끗한 분해+확장경로 명시.
- **프레임이 DISDE와 겹침**(가장 위험): "재탕" 공격. 방어=명시 인용+DISDE 퇴화 시연(깨지는 영역 확장)+
  실증/측정불가는 DISDE에 없음. "발명 아님"을 먼저 인정.
- **R2.3 경험·R2.5 미실행**: 지금은 그 주장 못 함. 상태=코드 완성, 서버 실행만(R2.3 정의적 절반은 코드로
  이미 반박불가).
- **redundancy는 가설**: "시간 입력 NN은 임의 P(y|x,t) 표현→구조 추가는 중복"은 *in-dist 예측만* 닫음
  (표현능력≠학습능력; 외삽/샘플효율 경로 안 닫음). 증명처럼 쓰면 반례(TabPFN)에 당함→"허용이지 증명
  아님 + 다른경로 이득의 경험적 부재"로만 진술. in-dist/외삽 뒤집힘이 *일관* 근거(증명 아님).
- 수렴: 네 한계가 모두 "A 리드 + DISDE 정밀 포지셔닝 + B 신중"으로 귀결.
