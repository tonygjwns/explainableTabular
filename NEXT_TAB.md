# NEXT_TAB — 인계 (이어서 작업할 새 탭용)

> 워크플로우: 로컬(이 repo)서 코드 작성→git push, 서버(`explaintab311` env, py3.11)서 pull→실행.
> 서버엔 Claude 없음 → 로컬서 완성해 push. 최신 커밋 = `git log --oneline -1`.

## ★★★ 현행 (2026-08-03 저녁) — **V5 채택 확정**, day-4 판독 완료, §3·§6·§7 초안 완료. 새 탭은 여기부터

**한 줄**: 열린 결정이던 계기-우선 재구성을 **채택**했고, day-4(E1~E4) 판독·기록·반영과
**V5 §1~§9 전 절 + 단일 원고 폴드까지 끝났다** → [PAPER_DRAFT_V5.md](PAPER_DRAFT_V5.md)(1224줄,
수치 404개 대조, unmatched 2 = 기존 출처 공백). **다음 = ①EN 냉각 통독 1회 → ②KO 미러 → ③tex.**
KO/tex는 EN이 굳기 전에 손대면 세 번 고치게 된다.

### V5 절별 출처 (폴드가 어디서 왔는지 — 되짚을 때 필요)

| V5 | 출처 | 비고 |
|---|---|---|
| §1 제목·초록·서론 | 신규 | 주어 교체. 기여 순서 = 채널 → 계기 → 맹점 지도 → 0/8 지도 |
| §2 | V4 §2 이동 | **"head-to-head 안 했다" 문장 수정**(§3.4가 그것임) |
| §3 실패 채널 | 신규(V4 §5.3·§4.1 승격) | 3.4 = E2 |
| §4 계기 | V4 §3 이동 | 3.1을 4.1/4.2로 분할, reading aid가 §3.2를 되가리킴 |
| §5 검증 | V4 §4.1 이동 + **신규 5.2** | envelope→§3.3, 클래스 행렬→§6.1로 갔으므로 **여기서 재진술 금지** |
| §6 맹점 | 신규(V4 한계 1·4·6 + ACS) | 6.2=E3, 6.3=E4, 6.4=E1 |
| §7 적용 | V4 §5+§6 강등 | 수치 불변, 한정어를 문장 안으로 |
| §8·§9 | 신규 | 한계가 §6으로 갔으므로 축소, §9에 재게이트·공시 2건 |
| 부록 A/B/C | V4 이동 | 참조 번호 기계 변환 |

⚠ 폴드 시 참조 번호 매핑: V4 §3→§4, §4→§5, §5→§7, §6→§7.3, §7→§8, §8→§9, **§4.2→§3.3,
§4.3·§5.4→§6.1, §5.3→§3.2**. `PREREG §N`은 **절대 변환 금지**(1차 시도에서 실제로 잘못 바뀌었다).

### 지금 상태 (커밋 `baa81b0`)

| | |
|---|---|
| 원고 | **V4 = v4.9 그대로 살아 있음**(EN/KO/tex 동기). **V5 = `PAPER_DRAFT_V5.md`(단일 EN 원고)** + `PAPER_DRAFT_V5_SECTIONS.md`(절별 초안·이동 명세) |
| 사전등록 | `PREREG_DEPLOYMENT_V2.md` §16·§17·§18·**§19(day-4)**, `PREREG_ACS_EXTENSION` §10·§11·**§12(E1)** |
| 헤드라인 | V5 기준 **노이즈 오탐 채널**이 리드, 0/8·0/5 지도는 §7 적용 사례로 강등 |
| 서버 | 유휴 — day-4 결과 `a8f1549`로 수령·판독 완료 |
| 장부 | RESULTS §31(수치 감사)·**§32(day-4 판독)** |

### day-4 판독 요지 (상세 = PREREG §19 / ACS 확장 §12, 장부 = RESULTS §32)

- **게이트**: 배터리 14/14 + map-env 참조와 **비트 동일** → 신규 플래그 2종 안전.
- **E1**: PA denoised가 AUC −0.009 → brier +0.002 → **logloss +0.021**(TX 대조 +0.006, 3.5×).
  기제(순위 기반 지표) 확인. **단 인증서 없음(learn −0.185)·대조도 양수·floor는 AUC 단위** — 셋 다 명시.
  예측 3건 중 ①부분 ②성립 ③**반증**. 산업 3셀은 지표 바꿔도 vacuity 안 풀림.
- **E2**: 프레임이 크기로는 7.5–14× 갈라냄 → **강한 주장 반증**. 남는 주장 = **부호로는 안 갈린다**.
- **E3**: D는 k와 함께 **상승**(예측 부호 반대), 회복은 **측정 불가**(셀이 D\* 아래로 내려가 주입 미라우팅).
  ρ=−0.47은 상관으로 유지. 부수: 표현이 식별가능성·부호를 동시에 정함(homecredit k=5 +0.013 @ D=0.721).
- **E4**: 12/12 회복. **subpop이 저-D(≈0.49)에선 회복** → subpop 맹점 = **패밀리 × 기하**.

### 이번 4일이 바꾼 것 (요지)

1. **분모 복구**: `weather`가 공허 → **verified no-concept**(저분산 캐리어에서 회복 +0.183/+0.193,
   확증 통과). `homesite`는 불안정 → **earned blindness** 안정화. 인증 분모 4 → **5**.
2. **인증서의 패밀리-상대성이 측정됨**: `weather_fullspan`에서 부분모집단-국소 규칙이 R² 0.560으로
   여유 있게 학습가능한데 **−0.050 미회복**(확증 재현), 같은 윈도우로 저분산 +0.128 · 상호작용
   +0.046은 회복. → 계기는 피처 *방향*엔 검정력이 있고 *부분모집단 소속*엔 없다.
3. **[C] full-span**: 산업 CONCEPT 0건, 8/8 확증 안정 → 헤드라인 재스코프 없음.
   `homecredit_fullspan` den **+0.018**(floor의 89%)이 패널 최대 양수이고 인증서가 *earned*.
4. **★ ACS 반증 (§10·§11)**: 문서화된 ACA 메디케이드 확대(PA 2015-01-01 vs TX 미채택)를
   **식별가능 영역(D=0.53)·최소 δ(0.00083)에서 놓쳤다.** 기제는 실행 전 등록된 대로 "AUC는
   순위 기반" — proper score로 재면 처치/대조 분리가 3.1× → 6.3×(Brier) → 6.6×(logloss) →
   15×(KL). **그러나 절반만**: 절대값은 두 계기의 모든 floor 아래이고 대조군도 placebo-유의 gap을
   낸다. → **한계 7** + **부록 C floor 비판**(0.02가 실제 정책 규칙 변화보다 크다; 상수는 안 바꿈).

### ⚠ 새 탭이 반드시 지킬 것

- **PREREG의 기존 절을 수정하지 말 것.** 결과는 새 절로만 추가(§9·§10·§12·§13·§17·§18 전례).
  `PREREG_ACS_EXTENSION`은 서명 조항이 §1–§9 수정을 금지 — 체크박스도 미체크로 둘 것.
- **임계값·캐스케이드·시드 프로토콜 변경 금지.** floor 0.02가 실제 정책 변화보다 크다는 게
  드러났지만 **바꾸지 않는다**(부록 C에 기록만).
- **계기(`run_deployment_decay.py`) 수정 시 PREREG §4 배터리 재게이트 필수** — map env
  (3.11.15/1.9.0)에서 14/14 + 수치 비트 동일. 아니면 되돌린다.
- **`which python`을 믿지 말 것.** 프롬프트가 `(explaintab311)`인데 PATH가 다른 프로젝트
  venv를 가리킨 사고가 있었다(§17.9). 항상 절대경로
  `$HOME/miniconda3/envs/explaintab311/bin/python`.
- **스레드 캡 필수**: `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=8`. 안 걸면 병렬도가 1로 붕괴한다
  (실측: 8시간 낭비 1회, 서버 로그상 4.4× 손해 1회).

### day-4 결과가 오면 (판독 규칙은 이미 커밋됨)

`run_day4.sh` 헤더에 E1~E4 예측과 판독 규칙이 실행 전 커밋돼 있다. 특히:

- **E1은 판정 라벨을 전부 버리고 `staleness_harm`/`denoised_staleness` 수치와 PA-vs-TX 대비만
  읽는다.** 캐스케이드 상수가 AUC 단위라 proper score에서 라벨은 무의미하다(로컬 확인:
  규칙이 진짜 회전하는 셀에서 staleness +0.214인데 판정 INCONCLUSIVE).
- **E2의 주장은 이미 약화됐다.** 로컬 예비에서 프레임이 크기로는 분리한다(규칙 0.818 vs
  노이즈 0.059, 14×). "필드 도구가 오독한다"는 **반증**. 남는 주장은 "부호만으로는 안 갈리고,
  임계값 판독은 둘을 같이 Y|X로 부르는데 함의된 조치는 한쪽에서만 듣는다".
- 결과는 `PREREG_DEPLOYMENT_V2.md` §19 / `PREREG_ACS_EXTENSION` §12로 **새 절 추가**.

### 서버 도는 동안 로컬에서 할 수 있는 것

1. ~~**수치 감사**~~ — **실행 완료 (2026-08-03, 장부 = RESULTS §31).** `audit_paper_numbers.py`가
   행 스키마 이전 아티팩트 35개 + 로그 27개를 건너뛰고 있던 것을 고치고, 값만 같으면 통과시키던
   판정을 **라벨 대조(CONFIRMED)** 로 바꿨다. unmatched 25 → 3. **오탈 2건 수정**(세 파일 동시):
   `hyperplane_incremental` +0.113 → **+0.112**, oldcap600 미탐 없음 +0.430 → **+0.429**.
   남은 3건은 오류가 아니라 **출처 공백**: envelope 표 절대 수준(−0.1343/−0.1222)과 부록 C의
   0.506/0.964/+0.195는 커밋된 JSON이 없다(각각 per-seed 미저장 / `AUDIT_FINAL` §C1이 출처).
   재현 패키지 만들 때 재실행 대상. **수치는 바꾸지 않았다.**
2. **V5 재구성 골격 검토** → 채택 시 §3·§6 신설 초안.
3. LaTeX 재컴파일 — 로컬 툴체인 없음. **급하지 않음**, 제출 직전 1회. ⚠ 위 2건 수정이 tex에도
   들어갔으므로 다음 컴파일 때 같이 나간다.

---

## (이력) 2026-07-18 — 리버탈 런 환류·판독·**부록 B** 반영 완료 → 남은 것 = 제출 역학 (+선택 FT 프로브) (2026-07-18 저녁, 새 탭 여기부터)

**한 줄**: 서버 리버탈 런 2건([A] δ(N) 스윕, [B] 두-렌즈 head-to-head)이 수동 환류(서버 push
인증 불가 → 루트에 raw 5개 커밋: summary_20260718T{040332,052219,055856}_f6e65b6.json,
representation_summary2.json, whyshift_summary2.json)되어 판독·반영 완료. **δ(N): 판정 전 셀·
전 N(실현 1.5k~24k, 윈도우 기하가 상한) 불변, floor로의 dose-response 없음** (homecredit
denoised ≈+0.005 = floor의 ¼서 평탄·상단 축소; weather −0.022→−0.002 수렴; raw는 전부 유의
음수). **head-to-head: overlap 성립 셀 2개(maps D 0.58 / ACS D 0.515)에서 두 렌즈 일치**
(maps gap −0.003 [−.004,−.002] 전 표현 ~0 / ACS gap −0.009 ≈ placebo −0.008). → 논문 **부록
B**(B.1 δ(N) 표 + B.2 렌즈 일치 표) 신설 + §6 ACS 포인터 + §7 한계(1b) 갱신, EN md→KO→tex
전파, 재컴파일 검증(16쪽, 오버풀 0). 장부 = RESULTS §30, 큐 상태 = REVIEW_ROUND2 (1·2 ✅,
3=FT 프로브만 잔여·선택). 예상 리뷰 공격 ①(스케일)·②(기존 렌즈 비교) 방어재 확보.

**라운드 3 반영 (같은 날 저녁, v4.3 — 기록 = REVIEW_ROUND3_2026-07-18.md)**: 2차 검증
보고서(✅44/⚠️9/❌1) 전 지적 유효 → 7건 즉시 반영: **❌ EMBER §2↔§6 수치 불일치 해소**(§2가
대체된 전체-이력 판독 −0.012/0.0014를 잔존 인용 → 아티팩트 재검증 후 인증서급 −0.008/0.0013
으로 통일) / 초록 인증서 회계 정밀화(식별가능 null·불안정 셀 포함) / §1 10-패널 회계 완결 /
§2 검출기 head-to-head 범위 문장 / §3.1 수위 완화("기대이지 정리 아님") / §4.1 5-시드 사유 /
Fig.2 색각 중복-인코딩 문구. EN→KO→tex 전파, 재컴파일 클린.

**라운드 4 (조판본 검증, 같은 날 — 기록 = REVIEW_ROUND4_2026-07-18.md)**: 판정 = **"사실상
제출 가능"** (수치 전수 대조 불일치 0, 형식·익명성·그림 통과). 신규 2건 반영 → v4.4:
고아 참고문헌 3건 인용 복원(Johansson→§2 positivity / Moreno-Torres→§3.1 estimand /
Vela→§3.1 decay; `\nocite` 제거) + Fig.2 집계 밴드 2행화(타일 원척 렌더). 재컴파일 클린.

**장부 채점 라운드 (리뷰어2 페르소나의 REVIEW_LEDGER 채점, 같은 날 심야 — v4.6)**: 6사유
매핑 전부 수용 확인 + 신규 지적 반영: ① **A3 구멍 수정** — sine_reoccur2가 논문 내
반례(늦은 재발 = recency 양수)임을 수용, §6 문장을 "지평-지배적 재발 배제"로 재스코프하고
sub-window 주기성(일중·요일) 미커버 명시 ② **[C] 판독 사전 규정 = PREREG §14 커밋**(발화 시
0/8 → train-구간 재스코프 + 신규 발견 프레이밍; [D] 양방향 판독도 동일 문서에 형식화 — 서버
실행 *전*에 푸시됨) ③ **부록 C 신설**: 판정 상수 7종 표(값·역할·캘리브레이션 출처·유효
범위) ④ A13 신설(Kaggle 인증 재현성, 5/10 셀은 인증 불요 명시) ⑤ **제출 게이트 5개 확정**
(REVIEW_LEDGER §3): [C]·[D] 반영 / A3 재스코프✅ / §14 사전 커밋✅ / A9 지금 진행 / 냉각
통독 1회. 선택 카드 우선순위 = 고전 탐지기 배터리 표 > MLP 튜닝 > FT(보존). git 태그
`prereg-cd-2026-07-18`(§14 시점 고정)·`v4.6` 푸시됨.

**라운드 6 ("리뷰어 2" 적대 리뷰, 같은 날 밤 — 기록 = REVIEW_ROUND6_2026-07-18.md, 대응 수위
= B)**: Reject 권고 6사유 판정 — #1 절반 반박(**신규 실측: 산업 10셀 전부 recency ≥ 0 = 재발
지문 침묵**, §6에 문장 추가) / #3 수용(전 라운드 최강: 주입 인증서 = 단일 패밀리 →
family-상대성 명시 + sweep 구현) / #2·세부 문장급 수정 / #4·5·6 반박 논거 기록. → **v4.5**
(초록 볼드 트리-스코프, 딥-아키텍처 비판정 문장, §7 한계(6), sub-floor 헤지 등 7건, EN·KO·tex
재컴파일 클린). **실험 2종 구현·스모크 완료**: `--tabred-span full`(배포 갭 감사) +
`--inj-family lowvar|interaction|subpop`(인증서 패밀리 sweep). **서버 커맨드 (tmux, 순차)**:
```bash
conda activate explaintab311 && cd ~/explainableTabular && git pull
# [C] 배포 held-out 구간 감사 — 8개 TabReD full-span (행 태그 _fullspan)
python scripts/run_deployment_decay.py --tabred sberbank_housing homecredit_default ecom_offers homesite_insurance weather cooking_time delivery_eta maps_routing --tabred-span full --n-seeds 10
# [D] 주입-패밀리 sweep — 인증서 셀 6개 + insects(양성 대조), 패밀리 3종
python scripts/run_deployment_decay.py --tabred cooking_time delivery_eta ecom_offers homecredit_default weather --elec2 --insects --n-seeds 10 --inj-family lowvar
python scripts/run_deployment_decay.py --tabred cooking_time delivery_eta ecom_offers homecredit_default weather --elec2 --insects --n-seeds 10 --inj-family interaction
python scripts/run_deployment_decay.py --tabred cooking_time delivery_eta ecom_offers homecredit_default weather --elec2 --insects --n-seeds 10 --inj-family subpop
```
환류 = `results/phase1/deployment_decay/summary_*.json` 새 파일 4개(각 런 1개) 루트 복사 후
commit·push (인증 안 되면 로컬 전달). 판독·부록 B.3/B.4 반영은 로컬 탭. 예상: [C] 반나절,
[D] 하룻밤. 판독 후 제출 진행.

**라운드 5 (최종 독립 검증, 같은 날 — 기록 = REVIEW_ROUND5_2026-07-18.md)**: **본문 수정
요구 결함 0건** — 검증 사이클 수렴(R3 ❌1 → R4 경미2 → R5 0), 본문 v4.4로 확정. 초록 밀도
지적은 라운드 2 반영으로 이미 충족 상태라 보류(기록에 사유). 고전 탐지기는 "배터리 통과"
방식으로 카드 ④ 설계 격상(river ADWIN/DDM/KSWIN/PageHinkley → 14-셀, type-blind 실증 표).
참조 깨짐·overfull 최종 실측 = 0. 검증자 착오 1건 기록(존재하지 않는 "Crosignani 2025").

**다음 = 제출 역학만 (사용자 행동)**: ① 익명 미러(anonymous.4open.science; 커밋 이력 보존 +
**서명 태그/OSF 등록으로 사전등록 제3자 증빙 격상** + 커밋 author 메타데이터 점검; tex §8
TODO) ② Overleaf pdfLaTeX 1회 확인 ③ 투고 직전 concurrent 문헌 재스캔("선행 없음" 주장 3곳
방어) ④ Broader Impact Statement(선택) 포함 여부 결정 ⑤ OpenReview 제출(양식 LLM 공시 확인).
(선택 리버탈 카드: FT-Transformer 프로브 / 고전 검출기 회고 1표 / MLP 튜닝 재실행 —
REVIEW_ROUND3 §큐.)

### (이력) 리뷰 라운드 2 반영 완료 (**v4.2**) (2026-07-18 낮)

**한 줄**: 외부 점검 보고서(11-섹션 체크리스트, KO본 대상) 접수 → 12개 지적 전부 판정·처리
(**기록 = REVIEW_ROUND2_2026-07-18.md**). 수용 9건 반영 = v4.2 (EN md 정본 → KO → tex 전파,
tectonic 재컴파일·육안 검증): 초록 압축 / §2 유령 "명제" 문구 제거 / §3 Reading aid(sberbank
접지) / **Figure 2 지도 시각화**(`paper/figures/fig2_map_body.tex`+래퍼, §5.2) / §5.3 17자리→
각주(anchor는 단어 뒤 — 숫자 뒤면 오독) / §5.4 MLP 발산 원인(per_seed 아티팩트: 3/10 시드
|stale|>10) / §6 sine_reoccur2 각주(아티팩트 검증: 최종 22%만 재발·recency +0.30 양수·자매쌍
≈0) / §8 Compute 문단(CPU-only, phase 로그 wall-clock) / 부록 A 라이선스 표(전 항목 배포처
검증: EMBER 데이터=MIT, elec2=OpenML Public, TabReD 도구=Apache-2.0, folktables=MIT, river=
BSD-3). "학회 불일치" 지적은 전제 오류로 기각(TMLR 확정은 ELEVATION_VERDICT 2026-07-04).

**다음 작업**: ① 제출 역학 — 익명 미러(anonymous.4open.science, 커밋 이력 보존; tex §8에
TODO 주석) + Overleaf pdfLaTeX 1회 확인. (PDF 메타데이터 = 이미 깨끗 확인됨(Author/XMP 전무);
TMLR author guide에 LLM-선언 요건 없음 확인 — 2026-07-18.)
② **리버탈 실험 = 서버에서 아래 커맨드 실행** (신규 코드 불필요 — δ(N)은 기존 `--max-train`,
`_load_tabred`는 max-n 미적용이라 캡만 올리면 됨; N=6000 앵커는 본 실행에 이미 커밋됨):
```bash
conda activate explaintab311 && cd ~/explainableTabular && git pull   # tmux에서, nohup 금지
# [A] δ(N) 스윕 — null 셀 3개(리뷰 공격 표적), N ∈ {1500, 24000} + homecredit만 96000
python scripts/run_deployment_decay.py --tabred homecredit_default weather maps_routing --n-seeds 10 --max-train 1500
python scripts/run_deployment_decay.py --tabred homecredit_default weather maps_routing --n-seeds 10 --max-train 24000
python scripts/run_deployment_decay.py --tabred homecredit_default --n-seeds 10 --max-train 96000
# [B] WhyShift 병행 분해 — overlap 성립 셀 2개 (within-overlap 렌즈, P0 ess-게이트 포함)
python scripts/run_representation.py --tabred maps_routing --n-seeds 5
python scripts/run_whyshift.py --states CA --years 2014 2018 --task income
```
환류 = 각 스크립트가 마지막에 `wrote ... <-- send me this`로 찍는 summary json들 commit·push
(δ(N)은 `results/phase1/deployment_decay/summary_*.json` 3개). 예상: [A] 합쳐 하룻밤(96k 셀이
최장), [B] 1~2h. 판독·부록 B 반영은 로컬 탭이 함. FT-Transformer급 프로브(GPU·신규 코드)는
[A][B] 환류 후 착수 여부 결정.
③ 수치·문구 수정은 md 정본 먼저 → tex 전파.

### (이력) 이전 현행 = 집필 파이프라인 종결 (LaTeX ✅ + KO ✅) (2026-07-16)

**한 줄**: 사전등록 §0~§13 전부 실행·판독·커밋 (실험 큐 종결, 2026-07-15와 동일). **② TMLR
LaTeX 전환 완료**: `paper/main.tex` + `paper/main.bib`(30항목, DOI 2건 doi.org 대조) +
공식 `tmlr.sty/tmlr.bst/fancyhdr.sty`(JmlrOrg 원본) + 표 4개(§3 용어 11행 / §4.1 배터리
14행 / §4.3 클래스 매트릭스 5클래스 / §5.2 지도 10행) + Figure 1 TikZ include. **로컬
tectonic(XeTeX)으로 컴파일·육안검증 완료**(14쪽 PDF, 인용 전부 해석, 표·그림 렌더 정상;
잔여 경고 = 긴 `\texttt` 경로 토큰의 underfull 2건, 무해). fig1은
`paper/figures/fig1_cascade_body.tex`(단일 소스) + `fig1_cascade.tex`(standalone 래퍼)로
분리, `fig1_cascade.pdf` 생성·확인 — **이전 탭의 "컴파일 미검증" 해소**. **④ 국문 전파
완료**: PAPER_DRAFT_V4_KO.md = v4.1 전문 동기화(누락이던 §2 pseudo-label 문단, §3 용어표,
§3.1 CI 방어, §4.1 표+floor 민감도, §4.3 MLP 행·문단, §5.4 MLP 패널, §6 EMBER 인증서급
격상+ACS 브리지, §7 (1b)스케일·향후과제, 표 4개 국문화 반영).

**다음 작업 (이 순서로)**:
1. (권장) Overleaf에 `paper/` 업로드 → **pdfLaTeX**로 최종 컴파일 확인 (로컬 검증은
   XeTeX(tectonic); 엔진 차이로 인한 문제는 예상 없음이나 제출 전 1회 확인 가치).
   main.bib의 DISDE(cai2023disde) 항목 = arXiv로 인용 중 — 카메라레디 전 OR 게재 여부 확인.
2. (선택) 리뷰 라운드 2: 완성된 LaTeX(`paper/main.pdf` 커밋됨)를 외부 검토자에게.
3. 제출 역학: TMLR OpenReview 계정/포털 확인, 익명 모드 확인(현재 `\usepackage{tmlr}` =
   submission = 자동 익명). preprint 전환 시 main.tex의 author 블록 TODO 채울 것.
4. 수치·문구 수정 시 **md(정본) 먼저 고치고 .tex에 전파** (헤더에 명시됨).
**읽는 순서**: 이 블록 → PAPER_DRAFT_V4.md (정본) → paper/main.tex (제출본) →
PREREG_DEPLOYMENT_V2.md §0~13 (전 결과 장부) → RESULTS.md §29. 감사 배경 = AUDIT_FINAL /
ELEVATION_VERDICT (2026-07-04). 서버 규칙: tmux에서 실행, nohup 금지 (메모리에 있음). 실험
재개 필요 시 드라이버 = run_prereg_phases.sh. 로컬 LaTeX 필요 시: tectonic 포터블
(github.com/tectonic-typesetting 릴리스 zip, 스크래치패드에 풀어 `tectonic main.tex`).

### (이력) 이전 현행 = 실험 큐 완전 종결 → LaTeX 전환 + 국문 전파 (2026-07-15)

**한 줄**: 사전등록 §0~§13 전부 실행·판독·커밋 (선택 실험 포함: ember2018=인증서급
DECAY-COVARIATE, folktables CA=최고검정력 null·WhyShift 브리지, MLP 프로브=카나리아 판정).
**논문 PAPER_DRAFT_V4.md = v4.1 전문 완성** (외부 리뷰 라운드 1 반영: MLP 실험·floor 민감도·
CI 방어·스케일 한계·denoised 문헌연결·초록 축약·용어표·Figure 1 mermaid). 서지 검증 완료
(Souza·Gower-Winter[IDA 2026 정식출판 격상]·TableShift·Lu). figures/fig1_cascade.tex =
스탠드얼론 TikZ (컴파일 미검증 → **2026-07-16 tectonic으로 검증 완료, paper/figures/로 이동**).

### (이력) 이전 현행 = v3 실행 완결 → 집필 단계 (2026-07-05)

**한 줄**: 사전등록(PREREG_DEPLOYMENT_V2.md §0~10) 하에 v3 계기(denoised staleness+noise gate+
group-aware D+학습가능성 주입; 합성 배터리 14/14)로 Phase 1~4 전부 실행·판독·커밋 완료.
**최종 지도 = 산업 mean-rule drift 0/8, insects 단독 CONCEPT, sberbank는 라벨노이즈-감쇠로 진단**
(9번째 dissolution, 최초의 계기-내 진단). 확증시드 10/10 재현·HGB↔RF 일치·앵커 단조 9/9 발화·
EMBER null. 정본: **RESULTS §29** + PREREG §7~10 + prereg_results/. 배경: AUDIT_FINAL_2026-07-04.md
(v2 3중 기각) + ELEVATION_VERDICT_2026-07-04.md (천장 = TMLR/D&B; main-track 이론 라인은 기각 —
Hinder 2023/Shimodaira/Loog 2019 선점).

**다음 작업 = 논문 (TMLR 즉시 / D&B 차기)**:
1. PAPER_DRAFT_V3 → **v4 재작성**: 골격 = identifiability map + 수리된 계기(유효 envelope 포함) +
   구조별 감도 프로파일(단조 발화/재발 침묵+음의 recency 지문) + 진단된 sberbank + EMBER null.
   Retired 주장(재발 niche +0.26/+0.21, RESULTS §26)은 v3 draft에 정정 배너로 표기됨 — v4에서 제외.
   Related work는 WhyShift 선두 + Hinder 2023/D'Amour/Johansson/Shimodaira/Window-Dilemma 인용.
2. 선택 실험: ember2018(2018-only by-value) 셀 업그레이드 / folktables 브리지(신규 로더 필요).
3. 워크플로우 변경 없음: 로컬 작성→push, 서버 pull→실행.

### (이력) 이전 현행 = V4 종료 + 도구 자기검증 필요 (2026-06-28)
**상황**: 1년/8도메인(TabReD8·ACS·elec2·insects·BAF·IEEE-CIS·EMBER) 전부 음성. 6번 dissolution(시간검색/concept/C1퍼즐/
공간시간/생성법칙/재발niche) + V4 positivity 척추 기각(benign 전처리 artifact) + V4-B 적대도메인 0/8. RESULTS §17~28, PLAN_V4.
**완성·검증된 자산**: 측정 도구킷(`drift_measure.py`: covariate_shift_auc·concept_within_overlap[ess-gated·placebo]·
disde_iw_degeneration), run_positivity_regime(harm: perf_drop·conformal), run_adversarial_probe(generic CSV verdict),
prep_ember(JSONL 직접파싱), river 패널. 적대리뷰 패키지·ideation 4제안 = `Desktop/ExplainableTab_ReviewPackage/`.

**★사용자 핵심 의문(미해결·최우선)**: "착취가능 concept drift가 *부재*라는 게 말이 되나? *내 측정 방법이 틀린* 게 더
현실적 아닌가?" → **타당. "absent"는 over-claim; earned된 건 "overlap 렌즈로 측정불가"뿐.**
**가장 의심스러운 자산 = within-overlap 기구 자체**: ① 구조적 맹점 — covariate 강하면 overlap이 sliver고 거기서 abstain →
concept 사는 *new 영역(non-overlap)*을 설계상 못 봄(DISDE term-ii를 초기 선택 후 무검증). ② malware=smoking gun: concept
drift 문헌상 확립인데 EMBER가 concept~0/drop0.030 → 도구/split 문제 가능성(balanced median, 非TESSERACT).

**다음 탭 최우선 = broad-negative 쓰기 *전에* 도구 자기검증**(within-overlap 버리고 현실 배포-decay):
1. **EMBER TESSERACT 충실 holdout**: train ≤ month M → test month M+k, 불균형 유지(malware ~10%), 시간별 AUC decay 곡선.
   (현 balanced-median 0.030 decay가 protocol artifact인지 판정.) prep_ember는 보유; split만 현실화.
2. **rolling-origin decay + adaptation-gain**: TabReD/fraud/EMBER서 과거학습→미래테스트 decay + recency/재학습이 복구하나
   (run_anchors의 recency·no_change 재활용). 우리가 "측정불가/부재"라 한 곳에서 **decay 크고 adaptation 복구**면
   → negative는 도구 artifact, **진짜 양성** + 스토리("overlap 기반 concept 측정은 중요 regime서 맹목").
3. 음성(decay 작음·adaptation 무효)이면 → broad-negative 논문 *확정*(이제 within-overlap+deployment 양쪽서 음성=단단).
**판정 후**: 양성→피벗(deployment-decay 프레임). 음성→측정/negative-results/D&B-tools 논문 작성(골격: 도구킷+8도메인
Shift Cards+positivity boundary+정직한 negative들). 지도교수 언급 금지(메모리).

---

## ☆☆☆ 현행 = V3 재건 진행 중 (2026-06-17 갱신) — 새 탭은 여기부터
> 읽는 순서: **이 블록 → `PLAN_V3.md`(현행 계획·게이트 판정) → `RED_TEAM.md`(7-에이전트 검토·왜 재건)
> → `PAPER_DRAFT_V3.md`/`_KO.md`(현행 초안) → RESULTS.md**. (PLAN_V2/PREREG_V2/PAPER_DRAFT(v0.1)는 역사.)

**무슨 일**: v0.1 초안을 7-에이전트 적대 검토(RED_TEAM.md)로 자가-red-team → 리드 청구(Claim A 보편형)에
구조적 구멍 발견(home-field 교락·표현 의존성·estimand 부재·퍼즐 미입증) → **재건 결정(PLAN_V3)**. 규율 =
nucleus 죽일 결정적 게이트 *먼저*, 형식 재작성 *나중*.

**V3 진행 상태**:
- **V3.0 게이트 전부 통과/재범위화 (PLAN_V3 상단 판정)**: G1 placebo → elec2 +0.146/insects +0.150이 placebo
  한참 위 = 진짜 concept(home-field 아님). G2 표현 → disjoint TabReD 4/5가 희소표현서 측정가능+concept≈0,
  ecom만 진짜 disjoint, concept 양성은 표현 바꿔도 생존. G3 도구킷 견고(노이즈 +0.034 caveat). G4 웹: 신규성=
  측정불가성+abstention로 좁힘(adversarial-validation 계보 인용 필요, WhyShift가 대조 지지).
- **V3.1 형식 척추 완료**: PAPER_DRAFT_V3(.md/_KO) — estimand(DISDE term-ii 시간축) + positivity 정리(§9
  자기모순 해결) + 표현-인지 §6 + 게이트 통제(§5 placebo 표).
- **V3.2 C1 완료**: cov_AUC가 TabReD per-dataset margin 예측 **못 함**(Spearman +0.22 p=.61; ecom 반례) →
  §7 정직 축소("TabReD 퍼즐 설명 주장 안 함"). `c1_ranking_summary.json`.
- **✅ V3.2 C5 완료 (2026-06-23, `whyshift_summary.json`, 커밋 81dab7f)**: ACSIncome/folktables서 SPATIAL
  cov-AUC 0.94/gap≈0(4/4 meas), TEMPORAL cov-AUC 0.68/gap≈0(5/5 meas). **예측한 공간=Y|X/시간=X 대조 *미지지***
  (둘 다 concept≈0, 공간이 *더* covariate=정반대) → 또 하나 overclaim retire. **WIN**: 도구킷 ACS 일반화 +
  ~10피처서 둘 다 measurable=§6 표현논점 외부 증거. PAPER_DRAFT_V3(_KO) §2·§6·§11 + RESULTS §17 + PLAN §C5 반영 완료.
- **✅ V3.3 위생 완료 (2026-06-23, `gap_hygiene_summary.json`)**: 5항목(seed-CI / ℓ-robustness / rolling-origin
  / BH-FDR / 민감도 그리드)을 `run_gap_hygiene.py` 한 스크립트로. **판정: elec2·insects 둘 다 CLAIM-A CONCEPT**
  (4조건 전부 + 민감도 18/18 셀 불변). bias-corr +0.181/+0.167, 메트릭 불변(Brier·log-loss·KL), 모든 cut 양성
  (insects는 cut에 단조감소=drift 앞쪽집중), BH 둘 다 reject. **한계 a/b/g 닫힘.** RESULTS §18 + PAPER_DRAFT_V3(_KO)
  §5 경화 문단 + 향후-작업 2·5 완료 반영.
- **✅ C2 앵커 완료 (2026-06-23, `anchors_summary.json`)**: knn±t/GBDT±t/no-change를 elec2+insects 2변종서.
  **2결과**: ① 신경 arm floor 통과(elec2 mlp_t≈0.905 > lgbm 0.887 > no_change 0.845 → Žliobaitė 비판 해소);
  ② **GBDT±t가 메커니즘 독립 재현**(시간피처 incremental +0.070 / abrupt −0.192 = 신경 훅 in-dist도움/외삽해).
  §8 외부보정 문단 + RESULTS §19 반영. "정품 TabR"=R0-경화 tabr_t 충족.
- **🔄 C3 §9 Cai&Ye faithful = 코드 완료, 서버 실행 대기 (2026-06-23)**: `run_modulation_adjudication.py --lr-grid`
  (양 arm val-튜닝). 서버: `... --all --elec2 --insects --mod-basis fourier --lr-grid 2e-3 1e-3 5e-4 2e-4 --n-seeds 10`
  → `modulation_adj/summary_fourier_tuned.json`.
- **🔄 V3.4 외부 적대리뷰 대응 (2026-06-24~) — 현 작업**: 별도 LLM 비판검증(`Desktop/ExplainableTab_ReviewPackage`).
  - **P0 ✅ 완료**: ess-floor 미집행(MISSED-1)+§6 cherry-pick(반론A)=같은 뿌리. `drift_measure` measurable 게이트에
    `ess_pct≥5.0` 추가(전 caller 일관)+None-guard. **ground-truth 로컬 재검증 PASS**(mu=0.70/1.5/3.0 abstain). `run_representation`
    §5급 엄격화(다중시드CI+사전등록 verdict). RESULTS §20, PLAN_V3 §V3.4.
    **서버 재실행 대기**(pull 후):
    ```
    python scripts/run_representation.py --tabred sberbank_housing homecredit_default ecom_offers \
      homesite_insurance weather --elec2 --insects --n-seeds 5
    python scripts/run_gap_hygiene.py --elec2 --insects   # elec2/insects가 ess≥5로 생존하는지=Claim A 확인
    ```
  - **P1 🔄 코드 완료, 서버 대기**: `run_elec2_decompose.py` — thinning(자기상관 파괴) + lagged-label ablation +
    Bayes-noise proxy로 Elec2 +0.146 분해. **결정적 go/no-go**(Elec2 생존=concept 2개 / 탈락=INSECTS 1개). 서버:
    `python scripts/run_elec2_decompose.py --elec2 --insects`. **P2 ⬜**: D'Amour 2021 등 인용·§6 신규성 재범위화.
    **P3 ⬜**: floor 단위·placebo 음수오프셋·C1 §7.
- **🔄 V3.4 적대리뷰 대응 = P0·P2·P3 완료, P1 서버 결과 중간 도착 (2026-06-25)**:
  - **P1 decompose 완료 (둘 다)**: **Elec2 ❌ 탈락** — de-time-leaked +0.073이 thinning(stride5 +0.033)·
    **lagged-label(+0.073→+0.013 below-floor, y_{t-1}→y AUC 0.849)**·noise(late +0.046)로 자기상관·노이즈가 대부분.
    **INSECTS ⚠️ 부분생존** — thinning 통과(stride1 +0.149/stride5 +0.157 둘 다 concept = *자기상관 아님*, Elec2와 차이)
    BUT **achievable-acc drift +0.192**(late 훨씬 쉬움)가 +0.149와 엉켜 깨끗이 분리 안 됨; lagged-label은 multiclass라 없음.
    → **깨끗한 concept 증거 얇음** → 생성 테스트(V3.5)가 진짜 심판. (INSECTS는 matched-noise-null로 추가 분리 필요.)
  - **P0/P2/P3 완료**: ess-gate 집행·ground-truth 재검증 / D'Amour 2021 인용·§6 재범위화 / floor 단위·placebo·C1 §7.
- **🔄 V3.5 = 생성 테스트 (사용자 falsification, 코드 완료·서버 대기)**: `run_correct_assumption.py` — recency-적응이
  concept 측정된 곳서만 정적 모델 이기나? 양수+패턴=진단 생성적(긍정 결과) / 무패턴=피벗 신호. 로컬 합성 통과
  (concept gain +0.91 / covariate +0.03). 서버: `... --tabred <8개> --elec2 --insects`. **현 최우선 go/no-go.**
- **✅ V3.5 생성 테스트 결과 (2026-06-26)**: `Spearman(concept_gap, recency_gain)=+0.60`(p=.28, n=5) — 방향 양수.
  **INSECTS(concept +0.144)→recency_gain +0.054** = 깨끗한 concept 유일 데이터셋이 유일하게 적응 이김; cooking/maps
  (≈0)→≈0. **프레임이 예측하는 모양이나 INSECTS 단일 의존(n=1)** → 피벗 아님, **"폭 확장" 신호.** gap_hygiene2 확정:
  elec2 abstain(claim_a False)/INSECTS measurable(claim_a True, floor 0.041).
- **🔄 폭 확장 = 코드 완료, 서버 대기 (2026-06-26, 현 최우선)**: `src/data/river_streams.py`(SEA/Agrawal/STAGGER/Sine/
  Hyperplane × {no-drift/abrupt/gradual/incremental} = ~12 스트림, concept 크기 다이얼) + `run_correct_assumption.py
  --river all --insects-variants ...`. 한 번에 concept_gap + recency_gain + Spearman. 로컬 river 미설치라 서버서 검증.
  서버: `pip install river && python scripts/run_correct_assumption.py --river all --insects-variants incremental_balanced
  abrupt_balanced incremental_abrupt_balanced incremental_reoccurring_balanced --tabred cooking_time maps_routing --n-seeds 5`
  → `correct_assumption_summary.json`(맨 끝 Spearman이 핵심).
- **✅ V3.5-C 1단계 POWERED 양성 (2026-06-26)**: retrieval(kNN) vs recency, reoccurring **+0.192 [+.026,+.358] CI>0** /
  monotonic −0.098 / nodrift ~0. 사전등록 패턴 통계 확정 = retrieval이 재발 drift 도구. RESULTS §24. **5번 dissolution 끝 첫 양성.**
- **🔄 V3.5-C 2단계 = 학습형 retrieval 부활 (코드 완료, 서버 GPU 대기)**: `run_learned_retrieval.py` — `tabr_t`(학습 retrieval
  구조) vs `mlp_t`(파라메트릭) vs recency, reoccurring 패널서. **사전등록**: reoccurring서 retrieval_struct_gain>0 ∧
  >recency, monotonic서 ≤0(원 Claim B 음성) → **죽은 Claim B가 *재발 niche* 조건부 양성으로 부활**. 서버(무거움, 부분셋 먼저):
  `python scripts/run_learned_retrieval.py --river stagger_reoccur stagger_reoccur2 sine_reoccur2 agrawal_reoccur sea_abrupt
  stagger_abrupt --insects-variants incremental_reoccurring_balanced incremental_balanced --n-seeds 3` → 검증 후 `--river all`.
- **⬜ 남은 것**: 2단계 결과 → 학습형이 kNN·recency 이기면 §8(Claim B) "재발 drift서 구조 우위"로 재작성 = 논문 새 척추;
  못 이기면 1단계 측정-결과로 마감. 적대리뷰 패키지 갱신.

**V3 코드맵(새로 추가)**: `run_gap_controls.py`(G1 placebo+nulls), `run_representation.py`(G2 표현),
`run_toolkit_adversarial.py`(G3), `run_c1_ranking.py`(C1), `run_whyshift.py`(C5), **`run_gap_hygiene.py`(V3.3
위생 5항목)**. 진단 코어 `src/analysis/drift_measure.py`에 `concept_within_overlap(permute_time=)`(placebo) +
**`concept_within_overlap_multi`(다중-메트릭 ℓ-robust + clf 선택)** 추가됨.
**결과 artifact(루트, _synth=합성/없음=실데이터)**: gap_controls_*, representation_*, toolkit_adversarial_,
c1_ranking_, disde_degeneration_, toolkit_validation_, modulation_adj_, summary_fourier, q1_verdict_*.

**다음 탭 첫 행동**: (1) 서버서 C5(`whyshift_summary.json`) + V3.3(`gap_hygiene/summary.json`) 회수·반영.
(2) 안 돌았으면 위 두 명령 전달. (3) 두 결과 반영 후 C2 앵커. 상세·우선순위 = PLAN_V3 §V3.2/§V3.3/우선순위.

---

## (이력) R0~R2 완료 (2026-06-17) — V3 재건 *이전* 상태
- **R1**: V2 재검정 = 구조 음성(유의, CI<0). **R2.1** 문헌(A 미선점). **R2.2** DISDE 퇴화표 10데이터셋 3분법(RESULTS §13).
  **R2.4** 도구킷 ground-truth 검증 4/4 PASS(§14). **R2.5** Q1 큰-회전: 바닥 0.894→0.017, 복원 0.988 10/10 PASS(§15).
- **R2.3 판결**: 정의적(label-free⇒X-side)=확정. 경험적(최소 재현 gain↔cov_AUC)=trend 외삽붕괴(−0.5 무효)→
  fourier도 ~null(+0.231 약함)=충실 재현 아님 → **PREREG §8대로 inconclusive, LAMDA repo gold 재현=future work**(§16).
- **남은 것**: ①지도교수 정렬(R1+R2 들고) ②D&B 범위확장(데이터셋·방법 sweep, 도구킷 패키징, R2.3 faithful 재현).
- 코드/실험 미실행 없음. 결과 artifact: `summary_fourier.json`, `q1_verdict_a6.28_fourier.json`, `disde_/toolkit_ 등`.

## (이력) R1 완료 — V2 재검정 판정 = 구조 음성(유의) (2026-06-14)
25시드 본 실행 완료. **PREREG §4 판정: 구조 우위 = NO**, 교락 없는 *유의* 음성.
주 대비 `time_tabr_t − tabr_t`(temporal, val-fair): incremental −0.0067 [CI −.012,−.001] p=.006,
incremental_abrupt −0.0205 [−.034,−.008] p<.001 — **두 clean 변종 모두 CI<0 유의 음성**. (abrupt ρ=−.43
게이트 탈락; reoccurring ρ=.33 경계지만 trend-기저 외삽 병리 time_tabr_t→0.19, §4 red flag → 크기 비사용; elec2 ρ=−.34 보조.)
- **교락 제거로 pre-V2보다 깨끗·강함**: ①기판 경쟁력(tabr_t−mlp_t≈0~+.011, −.038 적자 소멸) ②시간은 검색 도움(time_tabr_t−tabr=+.042)
  but 피처가 더 나름 ③**★in-dist vs 외삽 뒤집힘**: random서 훅 도움(+.005,+.021)/temporal서 해(−.007,−.021)
  = 시간-인덱싱 훅은 in-dist 장치, 외삽 장치 아님(redundancy 직접 뒷받침) ④trend기저는 비단조 drift서 외삽붕괴(Claim A 먹이).
- 결과 jsonl 환류·커밋. RESULTS §12, FINDINGS "V2 RE-TEST VERDICT" 참조.
- **R2 진행상황 (PLAN_V2 §R2)**:
  - ✅ **R2.1 문헌 검증(웹 원문)**: Claim A 코어 **미선점**(측정프레임만 DISDE와 PARTIAL). REFERENCES §0.
  - ✅ **R2.2 DISDE 퇴화**(`run_disde_degeneration.py`): 10데이터셋 3분법 확정. RESULTS §13. (서버 실행·환류 완료.)
  - 🔄 **R2.3 Cai&Ye 판결 인프라 완료**(아래 커밋). **정의적 절반 = 코드로 증명**(변조는 label-free X-side,
    REFERENCES §0.1). **경험적 절반 = 서버 실행 대기**:
    ```
    python scripts/smoke_test_modulation.py    # 배선/identity-init 검증 (먼저)
    python scripts/run_modulation_adjudication.py --config configs/phase1.yaml --all --elec2 --insects --n-seeds 5
    ```
    기대: gain↔cov_AUC 양의 상관, gain↔concept_gap ~0, **cooking/maps(concept≈0)서도 변조 이득>0 = X-side**.
    결과 `results/phase1/modulation_adj/summary.json` 환류.
  - ✅ **R2.4 도구킷 검증**(`run_toolkit_validation.py`): covariate×concept 4×4, **4/4 PASS**. RESULTS §14. (로컬 실행 완료.)
  - 🔄 **R2.5 Q1 큰-회전 인프라 완료**(`run_q1_faithfulness.py` 확장: `--angle-max`/`--basis`/`--n-harmonics`/`--tag`).
    기존 게이트(π/2+trend)는 기본값으로 불변. **R2.5 robustness = 큰 회전+Fourier 정합**(기저-불일치 교락 회피):
    ```
    python scripts/run_q1_faithfulness.py --angle-max 6.283 --basis fourier --n-harmonics 4   # 2π 전회전
    # (변형) --angle-max 3.1416 --basis fourier  # π 반회전(규칙 완전반전)
    ```
    기대: 큰 회전이면 바닥(shuffle-t)이 ~0으로 내려가 동적범위 넓어짐 → 메커니즘이 그래도 복원(≥PASS선)하면
    "충실"이 헤드라인 논거로 robust. 출력 `results/phase1/q1/q1_verdict_<tag>.json`. ⚠ 출력 파일명이 태그식으로
    바뀜(기존 `q1_verdict.json`→`q1_verdict_a1.57_trend.json`).
  → R1+R2 결과 들고 지도교수 정렬(워크숍 now / **NeurIPS D&B 주타깃**). Claim A 리드, B는 분해와 함께 보조.

## 한 문단 요약 (2026-06-12 대전환 — 위 ★★가 현행 최신)
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

## R1.1 튜닝 결과 (2026-06-13, 서버 3시드 — 본 실행 전 기록)
서버에서 0단계 smoke 3종 통과 + 앵커 + topk 튜닝(3시드) 완료. 결과는
`elec2_q2_diagnostics.jsonl` / `insects_q2_diagnostics.jsonl`(로컬 환류됨, repo 루트).

**① topk 선택 (PREREG_V2 §7 규칙 = 검색 3-arm best-mean-val 평균 최대 K):**
- incremental_balanced → **32**, incremental_abrupt_balanced → **128**, abrupt_balanced → 8(탈락예정),
  elec2(보조) → 128. ※ topk는 4째 소수점에서 갈리는 무의미 축(노이즈) — 해석 시 민감도 없음 보고.

**② ★abrupt_balanced가 val→test ρ 게이트 탈락 (PREREG_V2 §3):**
- ρ: incremental +0.80~0.95 ✅ / **abrupt −0.43~−0.64 ❌** / incremental_abrupt +0.57~0.70 ✅.
- 원인: abrupt drift + trend 외삽 상호작용(test t가 train 밖 → 틀린 regime으로 외삽 → 시간피처가 해침).
  앵커가 예고: abrupt에서 lgbm_t 0.459(lgbm 0.664 대비 폭락), V2 `*_t` arm도 0.55~0.59로 붕괴(tabr는 0.65 건재).
- → clean에서 제외, `incremental_reoccurring_balanced`로 보충. 부수발견: "trend 시간피처는 abrupt서 실패".

**③ 예비 신호(3시드, 판정 아님):** 주 대비 `time_tabr_t−tabr_t` ≈ 0(clean 변종 −0.001~−0.045),
  보조 `time_tabr_t−tabr`·`tabr_t−tabr`는 일관 +(+0.01~+0.04). **V2 기판 수정 작동**: INSECTS `tabr_t`≈0.685로
  pre-V2 mlp_t(0.670) 상회, elec2 `tabr_t`≈0.90으로 옛 mlp_t 동급(감사의 substrate 적자 −0.038 상당 해소).
  → 잠정 해석 "시간은 기판을 돕지만 *구조적 인덱싱*이 *피처*를 못 넘음"(교락 없는 유효 음성 후보). 25시드가 결정.

**앵커 기록 (1셀 seed0):** elec2 lgbm 0.887/lgbm_t 0.884/knn 0.851/**no_change AUC 0.845**(acc 0.846 — 문헌 비판대로 강함,
보고 의무) → 신경망 arm(mlp_t 0.90)이 그 위. INSECTS no_change 0.16~0.59(위협 아님). lgbm_t가 incremental서 0.679로
pre-V2 최고 arm(0.670) 상회 → 외부 바닥선 필수.

## ★ R1.2 본 실행 (다음, 서버) — 커맨드는 위 "다음 행동"이 아니라 아래
```bash
# (a) 보충 변종 topk 튜닝 3시드
for K in 8 32 128; do python scripts/run_elec2_q2.py --dataset insects \
  --insects-variant incremental_reoccurring_balanced --report-grid --n-seeds 3 --splits temporal \
  --bases trend --archs tabr tabr_t time_tabr_t --lr-grid 1e-3 5e-4 2e-4 \
  --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk $K; done
# (b) 본 실행 25시드 4-arm temporal: incremental(k32) / incremental_abrupt(k128) /
#     incremental_reoccurring(k=위 선택) / elec2 보조(k128) / abrupt(k8, 게이트가 잠긴규칙으로 탈락시키게 형식상 1회)
#     archs = mlp_t tabr tabr_t time_tabr_t. 명령 형식은 incremental 예:
python scripts/run_elec2_q2.py --dataset insects --insects-variant incremental_balanced \
  --report-grid --n-seeds 25 --splits temporal --bases trend --archs mlp_t tabr tabr_t time_tabr_t \
  --lr-grid 1e-3 5e-4 2e-4 --dropout 0.1 --weight-decay 1e-4 --min-epochs 20 --topk 32
# (c) random 대조 10시드(clean 통과 변종만): --splits random, 나머지 동일.
```
판정 = PREREG_V2 §4 (주 대비 time_tabr_t−tabr_t, valfair, clean ≥2/3, CI>0∧p<.05∧g_z≥.5). 결과 2 jsonl 환류.

## 미커밋 상태
없음. 다음 = 서버 R1.2 본 실행(위 ★).
