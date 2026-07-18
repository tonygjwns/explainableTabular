# 리뷰 라운드 2 대응 기록 (2026-07-18)

> 입력 = 외부 점검 보고서 (탑티어 셀프 체크리스트 11개 섹션, 점검 대상 = PAPER_DRAFT_V4_KO.md
> v4.1). 본 파일 = 항목별 판정·조치의 영구 기록. 반영 결과 = **v4.2** (EN md 정본 → KO → tex
> 순으로 전파, tectonic 재컴파일 검증 완료).

## 판정 요약

- **수용·즉시 반영 8건**: 컴퓨트 명시 / 명제 문구 / 초록 압축 / Figure 2 / sine_reoccur2 해명 /
  MLP 원인 / 17자리 각주화 / running-example 읽기 보조 / 라이선스 부록 (9건이나 #8·#11 통합 집계).
- **리뷰어 전제 오류로 기각 1건**: "학회-템플릿 불일치" — v4 초안 헤더가 이미 **TMLR(즉시) /
  D&B(차기)**로 확정 (ELEVATION_VERDICT 2026-07-04: main-track 이론 라인 기각). TMLR 템플릿이 맞음.
- **제출-시점 절차로 보류 2건**: 익명 미러 생성(§8 문구·tex TODO 주석은 선반영), LLM 선언 정책 확인.
- **리버탈 큐로 이관 3건**: WhyShift 병행 실행, δ(N) 부분 스케일링, FT-Transformer급 프로브.

## 항목별 처리

| # | 지적 | 판정 | 조치 (위치) |
|---|---|---|---|
| P0-1 | 학회 확정 | **기각(전제 오류)** | 이미 TMLR 확정 — 초안 헤더·ELEVATION_VERDICT. 조치 불요 |
| P0-2 | 컴퓨트 미기재 | **수용** | §8 Compute 문단: CPU-only sklearn / 1노드 / py3.11.15·sklearn1.9.0·numpy2.4.6(freeze 커밋) / phase-병렬 ≈17h + 패널 ≈2h + 선택 ≈13h — 커밋된 phase 로그 타임스탬프로 산출 |
| P0-3 | 익명 저장소 | **부분 반영+보류** | §8 "linked in anonymized form for review" + tex `% SUBMISSION TODO`(anonymous.4open.science, 커밋 이력 보존). 미러 생성은 제출 시 |
| P0-4 | 명제 진술 부재 | **수용(문구 수정)** | 실체: 명제는 어디에도 없음(v3 형식척추의 유령 참조). §2 → "we state no theorems of our own — measured boundary, not a proved one" |
| P1-5 | 초록 과밀 | **수용(보수적)** | 문장 분할·중첩 괄호 해소·(+0.098) 제거, 핵심 수치(+0.021 vs +0.024)·0/8·9/9 유지 |
| P1-6 | Figure 2 부재 | **수용** | `paper/figures/fig2_map_body.tex`(+standalone 래퍼) — 10타일 색상 그리드 + 0/8 집계 밴드, §5.2 배치, fig1 팔레트 |
| P1-7 | sine_reoccur2 모순 외견 | **수용(각주)** | 아티팩트 검증: 최종 ~22%만 재발(경계 0.35/0.78), 중간 레짐 = 라벨-반전형(함수 0↔3), recency **+0.299 양수**(재발 지문의 정반대), 자매쌍(reoccur/reoccur3) ≈0. §6 각주 — 렌즈 의미론과 정합, 모순 아님 |
| P1-8 | 용어 인지부하 | **수용(경량 대안)** | 통폐합 대신 리뷰어의 대안(2): §3 서두 *Reading aid* — sberbank 사례로 raw/gate/denoised/캐스케이드/injection 접지. 용어 이름은 PREREG·아티팩트와 결합돼 있어 개명 리스크 > 이득 |
| P1-9 | 라이선스 표 | **수용** | 부록 A (Table 5). 전 항목 배포처 검증: TabReD 도구 Apache-2.0(+Kaggle 대회규칙) / elec2 OpenML "Public"(API licence 필드) / river BSD-3 / **EMBER 데이터 = MIT**(코드 AGPL-v3 미사용 — LICENSE.txt 원문) / folktables MIT(GitHub API) |
| P2-10 | WhyShift 병행 | **리버탈 큐** | overlap-성립 셀 = maps_routing(D 0.58)·ACS(D 0.515). v3 도구 재사용: `drift_measure.concept_within_overlap`·`run_whyshift.py` |
| P2-11a | MLP 불안정 원인 | **수용** | per_seed 아티팩트(optional_raw 163057): sberbank staleness [+60.9, −11.0, −15.5, 나머지 7개 ±2 이내] → "3/10 시드 발산, 미튜닝 프로브의 간헐적 최적화 실패" §5.4 |
| P2-11b | 17자리 수치 | **수용** | 본문 +0.0239 + 각주 전체 정밀도. **각주 anchor는 단어 뒤**(숫자 뒤면 "+0.02391"로 오독) |
| P2-12 | FT-Transformer 프로브 | **리버탈 큐** | 배터리 = 사전등록 진입 관문(§7 한계 (1)이 이미 명문화). GPU·신규 코드 필요 |

기타 ⚠: 기여 4개 압축 — 각 항목 첫 문장이 이미 자립적이라 보류(과도한 재작성 리스크).
§5.3 "최초 사례" 헤지 — v4.1에서 이미 "We are not aware of a prior case"로 완화되어 있음(KO도 동기화).

## 리버탈 대비 실험 큐 (서버; 우선순위순) — 1·2 실행·반영 완료 (2026-07-18, git f6e65b6 런)

1. ✅ **WhyShift/DISDE 병행** — 실행 완료, **일치**: maps_routing gap −0.003 [−.004,−.002]
   (5/5 시드 measurable, ESS 93%, sparse MI@{5,10,20,50}도 전부 ~0) / ACS CA gap −0.009 ≈
   placebo −0.008 (cov-AUC 0.68). → **부록 B.2** + §6 포인터. raw = representation_summary2.json
   / whyshift_summary2.json. RESULTS §30.
2. ✅ **δ(N) 스윕** — 실행 완료(캡 1500/24000/96000, 실현 N ≈1.5k~24k — 윈도우 기하가 상한):
   판정 전 셀·전 N 불변, floor로의 dose-response 없음 (homecredit denoised ≈+0.005 평탄·축소,
   weather −0.022→−0.002 수렴, maps 안정 음수; raw는 전부 유의 음수 = old가 도움). → **부록
   B.1** + §7 한계(1b) 갱신. raw = summary_20260718T*.json ×3. RESULTS §30.
3. ⬜ **FT-Transformer급 프로브 배터리 1차** (2~3일, GPU): 5-컨트롤 분리 배터리 통과 여부만.
   통과→패널 승격 / 탈락→"두 번째 신경 카나리아"로 §4.3 강화. 어느 쪽이든 논문이 이김. (선택)
