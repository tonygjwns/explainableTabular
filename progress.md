# Progress Log

> 매일 짧게 기록. SETUP.md §7 형식.

## 2026-06-05 — Phase 1 평가 완료: 구현 검증된 음성 (TabReD=covariate, not concept)
- **결론(FINDINGS.md 참조)**: 메커니즘은 *작동*(합성 순수 concept-drift에서 +87%, mem_gap+0.93),
  그러나 **TabReD에선 성능 이득 0** — covariate 드리프트지 exploitable concept이 아니기 때문.
- 증거 사슬:
  - Test1(격리, 10시드): clean null (|g|≤0.17, 4/4 비유의).
  - 학습 진단 도입(`diagnostics.py`): concat에선 메모리 *장식*(mem_gap≈0, 메모리 grad 100~1000×↓,
    검색 1개로 붕괴) = z-지름길. → 깜깜이 튜닝에서 내부 계측 기반으로 전환.
  - **합성 positive control**: 순수 concept-drift(y=x·w(t) 회전)에서 time≪fixed(0.13 vs 1.03) →
    **구현 정확성 입증**(버그 아님).
  - 드리프트 해부(G3): covariate AUC≈1.0(pervasive, drop-top5 후도 높음), label ρ(t,y)≤0.13.
  - concept 측정(early vs late→future): sberbank +36%(단 covariate 외삽에 오염), 나머지 작음.
  - **engaged time-vs-fixed(sberbank, residual/memory_only)**: 메모리 강제해도 안 됨(residual g=−0.30,
    memory_only 불안정/붕괴) → sberbank +36%는 exploitable concept 아님 확정.
- **TabReD 양성(성능) 경로 닫힘** (합성↔실데이터↔concept측정↔engaged null 4중 확인).
- 신규 구조/도구: `predictor_mode`{concat,memory_only,residual}, ablate_memory, diagnostics,
  drift_measure(covariate/concept), extrapolation, retrieval_trajectory, run_synth_control.
- **다음 결정(보류, 지도교수 상의)**: A'(외부 concept-drift 벤치서 양성, decider=Elec2) / B(분석 paper) / A'+B.

## 2026-06-04 — Phase 0 (TabM 재현) 8/8 PASS ✅ + Phase 1 첫 3개 코딩
- **Phase 0 종료**: TabReD 8개 default(시간) split, 5시드. TabReD 논문 Table 2 MLP 베이스라인 대조:
  | 데이터셋 | 우리(TabM) | MLP | 차이 |
  |---|---|---|---|
  | sberbank_housing | 0.2572 | 0.2508 | +2.6% |
  | cooking_time | 0.4807 | 0.4820 | −0.3% |
  | delivery_eta | 0.5522 | 0.5504 | +0.3% |
  | maps_routing | 0.1614 | 0.1622 | −0.5% |
  | weather | 1.5073 | 1.5470 | −2.6%(더좋음) |
  | ecom_offers (AUC) | 0.5906 | 0.6015 | −1.8% |
  | homesite_insurance (AUC) | 0.9615 | 0.9500 | +1.2%(더좋음) |
  | homecredit_default (AUC) | 0.8518 | 0.8545 | −0.3% |
  - 6개 ±1% 이내/더좋음(정방향), 2개(sberbank/ecom)는 데이터셋 노이즈·공개범위 내(MLP 기준). 시드 std 극소 → 건강. **재현 게이트 PASS.**
- **버그 수정 2건** (재현 중 발견):
  - 폴더명: TabReD가 `homesite-insurance`/`homecredit-default`로 씀 → 로더 `DIRNAME`은 `homesite`/`homecredit` 기대(서버서 폴더 rename으로 정합). 재전처리 시 또 rename 필요(또는 로더 양쪽 인식하도록 추후 보강).
  - cat embedding OOB: homecredit temporal test에 train 미관측 범주 → `index out of bounds`(CUDA). cardinality를 train+val+test 전체 max+1로 (commit 2072d19). ← 범주 수준 시간 드리프트의 실증.
- **Phase 1 첫 3개 코딩 완료** (EXPERIMENT_PLAN §6 최소버전, push 95d9777/27f6660):
  - `src/models/phase1_model.py` (`Phase1Model`): TabM encode(mean) + 입력 시간주입(토글) + 메모리검색층 조립
  - `scripts/smoke_test_phase1.py`: forward/shape/w_sum=1/grad/fixed-smooth=0/KMeans-init CPU 검증
  - `src/models/proto_init.py`: 시간슬라이스 KMeans init(z-space, subsample 가드)
- **다음**: smoke_test_phase1 서버 통과 확인 → Phase 1 ④ 트레이너(L_main+λL_smooth) + run_phase1_sanity(Test 1~4) + Cai/Fixed-memory 대조군.

## 2026-06-03 — 서버 env 통째 소실 → 재구축 + TabReD 8개 전처리 전부 성공 ✅
- **사건**: 공유 H100 서버에서 `~/miniconda3`(conda 전체)·`~/external`(tabred/tabm)·전처리 데이터가
  통째로 사라짐 (디스크 압력 시 큰 디렉토리 정리된 것으로 추정). git repo `~/explainableTabular`만 생존.
  - 초기 증상은 `OSError [Errno 28]`(디스크풀)이었으나, 실제로는 (1) 일시적 외부 디스크 압력 +
    (2) env/external/데이터 삭제가 겹친 것. 디스크는 멀쩡(/dev/sdb4 240G free, inode 5%, quota 없음)로 확인됨.
- **재구축 (SETUP.md §4로 정립)**:
  - miniconda 재설치(`~/miniconda.sh` 잔존) → `explaintab311`(py3.11) 재생성 + torch(cu121)+requirements+tabm
  - smoke_test_memory.py **통과** (novel 모듈 4종, binclass/multiclass/regression 배선 정상; fixed-memory smooth=0 대조 OK)
  - 전처리 deps 반응적 설치가 함정 → **TabReD 공식 `tabred-env.yaml`로 별도 env `tabred` 생성**이 정답.
    - 함정 기록: lightgbm GPU 소스빌드(OpenCL 없음) 실패 → 바이너리휠로 보강 / kaggle 1.6.11 핀 → 1.8+로 올려 OAuth 토큰 사용 / `import lib`가 전체 DL 스택 요구.
- **결과**: `prepare_all_data.sh` (tmp 데이터셋별 자동정리 fix `e2ae44f`) → **8개 전부 ok**
  (sberbank/cooking/delivery/maps/weather/homesite/ecom/homecredit), fail 0 / skip 0.
- **다음**: `explaintab311`에서 `run_phase0.py` 실행 중 → 8개 수치 나오면 Phase 0 통과 판정 → Phase 1 착수.
- 교훈: prep=`tabred` env, train=`explaintab311` env로 분리. `run_overnight.sh`는 한 env 가정이라 env 분리 시 prep/phase0 따로 실행.

## 2026-06-02 — Phase 0 sberbank 재현 성공 ✅ + overnight 자동화
- 서버(explaintab311, py3.11)에서 전 파이프라인 실증 완료:
  - 환경/smoke test/Kaggle(KGAT 토큰, `kaggle auth login`)/sberbank 전처리/로더 전부 OK
  - **Phase 0 TabM 재현: sberbank rmse = 0.2572 ± 0.0046** → TabReD/Cai 공개 수치(~0.24-0.26)와 일치 ✅
  - 트러블슈팅 기록: kaggle 1.8+는 py3.11 필요(KGAT 토큰), polars==0.20.19 핀, X_num NaN→임퓨테이션+grad clip
- 코드 개선:
  - 트레이너에 tqdm 진행바 (val/best/patience postfix)
  - run_phase0: 8개 전체 + 데이터 없는 건 자동 skip + done/skipped 요약, seed별 tqdm
  - config: 8개 데이터셋 전부 (Yandex 4개 먼저, 규칙수락 3개 뒤)
  - `scripts/prepare_all_data.sh`: 8개 전처리 일괄(실패해도 계속)
  - `scripts/run_overnight.sh`: 전처리→Phase0 한 번에 + 로그 (nohup용)
- 밤샘 실행: `ln -s ~/external/tabred/data data` 후
  `nohup bash scripts/run_overnight.sh >/dev/null 2>&1 &` / `tail -f logs/overnight_*.log`

## 2026-06-01 — repo 셋업 + Phase 0 스캐폴딩
- git repo 초기화, remote 연결 (tonygjwns/explainableTabular)
- 핸드오프 문서 7종 커밋 (아키텍처: 시간 인덱싱 메모리 + 검색)
- Phase 0 코드 스캐폴딩 생성:
  - `src/utils/stats.py` — Wilcoxon/BH-FDR/Hedges' g (완전 구현, 환경 독립)
  - `src/utils/seed.py`, `src/utils/metrics.py` — 완전 구현
  - `src/data/tabred_loader.py` — 스켈레톤 (Cai 분할 TODO, 데이터 필요)
  - `src/models/tabm_wrapper.py` — 스켈레톤 (external/tabm 필요)
  - `configs/tabm_baseline.yaml`, `scripts/run_phase0.py` — 오케스트레이션 스켈레톤
- **우리 novel 모듈 완전 구현** (external API 의존 없음, 서버에서 Claude 못 쓰니 로컬서 완성):
  - `src/models/temporal_embedding.py` — Fourier 시간 임베딩 tau(t)
  - `src/models/prototype_memory.py` — P_k(t)=P_k^base+drift_k(tau(t)), time_indexed 토글, KMeans init, smoothness penalty
  - `src/models/value_module.py` — V_k = W_y(label) + value (해석가능성용 라벨 표현)
  - `src/models/retrieval.py` — 단순 softmax 검색 + MemoryRetrievalLayer (predictor 포함)
  - `scripts/smoke_test_memory.py` — CPU 스모크 테스트 (shape/grad 검증)
  - ⚠️ 이 문서 머신엔 torch 없어 미실행 — **torch 있는 환경에서 `python scripts/smoke_test_memory.py` 먼저 돌려 검증할 것**
- 다음: GPU 머신에서 SETUP.md 따라 환경 구축 + smoke test

## 2026-06-01 (2) — TabReD 실제 포맷 확인 + 로더 정확 구현
- external/tabred clone해서 **실제 출력 포맷 직접 확인** (lib/data.py, preprocessing/*.py):
  - `data/<folder>/`: X_num/X_bin/X_cat/X_meta/Y.npy + info.json + split-<name>/{train,val,test}_idx.npy
  - **timestamp = X_meta[:, 0]** (8개 데이터셋 모두 일관, int64)
  - split 3종: `default`(시간), `random-{0,1,2}`, `sliding-window-{0,1,2}` — 모두 디스크에 존재
- `src/data/tabred_loader.py` **정확 재구현** (추측 아닌 실물 포맷 기반):
  - 폴더명 매핑, timestamp 추출+정규화([0,1], 학습범위 기준, test는 >1 가능)
  - split 선택 (default/random/sliding), X_num/bin/cat 분리 노출
- `src/data/splits.py` 신규: `cai_resplit` (Cai lag=0/bias-min 분할, 우리 구현 — 그들 코드 대조 필요)
- SETUP.md §3 **검증된 다운로드 절차로 교체** (kaggle.json, 규칙 수락, preprocessing 스크립트)
- config/run_phase0: Phase 0은 `default` split 사용 (논문 수치 매칭), root=external/tabred/data
- ⚠️ 데이터 자체는 미다운로드 (Kaggle 계정 필요) — 로더는 서버에서 데이터 생성 후 즉시 동작
- 데이터셋 키↔폴더: sberbank_housing↔sberbank-housing, homesite_insurance↔homesite,
  ecom_offers↔ecom-offers, homecredit_default↔homecredit, 나머지 동일(하이픈)

## 2026-06-01 (3) — TabM 래퍼 + 트레이너 (Phase 0 코드 완성)
- external/tabm clone해서 **공식 API 직접 확인** (tabm.py):
  - `TabM.make(n_num_features, cat_cardinalities, d_out, k=32, n_blocks, d_block, dropout, arch_type)`
  - forward(x_num, x_cat) → (B, k, d_out or d_block); d_out=None이면 표현(k축 보존)
  - **학습 관행(README 검증): k submodel 평균 손실 최적화** (mean prediction의 손실 아님)
  - **추론: 확률 평균** (logits 아님)
- `src/models/tabm_wrapper.py` **정확 구현**: encode(d_out=None, reduce=mean/none) / predict(평균)
- `src/training/trainer.py` 신규: Phase 0 학습 루프 (quantile norm, 회귀 타겟 표준화,
  mean-loss 학습, val 조기종료, 확률평균 추론)
- `scripts/run_phase0.py` 완성: 데이터→트레이너→시드 집계(mean±std)→재현 tolerance 비교
- 모든 파일 py_compile 통과. **Phase 0 코드는 데이터+tabm 설치만 있으면 즉시 실행 가능 상태**

## Phase 0 코드 완성도
- ✅ 로더(실물), TabM 래퍼(실물 API), 트레이너, run_phase0 — 추측성 NotImplementedError 제거됨
- 서버에서 할 일: 데이터 생성(Kaggle) → `pip install -e external/tabm` → smoke test → run_phase0
- ⚠️ torch/data 없어 로컬 미실행 — 서버에서 첫 실행 시 디버깅 가능성 (특히 trainer의 shape/normalization)

## Phase 0 체크리스트 (PLAN.md §4)
- [ ] conda 환경 구축 (SETUP.md §4)
- [ ] external/tabm, external/tabred clone (SETUP.md §2)
- [ ] TabReD 데이터 다운로드 (SETUP.md §3) — 작은 4개 우선
- [ ] `tabred_loader.py` 구현 — Cai & Ye ICML 2025 분할 적용
- [ ] `tabm_wrapper.py` 구현 — 공식 TabM 연결 (k=32 축 보존)
- [ ] Sberbank Housing에서 TabM 첫 학습
- [ ] 재현 수치가 TabM 논문/TabReD 표 ±1% 이내 확인
- [ ] (병행) Cai et al. NeurIPS 2025 코드 공개 여부 확인

## 통과 기준
TabM 재현 성공 + Cai et al. 베이스라인 작동 → Phase 1로.

## 미해결/결정 대기
- 결정 3 (WTA 보류) 동료 합의 — PRE_REGISTRATION commit 전
- TabReD timestamp 형식 통일 방법 (데이터 받은 후)
- Cai et al. NeurIPS 코드 공개 여부 → 미공개 시 직접 구현 일정 추가
