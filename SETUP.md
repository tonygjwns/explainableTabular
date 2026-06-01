# SETUP: 환경 구축 및 실행 가이드

> 새 탭이 이 문서를 따라가면 Phase 0 시작 직전까지 갈 수 있도록 작성.

---

## 1. 디렉토리 구조 권장

```
~/Desktop/ExplainableTab/             ← 이 폴더 (문서들)
~/Desktop/explainableTabular/         ← 우리 GitHub repo (clone 후)
~/Desktop/external/                   ← 외부 baseline repo들
  ├── tabm/                           ← TabM 공식
  ├── tabred/                         ← TabReD 벤치마크
  ├── lamda-talent/                   ← ModernNCA (TALENT 벤치마크 포함)
  └── tabular-temporal-modulation/    ← Cai et al. NeurIPS 2025
~/data/                               ← 데이터셋 (큰 파일들)
  └── tabred/                         ← TabReD 데이터
```

---

## 2. GitHub Repository 링크

### 우리 작업 repo (이미 만들어 둠)
```bash
git clone https://github.com/tonygjwns/explainableTabular.git
cd explainableTabular
```

### 외부 baseline / 핵심 reference repos

| Repo | 용도 | URL |
|---|---|---|
| **TabM** | 우리 백본, 필수 | https://github.com/yandex-research/tabm |
| **TabReD** | 벤치마크 + 평가 프로토콜 | https://github.com/yandex-research/tabred |
| **LAMDA-TALENT** | ModernNCA + TALENT 벤치마크 (관련 도구) | https://github.com/qile2000/LAMDA-TALENT |
| **Tabular-Temporal-Modulation** | Cai et al. NeurIPS 2025 (직접 경쟁작) | https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation |

### Reference 전용 (clone 안 해도 됨, 코드 참조용)

| Repo | 용도 | URL |
|---|---|---|
| EvolveGCN | 시간 진화 가중치 — 인용용 | https://github.com/IBM/EvolveGCN |
| Latent ODE | 연속 시간 잠재 — 인용용 | https://github.com/YuliaRubanova/latent_ode |
| TimeMCL | WTA centroid 학습 — Phase 2에서 aMCL annealing 차용 가능 | https://github.com/Victorletzelter/timeMCL |

### Clone 명령

```bash
mkdir -p ~/Desktop/external
cd ~/Desktop/external

git clone https://github.com/yandex-research/tabm.git
git clone https://github.com/yandex-research/tabred.git
git clone https://github.com/qile2000/LAMDA-TALENT.git lamda-talent
git clone https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation.git tabular-temporal-modulation
```

---

## 3. 데이터셋

### TabReD 8개 데이터셋

> ⚠️ **데이터는 repo에 없음. GitHub = 스크립트, Kaggle = 실제 데이터.**
> (yandex-research/tabred 코드 직접 확인함. `preprocessing/<script>.py`가 `import kaggle`로
> Kaggle API를 통해 다운로드 후 전처리.)

**데이터셋 목록 (folder name → 우리 key)**:
| Kaggle competition (규칙 수락 필요) | folder | 우리 key | task |
|---|---|---|---|
| Sberbank Russian Housing | sberbank-housing | sberbank_housing | regression |
| Homesite Quote Conversion | homesite | homesite_insurance | binclass |
| Acquire Valued Shoppers (Ecom) | ecom-offers | ecom_offers | binclass |
| Home Credit Default Risk | homecredit | homecredit_default | binclass |

| Kaggle dataset (계정만) | folder | 우리 key | task |
|---|---|---|---|
| Cooking Time | cooking-time | cooking_time | regression |
| Delivery ETA | delivery-eta | delivery_eta | regression |
| Maps Routing | maps-routing | maps_routing | regression |
| Weather | weather | weather | regression |

**다운로드 절차 (실험 머신에서, 검증된 방법)**:
```bash
# 1. Kaggle API 준비
pip install kaggle
#    Kaggle 웹 → Settings → API → Create New Token → kaggle.json 다운로드
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. 위 4개 competition은 웹에서 "Join Competition / Accept Rules" 클릭 (1회, API로 불가)

# 3. tabred repo에서 데이터셋별 스크립트 실행 (repo root 기준)
cd external/tabred
mkdir -p data
python preprocessing/sberbank-housing.py
python preprocessing/homesite.py
python preprocessing/ecom-offers.py
python preprocessing/homecredit.py
python preprocessing/cooking-time.py
python preprocessing/delivery-eta.py
python preprocessing/maps-routing.py
python preprocessing/weather.py
```

**출력 포맷 (우리 `tabred_loader.py`가 직접 읽음)** — `data/<folder>/`:
- `X_num.npy`(float32) / `X_bin.npy`(float32) / `X_cat.npy`(int64) / `X_meta.npy`(int64) / `Y.npy`
- **`X_meta[:, 0]` = timestamp** (8개 모두 일관 — int64)
- `info.json` = {name, task_type, score?}
- `split-<name>/{train,val,test}_idx.npy` — 사용 가능 split:
  - **`default`** = 공식 시간 분할 (Phase 0 재현용 — 논문 수치 매칭)
  - **`random-{0,1,2}`** = 무작위 분할 (Cai "random vs temporal" 대조군, Test 4)
  - **`sliding-window-{0,1,2}`** = 슬라이딩 시간 윈도우

**우리 로더 사용**:
```python
from src.data.tabred_loader import load_tabred
from pathlib import Path
ds = load_tabred("sberbank_housing", Path("external/tabred/data"), split="default")
# ds.train.t 는 [0,1]로 정규화된 timestamp (test는 >1.0 가능 — 미래 외삽용)
```

**Cai 분할**: TabReD `default`(시간)·`random`은 디스크에 있지만, Cai & Ye ICML 2025의
*개선* 분할(lag=0, bias-min)은 `src/data/splits.py:cai_resplit`로 구현해둠 (그들 코드와 대조 검증 필요).

**대용량 처리**: Maps Routing(6.5M), Weather(13M), HomeCredit(1.5M)은 전처리 스크립트가
이미 서브샘플 버전을 만들 수 있음 (스크립트 내부 확인). 필요 시 추가 서브샘플.

---

## 4. Python 환경 셋업

### Conda 권장

```bash
# 새 환경 생성
conda create -n explaintab python=3.10 -y
conda activate explaintab

# PyTorch (CUDA 12.x 가정)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 핵심 패키지
pip install \
  numpy pandas scipy scikit-learn \
  optuna \
  tqdm omegaconf hydra-core \
  matplotlib seaborn umap-learn \
  pyarrow

# TabM 관련 의존성 (TabM repo의 requirements.txt 따르기)
cd ~/Desktop/external/tabm
pip install -r requirements.txt
```

### GPU 환경 확인

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 출력: True 2  (H100 ×2 확인)
```

---

## 5. 실험 코드 구현 순서

이 폴더의 우리 repo에 다음 순서로 모듈 작성:

### Week 1: 데이터 로더 + TabM wrapper

```
explainableTabular/
├── src/
│   ├── data/
│   │   ├── tabred_loader.py     ← TabReD 데이터 로더 (Cai 분할 적용)
│   │   └── splits.py             ← 학습/검증/테스트 분할 유틸
│   ├── models/
│   │   └── tabm_wrapper.py       ← TabM을 우리 파이프라인에 맞게 래핑
│   ├── training/
│   │   └── trainer.py            ← 기본 학습 루프
│   └── utils/
│       ├── seed.py
│       └── metrics.py            ← AUC, RMSE 등
├── configs/
│   └── tabm_baseline.yaml
└── scripts/
    └── run_phase0.py             ← Phase 0 entry point
```

**구현 우선순위**:
1. `tabred_loader.py`: TabReD 데이터 로딩 + Cai & Ye 2025 분할 protocol
2. `tabm_wrapper.py`: TabM 코드 직접 import 또는 fork
3. `metrics.py`: Wilcoxon, FDR, Hedges' g 함수
4. `run_phase0.py`: 작은 데이터셋에서 TabM 재현 검증

### Week 2-3: Cai et al. modulation 추가 + Phase 0 완료

```
src/
├── models/
│   ├── temporal_embedding.py    ← Fourier 시간 임베딩 (Cai & Ye ICML 2025)
│   └── feature_modulator.py     ← Yeo-Johnson 변조 (Cai et al. NeurIPS 2025)
└── scripts/
    └── run_phase0_cai.py         ← Cai 베이스라인 재현
```

### Week 4-5: 시간 인덱싱 메모리 + 검색 (최소 버전) 구현 + Phase 1 sanity check

```
src/
├── models/
│   ├── prototype_memory.py       ← 시간 인덱싱 메모리 P_k(t) = P_k^base + drift_k(Fourier(t))
│   │                                 (fixed/time-indexed 두 변형 토글)
│   ├── retrieval.py              ← 단순 softmax 검색 + 집계 (WTA·보정항 없음, 결정 3)
│   └── value_module.py           ← V_k = 라벨분포 임베딩 + 학습벡터 (결정 2)
├── analysis/
│   ├── extrapolation_test.py    ← Sanity Test 2 (외삽 검증)
│   ├── retrieval_analysis.py    ← Sanity Test 3 (검색 가중치 집중도)
│   ├── trajectory_viz.py        ← Sanity Test 3 (궤적 PCA/UMAP)
│   └── time_injection_ablation.py ← Sanity Test 4 (메모리/입력/둘 다 비교)
└── scripts/
    └── run_phase1_sanity.py
```

**구현 주의** (EXPERIMENT_PLAN.md §6 최소 버전 엄수):
- 프로토타입은 **시간 슬라이스별 KMeans 초기화** (결정 5, 랜덤 아님)
- 검색은 **단순 softmax** — WTA/annealing/TabR 보정항/외적 게이팅은 Phase 1에 **넣지 않음**
- 시간은 **메모리(주) + 입력(보조)** 양쪽 — head 출력단에만 넣지 말 것

**중요**: 이 단계에서 PRE_REGISTRATION.md commit 완료 상태여야 함.

### Week 6+: Phase 2a 또는 2b

Sanity check 결과에 따라 분기.

---

## 6. 통계 분석 유틸 (구현 필수)

```python
# src/utils/stats.py

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import numpy as np

def paired_wilcoxon(scores_a, scores_b):
    """
    scores_a, scores_b: shape (n_datasets, n_seeds)
    같은 시드끼리 페어드 비교.
    """
    # 시드 평균 또는 모든 (dataset, seed) 쌍
    diffs = scores_a - scores_b
    return wilcoxon(diffs.flatten())

def benjamini_hochberg(p_values, alpha=0.05):
    """다중 비교 보정."""
    return multipletests(p_values, alpha=alpha, method='fdr_bh')

def hedges_g(scores_a, scores_b):
    """효과 크기 (작은 표본 보정)."""
    n_a, n_b = len(scores_a), len(scores_b)
    s_pooled = np.sqrt(((n_a-1)*np.var(scores_a, ddof=1) + (n_b-1)*np.var(scores_b, ddof=1)) / (n_a+n_b-2))
    cohen_d = (np.mean(scores_a) - np.mean(scores_b)) / s_pooled
    correction = 1 - 3/(4*(n_a+n_b)-9)
    return cohen_d * correction
```

---

## 7. 실험 추적

### 매일 짧게 기록할 것

`progress.md` 파일 권장:
```markdown
# Progress Log

## 2026-XX-XX
- Phase 0 시작
- Sberbank Housing 데이터 로드 성공
- TabM 첫 시드 학습: AUC 0.86 (논문 수치 0.86 일치 ✓)
- Issues: ...
```

### 결과 저장

```
results/
├── phase0/
│   └── tabm_baseline/
│       └── sberbank_housing_seed0.json
├── phase1/
└── phase2/
```

JSON 형식: `{model, dataset, seed, metrics, hyperparams, timestamp}`

---

## 8. 흔한 함정 미리 알기

1. **TabM은 BatchEnsemble 구조**. 메모리+검색 레이어를 얹을 때 32 submodel을 단순 평균하지 말 것 (TabM 원래 설계 깨짐). Phase 1은 단일 백본 표현으로 시작하고, 32 결합은 **EXPERIMENT_PLAN.md §9 Phase 2 ablation**에서 결정 (옵션: submodel별 메모리 / 공유 메모리 + submodel별 query).

2. **시간은 head(출력단)에만 넣지 말 것**. Cai 증거상 출력단은 12.6% 효과. 시간은 **메모리 P_k(t)(주) + 입력(보조)** 로. (준이 질문의 핵심 — Test 4로 검증)

3. **Phase 1에 화려한 요소 금지** (ablation factory 함정). WTA/annealing(TimeMCL), TabR 보정항, 외적 게이팅은 전부 Phase 2 ablation으로. Phase 1은 단순 softmax 검색만.

4. **Cai et al.의 Yeo-Johnson 변환은 분포 모양 변경**. PLR 임베딩과 충돌. PLR 사용 모델엔 입력 단 변조만 권장.

5. **TabReD 데이터셋별 timestamp 형식이 다름**. 처음에 통일된 datetime 형식으로 변환 필요. (메모리 P_k(t)의 t 인덱스로 직접 쓰이므로 정규화 중요)

6. **회귀 데이터셋 5개 (Sberbank, Cooking, Delivery, Maps, Weather)** 에서 label-conditional smoothness 적용 시 `|y_i - y_j| < δ` 사용. δ는 데이터셋별 표준편차의 0.1배 정도부터 시작.

7. **Optuna 튜닝 비용 큼**. Phase 0에서 TabM 권장 하이퍼파라미터부터 시작, 본격 튜닝은 Phase 2 메인 결과에서만.

---

## 9. 자주 쓰일 명령어

```bash
# GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 백그라운드 학습
nohup python scripts/run_phase0.py --dataset sberbank > logs/phase0_sberbank.log 2>&1 &

# 결과 종합
python scripts/aggregate_results.py --phase 0
```

---

## 10. 도움 요청 채널

- TabM 관련: `yandex-research/tabm` issues
- TabReD 관련: `yandex-research/tabred` issues
- Cai et al. 코드 관련: `LAMDA-Tabular/Tabular-Temporal-Modulation` issues
- 우리 작업 관련: `tonygjwns/explainableTabular` issues

핵심 인물 이메일 (정중하게, 구체적 질문만):
- Yury Gorishniy (TabM 저자): firstnamelastname@gmail.com
- Han-Jia Ye (LAMDA): yehj@lamda.nju.edu.cn
- Hao-Run Cai: caihr@lamda.nju.edu.cn
