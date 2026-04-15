# Railway 전용 구조 — 넥스트 팔란티어 스캐너

이 저장소는 GitHub Actions용 워크플로우를 제거하고, Railway에서 바로 실행되도록 바꾼 배치형 구조입니다.

## 포함 파일

- `next_palantir_scanner.py` — 기본 실행기 (`SCANNER_MODE=nps`)
- `high_growth_valuation.py` — 보조 실행기 (`SCANNER_MODE=hgv`)
- `ticker.txt` — 기본 종목 목록
- `requirements.txt` — Python 의존성
- `start.sh` — Railway 시작 스크립트
- `Procfile` — worker 엔트리
- `Dockerfile` — Railway 빌드 안정화용
- `railpack.json` — start command 명시

## Railway 배포 순서

### 1) 새 서비스 생성
Railway에서 GitHub 저장소를 연결하고 이 폴더를 루트로 배포합니다.

### 2) Variables 설정
다음 변수를 Railway Variables에 넣습니다.

- `SCANNER_MODE` : `nps` 또는 `hgv`
- `WORKERS` : 기본 `5`
- `LOG_LEVEL` : `INFO` 또는 `DEBUG`
- `OUTPUT_DIR` : 기본 `results`
- `TICKERS` : 선택. 공백 구분 티커 직접 지정
- `NO_DETAIL` : `1`이면 상세 컬럼 축소
- `QUIET` : `1`이면 진행 로그 축소

예시:

```bash
SCANNER_MODE=nps
WORKERS=5
LOG_LEVEL=INFO
OUTPUT_DIR=results
```

### 3) Start Command
이 구조는 `Dockerfile`과 `railpack.json`이 이미 포함되어 있어서 별도 입력 없이 실행 가능하게 맞춰져 있습니다.
수동 지정이 필요하면 아래로 넣으면 됩니다.

```bash
bash ./start.sh
```

## 동작 방식

- `SCANNER_MODE=nps` → `next_palantir_scanner.py` 실행
- `SCANNER_MODE=hgv` → `high_growth_valuation.py` 실행
- `TICKERS`가 비어 있으면 `ticker.txt` 사용
- 실행 결과는 `results/`에 타임스탬프 CSV로 저장
- 최신 결과는 `results/latest.csv`로 복사

## Railway Cron 권장

이 프로젝트는 웹서비스가 아니라 **배치 스캐너**입니다.
그래서 Railway에서는 일반 Web Service보다 **Cron/Worker 방식**이 더 적합합니다.

권장 패턴:
- 서비스 타입: Worker
- 시작 명령: `bash ./start.sh`
- 스케줄: Railway Cron에서 설정

## 로컬 테스트

```bash
pip install -r requirements.txt
bash ./start.sh
```

특정 티커만 테스트:

```bash
TICKERS="PLTR SNOW RKLB" bash ./start.sh
```

보조 스캐너 테스트:

```bash
SCANNER_MODE=hgv bash ./start.sh
```
