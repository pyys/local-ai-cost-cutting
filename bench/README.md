# bench - 비용/성능 벤치마크

[부록 - 벤치마크](../docs/ko/98-benchmark.md)의 모든 수치를 만들어낸 스크립트다.

```
bench.py             조건 하나를 콜드 스타트로 N회 측정
bench_all.py         전 조합 무인 실행 + 요약
thread_scaling.py    CPU 스레드 스케일링 - 인코딩 (아래 별도 설명)
mps_scaling.py       GPU SM 스케일링 - MPS 로 할당 비율을 바꿔 가며 인코딩 (아래 별도 설명)
```

---

## 답하려는 질문

> **LLM용 32GB 카드는 이미 있다. 이미지 생성을 위해 추가로 무엇을 살 것인가?**

| 조건 | 워커 카드 | 인코더 | 워커 |
|---|---|---|---|
| 1 | V100 32GB x1 | 같은 카드 (통째로 마운트) | 1 |
| 2 | V100 16GB x1 | RAM | 1 |
| 3 | V100 16GB x1 | P104-100 | 1 |
| 4 | V100 16GB x2 | P104-100 | 2 |
| 6 | V100 32GB x1 | P104-100 | 2 (같은 카드) |
| 7 | V100 32GB x1 | P104-100 | 1 (진단용) |

*조건 5(워커 3)는 카드가 없어 측정하지 못했고 조건 3/4에서 역산한다.*

---

## 준비

**1. 전제 조건**

- [`patches/sdcpp/`](../patches/) 세 패치를 적용해 빌드한 `sd-server`
- 측정 중 다른 서비스를 전부 정지 (LLM 서버 포함)
- **root 권한** - 콜드 스타트를 위해 `/proc/sys/vm/drop_caches` 에 쓴다

**2. GPU UUID 확인**

```
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

**3. `bench.py` 상단 설정 블록 수정**

`----- 여기부터 -----` 와 `----- 여기까지 -----` 사이의 경로와 UUID를 자기 환경 값으로 바꾼다.

---

## 실행

조건 하나만:

```
python3 bench.py --condition 3 --count 8 --repeat 3
```

전 조합 무인 실행 (약 1시간):

```
nohup python3 bench_all.py > /root/bench/console.log 2>&1 &
```

진행 확인:

```
tail -40 /root/bench/run.log
```

조건 6/7만 추가로 (기존 결과에 덧붙는다):

```
nohup python3 bench_all.py --extra > /root/bench/console6.log 2>&1 &
```

요약 재출력 - `results.csv` 에서 다시 계산하므로 몇 번이든 가능하다:

```
python3 bench_all.py --summary
```

---

## 부하 중 클럭과 온도를 반드시 함께 기록할 것

별도 터미널에서:

```
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv -l 5
```

이 프로젝트에서는 **같은 모델/같은 폼팩터의 카드가 냉각 조건 때문에 12% 느렸다.** 총 소요 시간만 기록했다면 그 차이를 "모듈 분리 덕분"이라고 잘못 귀속했을 것이다 -> [부록 9장](../docs/ko/98-benchmark.md#9-반증---32gb-카드가-느렸던-이유)

---

## 측정 설계에서 주의한 것

**타이머는 서비스 기동이 아니라 첫 요청 시점에서 시작한다.**
`sd-server` 는 가중치를 lazy 로드하므로 포트가 열려도 모델은 아직 안 올라와 있다. 기동 시점에 타이머를 걸면 모델 로딩이 빠진 값이 나온다.

**콜드 스타트는 프로세스 종료 + 페이지 캐시 비우기 둘 다 해야 한다.**

```python
pkill -f sd-server  ->  종료 확인  ->  sync  ->  drop_caches=3
```

**"8장"은 1장을 만든 뒤 7장을 더 만든 값이 아니다.**
명령 한 번으로 8장을 요청했을 때의 총 소요다. 실제 사용 패턴과 맞추기 위해서다.

**조건 1만 `batch_count=8` 을 쓴다.**
인코더가 같은 프로세스에 있는 구성에서 8개 요청으로 쪼개면 8번 재인코딩하게 되어 불공정하다. 배치가 그 구조의 자연스러운 운용이다.

**프롬프트 길이를 조건 파일 크기로 검증한다.**

```
토큰 수 = (파일크기 - 3328) / 16384
```

`results.csv` 의 `cond_bytes` 열이 전 행에서 같아야 같은 프롬프트로 측정된 것이다.

---

## `pgrep -f` 를 셸 경유로 실행하지 말 것

```python
subprocess.run("pgrep -f sd-server", shell=True)   # X 항상 발견됨
subprocess.run(["pgrep", "-f", "sd-server"])       # O
```

`shell=True` 는 `/bin/sh -c pgrep -f sd-server` 를 띄우는데 **그 래퍼 셸의 명령줄에 검색어가 들어 있다.** `pgrep` 은 자기 PID 만 제외하므로 대상이 하나도 없어도 "살아 있음"이 된다.

이 한 줄 때문에 **첫 무인 실행 24회가 전부 즉시 실패했다.** 파생 원칙 - **무인 실행에 넣기 전에 정리 로직을 최소 1회는 실제로 태워 볼 것.**

---

## `thread_scaling.py` - CPU 스레드 스케일링

**"CPU 코어를 늘리면 저가 GPU를 따라잡을 수 있는가"** 에 답하는 별도 측정이다. 텍스트 인코딩만 재며 디퓨전 워커는 띄우지 않는다.

```
python3 thread_scaling.py
```

스레드 1/2/4/8 각각에 대해 256토큰/512토큰을 재고, 마지막에 P104를 같은 프롬프트로 측정해 비교군으로 둔다. **약 25분** (1스레드가 대부분을 차지한다).

SMT 구간만 추가로 재려면:

```
python3 thread_scaling.py --smt
```

16스레드와 **기본 스레드 설정**(`-t` 미지정)을 잰다. 기존 결과에 덧붙으므로 `thread_scaling.csv` 를 지우지 말 것. 약 4분.

```
python3 thread_scaling.py --summary
```

### 이 스크립트에서 놓치기 쉬운 것

**조건 파일 캐시를 매번 지운다.** 파일명이 프롬프트 해시라서, 안 지우면 두 번째 요청이 인코딩을 건너뛰고 0초가 나온다. **이것 하나로 실험 전체가 무의미해진다.**

**워밍 1회를 버린다.** 첫 요청에 모델 로드가 섞이므로 그대로 두면 스레드가 적은 구성이 부당하게 불리해진다.

**시간은 HTTP 왕복이 아니라 로그에서 파싱한다.** `sd_encode_conditioning completed, taking Xs` 를 읽으므로 다른 측정과 계측 지점이 같다.

**토큰 수를 조건 파일 크기로 검증한다.** 프롬프트가 의도한 길이가 아니면 로그에 그대로 찍힌다.

결과 해석은 [부록 7-3](../docs/ko/98-benchmark.md#7-3-왜-cpu-대비-25배인가---연산-강도와-분기점)에 있다.

---

이 디렉터리의 코드는 [MIT](../LICENSE) 다.


## `mps_scaling.py` - GPU SM 스케일링

**답하려는 질문은 하나다 - GPU 에서 코어 수와 성능이 선형인가.** 선형이면 "총 처리량 / 코어 수 = 코어당 평균" 이라는 지표를 쓸 수 있다.

MPS 로 SM 할당 비율을 100 / 50 / 25 / 12.5% 로 바꿔 가며 512토큰 인코딩을 잰다.

```
python3 mps_scaling.py
```

MPS 없는 기준선도 함께 잰다. 약 10분. 결과는 `/root/bench-mps/result.txt` 에 누적된다.

**측정 대상은 V100 이다.** MPS 의 스레드 비율 제한은 Volta 이상에서만 동작하므로 **P104(Pascal) 에서는 같은 실험을 할 수 없다.**

주의할 점 셋.

- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 는 클라이언트가 뜰 때 읽힌다. **지점마다 sd-server 를 새로 띄운다** (스크립트가 알아서 한다)
- sd-server 는 가중치를 lazy 로드한다. **1회차는 모델 로드가 섞이므로 버린다**
- MPS 의 비율 제한은 SM 을 물리적으로 끄는 것이 아니라 할당을 제한한다. **클럭과 대역폭은 그대로 남는다**

측정값에는 HTTP 왕복과 조건 gguf 8.39MB 쓰기가 포함된다. `시간 = 고정비용 + k/SM` 으로 회귀하면 이 고정 비용이 분리되고, **연산 부분은 SM 수에 반비례한다.** 해석은 [부록 12장](../docs/ko/98-benchmark.md#12-sm-스케일링과-커널-경로-실측) 에 있다.

### 커널 처리율은 llama.cpp 로 잰다

같은 부록의 12-2 는 `test-backend-ops` 결과다. 운영 빌드와 섞이지 않도록 별도 디렉터리에 빌드한다.

```
cd /root/llama.cpp && cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70" -DLLAMA_BUILD_TESTS=ON
```

```
CUDA_VISIBLE_DEVICES=<카드 UUID> ./build-bench/bin/test-backend-ops perf -o MUL_MAT
```

아키텍처를 빠뜨리면 그 카드에서 실행 자체가 안 된다. **재는 카드가 전부 들어가야 한다.**
