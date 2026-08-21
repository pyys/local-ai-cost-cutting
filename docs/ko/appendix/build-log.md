[English](../../en/appendix/build-log.md) / **한국어**

# 부록 - 구축 기록

> 이종 GPU 서버를 **바닥부터 세운 실제 절차**다. 본문이 "왜 그렇게 하는가"라면 이 문서는 "무엇을 어떤 순서로 쳤는가"다.
> 명령은 이 프로젝트의 환경에 맞춰져 있으므로 **그대로 복사하지 말고 자기 경로로 바꿔야 한다.**

관련 문서: [공통 방법론](../00-method.md) / [No.1 역할 배정](../01-role-assignment.md) / [함정 모음](../99-pitfalls.md) / [설계 기록](design-record.md)

---

## 배경 - 여기에 도달하기까지

이 프로젝트의 최종 구성은 처음부터 설계된 것이 아니라 **세 차례에 걸쳐 변경된 결과**다. 시행착오를 포함해 시계열로 정리한다.

### 발단

상용 LLM에는 여러 형태의 제약이 걸려 있다. 문제가 된 것은 **정액제 토큰 사용량이 부족했던 것**, **창작물 생성에 걸리는 내용 제약**, 그리고 **특정 유형의 코드 분석이 거부되는 것** 세 가지였다. 로컬 구축이 유일한 선택지였고, 이를 위해 프로젝트를 시작했으며, 프로젝트 대상으로는 Qwen 27B 계열을 골랐다.

이 모델을 **양자화하지 않고(BF16)** 돌리려면 가중치만 54GB, KV 캐시까지 포함하면 65GB 이상의 VRAM이 필요했다.

첫 후보는 **A100 80GB**였다. 클라우드로 대여해 테스트한 결과 **구동 자체는 확인**했으나, 로컬 구축 시의 가격이 예산을 크게 넘겨 포기했다.

> 당시(2026년 3월) SXM4 개조 PCIe 타입의 가격은 약 **8,000 USD**였다. **이 저장소의 다른 가격은 모두 2026년 8월 기준이며, 같은 판매자의 같은 제품이 8월에는 11,000 USD였다.** 반년 사이 시세가 크게 움직였다는 뜻이므로, 중고 기반 구축에서는 이와 같은 가격 변동이 있을 수 있음을 상정해야 한다.
>
> 클라우드 테스트는 비양자화 모델의 **구동 여부만 확인**했고 성능 측정은 하지 않았다. 비교 자료로 쓸 수 없다 -> [부록 - 측정하지 않은 것](../98-benchmark.md#11-측정하지-않은-것)

### 1차 구성 - NVLink V100 32GB x2

최초로 구입한 실물 GPU는 **SXM2 V100 32GB 2장**이다. 중국에서 역설계한 NVLink 지원 보드에 카드 2장이 **NVLink 300GB/s**로 연결되고, 이 GPU 모듈이 **외장 PCIe 스위치 카드**를 통해 메인 시스템의 x16 슬롯 하나에 물리는 구조였다. 연결 케이블은 **SFF-8654** 규격이다.

> 스위치 카드는 PLX/PEX 계열로 보이나 **세부 모델명은 확인하지 못했다.** 재현할 계획이라면 확인 후 채워 넣을 것.

```
CPU      Ryzen 4350G
M/B      ASRock A320
RAM      DDR4 16GB UDIMM x2
GPU      NVLink 보드 + SXM2 V100 32GB x2
PSU      800W x2
CASE     ANiX m6509 mini
```

가격은 GPU 모듈(보드 + PSU + V100 x2 + PLX8749 + 케이블)이 약 2,100 USD, 메인 시스템을 합쳐 약 2,500 USD였다.

**이 구성도 의도한 기능을 모두 구현했다.** Qwen 27B는 Q6 양자화로 32GB 한 장에 올라갔고, 다른 32GB 카드에서 FLUX D가 돌았다. **수년 단위 운용을 예상하고 있었다.**

개선 요구가 없지는 않았다.

- 이미지 생성이 느렸다 - 768x768 / 15스텝 기준 **8장에 261.7초**(장당 32.7초)였다
- **NVLink를 실제로 쓴 적이 없었다** [관련 고지 1)](#관련-고지)

**그러나 직접적인 교체 계기는 다른 곳에서 왔다.** 원인 불명의 메인보드 파손(사운드 칩셋 완전 무응답), 그리고 **GPU 모듈만 중고로 매입하겠다는 구매자의 등장**이다. 이 둘이 겹치면서 전체 재설계로 넘어갔다.

### 재구성 요구 조건

재설계 시 세운 조건은 셋이었다.

```
1. 실행 모델   Qwen 3.6 27B Q6 x1, FLUX D x2
2. 사용 GPU    V100 32GB x1, V100 16GB x2
3. 슬롯        PCIe x16 물리 슬롯 3개 이상
```

**모델이 먼저 결정되고, 그 모델을 올릴 GPU가 결정되고, 그 GPU를 장착할 메인보드와 케이스와 전원이 차례로 결정됐다.** 순서를 뒤집지 않은 것이 이 프로젝트의 기본 방침이다 -> [작업 분석](../00-method.md)

**2번 조건에서 16GB 카드가 등장할 수 있었던 것은 그 전에 결론이 하나 나와 있었기 때문이다.** 1차 시스템에서는 FLUX가 32GB 카드 한 장을 통째로 쓰고 있었고 인코더를 분리할 이유가 없었다. 재설계 과정에서 **그 모델을 분할 적재할 수 있다는 것을 먼저 확인한 뒤에야**, 워커용으로 16GB 카드를 후보에 올릴 수 있었다 -> [No.2 인코더 분리](../02-encoder-separation.md)

### 2차 구성 - 페이퍼 테스트를 통과했으나 실물에서 탈락

x16 슬롯 3개 이상을 가진 보드 중 가장 저렴한 것을 고르고, 거기에 맞는 CPU와 RAM을 조합했다.

```
CPU      Xeon E5-2680 v4
M/B      GIGABYTE GA-X99-UD4P
RAM      DDR4 16GB ECC x2
PSU      2000W
CASE     마이크로닉스 WIZMAX 우드리안 맥스
```

**모든 품목이 페이퍼 테스트를 통과했다.** 문제는 실물에서 나왔다.

메인보드의 F22 펌웨어에는 **Above 4G Decoding 옵션이 아예 없었다.** BIOS 모딩으로 이 옵션을 활성화했고, **모딩은 실제로 작동했다.** P104를 단독 설치한 뒤 우분투에서 확인하니 **BAR가 `0x100000000`, 즉 4GiB 경계 바로 위의 64비트 주소 영역에 할당돼 있었다.** 4G Decoding이 꺼져 있으면 BAR는 32비트 공간(4GiB 미만) 안에만 잡히므로, **모딩이 실제로 반영됐다는 직접 증거다.**

> 할당되는 주소는 플랫폼마다 다르다. 이 값은 X99에서 관측한 것이고, EPYC 계열은 훨씬 높은 대역을 쓰기도 한다. **자릿수가 8자리를 넘어 64비트 영역에 잡히는지**가 판단 기준이지 특정 값이 아니다.

**그럼에도 V100을 장착한 어떤 조합에서도 POST를 통과하지 못했다.**

| 구성 (전부 4G Decoding 활성 후) | 결과 |
|---|---|
| P104 단독 | **동작** |
| V100 32GB x1 + 16GB x2 | POST 실패 |
| V100 16GB x2 | POST 실패 |
| V100 32GB 단독 | **POST 실패** |

> 위는 테스트한 순서다. **V100 16GB 단독은 테스트하지 않았다.**

**4G Decoding은 원인이 아니었다.** 모딩으로 해결하고 P104로 검증까지 했는데 증상이 남았기 때문이다. **원인은 규명하지 않았다.** 부품별 테스트와 해결 방안 설계에 약 4시간을 쓴 뒤, 추가로 투입해야 할 자원과 그 결과로 얻을 것을 추정한 뒤, 비용 대비 효용이 크지 않다고 판단해 **CPU와 메인보드 교체를 결정했다.**

이 단계에서 **X79와 X99 기반 플랫폼 전체가 후보에서 빠졌다.** 주된 이유는 플랫폼의 근본적인 PCIe 레인 부족이다 -> [No.1 7-2](../01-role-assignment.md#7-2-상위-등급-gpu를-하위-등급-gpu로-분할할-때는-pcie-슬롯-비용을-감안해야-한다)

> **페이퍼 테스트는 실물 테스트를 대체하지 못한다.** 슬롯 개수와 레인 수와 펌웨어 버전이 스펙 시트에서 모두 통과해도 POST에서 막힐 수 있고, 그 원인이 스펙 시트에 적힌 항목이 아닐 수 있다. **같은 조합으로 실제 구축한 사례를 찾는 것이 유일하게 확실한 검증이다.**

### 3차 구성 - 최종

CPU와 메인보드만 교체한 구성이 최종이 됐다.

```
CPU      AMD EPYC 7232P                  PCIe 128레인
M/B      ASRock Rack EPYCD8-2T
RAM      DDR4 16GB ECC x2
PSU      2000W
CASE     마이크로닉스 WIZMAX 우드리안 맥스
OS       Ubuntu 24.04.1
드라이버 580 계열                         <- Maxwell/Pascal/Volta 를 덮는 마지막 브랜치
```

GPU를 장착한 뒤 **1차 구성에서부터 쓰던 NVMe 250GB를 이식**해 모델 저장소로 쓰고, OS는 별도 SATA SSD 480GB에 설치했다.

이 프로젝트에서는 **1차 구성의 NVMe를 손상 없이 옮길 수 있었다.** 모델 파일과 소스 트리를 다시 받지 않아도 됐던 것이 시간을 크게 아꼈다. [관련 고지 2)](#관련-고지)

대체 구성 설계 단계에서 같은 예산으로 카드를 다시 사야 했고, 그 과정에서 "각 작업에 필요한 만큼만 배정한다"는 원칙이 나왔다 -> [No.1 역할 배정](../01-role-assignment.md)

### 1차 대 3차 - 무엇이 달라졌나

| 항목 | 1차 (V100 32GB x2 NVLink) | 3차 (32GB x1 + 16GB x2 + P104) |
|---|---|---|
| **시스템 구축비** | 약 2,500 USD | **약 2,000 USD** |
| LLM 생성 속도 | 31.5 t/s | **33.5 t/s** |
| MTP 수락률 | 41% | **45.5%** |
| 이미지 8장 (768x768 / 15스텝) | 261.7초 | **134.0초** |

> **1차의 이미지 생성 시간은 당시 실측 기록이 남아 있지 않다.** 위 값은 같은 조건(V100 32GB 한 장에 통째로 적재, 768x768 / 15스텝 / PuLID 활성)을 3차 시스템에서 측정한 값이다 -> [No.2 8장 조건 1](../02-encoder-separation.md#8-결과). 이미지 생성은 GPU에서 끝나는 작업이라 플랫폼이 바뀌어도 오차 범위 안이지만, **엄밀히는 1차에서 잰 값이 아니다.**

**카드 수는 늘었지만 GPU 등급은 내려갔다.** NVLink 커스텀 보드가 빠지고 V100 32GB 한 장이 16GB 두 장으로 대체되면서 **구축비도 함께 내려갔다.** 그럼에도 세 지표 모두 개선된 것이 이 저장소가 다루는 내용이다.

---

## 0. 순서를 지켜야 하는 이유

```
0. 네트워크 인터페이스 이름을 MAC 주소로 고정한다   <- 카드를 꽂기 전에
1. OS 업데이트를 먼저 끝낸다
2. 드라이버 인식을 확인한다
3. CUDA 툴킷을 설치한다
4. 그 다음에 무엇이든 빌드한다
```

**3번과 4번을 먼저 하면 나중에 커널/드라이버가 어긋나 전부 다시 빌드하게 된다.**

### 0번을 빠뜨리면 서버에 접속조차 못 한다

리눅스의 네트워크 인터페이스 이름(`enp2s0`)은 **PCI 슬롯 위치에서 유도된다.** 카드를 추가하거나 슬롯을 바꾸면 이름이 바뀌고, 그 이름으로 고정해 둔 네트워크 설정이 통째로 어긋난다.

**헤드리스 서버에서는 이게 치명적이다.** GPU를 꽂았을 뿐인데 SSH가 안 되고, 화면이 없으니 원인을 볼 수도 없다. 이 프로젝트에서 실제로 겪었다.

MAC 주소로 이름을 고정한다.

```
/etc/systemd/network/10-nicname.link      <- [Match] MACAddress= / [Link] Name=
```

그리고 netplan 등의 설정에서 그 고정된 이름을 쓴다.

**GPU UUID와 완전히 같은 문제다.** 위치 기반 식별자는 전부 하드웨어 구성 변경에 취약하다 -> [함정 모음](../99-pitfalls.md#위치-기반-식별자는-전부-같은-함정이다)

> 참고
> - [Predictable Network Interface Names - systemd](https://systemd.io/PREDICTABLE_INTERFACE_NAMES/) - 이름 결정 규칙
> - [PCI-E 장치 교체 후 이더넷 연결 끊김 - 증상과 복구 절차](https://pyys.cafe24.com/wp/?p=729) - 진단과 복구 절차 상세

### 하드웨어 인식 확인

```
nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version --format=csv
```

**이름이 `Unknown` 이거나 목록에 없으면 그 드라이버 브랜치가 해당 아키텍처를 버린 것이다.** 여기서 걸리면 더 진행할 이유가 없다.

**UUID를 이 시점에 기록해 둔다.** 카드를 추가하면 인덱스가 밀리므로, 이후 모든 설정은 UUID로 쓴다 -> [No.4 1장](../04-orchestration.md#1-gpu-할당은-반드시-uuid로-지정한다)

PCIe 링크 폭/세대도 함께 본다. 채굴 카드나 라이저에 물린 카드는 좁은 레인에 걸려 있는 경우가 많다.

```
nvidia-smi --query-gpu=index,name,pcie.link.width.max,pcie.link.gen.max --format=csv
```

---

## 1. CUDA 툴킷

> **CUDA 13.0은 Volta(sm_70) 지원이 삭제되어 V100에서 쓸 수 없다.** 12.x를 써야 한다. 12.8은 Ubuntu 24.04 + GCC 13.3을 공식 지원하는 라인이라 이걸 골랐다 -> [No.1 6-1](../01-role-assignment.md#6-1-지원-창support-window이-구성-가능-여부를-결정한다)

### 1-1. 저장소 등록

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb
```

### 1-2. 드라이버 패키지 차단 - 필수

NVIDIA 저장소는 드라이버도 공급한다. 그대로 두면 `apt upgrade` 가 `libnvidia-gl-580` 과 `libnvidia-egl-gbm1` 충돌로 통째로 막힌다.

```
printf 'Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* libxnvctrl*\nPin: origin developer.download.nvidia.com\nPin-Priority: -1\n' > /etc/apt/preferences.d/nvidia-no-driver
```

**CUDA 툴킷만 받고 드라이버는 배포판 것을 쓴다.**

### 1-3. 업데이트

```
apt update && apt upgrade -y
```

> 드라이버가 DKMS에 등록돼 있으면(`dkms status`) 커널이 올라가도 자동 재빌드된다. 드라이버 제거/재설치는 불필요하다.

### 1-4. 툴킷 설치

**`cuda` 나 `cuda-12-8` 이 아니라 `cuda-toolkit-12-8` 이다.** 앞의 것들은 드라이버까지 교체한다.

```
apt install -y cuda-toolkit-12-8
```

### 1-5. 환경변수와 검증

```
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```

```
nvcc -V && nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

### 1-6. 빌드 도구

```
apt install -y build-essential cmake git ccache libcurl4-openssl-dev pkg-config
```

---

## 2. 소스와 모델 확보

### 2-1. 소스는 커밋을 고정해서 받는다

**`git pull` 을 하지 않는다.** 검증된 커밋을 그대로 쓴다. 옵션명이 바뀌었거나 GGUF 호환이 깨졌을 위험이 있다.

```
git clone https://github.com/ggml-org/llama.cpp /root/llama.cpp && cd /root/llama.cpp && git checkout 1593d5684
```

```
git clone https://github.com/leejet/stable-diffusion.cpp /root/stable-diffusion.cpp && cd /root/stable-diffusion.cpp && git submodule update --init --recursive && git checkout f440ad9c
```

이 프로젝트가 고정한 커밋은 llama.cpp **`1593d5684`**(b9602, 2026-06-12), stable-diffusion.cpp **`f440ad9c`**(2026-06-23) 다. [`patches/`](../../../patches/) 의 앵커가 후자를 기준으로 한다.

확인은 이렇게 한다.

```
cd /root/stable-diffusion.cpp && git log -1 --format='%H %ad %s'
```

**`--depth 1` 로 클론하면 `git pull` 이 "Already up to date" 라고 답하면서도 수백 커밋 뒤처져 있을 수 있다.** 커밋을 지정해 받을 것이라면 얕은 클론을 쓰지 않는다.

### 2-2. 모델 내려받기

```
mkdir -p /root/models/flux-sd-cpp /root/models/pulid /root/loras
```

**LLM** - 이 프로젝트가 쓴 것은 Qwen 3.6 27B 계열을 Q6_K로 양자화하고 MTP 가중치를 포함시킨 GGUF다.

| 파일 | 크기 | 출처 |
|---|---|---|
| `qwen3.6-27b-Q6K-mtp.gguf` | 21.4 GB | [pyys/Qwen3.6-27B-AEON-Ultimate-Uncensored-Q6K-MTP-GGUF](https://huggingface.co/pyys/Qwen3.6-27B-AEON-Ultimate-Uncensored-Q6K-MTP-GGUF) |
| `qwen3.6-27b-mmproj.gguf` | 931 MB | [pyys/Qwen3.6-27B-mmproj-GGUF](https://huggingface.co/pyys/Qwen3.6-27B-mmproj-GGUF) |

```
pip install -U "huggingface_hub[cli]"
```

```
hf download pyys/Qwen3.6-27B-AEON-Ultimate-Uncensored-Q6K-MTP-GGUF qwen3.6-27b-Q6K-mtp.gguf --local-dir /root/models
```

```
hf download pyys/Qwen3.6-27B-mmproj-GGUF qwen3.6-27b-mmproj.gguf --local-dir /root/models
```

> **다른 모델을 쓴다면 MTP 가중치가 포함된 GGUF인지 확인할 것.** MTP가 없으면 투기적 디코딩이 동작하지 않아 이 문서의 t/s 수치가 재현되지 않는다.

**이미지 생성** - 이 프로젝트가 쓴 것은 FLUX.1 [dev] 다. 확산 모델 본체, VAE, 텍스트 인코더 2종, PuLID 가중치가 필요하다.

| 파일 | 크기 | 출처 |
|---|---|---|
| `flux1-dev-Q8_0.gguf` | 12.7 GB | [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf) |
| `ae.safetensors` | 335 MB | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `clip_l.safetensors` | 246 MB | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |
| `t5xxl_fp16.safetensors` | 9.79 GB | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |
| `pulid_flux_v0.9.1.safetensors` | 1.14 GB | [guozinan/PuLID](https://huggingface.co/guozinan/PuLID) |

> **위는 파일 크기이고, 다른 문서의 19.1GB / 13.6GB는 적재 시점의 VRAM 실측값이다.** 이 프로젝트가 실측에 쓴 Q8_0 가중치는 위 배포본과 파일 크기가 소폭 다르다. 양자화 방식이 같으면 판정(16GB 카드에 통째로는 안 들어간다)은 바뀌지 않지만, **배포본에 따라 소수점 자리는 달라질 수 있다.**

> **`ae.safetensors` 만 게이트가 걸려 있다.** `black-forest-labs/FLUX.1-dev` 는 HF 계정으로 로그인해 비상업 라이선스에 동의해야 파일을 받을 수 있다. 웹에서 동의를 마친 뒤 `hf auth login` 으로 토큰을 등록해야 CLI 다운로드가 통과한다. 나머지 넷은 게이트가 없다.

```
hf auth login
```

```
hf download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir /root/models/flux-sd-cpp
```

```
hf download city96/FLUX.1-dev-gguf flux1-dev-Q8_0.gguf --local-dir /root/models
```

```
hf download comfyanonymous/flux_text_encoders clip_l.safetensors t5xxl_fp16.safetensors --local-dir /root/models/flux-sd-cpp
```

```
hf download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir /root/models/pulid
```

T5-XXL은 받은 fp16을 **q8_0으로 직접 변환해서 쓴다.** 이 프로젝트가 인코더를 8GB 카드에 올릴 수 있었던 것이 이 변환 덕분이고, 품질 비교 기록은 [설계 기록 0-2](design-record.md#0-2-양자화-ab---카드-없이-지금-스택에서)에 있다.

```
/root/stable-diffusion.cpp/build/bin/sd-cli -M convert -m /root/models/flux-sd-cpp/t5xxl_fp16.safetensors -o /root/models/flux-sd-cpp/t5xxl-q8_0.gguf --type q8_0
```

**이 변환만은 [4-1](#4-1-빌드---모든-대상-아키텍처를-넣는다) 의 빌드가 끝난 뒤에 실행한다.** `sd-cli` 가 아직 없기 때문이다. 변환이 끝나면 배치는 다음과 같다.

```
/root/models/flux1-dev-Q8_0.gguf                   확산 모델 (12.7 GB)
/root/models/flux-sd-cpp/ae.safetensors            VAE (335 MB)
/root/models/flux-sd-cpp/clip_l.safetensors        CLIP-L (246 MB)
/root/models/flux-sd-cpp/t5xxl-q8_0.gguf           T5-XXL q8_0 (5.2 GB)
/root/models/pulid/pulid_flux_v0.9.1.safetensors   PuLID 가중치 (1.14 GB)
```

**`/root/loras` 는 비어 있어야 한다.** sd-server가 요청마다 이 디렉터리를 재귀 스캔하므로, 기본값(`.`)을 쓰거나 큰 디렉터리를 지정하면 크래시하거나 느려진다.

## 3. LLM 추론 엔진

### 3-1. 빌드

V100은 Compute Capability 7.0이므로 아키텍처를 70으로 고정한다. 불필요한 아키텍처를 건너뛰어 빌드 시간이 크게 준다.

```
cd /root/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
```

```
cmake --build build --config Release -j 8
```

> **프리빌트 프론트엔드 자산이 404로 죽어 있었다.** WebUI가 필요하면 Node를 설치해 직접 빌드해야 한다. 배포처가 사라지는 것은 흔한 일이므로 **빌드 경로를 확보해 두는 편이 안전하다.**

### 3-2. 검증

```
/root/llama.cpp/build/bin/llama-server --list-devices
```

**CUDA 장치가 나열되지 않으면 CPU 빌드가 된 것이다.** 여기서 걸러야 나중에 "왜 이렇게 느리지"로 헤매지 않는다.

### 3-3. 기동 - 메인 GPU 지정만으로는 부족하다

```
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-... llama-server -m <모델> --mmproj <프로젝터> -ngl 99 -c 80000 --split-mode none -mg 0
```

`-mg 1` 로 메인 GPU만 지정했더니 **비전 프로젝터가 GPU0에 1.4GB 얹혀** 나중에 이미지 생성 서버가 OOM 났다. `CUDA_VISIBLE_DEVICES` 로 **카드 하나만 보이게 만든 뒤 인덱스 0** 을 주는 것이 확실하다.

---

## 4. 이미지 생성 엔진

### 4-1. 빌드 - 모든 대상 아키텍처를 넣는다

```
cd /root/stable-diffusion.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70"
```

```
cmake --build build --config Release -j 8
```

**`"61;70"` 의 따옴표는 필수다.** 없으면 셸이 명령을 두 개로 쪼갠다.

**아키텍처를 빠뜨리면 실행이 안 되고, 더 나쁘게는 최적 커널이 조용히 꺼진다.** 처음에는 `70` 만 넣어 빌드했다가 P104를 꽂고 나서 실행 자체가 안 돼 다시 빌드했다 -> [No.1 5-1](../01-role-assignment.md#5-1-양자화-행렬곱의-세-경로)

### 4-2. 반드시 켜야 하는 것

**`--vae-tiling`** - VAE 디코드 컴퓨트 버퍼를 3,744MB에서 416MB로 줄인다. 없으면 **샘플링을 다 마치고 마지막 VAE 단계에서 OOM으로 죽는다.** 비용은 약 1.4초다.

**`--lora-model-dir <빈 디렉터리>`** - 기본값이 `.` 이라 서버를 `/` 에서 띄우면 요청마다 루트 전체를 재귀 스캔하다가 심볼릭 링크에서 크래시한다.

```
{"error":"server_error","message":"filesystem error: status: Too many levels of symbolic links [./run/udev/watch/22]"}
```

### 4-3. 쓰면 안 되는 것

**`--mmap`** - RAM을 아껴 준다고 켰다가 모델 로드가 **8초에서 110초로** 늘었다. 페이지 폴트 단위 랜덤 읽기가 순차 읽기보다 훨씬 느리다. 여러 프로세스가 같은 가중치를 공유하는 상황이 아니면 손해다.

**`txt_cfg` 기본값 7.0** - FLUX는 distilled 모델이라 CFG를 쓰지 않는다. 기본값을 두면 빈 네거티브 프롬프트를 인코딩하고 스텝마다 두 번 계산해서 **3.3배 느려진다.** 요청에 `"guidance": {"txt_cfg": 1.0}` 을 명시한다.

### 4-4. 인자 이름을 추측하지 않는다

```
sd-server --help
```

`--host` / `--port` 가 아니라 `--listen-ip` / `--listen-port` 였다. **틀린 인자를 주면 usage만 출력하고 조용히 종료**하므로, 로그를 안 보면 "왜 안 뜨지"로 시간을 버린다.

> 4-2부터 4-4까지의 다섯 항목은 [함정 모음 - 추론 엔진](../99-pitfalls.md#3-추론-엔진)에 증상/대응 표로 정리돼 있다. 여기서는 기동 절차의 일부로 남긴다.

### 4-5. 기동 로그에서 확인할 것

```
grep -E "total params memory size|loading tensors completed" <로그>
```

**로그의 메모리 예상치와 실제 할당은 다르다.** 반드시 `nvidia-smi` 로 확인한다.

```
보고되는 예상치 : 17346 MB
실제 할당       :  5181 MB
```

**포트가 열려도 가중치는 아직 안 올라와 있다.** 가중치가 lazy 로드라 첫 요청에서 모델을 읽는다. `listening` 로그를 기동 완료로 착각하면 벤치마크가 통째로 틀어진다.

**이 문서의 기동 명령은 모두 포그라운드 실행이다.** 터미널을 닫으면 서비스가 죽고 재부팅 시 자동으로 뜨지 않는다. 상시 운용하려면 `systemd` 로 등록하거나 최소한 `tmux` / `nohup` 으로 감싼다.

---

## 5. 얼굴 일관성 확장 (PuLID)

이미지 생성 시 캐릭터의 얼굴을 여러 이미지에 일관되게 유지하기 위해 [PuLID](https://github.com/ToTheBeginning/PuLID)를 붙였다. 얼굴 임베딩 추출은 **캐릭터당 1회**만 실행된다.

### 5-1. 의존성 설치 - onnxruntime은 CPU판을 쓴다

임베딩 추출은 Python 쪽이다. 필요한 것은 torch와 InsightFace, 그리고 ONNX 런타임이다.

```
pip install torch insightface onnxruntime
```

**`onnxruntime-gpu` 가 아니라 `onnxruntime` 이다.** V100에서는 GPU판이 동작하지 않으므로([5-3](#5-3-얼굴-임베딩-추출은-cpu로)) 처음부터 CPU판을 넣는 편이 낫다. 추출은 캐릭터당 1회뿐이라 CPU로 처리해도 운용 단계의 시간 비용은 거의 없다.

```
python3 -c "import torch, insightface; print(insightface.__version__)"
```

> 구 시스템에서 옮겨 오는 경우라면 재설치가 필요 없을 수도 있다. **OS와 Python 마이너 버전이 같으면 ABI가 맞고 torch는 자체 CUDA 런타임을 번들하므로, `PYTHONPATH=/old/path/dist-packages` 로 가리키기만 해도 동작한다.** 이 프로젝트가 실제로 쓴 방법이다.

### 5-2. 얼굴 인식 모델 (antelopev2)

InsightFace는 `~/.insightface/models/` 를 찾고, 없으면 **약 400MB를 자동으로 받는다.** 네트워크가 되면 따로 할 일이 없다.

자동 다운로드가 막히는 환경이면 직접 받아서 풀어 둔다.

```
mkdir -p /root/.insightface/models && cd /root/.insightface/models && curl -LO https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip && unzip antelopev2.zip
```

`antelopev2/` 아래에 `.onnx` 파일들이 놓이면 된다. **한 단계 더 깊은 `antelopev2/antelopev2/` 로 풀리는 경우가 있으니 확인할 것.**

```
ls /root/.insightface/models/antelopev2/
```

### 5-3. 얼굴 임베딩 추출은 CPU로

**onnxruntime-gpu 1.27은 Volta(sm_70)를 지원하지 않는다.** GPU로 시도하면 이렇게 실패한다.

```
CUBLAS failure 8: the function requires an architectural feature absent from the device
```

`CUDA_VISIBLE_DEVICES=""` 로 두면 CPU로 자동 전환된다. **캐릭터당 1회뿐이라 CPU로 충분하다.**

> 이것도 [지원 창](../01-role-assignment.md#6-1-지원-창support-window이-구성-가능-여부를-결정한다) 문제의 한 형태다. 추론 엔진만이 아니라 **부속 라이브러리도 아키텍처를 버린다.** 이종 구성을 계획할 때는 의존 라이브러리까지 확인해야 한다.

---

## 6. 최종 상태

| GPU | 역할 | VRAM 사용 |
|---|---|---|
| V100 32GB | LLM 서버 | 30.3GB / 32.7GB |
| V100 16GB x2 | 디퓨전 워커 | 각 13.6GB / 16.1GB |
| P104-100 8GB | 텍스트 인코더 | 5.2GB / 8.0GB |

이미지 생성 부분은 이 구축 이후 **텍스트 인코더를 1벌로 분리하는 구조로 다시 개편했다.** 그 과정은 [설계 기록](design-record.md)에, 결과는 [부록 - 벤치마크](../98-benchmark.md)에 있다.

---

## 7. 이 구축에서 실제로 시간을 잃은 지점

| 함정 | 증상 | 대응 |
|---|---|---|
| **페이퍼 테스트만 통과한 보드** | 스펙은 맞는데 **POST 실패**, 원인 미규명 | 같은 조합의 실제 구축 사례 확보 |
| **네트워크 인터페이스 이름** | 카드 교체 후 **SSH 불가** | MAC 주소로 고정 |
| CUDA 13 설치 | Volta 미지원 | 12.8 고정 |
| NVIDIA 저장소 드라이버 충돌 | `apt upgrade` 실패 | apt pinning |
| rsync `--exclude 'build'` | 서드파티 디렉터리 삭제 -> 빌드 실패 | `--exclude "/build"` |
| 아키텍처 누락 | P104에서 실행 자체가 안 됨 | 모든 대상 카드 포함 |
| cmake 인자 따옴표 누락 | 명령이 둘로 쪼개짐 | `"61;70"` |
| 메인 GPU 지정만 | 보조 모듈이 다른 카드 점유 -> OOM | `CUDA_VISIBLE_DEVICES` 로 격리 |
| `--vae-tiling` 누락 | 샘플링 완료 후 VAE에서 OOM | 옵션 추가 |
| `txt_cfg` 기본값 | 3.3배 느림 | 1.0 지정 |
| `--mmap` | 로드 8초 -> 110초 | 사용 금지 |
| LoRA 디렉터리 기본값 `.` | 요청마다 재귀 스캔, 루트에서 크래시 | 빈 디렉터리 명시 |
| 인자 이름 추측 | usage만 출력하고 종료 | `--help` 확인 |
| onnxruntime-gpu 1.27 | 얼굴 추출 시 CUBLAS failure 8 | CPU 추출 |
| 프리빌트 자산 404 | WebUI 접속 시 404 | Node 설치 후 직접 빌드 |

분류와 대응 원칙은 [함정 모음](../99-pitfalls.md)에 정리했다.

### 터미널 붙여넣기 문자 손실

작업 중 터미널이 붙여넣기에서 `*`, `\`, `__` 를 삭제하는 현상이 반복됐다. **와일드카드나 백슬래시 이스케이프가 들어간 명령은 파일명을 명시하거나 `grep -E` 같은 대체 형태로 쓰는 편이 안전하다.** 원인을 모르는 채로 "명령이 왜 안 먹지"를 여러 번 겪었다.

---

## 관련 고지

**1) NVLink를 갖추고도 쓰지 않았다**

**당시 운용 설계를 다시 보면 두 카드는 처음부터 역할이 나뉘어 있었다.**

```
GPU0    llama-server (LLM)
GPU1    sd-server (FLUX 이미지 생성) + PuLID 추출
```

llama-server는 `--split-mode none -mg 0` 으로 **단일 GPU 전용**으로 띄웠고, sd-server는 `CUDA_VISIBLE_DEVICES=1` 로 다른 카드에 고정했다. **NVLink로 VRAM을 합친 것이 아니라 카드별로 다른 모델을 올린 구조**였다.

**더 정확히 말하면, 당시에는 텐서 병렬을 쓰는 방법 자체를 몰랐다.** 멀티 GPU 적재를 시도하며 만든 명령은 이런 형태였는데,

```
llama-server -m <모델> --mmproj <프로젝터> -ngl 99 -c 131072 --host 0.0.0.0 --port 8080
```

**여기에는 `--split-mode row` 가 없다.** llama.cpp의 기본값은 레이어 분할이므로, 카드가 두 장 보여도 **레이어를 나눠 싣기만 할 뿐 텐서 병렬로 동작하지 않는다.** 이 사실은 한참 뒤에야 확인했다.

> **NVLink 하드웨어를 갖추는 것과 그 대역폭을 실제로 쓰는 것은 별개다.** 이 프로젝트는 NVLink 보드를 사서 한 달을 운용하고도 **그 링크를 한 번도 쓰지 않았다.** 재구성 때 NVLink 비용을 포기한 판단은 솔루션의 한 가지 방향으로는 문제가 없었지만, **이 단계에서 텐서 병렬을 경험했고 그 이득이 결정적이었다면 다른 방향의 솔루션을 채택했을 가능성도 배제할 수 없다** -> [No.1 1-3](../01-role-assignment.md#1-3-텐서-병렬은-별개-문제다)

**2)** 백업은 거의 모든 경우에 시간 비용을 절약하는 방법이다. 백업에 필요한 스토리지 비용과 솔루션 도입 비용은 문제가 발생했을 때의 인건비보다 거의 항상 싸다.
