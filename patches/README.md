# patches - stable-diffusion.cpp 개조

[No.2 인코더 분리](../docs/ko/02-encoder-separation.md)를 구현하기 위해 추론 엔진에 가한 변경이다.

세 패치 모두 **적용 전 백업을 만들고 `--revert` 로 되돌릴 수 있다.**

---

## 대상 커밋

```
https://github.com/leejet/stable-diffusion.cpp
commit f440ad9c
```

**문자열 앵커 기반 패치다.** 대상 커밋이 다르면 앵커를 못 찾고 `[FAIL]` 로 중단된다. **파일을 망가뜨리지는 않지만** 상류 코드가 바뀌었다면 앵커를 직접 갱신해야 한다.

각 스크립트는 앵커가 **정확히 1회** 발견될 때만 진행한다. 0회거나 2회 이상이면 중단한다. 이 조건이 상류 변경을 자동으로 감지하는 안전장치다.

---

## 적용 순서

의존 관계가 있으므로 **아래 순서를 지켜야 한다.**

```
1. cond_serialize_patch.py     조건 텐서 직렬화 / 역직렬화
2. cond_server_patch.py        /sdcpp/v1/encode 엔드포인트   <- 1 에 의존
3. pulid_perrequest_patch.py   PuLID 를 요청별 파라미터로     <- 독립
```

소스 트리 위치는 `SD_CPP_ROOT` 로 지정한다. 생략하면 `/root/stable-diffusion.cpp` 다.

```
export SD_CPP_ROOT=/path/to/stable-diffusion.cpp
```

**먼저 `--dry-run` 으로 앵커가 잡히는지 확인한다.**

```
python3 cond_serialize_patch.py --dry-run
```

문제가 없으면 적용한다.

```
python3 cond_serialize_patch.py
```

```
python3 cond_server_patch.py
```

```
python3 pulid_perrequest_patch.py
```

세 개를 적용한 뒤 **한 번만 재빌드한다.**

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70"
```

```
cmake --build build -j 8
```

**`"61;70"` 의 따옴표는 필수다.** 없으면 셸이 명령을 두 개로 쪼갠다.

**모든 대상 카드의 아키텍처를 넣어야 한다.** 빠뜨리면 실행이 안 되거나, 더 나쁘게는 **최적 커널이 조용히 꺼진다** -> [No.1 5장](../docs/ko/01-role-assignment.md#5-이종-gpu에서-성능이-실제로-갈리는-지점)

되돌리려면:

```
python3 cond_serialize_patch.py --revert
```

---

## 각 패치가 하는 일

### 1. `cond_serialize_patch.py`

**텍스트 인코딩 결과(조건 텐서)를 파일로 저장하고 다시 읽어들이는 경로를 만든다.**

| 추가 | 내용 |
|---|---|
| `sd_img_gen_params_t` | `conditioning_path` / `save_conditioning_path` 필드 |
| gguf 헬퍼 | `sd_cond_save()` / `sd_cond_load()` |
| 분기 | 조건 계산 지점에서 "계산할지 / 읽을지 / 저장할지" |

설계의 핵심은 **축 규약을 해석하지 않는 것**이다. shape 벡터와 flat 데이터를 있는 그대로 왕복시키면 규약이 어느 쪽이든 정확히 복원된다.

ggml의 `ggml_n_dims()` 는 뒤쪽 차원의 1을 잘라내므로, **원래 차원 수를 KV 메타데이터에 따로 기록**해야 왕복이 정확하다.

### 2. `cond_server_patch.py`

**인코딩만 수행하는 공개 API와 HTTP 엔드포인트를 추가한다.**

```
sd_encode_conditioning()        공개 API
POST /sdcpp/v1/encode           HTTP 엔드포인트
```

출력 파일명은 `프롬프트 | 해상도 | clip_skip` 의 해시다. 같은 입력이면 같은 파일이 되므로 캐시로도 동작한다.

### 3. `pulid_perrequest_patch.py`

**두 줄짜리 패치다.**

```c
load_if_exists("pulid_id_embedding_path", pulid_id_embedding_path);
load_if_exists("pulid_id_weight", pulid_id_weight);
```

[PuLID](https://github.com/ToTheBeginning/PuLID) ID 임베딩은 원래 기동 인자로만 넘길 수 있어, 캐릭터를 바꿀 때마다 워커를 전부 재기동해야 하는 것으로 보였다. 그런데 **엔진 내부에서는 처음부터 요청별 파라미터였고 HTTP로 넘길 통로만 없었다.**

**워커 수 N이 커질수록 재기동 비용은 N배가 된다.** 확장을 전제한 설계에서 이런 항목을 미리 찾아내는 것의 가치가 여기 있다 -> [No.4 4장](../docs/ko/04-orchestration.md#4-기동-인자가-정말-전역-상태인지-의심할-것)

---

## 검증

**HTTP를 만들기 전에 CLI로 먼저 증명한다.** 이미지가 틀렸을 때 직렬화/주입/네트워크 중 어디가 문제인지 구분이 안 되기 때문이다.

```
1) 정상 생성 + 조건 저장                    -> 이미지 A
2) 텍스트 인코더 인자를 빼고 조건만 주입     -> 이미지 B
3) A 와 B 가 픽셀 단위로 동일한가
```

**PNG 바이트 비교는 메타데이터 청크 때문에 무조건 다르게 나온다.** 픽셀로 비교해야 한다.

```
python3 -c "from PIL import Image; a=Image.open('a.png').tobytes(); b=Image.open('b.png').tobytes(); print('IDENTICAL' if a==b else 'DIFFERENT')"
```

이 한 번으로 직렬화 왕복/주입 지점/**워커가 텍스트 인코더 없이 기동되는지**가 동시에 증명된다.

---

## 오케스트레이터 쪽 변경

인코더 서비스와 워커들을 실제로 띄우고 작업을 분배하는 코드는 **프로젝트마다 다르므로 패치 형태로 배포하지 않는다.** 대신 일반화한 참조 구현을 [`reference/`](../reference/) 에 두었다.

---

## 상류 프로젝트

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 의 작업 위에 만들어졌다. **원저작자의 코드가 없었다면 이 프로젝트는 성립하지 않는다.**

이 디렉터리의 코드는 [MIT](../LICENSE) 다.
