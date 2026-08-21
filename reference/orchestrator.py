#!/usr/bin/env python3
"""
인코더 1벌 + 디퓨전 워커 N벌 오케스트레이션 — 참조 구현

[④ 오케스트레이션](../docs/ko/04-orchestration.md) 이 설명하는 패턴을 한 파일로 모았다.
실제 프로젝트에서는 웹 서버·큐·상태 저장소가 붙지만, 그 부분은 프로젝트마다 다르므로 뺐다.

여기 담긴 것은 넷이다.

  1. GPU 를 UUID 로 지정한다              (인덱스는 장비 수가 바뀌면 밀린다)
  2. 워커 수를 상수로 가정하지 않는다      (목록 길이로만 결정)
  3. 기동을 락으로 직렬화하고 종료에 kill 폴백을 둔다
  4. 1장 = 1작업으로 쪼개 라운드로빈한다   (조기 표시 + 부하 분산)

전제: patches/sdcpp/ 의 세 패치를 적용해 빌드한 sd-server

사용법
    python3 orchestrator.py --prompt "..." --count 8
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request


# ---------------------------------------------------------------- 설정
#
# 여기부터 자기 환경에 맞게 바꾼다.
#
# GPU UUID 는 아래 명령으로 얻는다. 인덱스가 아니라 UUID 를 쓰는 이유는
# 카드를 추가하면 인덱스가 밀려 엉뚱한 카드에 모델이 올라가기 때문이다.
#
#     nvidia-smi --query-gpu=index,name,uuid --format=csv

GPU_UUIDS = {
    "encoder": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",   # 가장 싼 카드
    "worker0": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "worker1": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # 카드를 늘리면 여기에 항목만 추가한다
}

SD_SERVER = "/path/to/stable-diffusion.cpp/build/bin/sd-server"

DIT = "/path/to/models/flux-dev-q8_0.gguf"
VAE = "/path/to/models/ae.safetensors"
CLIP_L = "/path/to/models/clip_l.safetensors"
T5 = "/path/to/models/t5xxl-q8_0.gguf"

# sd-server 는 요청마다 이 디렉터리를 재귀 스캔한다.
# 기본값이 "." 이라 루트에서 띄우면 심볼릭 링크를 만나 크래시한다. 반드시 명시할 것.
LORA_DIR = "/path/to/loras"

# 조건 텐서 저장 위치. tmpfs 를 쓰면 디스크를 거치지 않는다.
COND_DIR = "/dev/shm"
COND_TTL = 3600          # 이보다 오래된 조건 파일은 청소한다

ENCODER_PORT = 8090
WORKER_PORTS = [8081, 8082]          # 워커 수는 이 목록의 길이로만 결정된다

LOG_DIR = "/tmp/sd-orchestrator"

# 인코더를 GPU 가 아니라 RAM 에 올리려면 True.
# 추가 하드웨어 없이 분리 구조를 검증할 때 쓴다. 대신 인코딩이 크게 느려진다.
ENCODER_ON_CPU = False


# ---------------------------------------------------------------- 프로세스 관리

_procs = {}                          # port -> Popen
_start_lock = threading.Lock()       # 기동 직렬화. 아래 주석 참조


def _launch(name, gpu_uuid, args):
    os.makedirs(LOG_DIR, exist_ok=True)
    env = os.environ.copy()

    # 카드 하나만 보이게 만든 뒤 인덱스 0 을 쓴다.
    # "메인 GPU 지정" 옵션만 믿으면 보조 모듈이 다른 카드에 얹혀 OOM 을 낸다.
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    log = open(os.path.join(LOG_DIR, name + ".log"), "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)


def _encoder_args():
    a = [
        # 디퓨전 모델 경로가 필요하다. 엔진이 모델 버전을 보고 conditioner 종류를
        # 고르기 때문이다. 가중치는 lazy 로드라 실제로 올라가지 않는다.
        "--diffusion-model", DIT,
        "--vae", VAE,
        "--clip_l", CLIP_L,
        "--t5xxl", T5,
        "--lora-model-dir", LORA_DIR,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(ENCODER_PORT),
        "-v",
    ]
    if ENCODER_ON_CPU:
        a += ["--backend", "te=cpu"]
    return a


def _worker_args(port):
    # 텍스트 인코더 인자를 주지 않는다. 워커는 조건 텐서를 받아 쓴다.
    return [
        "--diffusion-model", DIT,
        "--vae", VAE,
        "--lora-model-dir", LORA_DIR,
        "--vae-tiling",              # 컴퓨트 버퍼 3744MB -> 416MB. 비용 약 1.4초
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(port),
        "-v",
    ]


def start_all():
    """인코더 1벌 + 워커 N벌 기동.

    락으로 직렬화한다. 모델 로딩에 20초 이상 걸리는데, 그 사이에 들어온
    두 번째 요청이 "안 떠 있다"고 판정하면 중복 세트가 뜬다.
    """
    with _start_lock:
        return _start_all_locked()


def _start_all_locked():
    stop_all()

    _procs[ENCODER_PORT] = _launch("encoder", GPU_UUIDS["encoder"], [SD_SERVER] + _encoder_args())

    for i, port in enumerate(WORKER_PORTS):
        key = "worker%d" % i
        _procs[port] = _launch(key, GPU_UUIDS[key], [SD_SERVER] + _worker_args(port))

    for port in [ENCODER_PORT] + WORKER_PORTS:
        _wait_listening(port)


def stop_all():
    """전 프로세스 종료.

    terminate 후 짧게 기다리고 끝내면 안 된다. 모델 로딩 중인 프로세스는
    SIGTERM 을 늦게 처리하므로, 타임아웃되면 kill 로 확실히 죽여야 한다.
    살아남은 프로세스가 포트를 잡고 있으면 다음 기동이 통째로 오염된다.
    """
    for proc in _procs.values():
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
    _procs.clear()


def _wait_listening(port, limit=180):
    """포트가 열릴 때까지 대기.

    주의: 포트가 열려도 가중치는 아직 안 올라온 상태다. sd-server 는 lazy
    로드라 첫 요청이 들어와야 모델을 읽는다. 따라서 "listening" 로그를
    기동 완료로 착각하면 안 된다.
    """
    for _ in range(limit * 5):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=1)
            return
        except urllib.error.HTTPError:
            return                       # 404 도 살아있다는 뜻
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("port %d 기동 실패. %s 확인" % (port, LOG_DIR))


# ---------------------------------------------------------------- HTTP

def _post(port, path, body, timeout=600):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(port, path, timeout=30):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- 조건 텐서

def purge_old_conditionings():
    """오래된 조건 파일 청소. 512토큰이면 1개당 약 8MB 라 쌓이면 tmpfs 가 찬다."""
    now = time.time()
    for name in os.listdir(COND_DIR):
        if not name.startswith("sdcond-"):
            continue
        path = os.path.join(COND_DIR, name)
        try:
            if now - os.path.getmtime(path) > COND_TTL:
                os.remove(path)
        except OSError:
            pass


def encode(prompt, width, height):
    """프롬프트를 1회 인코딩하고 조건 텐서 경로를 돌려준다.

    같은 프롬프트·해상도면 같은 파일이 되므로 캐시로도 동작한다.
    """
    purge_old_conditionings()
    res = _post(ENCODER_PORT, "/sdcpp/v1/encode", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "dir": COND_DIR,
    })
    return res["conditioning_path"]


def conditioning_tokens(path):
    """조건 파일 크기에서 토큰 수를 역산한다.

    인코딩이 느릴 때 원인이 청크 수인지 다른 것인지 구분하는 데 쓸모가 있다.
    hidden states 4096 x 토큰수 x 4 byte + pooled 768 x 4 + gguf 헤더 256
    """
    return round((os.path.getsize(path) - 3328) / 16384)


# ---------------------------------------------------------------- 작업 분배

def generate(prompt, count, width=768, height=768, steps=15, seed=42, on_done=None):
    """N 장을 N 개의 독립 작업으로 쪼개 워커에 라운드로빈한다.

    배치로 묶지 않는 이유는 둘이다.

      1. 완성되는 대로 1장씩 내보낼 수 있다 (사람의 평가 시간과 겹친다)
      2. 워커 간 불균형이 최대 1작업으로 제한되어 정적 라운드로빈으로 충분하다

    쪼개기의 대가는 실측상 장당 0.03초로 사실상 0이다. 인코딩 결과를
    워커들이 공유하기 때문이며, 장당 재인코딩을 하는 구조에서는 성립하지 않는다.
    """
    cond_path = encode(prompt, width, height)

    jobs = []
    for i in range(count):
        port = WORKER_PORTS[i % len(WORKER_PORTS)]      # 워커 수를 상수로 가정하지 않는다
        res = _post(port, "/sdcpp/v1/img_gen", {
            "conditioning_path": cond_path,             # 프롬프트 대신 조건 텐서를 넘긴다
            "width": width,
            "height": height,
            "batch_count": 1,
            "seed": seed + i,
            "sample_params": {
                "sample_steps": steps,
                "sample_method": "euler",
                # distilled 모델은 CFG 가 필요 없다. 1.0 이 아니면 3.3배 느려진다.
                "guidance": {"txt_cfg": 1.0, "distilled_guidance": 3.5},
            },
        })
        jobs.append((port, res["id"]))

    done = set()
    results = [None] * len(jobs)

    while len(done) < len(jobs):
        for idx, (port, jid) in enumerate(jobs):
            if idx in done:
                continue
            st = _get(port, "/sdcpp/v1/jobs/" + jid)
            status = (st.get("status") or "").lower()
            if status in ("completed", "done"):
                done.add(idx)
                results[idx] = st
                if on_done:
                    on_done(idx, st)        # 완성 즉시 사용자에게 내보낸다
            elif status in ("failed", "error"):
                done.add(idx)
                results[idx] = None
        time.sleep(0.2)

    return results


# ---------------------------------------------------------------- 진입점

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    a = ap.parse_args()

    t0 = time.time()
    start_all()

    def show(idx, st):
        print("[%5.1f초] %d 번째 완성" % (time.time() - t0, idx + 1))

    try:
        results = generate(a.prompt, a.count, a.width, a.height, on_done=show)
    finally:
        stop_all()

    ok = sum(1 for r in results if r)
    print("\n%d/%d 장, 총 %.1f초" % (ok, a.count, time.time() - t0))
    return 0 if ok == a.count else 1


if __name__ == "__main__":
    sys.exit(main())
