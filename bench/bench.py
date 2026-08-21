#!/usr/bin/env python3
"""
이종 GPU 구성 비용·성능 벤치마크

질문: LLM용 32GB 카드가 이미 있다. 이미지 생성을 위해 추가로 무엇을 살 것인가?

조건
  1  V100 32GB ×1, 올인 마운트              700 USD, 워커 1
  2  V100 16GB ×1 + 인코더 RAM              200 USD (+RAM 30), 워커 1
  3  V100 16GB ×1 + 인코더 P104             215 USD, 워커 1
  4  V100 16GB ×2 + 인코더 P104             415 USD, 워커 2
  (5 V100 16GB ×3 + 인코더 P104            615 USD, 워커 3  <- 카드 없음. 3·4에서 계산)
  6  V100 32GB ×1 + 인코더 P104, 워커 2벌   715 USD, 워커 2
  7  V100 32GB ×1 + 인코더 P104, 워커 1벌   (진단용. 구매 선택지가 아니다)
  8  V100 32GB ×1, 올인 마운트 + 낱장 재인코딩 (통제 실험. 구매 선택지가 아니다)

조건 6 은 "같은 예산을 32GB 한 장에 쓰면 어떻게 되는가" 를 본다.
  6 ≈ 3  ->  한 카드에 워커를 2벌 올려도 산출량이 늘지 않는다
  6 vs 4 ->  715 USD(32GB 1장) 와 415 USD(16GB 2장) 의 직접 대결

조건 8 은 조건 1 과 카드·인코더 위치·워커 수가 모두 같고 표시 방식만 다르다.
배치를 낱장으로 쪼개면 장마다 재인코딩이 일어나지만, 그 대가로 첫 장을 훨씬 빨리 내보낸다.
  1 대 8  ->  독립변인이 표시 방식 하나뿐이므로 조기 표시의 순수한 효과가 나온다

조건 7 은 조건 1 의 장당 생성이 느린 원인을 가른다.
조건 1 과 카드가 같고, 조건 3 과 운용이 같다.
  7 ≈ 3  ->  원인은 인코더 동거 또는 배치 경로. 분리가 실제로 기여했다
  7 ≈ 1  ->  원인은 카드 개체차. 분리의 공이 아니다
또한 조건 1 대 조건 7 은 카드가 동일하므로 "배치를 개별 작업으로 쪼개는 비용"
자체를 순수하게 분리해 준다.

측정 (둘 다 콜드, 명령 1회)
  --count 1   "1장 생성" -> 완료까지
  --count 8   "8장 생성" -> 완료까지

타이머는 **서비스 기동이 아니라 첫 요청 시점**에서 시작한다.
sd-server 는 가중치를 lazy 로드하므로 모델 로딩이 요청 시간에 포함된다.

사용법
    python3 bench.py --condition 3 --count 8
    python3 bench.py --condition 3 --count 8 --repeat 3
    python3 bench.py --condition 8 --count 8 --repeat 3
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

# ----------------------------------------------------------------- 설정

# ----- 여기부터 자기 환경에 맞게 바꾼다 --------------------------------

SD = "/path/to/stable-diffusion.cpp/build/bin/sd-server"
DIT = "/path/to/models/flux-dev-q8_0.gguf"
VAE = "/path/to/models/ae.safetensors"
CLIP = "/path/to/models/clip_l.safetensors"
T5 = "/path/to/models/t5xxl-q8_0.gguf"
PULID = "/path/to/models/pulid_flux_v0.9.1.safetensors"

# sd-server 가 요청마다 재귀 스캔한다. 빈 디렉터리를 명시할 것 (기본값 "." 은 위험)
LORA = "/path/to/loras"

LOGDIR = "/root/bench"
RESULTS = "/root/bench/results.csv"

# PuLID ID 임베딩 (.pulidembd) 을 찾을 디렉터리들. 먼저 발견되는 파일을 쓴다.
# 특정 파일을 강제하려면 EMBED 에 절대경로를 직접 넣는다.
# 디렉터리가 없으면 PuLID 없이 측정한다 (장당 약 4.7초 짧아진다).
EMBED_DIRS = [
    "/path/to/pulid-embeddings",
]
EMBED = None   # None 이면 EMBED_DIRS 에서 자동 탐색

# GPU UUID. 인덱스가 아니라 UUID 를 쓰는 이유는 카드를 추가하면 인덱스가 밀리기 때문이다.
#     nvidia-smi --query-gpu=index,name,uuid --format=csv
GPU32 = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"   # V100 32GB
GPU16A = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # V100 16GB
GPU16B = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # V100 16GB
GPUP104 = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # P104-100

# ----- 여기까지 -------------------------------------------------------

ENCODER_PORT = 8090
WORKER_PORTS = [8081, 8082]

W, H, STEPS, SEED = 768, 768, 15, 42
POLL = 0.2

# 512토큰(2청크)을 노리는 긴 프롬프트.
# 조건 2·3·4 에서 생성되는 조건 gguf 크기로 토큰 수를 검증한다.
#   4,197,632 B = 256토큰(1청크) / 8,391,936 B = 512토큰(2청크)
PROMPT = (
    "a photorealistic portrait of a young woman with light freckles across her nose and "
    "cheekbones, wearing a hand knitted crimson scarf wrapped twice around her neck, standing "
    "in front of a tall walnut bookshelf filled with worn hardcover volumes and a few small "
    "ceramic figurines, holding a chipped blue stoneware mug in her left hand at chest height, "
    "steam rising faintly from the surface of the tea inside, warm late afternoon sunlight "
    "entering through a tall sash window on the right side of the frame, casting long soft "
    "shadows across the wooden floorboards and illuminating dust motes suspended in the air, "
    "her expression calm and slightly amused, looking just past the camera rather than directly "
    "at it, shoulder length auburn hair loosely tied back with a few strands falling forward, "
    "wearing a cream colored cable knit sweater with visible texture and a slightly oversized "
    "fit, shallow depth of field with the bookshelf softly blurred behind her, natural skin "
    "texture with visible pores and fine detail, subtle color grading toward warm amber tones "
    "in the highlights and cool blue tones in the shadows, shot on a full frame camera with an "
    "85mm prime lens at f/1.8, careful attention to the fall of light across fabric folds and "
    "the reflection in the glazed surface of the ceramic mug, quiet domestic atmosphere, "
    "unhurried and intimate composition, film grain barely perceptible"
)


# ----------------------------------------------------------------- 유틸

def sh(cmd):
    subprocess.run(cmd, shell=True, capture_output=True)


def proc(args):
    """셸을 거치지 않고 실행.

    shell=True 로 pgrep/pkill 을 돌리면 래퍼 셸의 명령줄에 검색어가 들어 있어
    자기 자신을 매칭한다. 프로세스 탐지·종료는 반드시 리스트 형태로 실행할 것.
    """
    return subprocess.run(args, capture_output=True)


def post(port, path, body, timeout=600):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(port, path, timeout=30):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
        return json.loads(r.read())


def wait_listening(port, logfile, limit=180):
    """서비스가 포트를 열 때까지 대기. 가중치는 아직 안 올라온 상태다."""
    for _ in range(limit * 5):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=1)
            return True
        except urllib.error.HTTPError:
            return True          # 404 도 살아있다는 뜻
        except Exception:
            time.sleep(0.2)
    print("[FAIL] port %d 기동 실패. %s 확인" % (port, logfile))
    sys.exit(1)


def launch(name, gpu, args):
    os.makedirs(LOGDIR, exist_ok=True)
    logfile = os.path.join(LOGDIR, name + ".log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with open(logfile, "w") as f:
        subprocess.Popen([SD] + args, env=env, stdout=f, stderr=subprocess.STDOUT)
    return logfile


def sd_running():
    return proc(["pgrep", "-f", "sd-server"]).returncode == 0


def cold():
    """모든 서비스 종료 + 페이지 캐시 비우기

    sd-server 는 상황에 따라 SIGTERM 을 늦게 처리한다. 죽지 않은 프로세스가
    포트를 잡고 있으면 다음 측정이 통째로 오염되므로 확실히 종료시킨다.
    """
    proc(["pkill", "-f", "sd-server"])
    for _ in range(30):                 # 최대 15초 대기
        if not sd_running():
            break
        time.sleep(0.5)
    else:
        proc(["pkill", "-9", "-f", "sd-server"])   # 안 죽으면 SIGKILL
        time.sleep(2)
        if sd_running():
            print("[FAIL] sd-server 종료 실패. 수동 확인 필요")
            sys.exit(1)

    time.sleep(1)                       # VRAM 반환 대기
    sh("sync")
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")
    time.sleep(1)


def find_embed():
    """PuLID ID 임베딩 자동 탐색. 없으면 None (PuLID 비활성)"""
    if EMBED:
        return EMBED if os.path.exists(EMBED) else None
    for d in EMBED_DIRS:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".pulidembd"):
                return os.path.join(d, name)
    return None


EMBED_PATH = None   # main() 에서 설정


def tokens_from_size(nbytes):
    """조건 gguf 크기에서 토큰 수 역산. 4096×토큰×4 + 768×4 + 헤더"""
    return round((nbytes - 3328) / 16384)


# ----------------------------------------------------------------- 조건별 기동

def worker_args(port, with_encoder):
    a = [
        "--diffusion-model", DIT,
        "--vae", VAE,
        "--pulid-weights", PULID,
        "--lora-model-dir", LORA,
        "--vae-tiling",
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(port),
        "-v",
    ]
    if with_encoder:                      # 조건 1: 텍스트 인코더까지 같은 프로세스
        a += ["--clip_l", CLIP, "--t5xxl", T5]
    return a


def encoder_args(port, te_cpu):
    a = [
        "--diffusion-model", DIT,          # 버전 감지용. 가중치는 안 올라감
        "--vae", VAE,
        "--clip_l", CLIP,
        "--t5xxl", T5,
        "--lora-model-dir", LORA,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(port),
        "-v",
    ]
    if te_cpu:
        a += ["--backend", "te=cpu"]
    return a


CONDITIONS = {
    1: dict(price=700, workers=1, desc="V100 32GB x1, 올인 마운트"),
    2: dict(price=230, workers=1, desc="V100 16GB x1 + 인코더 RAM (RAM 30 USD 포함)"),
    3: dict(price=215, workers=1, desc="V100 16GB x1 + 인코더 P104"),
    4: dict(price=415, workers=2, desc="V100 16GB x2 + 인코더 P104"),
    6: dict(price=715, workers=2, desc="V100 32GB x1 + 인코더 P104, 워커 2벌 (같은 카드)"),
    7: dict(price=715, workers=1, desc="V100 32GB x1 + 인코더 P104, 워커 1벌 (진단용)"),
    8: dict(price=700, workers=1, desc="V100 32GB x1, 올인 마운트 + 낱장 재인코딩 (통제 실험)"),
}


def start(cond):
    logs = {}
    if cond in (1, 8):
        # 조건 8 은 조건 1 과 기동이 완전히 같다. 다른 것은 요청을 넣는 방식뿐이다.
        logs["worker0"] = launch("c%d-worker0" % cond, GPU32, worker_args(WORKER_PORTS[0], True))
        wait_listening(WORKER_PORTS[0], logs["worker0"])
        return logs, None, [WORKER_PORTS[0]]

    if cond == 2:
        # 인코더는 RAM(te=cpu). CUDA 컨텍스트만 워커 카드에 얹힌다 (약 300MB)
        logs["encoder"] = launch("c2-encoder", GPU16A, encoder_args(ENCODER_PORT, True))
        logs["worker0"] = launch("c2-worker0", GPU16A, worker_args(WORKER_PORTS[0], False))
        wait_listening(ENCODER_PORT, logs["encoder"])
        wait_listening(WORKER_PORTS[0], logs["worker0"])
        return logs, ENCODER_PORT, [WORKER_PORTS[0]]

    if cond == 3:
        logs["encoder"] = launch("c3-encoder", GPUP104, encoder_args(ENCODER_PORT, False))
        logs["worker0"] = launch("c3-worker0", GPU16A, worker_args(WORKER_PORTS[0], False))
        wait_listening(ENCODER_PORT, logs["encoder"])
        wait_listening(WORKER_PORTS[0], logs["worker0"])
        return logs, ENCODER_PORT, [WORKER_PORTS[0]]

    if cond == 4:
        logs["encoder"] = launch("c4-encoder", GPUP104, encoder_args(ENCODER_PORT, False))
        logs["worker0"] = launch("c4-worker0", GPU16A, worker_args(WORKER_PORTS[0], False))
        logs["worker1"] = launch("c4-worker1", GPU16B, worker_args(WORKER_PORTS[1], False))
        wait_listening(ENCODER_PORT, logs["encoder"])
        for p in WORKER_PORTS:
            wait_listening(p, logs["worker%d" % WORKER_PORTS.index(p)])
        return logs, ENCODER_PORT, list(WORKER_PORTS)

    if cond == 6:
        # 32GB 카드 한 장에 워커 2벌. 13.6GB x 2 = 27.2GB 로 32GB 안에 들어간다.
        # 여유가 4.8GB 뿐이므로 생성 중 nvidia-smi 로 실제 점유를 확인할 것.
        logs["encoder"] = launch("c6-encoder", GPUP104, encoder_args(ENCODER_PORT, False))
        logs["worker0"] = launch("c6-worker0", GPU32, worker_args(WORKER_PORTS[0], False))
        logs["worker1"] = launch("c6-worker1", GPU32, worker_args(WORKER_PORTS[1], False))
        wait_listening(ENCODER_PORT, logs["encoder"])
        for i, p in enumerate(WORKER_PORTS):
            wait_listening(p, logs["worker%d" % i])
        return logs, ENCODER_PORT, list(WORKER_PORTS)

    if cond == 7:
        # 조건 3 과 운용이 같고 워커 카드만 32GB. 조건 1 의 장당 지연 원인 판별용.
        logs["encoder"] = launch("c7-encoder", GPUP104, encoder_args(ENCODER_PORT, False))
        logs["worker0"] = launch("c7-worker0", GPU32, worker_args(WORKER_PORTS[0], False))
        wait_listening(ENCODER_PORT, logs["encoder"])
        wait_listening(WORKER_PORTS[0], logs["worker0"])
        return logs, ENCODER_PORT, [WORKER_PORTS[0]]

    raise ValueError(cond)


# ----------------------------------------------------------------- 실행

def sample_params():
    return {
        "sample_steps": STEPS,
        "sample_method": "euler",
        "guidance": {"txt_cfg": 1.0, "distilled_guidance": 3.5},
    }


def base_req():
    r = {"width": W, "height": H, "sample_params": sample_params()}
    if EMBED_PATH:
        r["pulid_id_embedding_path"] = EMBED_PATH
        r["pulid_id_weight"] = 0.5
    return r


def run_once(cond, count):
    cold()
    logs, enc_port, workers = start(cond)

    t0 = time.time()
    first_image_at = None
    cond_bytes = None

    if cond == 1:
        # 올인 마운트: 프롬프트를 직접 넣고 batch_count 로 N장
        # (8개 요청으로 쪼개면 8번 재인코딩하므로 불공정. 배치가 이 구조의 자연스러운 운용이다)
        req = base_req()
        req["prompt"] = PROMPT
        req["batch_count"] = count
        req["seed"] = SEED
        jobs = [(WORKER_PORTS[0], post(WORKER_PORTS[0], "/sdcpp/v1/img_gen", req)["id"])]
    elif cond == 8:
        # 통제 실험: 자원은 조건 1 과 동일하고 표시 방식만 낱장이다.
        # 프롬프트를 요청마다 넣으므로 장마다 재인코딩이 일어난다. 그 비용을 포함해 잰다.
        jobs = []
        for i in range(count):
            req = base_req()
            req["prompt"] = PROMPT
            req["batch_count"] = 1
            req["seed"] = SEED + i
            jobs.append((WORKER_PORTS[0], post(WORKER_PORTS[0], "/sdcpp/v1/img_gen", req)["id"]))
    else:
        # 인코딩 1회 -> 1장 = 1작업 N개를 워커에 라운드로빈
        enc = post(enc_port, "/sdcpp/v1/encode",
                   {"prompt": PROMPT, "width": W, "height": H, "dir": "/dev/shm"})
        cond_path = enc["conditioning_path"]
        try:
            cond_bytes = os.path.getsize(cond_path)
        except Exception:
            pass
        jobs = []
        for i in range(count):
            port = workers[i % len(workers)]
            req = base_req()
            req["conditioning_path"] = cond_path
            req["batch_count"] = 1
            req["seed"] = SEED + i
            jobs.append((port, post(port, "/sdcpp/v1/img_gen", req)["id"]))

    done = set()
    while len(done) < len(jobs):
        for idx, (port, jid) in enumerate(jobs):
            if idx in done:
                continue
            st = (get(port, "/sdcpp/v1/jobs/" + jid).get("status") or "").lower()
            if st in ("completed", "done"):
                done.add(idx)
                if first_image_at is None:
                    first_image_at = time.time() - t0
            elif st in ("failed", "error"):
                print("[FAIL] job %s 실패" % jid)
                sys.exit(1)
        time.sleep(POLL)

    total = time.time() - t0
    proc(["pkill", "-f", "sd-server"])
    return total, first_image_at, logs, cond_bytes


def dump_logs(logs):
    pats = [
        "loading tensors completed",
        "sd_encode_conditioning completed",
        "get_learned_condition completed",
        "sampling completed",
        "decode_first_stage completed",
        "generate_image completed",
    ]
    for name, path in sorted(logs.items()):
        print("\n--- %s ---" % name)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if any(p in line for p in pats):
                        print("  " + line.strip())
        except Exception as e:
            print("  (읽기 실패: %s)" % e)


def record(cond, count, idx, total, first, cond_bytes):
    os.makedirs(LOGDIR, exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new:
            f.write("time,condition,price_usd,workers,count,run,total_s,first_image_s,"
                    "cond_bytes,tokens,pulid\n")
        f.write("%s,%d,%d,%d,%d,%d,%.2f,%s,%s,%s,%s\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            cond, CONDITIONS[cond]["price"], CONDITIONS[cond]["workers"],
            count, idx, total,
            ("%.2f" % first) if first is not None else "",
            cond_bytes if cond_bytes else "",
            tokens_from_size(cond_bytes) if cond_bytes else "",
            "on" if EMBED_PATH else "off",
        ))


def main():
    global EMBED_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, required=True, choices=[1, 2, 3, 4, 6, 7, 8])
    ap.add_argument("--count", type=int, required=True, choices=[1, 8])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-logs", action="store_true")
    a = ap.parse_args()

    EMBED_PATH = find_embed()

    c = CONDITIONS[a.condition]
    print("=" * 64)
    print("조건 %d — %s" % (a.condition, c["desc"]))
    print("추가 비용 %d USD / 워커 %d개 / 이미지 %d장 / 콜드 / %d회"
          % (c["price"], c["workers"], a.count, a.repeat))
    if EMBED_PATH:
        print("PuLID  : 활성 — %s" % EMBED_PATH)
    else:
        print("PuLID  : ⚠ 비활성 (.pulidembd 를 못 찾음). 장당 시간이 약 4.7초 짧게 나온다")
    print("=" * 64)

    results = []
    for i in range(a.repeat):
        total, first, logs, cond_bytes = run_once(a.condition, a.count)
        results.append(total)
        record(a.condition, a.count, i + 1, total, first, cond_bytes)

        print("\n[%d/%d] 총 %.2f초" % (i + 1, a.repeat, total), end="")
        if a.count > 1 and first is not None:
            print("  (첫 장 %.2f초)" % first, end="")
        if cond_bytes:
            print("  [조건 %d B = %d토큰]" % (cond_bytes, tokens_from_size(cond_bytes)), end="")
        print()

        if not a.no_logs and i == 0:
            dump_logs(logs)

    if a.repeat > 1:
        s = sorted(results)
        print("\n" + "-" * 64)
        print("중앙값 %.2f초   (min %.2f / max %.2f)" % (s[len(s) // 2], s[0], s[-1]))

    print("\n결과 누적: %s" % RESULTS)


if __name__ == "__main__":
    main()
