#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPS 로 SM 할당 비율을 바꿔 가며 텍스트 인코딩 시간을 잰다.

목적은 하나다 - GPU 에서 코어 수와 성능이 선형인가.
선형이면 "총 처리량 / 코어 수 = 코어당 평균" 이라는 지표가 유효하다는 근거가 된다.

측정 대상은 V100 32GB 다. MPS 의 스레드 비율 제한은 Volta 이상에서만 동작하므로
P104(Pascal) 에서는 같은 실험을 할 수 없다.

  주의
  - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 는 클라이언트가 뜰 때 읽힌다.
    따라서 지점마다 sd-server 를 새로 띄운다.
  - sd-server 는 가중치를 lazy 로드한다. 첫 요청에 모델 로드가 섞이므로 1회차는 버린다.
  - MPS 는 SM 을 물리적으로 끄는 것이 아니라 할당 비율을 제한한다.
    클럭과 메모리 대역폭은 그대로 남으므로 완전한 선형을 기대하면 안 된다.

사용법
  python3 mps_scaling.py                 전체 측정
  python3 mps_scaling.py --no-mps        MPS 없이 기준선만
  python3 mps_scaling.py --repeat 6      지점당 측정 횟수 변경
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------- 설정

SD_SERVER_BIN = "/root/stable-diffusion.cpp/build/bin/sd-server"
DIFFUSION_MODEL = "/root/models/Persephone_Flux_2.0_Q8_0.gguf"
VAE_MODEL = "/root/models/flux-sd-cpp/ae.safetensors"
CLIP_L_MODEL = "/root/models/flux-sd-cpp/clip_l.safetensors"
T5XXL_MODEL = "/root/models/flux-sd-cpp/t5xxl-q8_0.gguf"
LORA_DIR = "/root/loras"

# V100 32GB. 인코더는 5.2GB 만 쓰므로 여유 있게 들어간다.
GPU_UUID = "GPU-6eaed7a1-7a66-e4e3-7ef4-a15086cf0bfd"
SM_TOTAL = 80                      # V100 은 80 SM

PORT = 8090
COND_DIR = "/dev/shm"
W = H = 512
LOGDIR = "/root/bench-mps"

# 100 을 기준선으로 삼는다. MPS 자체 오버헤드가 여기에 포함되므로
# 절대값이 아니라 이 값 대비 비율로 읽어야 한다.
PERCENTAGES = [100, 50, 25, 12.5]

# 512토큰(2청크)을 노리는 긴 프롬프트. bench/bench.py 와 동일해야 비교가 성립한다.
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

EXPECT_BYTES = 8391936             # 512토큰일 때의 조건 gguf 크기


# ----------------------------------------------------------------- 유틸

def emit(fh, msg):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=" + GPU_UUID,
             "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception:
        return "?"


def port_alive(timeout=1.0):
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/" % PORT, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True                # 404 여도 떠 있는 것
    except Exception:
        return False


def wait_port(limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        if port_alive():
            return True
        time.sleep(0.5)
    return False


def encode_once():
    """인코딩 1회. (소요 초, 조건 파일 크기) 를 돌려준다."""
    body = json.dumps({"prompt": PROMPT, "width": W, "height": H,
                       "dir": COND_DIR}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/sdcpp/v1/encode" % PORT,
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read().decode())
    dt = time.time() - t0
    path = res.get("conditioning_path")
    size = 0
    if path:
        try:
            size = os.path.getsize(path)
        except OSError:
            pass
    return dt, size


def encoder_cmd():
    return [
        SD_SERVER_BIN,
        "--diffusion-model", DIFFUSION_MODEL,
        "--vae", VAE_MODEL,
        "--clip_l", CLIP_L_MODEL,
        "--t5xxl", T5XXL_MODEL,
        "--lora-model-dir", LORA_DIR,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(PORT),
        "-v",
    ]


def purge_cond():
    for n in os.listdir(COND_DIR):
        if n.startswith("sdcond-"):
            try:
                os.remove(os.path.join(COND_DIR, n))
            except OSError:
                pass


# ----------------------------------------------------------------- MPS

def mps_start(fh):
    if subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"],
                      capture_output=True).returncode == 0:
        emit(fh, "[MPS] 이미 떠 있음")
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env,
                   capture_output=True, timeout=30)
    time.sleep(2)
    ok = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"],
                        capture_output=True).returncode == 0
    emit(fh, "[MPS] 데몬 기동 %s" % ("OK" if ok else "실패"))
    if not ok:
        sys.exit("MPS 데몬을 띄우지 못했다. 중단한다.")


def mps_stop(fh):
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n",
                       text=True, capture_output=True, timeout=30)
    except Exception:
        pass
    time.sleep(1)
    emit(fh, "[MPS] 데몬 정지")


# ----------------------------------------------------------------- 측정

def measure(pct, repeat, fh, use_mps):
    """한 지점 측정. sd-server 를 새로 띄우고, 재고, 내린다."""
    tag = ("mps%s" % str(pct).replace(".", "_")) if use_mps else "nomps"
    logpath = os.path.join(LOGDIR, "sd-%s.log" % tag)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID)
    if use_mps:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

    purge_cond()
    emit(fh, "")
    emit(fh, "=== %s (SM %s / %d) 기동 ... temp/clk %s" %
         (tag, ("%.0f" % (SM_TOTAL * pct / 100.0)) if use_mps else str(SM_TOTAL),
          SM_TOTAL, gpu_temp()))

    lf = open(logpath, "wb")
    p = subprocess.Popen(encoder_cmd(), env=env, stdout=lf, stderr=lf,
                         preexec_fn=os.setsid)
    try:
        if not wait_port():
            emit(fh, "  포트가 열리지 않았다. 로그: %s" % logpath)
            return None

        times, sizes = [], []
        for i in range(repeat):
            try:
                dt, size = encode_once()
            except Exception as e:
                emit(fh, "  %d회차 실패: %s" % (i + 1, e))
                return None
            times.append(dt)
            sizes.append(size)
            mark = " (버림 - 모델 로드 포함)" if i == 0 else ""
            emit(fh, "  %d회차 %7.3f초  cond %d B%s" % (i + 1, dt, size, mark))
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        lf.close()
        time.sleep(3)              # VRAM 반환 대기

    warm = times[1:]
    if not warm:
        return None
    med = statistics.median(warm)

    bad = [s for s in sizes[1:] if s != EXPECT_BYTES]
    if bad:
        emit(fh, "  주의: 조건 파일 크기가 %d B 가 아니다 -> %s" % (EXPECT_BYTES, bad))
        emit(fh, "        512토큰이 아닐 수 있다. 프롬프트를 확인할 것.")

    emit(fh, "  중앙값(1회차 제외) %.3f초   콜드 %.3f초   temp/clk %s" %
         (med, times[0], gpu_temp()))
    return {"pct": pct, "cold": times[0], "warm": warm, "median": med,
            "sizes": sizes, "log": logpath}


# ----------------------------------------------------------------- 본체

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=4, help="지점당 측정 횟수 (1회차는 버린다)")
    ap.add_argument("--no-mps", action="store_true", help="MPS 없이 기준선만 잰다")
    a = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    fh = open(os.path.join(LOGDIR, "result.txt"), "a")
    emit(fh, "=" * 72)
    emit(fh, "MPS SM 스케일링 측정   %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    emit(fh, "대상 %s   프롬프트 512토큰   %dx%d" % (GPU_UUID[:20], W, H))

    if port_alive():
        sys.exit("포트 %d 가 이미 열려 있다. 기존 서비스를 내리고 다시 실행할 것." % PORT)

    rows = []
    try:
        if a.no_mps:
            r = measure(None, a.repeat, fh, use_mps=False)
            if r:
                rows.append(r)
        else:
            # MPS 없는 기준선을 먼저 잡는다. MPS 자체 오버헤드를 분리하기 위해서다.
            base = measure(None, a.repeat, fh, use_mps=False)
            if base:
                rows.append(base)
            mps_start(fh)
            try:
                for pct in PERCENTAGES:
                    r = measure(pct, a.repeat, fh, use_mps=True)
                    if r:
                        rows.append(r)
            finally:
                mps_stop(fh)
    except KeyboardInterrupt:
        emit(fh, "\n중단됨")

    # ---- 요약
    emit(fh, "")
    emit(fh, "=" * 72)
    emit(fh, "요약 - 인코딩 시간 (512토큰, 1회차 제외 중앙값)")
    emit(fh, "")
    emit(fh, "| 설정 | 유효 SM | 인코딩 | MPS 100%% 대비 | 선형 기대치 |")
    emit(fh, "|---|---|---|---|---|")

    ref = None
    for r in rows:
        if r["pct"] == 100:
            ref = r["median"]
    for r in rows:
        if r["pct"] is None:
            name, sm, expect = "MPS 없음", str(SM_TOTAL), "-"
        else:
            name = "MPS %s%%" % r["pct"]
            sm = "%.0f" % (SM_TOTAL * r["pct"] / 100.0)
            expect = ("%.3f초" % (ref * 100.0 / r["pct"])) if ref else "-"
        ratio = ("%.2f배" % (r["median"] / ref)) if ref else "-"
        emit(fh, "| %s | %s | %.3f초 | %s | %s |" % (name, sm, r["median"], ratio, expect))

    emit(fh, "")
    emit(fh, "읽는 법")
    emit(fh, "  '선형 기대치' 는 SM 이 절반이면 시간이 2배가 된다는 가정으로 계산한 값이다.")
    emit(fh, "  실측이 이 값에 가까우면 코어 수와 성능이 선형이라는 뜻이고,")
    emit(fh, "  '총 처리량 / 코어 수' 를 코어당 평균으로 쓰는 것이 정당해진다.")
    emit(fh, "  실측이 기대치보다 빠르면 SM 이외의 자원(클럭/대역폭)이 남아 있다는 뜻이다.")
    emit(fh, "")
    emit(fh, "결과 파일 %s" % os.path.join(LOGDIR, "result.txt"))
    fh.close()


if __name__ == "__main__":
    main()
