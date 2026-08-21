#!/usr/bin/env python3
"""
CPU 스레드 스케일링 측정 — 텍스트 인코딩

질문: "CPU 코어를 늘리면 P104 를 따라잡을 수 있는가?"

  - 선형이면  -> "P104 를 따라잡으려면 N 코어가 필요하다" 가 실측 기반이 된다
  - 포화하면  -> "코어를 늘려도 못 따라잡는다" 로 결론이 더 강해진다

어느 쪽이 나와도 논지에 유리하다.

측정
  스레드 1 / 2 / 4 / 8 각각에 대해
    256토큰 1회(버림, 모델 로드 포함) -> 256토큰 2회 -> 512토큰 2회
  마지막에 P104 를 같은 프롬프트로 측정해 비교군으로 둔다

시간은 HTTP 왕복이 아니라 로그의
    sd_encode_conditioning completed, taking X.XXs
를 파싱한다. 기존 13.97초 기준값과 같은 계측 지점이다.

사용법
    nohup python3 /root/thread_scaling.py > /root/bench/ts_console.log 2>&1 &

진행 확인
    tail -30 /root/bench/thread_scaling.log

요약만 다시 보기
    python3 /root/thread_scaling.py --summary
"""

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

# ----- 여기부터 자기 환경에 맞게 --------------------------------------

SD = "/path/to/stable-diffusion.cpp/build/bin/sd-server"
DIT = "/path/to/models/flux-dev-q8_0.gguf"
VAE = "/path/to/models/ae.safetensors"
CLIP = "/path/to/models/clip_l.safetensors"
T5 = "/path/to/models/t5xxl-q8_0.gguf"
LORA = "/path/to/loras"

GPUP104 = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # P104-100

LOGDIR = "/root/bench"
RESULTS = os.path.join(LOGDIR, "thread_scaling.csv")
RUNLOG = os.path.join(LOGDIR, "thread_scaling.log")

# ----- 여기까지 -------------------------------------------------------

PORT = 8090
THREADS = [1, 2, 4, 8]      # EPYC 7232P = 8 코어. 16 을 넣으면 SMT 효과도 본다
REPEAT = 2                  # 워밍 이후 측정 횟수
COND_DIR = "/dev/shm"

# T5-XXL 파라미터 수. 연산량 = 2 x PARAMS x 토큰수
PARAMS = 4.7e9

# P104-100 공개 스펙 (INT8 DP4A)
P104_TOPS = 22.0

# 256토큰을 노리는 프롬프트
PROMPT_A = (
    "a photorealistic portrait of a young woman with light freckles across her nose and "
    "cheekbones, wearing a hand knitted crimson scarf wrapped twice around her neck, standing "
    "in front of a tall walnut bookshelf filled with worn hardcover volumes and a few small "
    "ceramic figurines, holding a chipped blue stoneware mug in her left hand at chest height, "
    "steam rising faintly from the surface of the tea inside, warm late afternoon sunlight "
    "entering through a tall sash window on the right side of the frame"
)

# 512토큰(2청크)을 노리는 프롬프트. 실사용 길이에 해당한다.
PROMPT_B = PROMPT_A + (
    ", casting long soft shadows across the wooden floorboards and illuminating dust motes "
    "suspended in the air, her expression calm and slightly amused, looking just past the "
    "camera rather than directly at it, shoulder length auburn hair loosely tied back with a "
    "few strands falling forward, wearing a cream colored cable knit sweater with visible "
    "texture and a slightly oversized fit, shallow depth of field with the bookshelf softly "
    "blurred behind her, natural skin texture with visible pores and fine detail, subtle color "
    "grading toward warm amber tones in the highlights and cool blue tones in the shadows, "
    "shot on a full frame camera with an 85mm prime lens at f/1.8, careful attention to the "
    "fall of light across fabric folds and the reflection in the glazed surface of the ceramic "
    "mug, quiet domestic atmosphere, unhurried and intimate composition"
)

PROMPTS = [("A", PROMPT_A), ("B", PROMPT_B)]

ENCODE_RE = re.compile(r"sd_encode_conditioning completed, taking ([\d.]+)s")


# ----------------------------------------------------------------- 유틸

def emit(log, text):
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def proc(args):
    """셸을 거치지 않고 실행.

    shell=True 로 pgrep/pkill 을 돌리면 래퍼 셸의 명령줄에 검색어가 들어 있어
    자기 자신을 매칭한다. 프로세스 탐지·종료는 반드시 리스트 형태로 실행할 것.
    """
    return subprocess.run(args, capture_output=True)


def sd_running():
    return proc(["pgrep", "-f", "sd-server"]).returncode == 0


def stop_sd():
    proc(["pkill", "-f", "sd-server"])
    for _ in range(30):
        if not sd_running():
            break
        time.sleep(0.5)
    else:
        proc(["pkill", "-9", "-f", "sd-server"])
        time.sleep(2)
    time.sleep(1)


def launch(name, threads, gpu):
    """인코더 기동.

    threads 가 None 이면 -t 를 주지 않는다 (P104 비교군).
    gpu 가 None 이면 CPU 백엔드로 텍스트 인코더를 올린다.
    """
    os.makedirs(LOGDIR, exist_ok=True)
    logfile = os.path.join(LOGDIR, name + ".log")

    args = [
        SD,
        "--diffusion-model", DIT,      # 버전 감지용. 가중치는 안 올라감
        "--vae", VAE,
        "--clip_l", CLIP,
        "--t5xxl", T5,
        "--lora-model-dir", LORA,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(PORT),
        "-v",
    ]
    if gpu is None:
        args += ["--backend", "te=cpu"]
    if threads is not None:
        args += ["-t", str(threads)]

    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)     # ggml 이 OpenMP 로 빌드된 경우 대비
    env["CUDA_VISIBLE_DEVICES"] = gpu if gpu else ""

    with open(logfile, "w") as f:
        subprocess.Popen(args, env=env, stdout=f, stderr=subprocess.STDOUT)

    for _ in range(180 * 5):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/" % PORT, timeout=1)
            return logfile
        except urllib.error.HTTPError:
            return logfile                        # 404 도 살아있다는 뜻
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("포트 %d 기동 실패. %s 확인" % (PORT, logfile))


def encode(prompt, timeout=1200):
    """조건 텐서 1회 생성. (조건 파일 크기, 토큰 수) 를 돌려준다."""
    req = urllib.request.Request(
        "http://127.0.0.1:%d/sdcpp/v1/encode" % PORT,
        data=json.dumps({"prompt": prompt, "width": 768, "height": 768,
                         "dir": COND_DIR}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    path = res["conditioning_path"]
    nbytes = os.path.getsize(path)
    return nbytes, round((nbytes - 3328) / 16384)


def last_encode_time(logfile):
    """로그에서 가장 마지막 인코딩 소요 시간을 뽑는다."""
    times = []
    with open(logfile, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ENCODE_RE.search(line)
            if m:
                times.append(float(m.group(1)))
    return times[-1] if times else None


def purge_cond():
    """조건 파일 캐시를 지운다. 남아 있으면 재인코딩을 건너뛸 수 있다."""
    for name in os.listdir(COND_DIR):
        if name.startswith("sdcond-"):
            try:
                os.remove(os.path.join(COND_DIR, name))
            except OSError:
                pass


# ----------------------------------------------------------------- 측정

def record(label, threads, tag, tokens, run, secs):
    new = not os.path.exists(RESULTS)
    gops = (2 * PARAMS * tokens) / secs / 1e9
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new:
            f.write("time,label,threads,prompt,tokens,run,encode_s,gops\n")
        f.write("%s,%s,%s,%s,%d,%d,%.3f,%.1f\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S"), label,
            threads if threads is not None else "", tag, tokens, run, secs, gops))
    return gops


def measure_config(log, label, threads, gpu):
    """한 구성에 대해 워밍 1회 + 프롬프트별 REPEAT 회 측정."""
    emit(log, "")
    emit(log, "#" * 68)
    emit(log, "# %s   %s" % (label, time.strftime("%H:%M:%S")))
    emit(log, "#" * 68)

    stop_sd()
    logfile = launch("ts-" + label, threads, gpu)

    # 워밍 — 모델 로드가 여기 포함된다. 버린다.
    purge_cond()
    emit(log, "  워밍 (모델 로드 포함, 버림) ...")
    encode(PROMPT_A)
    warm = last_encode_time(logfile)
    emit(log, "    %s초" % (("%.2f" % warm) if warm else "?"))

    out = {}
    for tag, prompt in PROMPTS:
        vals = []
        for i in range(REPEAT):
            purge_cond()                    # 캐시 히트로 인코딩을 건너뛰지 않게
            nbytes, tokens = encode(prompt)
            secs = last_encode_time(logfile)
            if secs is None:
                emit(log, "  [FAIL] 로그에서 인코딩 시간을 못 찾음: %s" % logfile)
                stop_sd()
                return None
            gops = record(label, threads, tag, tokens, i + 1, secs)
            vals.append(secs)
            emit(log, "  프롬프트 %s (%d토큰, %d B)  [%d/%d]  %.2f초  %.0f GOPS"
                 % (tag, tokens, nbytes, i + 1, REPEAT, secs, gops))
        out[tag] = (statistics.median(vals), tokens)

    stop_sd()
    return out


def summary():
    if not os.path.exists(RESULTS):
        print("결과 파일이 없다: %s" % RESULTS)
        return

    rows = list(csv.DictReader(open(RESULTS, encoding="utf-8")))

    def pick(label, tag):
        v = [float(r["encode_s"]) for r in rows
             if r["label"] == label and r["prompt"] == tag]
        return statistics.median(v) if v else None

    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])

    # 스레드 수는 THREADS 상수가 아니라 CSV 에서 뽑는다.
    # --smt 처럼 일부만 다시 돌려도 요약이 전체를 보여줘야 하기 때문이다.
    tnums = sorted(int(m.group(1)) for m in
                   (re.fullmatch(r"cpu-t(\d+)", l) for l in labels) if m)

    print("\n" + "=" * 72)
    print("CPU 스레드 스케일링 — 텍스트 인코딩 (중앙값)")
    print("=" * 72)
    print("%-10s %-12s %-12s %-12s %-10s" % (
        "구성", "256토큰", "512토큰", "256 GOPS", "1스레드 대비"))
    print("-" * 72)

    base = pick("cpu-t1", "A")
    p104 = pick("p104", "A")

    for label in labels:
        a, b = pick(label, "A"), pick(label, "B")
        if a is None:
            continue
        gops = (2 * PARAMS * 256) / a / 1e9
        sp = ("%.2f배" % (base / a)) if base else "—"
        print("%-10s %-12s %-12s %-12.0f %-10s" % (
            label,
            "%.2f초" % a,
            ("%.2f초" % b) if b else "—",
            gops, sp))

    print("-" * 72)

    # 병렬 효율 — 선형이면 100%
    if base and tnums:
        print("\n[병렬 효율]  1스레드 대비 가속 ÷ 스레드 수")
        for n in tnums:
            t = pick("cpu-t%d" % n, "A")
            if t:
                print("  %2d 스레드   가속 %.2f배   효율 %5.1f%%"
                      % (n, base / t, (base / t) / n * 100))
        # 물리 코어(8)를 넘는 구간은 SMT 다. 손해인지 이득인지 직접 본다.
        phys, smt = pick("cpu-t8", "A"), pick("cpu-t16", "A")
        if phys and smt:
            print("\n[SMT 판정]  물리 8코어 vs 논리 16스레드")
            print("   8 스레드 %.2f초 / 16 스레드 %.2f초" % (phys, smt))
            if smt > phys:
                print("   -> SMT 가 %.0f%% 느리다. 연산 바운드에서 SMT 는 손해다."
                      % ((smt / phys - 1) * 100))
            else:
                print("   -> SMT 가 %.0f%% 빠르다." % ((phys / smt - 1) * 100))
        dflt = pick("cpu-default", "A")
        if dflt and phys:
            print("\n[기본 스레드 설정]  -t 를 주지 않았을 때")
            print("   기본값 %.2f초 / 명시 8스레드 %.2f초  (차이 %+.0f%%)"
                  % (dflt, phys, (dflt / phys - 1) * 100))

    # 핵심 결론 — P104 를 따라잡는 데 필요한 코어 수.
    # SMT 구간은 코어 수 환산에 쓰면 안 되므로 물리 코어 상한(8)을 쓴다.
    top = pick("cpu-t8", "A")
    if p104 and top:
        gops_cpu = (2 * PARAMS * 256) / top / 1e9
        gops_p104 = (2 * PARAMS * 256) / p104 / 1e9
        per_core = gops_cpu / 8
        print("\n[결론]")
        print("  CPU 8코어     %.2f초   %.0f GOPS   (코어당 %.1f GOPS)"
              % (top, gops_cpu, per_core))
        print("  P104-100      %.2f초   %.0f GOPS   (공개 스펙 %.0f TOPS 의 %.0f%%)"
              % (p104, gops_p104, P104_TOPS, gops_p104 / (P104_TOPS * 1000) * 100))
        print("  격차          %.1f배" % (top / p104))
        print("  선형 가정 시 P104 를 따라잡는 데 필요한 코어 수 : 약 %.0f 개"
              % (gops_p104 / per_core))
        print("  ※ 위 병렬 효율이 100%% 에서 멀수록 이 값은 낙관적 하한이다.")

    # 512토큰에서 격차가 줄어드는지
    if top and p104:
        ta, tb = pick("cpu-t8", "B"), pick("p104", "B")
        if ta and tb:
            print("\n[프롬프트 길이 영향]")
            print("  256토큰  CPU %.2f초 / P104 %.2f초  = %.1f배" % (top, p104, top / p104))
            print("  512토큰  CPU %.2f초 / P104 %.2f초  = %.1f배" % (ta, tb, ta / tb))
            print("  CPU 는 토큰이 2배가 될 때 %.2f배 늘었다 (2.0 이면 선형)" % (ta / top))

    print("\n원본: %s" % RESULTS)
    print("=" * 72)


def main():
    if "--summary" in sys.argv:
        summary()
        return

    # --smt : 16스레드(SMT 전부)와 기본 스레드 설정만 추가로 잰다.
    #         기존 결과에 덧붙으므로 results.csv 를 지우지 말 것. 약 4분.
    smt_only = "--smt" in sys.argv

    os.makedirs(LOGDIR, exist_ok=True)
    start = time.time()

    mode = "a" if smt_only else "w"
    with open(RUNLOG, mode, encoding="utf-8") as log:
        emit(log, "=" * 68)
        emit(log, "CPU 스레드 스케일링  시작 %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        if smt_only:
            emit(log, "SMT 검증 모드 — 16스레드 + 기본 스레드 설정. 약 4분")
        else:
            emit(log, "스레드 %s + P104 비교군 / 프롬프트 2종 / 각 %d회"
                 % (THREADS, REPEAT))
            emit(log, "1스레드가 가장 오래 걸린다. 전체 25분 내외 예상")
        emit(log, "=" * 68)

        if smt_only:
            # 논리 스레드 전부
            if measure_config(log, "cpu-t16", 16, None) is None:
                emit(log, "!! 16스레드 측정 실패")
            # -t 를 아예 주지 않았을 때. 기존 13.97초의 출처를 확인한다.
            if measure_config(log, "cpu-default", None, None) is None:
                emit(log, "!! 기본 설정 측정 실패")
        else:
            for n in THREADS:
                if measure_config(log, "cpu-t%d" % n, n, None) is None:
                    emit(log, "!! 실패. 다음 구성으로 진행한다.")

            # 비교군 — 같은 세션, 같은 프롬프트
            if measure_config(log, "p104", None, GPUP104) is None:
                emit(log, "!! P104 측정 실패")

        emit(log, "")
        emit(log, "=" * 68)
        emit(log, "완료  %s   총 %.1f분" % (time.strftime("%H:%M:%S"),
                                          (time.time() - start) / 60))
        emit(log, "=" * 68)

    summary()


if __name__ == "__main__":
    main()
