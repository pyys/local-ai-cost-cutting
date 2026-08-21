#!/usr/bin/env python3
"""
벤치마크 무인 실행 — 단계 2(1장) + 단계 3(8장) 전체

조건 1~4 × {1장, 8장} × 3회 = 24회. 약 1시간.
한 조합이 실패해도 멈추지 않고 다음으로 넘어간다.

사용법
    nohup python3 /root/bench_all.py > /root/bench/console.log 2>&1 &

조건 6 만 추가로 (기존 결과에 덧붙는다. results.csv 지우지 말 것)
    nohup python3 /root/bench_all.py --extra > /root/bench/console6.log 2>&1 &

진행 확인
    tail -40 /root/bench/run.log

완료 후 요약만 다시 보기
    python3 /root/bench_all.py --summary
"""

import csv
import os
import subprocess
import sys
import time

# bench.py 는 이 스크립트와 같은 디렉터리에 있다고 본다.
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
LOGDIR = "/root/bench"          # 로그·결과 출력 위치
RUNLOG = os.path.join(LOGDIR, "run.log")
RESULTS = os.path.join(LOGDIR, "results.csv")

REPEAT = 3
PLAN = [(c, 1) for c in (1, 2, 3, 4)] + [(c, 8) for c in (1, 2, 3, 4)]
PLAN_EXTRA = [(7, 1), (7, 8), (6, 1), (6, 8)]

DESC = {
    1: "V100 32GB x1, 올인 마운트          700 USD, 워커 1",
    2: "V100 16GB x1 + 인코더 RAM          230 USD, 워커 1",
    3: "V100 16GB x1 + 인코더 P104         215 USD, 워커 1",
    4: "V100 16GB x2 + 인코더 P104         415 USD, 워커 2",
    6: "V100 32GB x1 + P104, 워커 2벌      715 USD, 워커 2",
    7: "V100 32GB x1 + P104, 워커 1벌      진단용, 워커 1",
}
PRICE = {1: 700, 2: 230, 3: 215, 4: 415, 6: 715, 7: 715}
WORKERS = {1: 1, 2: 1, 3: 1, 4: 2, 6: 2, 7: 1}
CONDS = (1, 2, 3, 4, 6, 7)


def emit(log, text):
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def summary():
    if not os.path.exists(RESULTS):
        print("결과 파일이 없다: %s" % RESULTS)
        return

    rows = []
    with open(RESULTS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def pick(cond, count):
        v = [float(r["total_s"]) for r in rows
             if int(r["condition"]) == cond and int(r["count"]) == count]
        return median(v) if v else None

    def pickfirst(cond, count):
        v = [float(r["first_image_s"]) for r in rows
             if int(r["condition"]) == cond and int(r["count"]) == count
             and r["first_image_s"]]
        return median(v) if v else None

    print("\n" + "=" * 72)
    print("요약 — 조건별 중앙값 (콜드, 명령 1회)")
    print("=" * 72)
    print("%-4s %-6s %-6s %-10s %-10s %-10s %-10s" % (
        "조건", "USD", "워커", "1장", "8장", "첫장", "라운드당"))
    print("-" * 72)

    def per_round(c):
        """(8장 − 1장) / (라운드 수 − 1).  워커 1개면 장당과 같다."""
        one, eight = pick(c, 1), pick(c, 8)
        if not one or not eight:
            return None
        rounds = -(-8 // WORKERS[c])          # ceil(8/W)
        return (eight - one) / (rounds - 1) if rounds > 1 else None

    for c in CONDS:
        one, eight, first = pick(c, 1), pick(c, 8), pickfirst(c, 8)
        if one is None and eight is None:
            continue
        pr = per_round(c)
        print("%-4d %-6d %-6d %-10s %-10s %-10s %-10s" % (
            c, PRICE[c], WORKERS[c],
            ("%.1f초" % one) if one else "—",
            ("%.1f초" % eight) if eight else "—",
            ("%.1f초" % first) if first else "—",
            ("%.2f초" % pr) if pr else "—",
        ))

    print("-" * 72)

    t1, t3, t4, t6, t7 = (pick(x, 8) for x in (1, 3, 4, 6, 7))
    p1, p3, p7 = per_round(1), per_round(3), per_round(7)

    # 조건 7 — 조건 1 의 장당 지연 원인 판별
    if p7 and p1 and p3:
        print("\n[조건 7 — 조건 1 의 장당 지연은 카드 탓인가 구조 탓인가]")
        print("  라운드당   조건 1 %.2f초 / 조건 7 %.2f초 / 조건 3 %.2f초" % (p1, p7, p3))
        d1, d3 = abs(p7 - p1), abs(p7 - p3)
        if d3 < d1:
            print("  -> 조건 3 쪽에 가깝다. 원인은 인코더 동거 또는 배치 경로.")
            print("     같은 32GB 카드에서도 분리·쪼개기가 장당 %.2f초를 줄였다." % (p1 - p7))
        else:
            print("  -> 조건 1 쪽에 가깝다. 원인은 카드 개체차다.")
            print("     조건 3 이 빨랐던 것은 분리의 공이 아니므로 서술을 수정할 것.")
    if t1 and t7:
        print("  배치→개별 쪼개기 비용 (카드 동일) : 8장 %.1f초 vs %.1f초  (차 %+.1f초)"
              % (t1, t7, t7 - t1))

    # 조건 6 — 같은 예산을 32GB 한 장에 쓰면
    if t6:
        print("\n[조건 6 — 같은 예산을 32GB 한 장에 쓰면]")
        base = t7 or t3
        if base:
            print("  6 대 %s   워커 2벌 대 1벌, 같은 카드 : %.1f초 vs %.1f초  (%.2f배)"
                  % ("7" if t7 else "3", t6, base, base / t6))
            print("            1.0배에 가까우면 '한 카드에 워커를 더 올려도 소용없다'")
        if t4:
            print("  6 대 4   715 USD 대 415 USD, 워커 동수 : %.1f초 vs %.1f초"
                  % (t6, t4))
            print("            4가 빠르면 저비용 카드 혼합이 비용·산출 양쪽에서 이긴다")

    # 조건 5 산출 (측정 아님)
    if t3 and t4:
        # 워커 1개 -> 2개에서 줄어든 생성 구간으로 장당 시간을 역산
        # t(N) = 기동+인코딩 + ceil(8/N) x 장당
        # t3 = S + 8a , t4 = S + 4a  ->  a = (t3-t4)/4 , S = t4 - 4a
        a = (t3 - t4) / 4.0
        S = t4 - 4 * a
        t5 = S + 3 * a          # 워커 3개 -> ceil(8/3)=3라운드
        print("\n[산출값 — 실측 아님]")
        print("  장당 생성      a = %.1f초" % a)
        print("  고정 구간      S = %.1f초  (기동 + 인코딩)" % S)
        print("  조건 5 (615 USD, 워커 3) 8장 ≈ %.1f초" % t5)
        print("  ※ V100 16GB 3장을 보유하지 않아 측정하지 못했다.")
        print("     조건 3·4의 워커 확장 계수로 계산한 추정값이다.")

    # 워밍 추정 (실측 아님)
    print("\n[워밍 추정 — 실측 아님]")
    print("  콜드 측정 로그에서 기동 구간을 분리해 계산한 값이다.")
    print("  각 조건의 로그에서 다음을 빼면 된다:")
    print("    - T5 로드 (loading tensors, 인코더 쪽)")
    print("    - DiT 로드 (loading tensors, 워커 쪽)")
    print("    - 첫 생성의 클럭 램프업분 (첫 장 sampling - 이후 장 sampling)")

    print("\n원본: %s" % RESULTS)
    print("=" * 72)


def main():
    if "--summary" in sys.argv:
        summary()
        return

    extra = "--extra" in sys.argv
    plan = PLAN_EXTRA if extra else PLAN
    runlog = RUNLOG.replace(".log", "6.log") if extra else RUNLOG

    os.makedirs(LOGDIR, exist_ok=True)
    start = time.time()

    with open(runlog, "w", encoding="utf-8") as log:
        emit(log, "=" * 72)
        emit(log, "벤치마크 무인 실행  시작 %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        emit(log, "%d개 조합 × %d회 = %d회" % (len(plan), REPEAT, len(plan) * REPEAT))
        emit(log, "=" * 72)

        for i, (cond, count) in enumerate(plan, 1):
            emit(log, "")
            emit(log, "#" * 72)
            emit(log, "# [%d/%d]  조건 %d / %d장   %s" % (
                i, len(plan), cond, count, time.strftime("%H:%M:%S")))
            emit(log, "#        %s" % DESC[cond])
            emit(log, "#" * 72)

            p = subprocess.run(
                [sys.executable, BENCH,
                 "--condition", str(cond), "--count", str(count),
                 "--repeat", str(REPEAT)],
                capture_output=True, text=True)

            emit(log, (p.stdout or "") + (p.stderr or ""))

            if p.returncode != 0:
                emit(log, "!! 실패 (exit %d). 다음 조합으로 진행한다." % p.returncode)

            # 셸을 거치면 래퍼 셸이 검색어에 매칭되므로 리스트 형태로 실행
            subprocess.run(["pkill", "-f", "sd-server"], capture_output=True)
            time.sleep(3)

        el = (time.time() - start) / 60.0
        emit(log, "")
        emit(log, "=" * 72)
        emit(log, "전체 완료  %s   총 %.1f분" % (time.strftime("%H:%M:%S"), el))
        emit(log, "=" * 72)

    summary()


if __name__ == "__main__":
    main()
