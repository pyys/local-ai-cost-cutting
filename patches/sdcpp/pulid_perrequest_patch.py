#!/usr/bin/env python3
"""
PuLID ID 임베딩을 요청별 파라미터로 노출한다.

배경:
  sd_pulid_params_t 는 sd_img_gen_params_t 안에 있어 원래부터 생성별 파라미터다.
  SDGenerationParams 에도 pulid_id_embedding_path / pulid_id_weight 멤버가 있고
  to_sd_img_gen_params_t() 가 이를 조립한다. 다만 from_json_str() 에 등록되어
  있지 않아 HTTP 요청으로는 지정할 수 없었다.

효과:
  캐릭터를 바꿀 때마다 sd-server 를 재기동하던 것을 없앨 수 있다.
  워커 수 N 이 커질수록 이득이 커지고(재기동 비용이 N 배가 된다),
  서로 다른 캐릭터의 이미지를 같은 워커 큐에 섞어 넣을 수 있게 된다.

주의:
  --pulid-weights (PuLID 모델 자체) 는 컨텍스트 파라미터라 기동 시 고정이다.
  바뀌는 것은 ID 임베딩(.pulidembd)뿐이다.

전제: cond_serialize_patch.py 적용 (앵커로 사용)

사용법:
    python3 pulid_perrequest_patch.py --dry-run
    python3 pulid_perrequest_patch.py
    python3 pulid_perrequest_patch.py --revert
"""

import os
import sys
import glob
import shutil
import datetime

# 소스 트리 위치. 환경변수 SD_CPP_ROOT 로 덮어쓸 수 있다.
ROOT = os.environ.get("SD_CPP_ROOT", "/root/stable-diffusion.cpp")
COMMON_CPP = os.path.join(ROOT, "examples/common/common.cpp")

ANCHOR = '    load_if_exists("conditioning_path", conditioning_path);'
ADDED = ('\n    load_if_exists("pulid_id_embedding_path", pulid_id_embedding_path);'
         '\n    load_if_exists("pulid_id_weight", pulid_id_weight);')
MARK = "pulid_id_embedding_path\", pulid_id_embedding_path"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    if "--revert" in sys.argv:
        backups = sorted(glob.glob(COMMON_CPP + ".bak3-*"))
        if not backups:
            print("[SKIP] 백업 없음")
            return 1
        shutil.copy2(backups[-1], COMMON_CPP)
        print("[OK] 복원: %s" % os.path.basename(backups[-1]))
        return 0

    src = read(COMMON_CPP)

    if MARK in src:
        print("[FAIL] 이미 적용됨")
        return 1
    if src.count(ANCHOR) != 1:
        print("[FAIL] 앵커를 찾지 못함. cond_serialize_patch.py 를 먼저 적용할 것.")
        return 1

    out = src.replace(ANCHOR, ANCHOR + ADDED)

    if "--dry-run" in sys.argv:
        print("[OK] 앵커 확인. 추가될 내용:")
        print(ADDED.strip())
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(COMMON_CPP, "%s.bak3-%s" % (COMMON_CPP, stamp))
    with open(COMMON_CPP, "w", encoding="utf-8") as f:
        f.write(out)

    print("[OK] %s" % COMMON_CPP)
    print("백업 접미사: .bak3-%s" % stamp)
    print("재빌드: cmake --build %s/build -j 8" % ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
