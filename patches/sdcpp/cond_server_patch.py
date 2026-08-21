#!/usr/bin/env python3
"""
stable-diffusion.cpp 개조 2단계 — 인코더 서비스화.

전제: cond_serialize_patch.py 가 이미 적용되어 있어야 한다.

추가되는 것:
  1. sd_encode_conditioning()  공개 API — 인코딩만 수행하고 gguf 로 저장
  2. POST /sdcpp/v1/encode     동기 엔드포인트
  3. img_gen 요청의 "conditioning_path" JSON 필드

구조:
  인코더 sd-server (P104, 확산 미수행 -> DiT 가 VRAM 에 안 올라감)
      POST /sdcpp/v1/encode {"prompt":..., "width":768, "height":768}
      -> {"conditioning_path": "/dev/shm/sdcond-<hash>.gguf"}
  워커 sd-server (V100, --clip_l / --t5xxl 없이 기동)
      POST /sdcpp/v1/img_gen {"conditioning_path": "...", ...}

파일명은 프롬프트+해상도 해시라서 같은 입력이면 같은 경로가 된다.
오래된 파일 정리는 호출자(오케스트레이터) 책임이다. 512토큰이면 1개당 약 8MB 라
/dev/shm 에 쌓이면 tmpfs 가 찬다.

사용법:
    python3 cond_server_patch.py --dry-run
    python3 cond_server_patch.py
    python3 cond_server_patch.py --revert
"""

import os
import sys
import glob
import shutil
import datetime

# 소스 트리 위치. 환경변수 SD_CPP_ROOT 로 덮어쓸 수 있다.
ROOT = os.environ.get("SD_CPP_ROOT", "/root/stable-diffusion.cpp")

SD_H = os.path.join(ROOT, "include/stable-diffusion.h")
SD_CPP = os.path.join(ROOT, "src/stable-diffusion.cpp")
COMMON_CPP = os.path.join(ROOT, "examples/common/common.cpp")
ROUTES = os.path.join(ROOT, "examples/server/routes_sdcpp.cpp")

MARK = "sd_encode_conditioning"
PREREQ = "sd_cond_io"


# ---------------------------------------------------------------- 삽입 내용

API_DECL = '''SD_API bool sd_encode_conditioning(sd_ctx_t* sd_ctx,
                                  const char* prompt,
                                  int clip_skip,
                                  int width,
                                  int height,
                                  const char* out_path);
'''

API_IMPL = r'''// sd_cond_io: 인코딩만 수행해 조건 텐서를 gguf 로 저장한다.
// prepare_image_generation_embeds() 의 cond 계산부와 같은 경로를 쓰되
// prepare_generation_extensions() 는 호출하지 않는다. PuLID 는 conditioning 을
// 수정하지 않으므로 결과가 같다 (PhotoMaker 는 수정하므로 이 API 로는 불가).
bool sd_encode_conditioning(sd_ctx_t* sd_ctx,
                            const char* prompt,
                            int clip_skip,
                            int width,
                            int height,
                            const char* out_path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || out_path == nullptr) {
        return false;
    }
    if (sd_ctx->sd->cond_stage_model == nullptr) {
        LOG_ERROR("sd_encode_conditioning: no conditioner loaded");
        return false;
    }
    ConditionerRunnerDoneOnExit conditioner_runner_done{sd_ctx->sd->cond_stage_model.get()};

    std::vector<sd::Tensor<float>> no_ref_images;
    ConditionerParams condition_params;
    condition_params.text            = SAFE_STR(prompt);
    condition_params.clip_skip       = clip_skip;
    condition_params.width           = width;
    condition_params.height          = height;
    condition_params.ref_images      = &no_ref_images;
    condition_params.zero_out_masked = false;

    int64_t t0 = ggml_time_ms();
    auto cond  = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                    condition_params);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("sd_encode_conditioning completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    return sd_cond_save(out_path, cond);
}

'''

ROUTE = r'''    // sd_cond_io: 인코딩 전용 엔드포인트. 확산을 수행하지 않으므로
    // 이 서비스에서는 DiT 가중치가 VRAM 에 올라가지 않는다 (lazy load).
    svr.Post("/sdcpp/v1/encode", [runtime](const httplib::Request& req, httplib::Response& res) {
        static std::mutex sd_encode_mutex;
        try {
            json body = req.body.empty() ? json::object() : json::parse(req.body);

            std::string prompt = body.value("prompt", std::string());
            int clip_skip      = body.value("clip_skip", -1);
            int width          = body.value("width", 512);
            int height         = body.value("height", 512);
            std::string dir    = body.value("dir", std::string("/dev/shm"));

            // 같은 프롬프트·해상도면 같은 경로가 되도록 해시를 쓴다.
            std::string key = prompt + "|" + std::to_string(width) + "x" + std::to_string(height) +
                              "|" + std::to_string(clip_skip);
            char name[64];
            snprintf(name, sizeof(name), "/sdcond-%016llx.gguf",
                     (unsigned long long)std::hash<std::string>{}(key));
            std::string path = dir + name;

            bool ok;
            {
                std::lock_guard<std::mutex> lock(sd_encode_mutex);
                ok = sd_encode_conditioning(runtime->sd_ctx,
                                            prompt.c_str(),
                                            clip_skip,
                                            width,
                                            height,
                                            path.c_str());
            }
            if (!ok) {
                res.status = 500;
                res.set_content(R"({"error":"encode_failed"})", "application/json");
                return;
            }

            json out;
            out["conditioning_path"] = path;
            out["width"]             = width;
            out["height"]            = height;
            res.status = 200;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid request"}, {"message", e.what()}}).dump(),
                            "application/json");
        }
    });

'''


# ---------------------------------------------------------------- 유틸

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def once(text, needle, label):
    n = text.count(needle)
    if n != 1:
        raise RuntimeError("%s: 앵커가 %d 번 발견됨 (1 이어야 함)\n  %r" % (label, n, needle[:70]))


# ---------------------------------------------------------------- 각 파일 패치

def patch_sd_h(text):
    anchor = "SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);"
    once(text, anchor, "stable-diffusion.h")
    return text.replace(anchor, anchor + "\n" + API_DECL.rstrip("\n"))


def patch_sd_cpp(text):
    anchor = "static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,"
    once(text, anchor, "stable-diffusion.cpp")
    return text.replace(anchor, API_IMPL + anchor)


def patch_common_cpp(text):
    anchor = '    load_if_exists("negative_prompt", negative_prompt);'
    once(text, anchor, "common.cpp / from_json_str")
    return text.replace(anchor, anchor + '\n    load_if_exists("conditioning_path", conditioning_path);')


def patch_routes(text):
    anchor = '    svr.Get("/sdcpp/v1/capabilities",'
    once(text, anchor, "routes_sdcpp.cpp")
    return text.replace(anchor, ROUTE + anchor)


TARGETS = [
    (SD_H, patch_sd_h),
    (SD_CPP, patch_sd_cpp),
    (COMMON_CPP, patch_common_cpp),
    (ROUTES, patch_routes),
]


# ---------------------------------------------------------------- main

def revert():
    n = 0
    for path, _ in TARGETS:
        backups = sorted(glob.glob(path + ".bak2-*"))
        if not backups:
            print("[SKIP] 백업 없음: %s" % path)
            continue
        shutil.copy2(backups[-1], path)
        print("[OK] 복원: %s <- %s" % (path, os.path.basename(backups[-1])))
        n += 1
    print("%d 개 파일 복원. 재빌드 필요." % n)
    return 0


def main():
    if "--revert" in sys.argv:
        return revert()

    dry = "--dry-run" in sys.argv

    if PREREQ not in read(SD_CPP):
        print("[FAIL] 선행 패치가 없음. cond_serialize_patch.py 를 먼저 적용할 것.")
        return 1

    for path, _ in TARGETS:
        if not os.path.exists(path):
            print("[FAIL] 파일 없음: %s" % path)
            return 1
        if MARK in read(path):
            print("[FAIL] 이미 적용된 것으로 보임: %s" % path)
            print("       되돌리려면: python3 cond_server_patch.py --revert")
            return 1

    results = []
    for path, fn in TARGETS:
        try:
            results.append((path, fn(read(path))))
        except RuntimeError as e:
            print("[FAIL] %s" % e)
            return 1

    if dry:
        for path, _ in results:
            print("[OK] 앵커 확인: %s" % path)
        print("\n--dry-run 이므로 쓰지 않음.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for path, out in results:
        shutil.copy2(path, "%s.bak2-%s" % (path, stamp))
        write(path, out)
        print("[OK] %s" % path)

    print("\n백업 접미사: .bak2-%s" % stamp)
    print("재빌드: cmake --build %s/build -j 8" % ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
