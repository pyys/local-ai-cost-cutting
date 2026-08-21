#!/usr/bin/env python3
"""
stable-diffusion.cpp 개조 — 조건 텐서(conditioning) 직렬화/주입.

목적:
  텍스트 인코딩 결과를 gguf 파일로 저장하고, 다른 프로세스가 그 파일을 읽어
  인코딩 없이 확산만 수행할 수 있게 한다. (인코더 1벌 + 워커 N개 구조의 기반)

추가되는 CLI 인자:
  --save-conditioning <path>   인코딩 후 조건 텐서를 gguf 로 저장
  --conditioning <path>        조건 텐서를 읽어 텍스트 인코딩을 건너뜀

검증 절차 (Phase 2-3 게이트):
  1) 정상 실행 + --save-conditioning cond.gguf   -> 이미지 A
  2) --t5xxl / --clip_l 없이 --conditioning cond.gguf -> 이미지 B
  A 와 B 가 픽셀 단위로 동일해야 한다.

사용법:
    python3 cond_serialize_patch.py --dry-run
    python3 cond_serialize_patch.py
    python3 cond_serialize_patch.py --revert
"""

import os
import re
import sys
import glob
import shutil
import datetime

# 소스 트리 위치. 환경변수 SD_CPP_ROOT 로 덮어쓸 수 있다.
ROOT = os.environ.get("SD_CPP_ROOT", "/root/stable-diffusion.cpp")

SD_H = os.path.join(ROOT, "include/stable-diffusion.h")
SD_CPP = os.path.join(ROOT, "src/stable-diffusion.cpp")
COMMON_H = os.path.join(ROOT, "examples/common/common.h")
COMMON_CPP = os.path.join(ROOT, "examples/common/common.cpp")

MARK = "sd_cond_io"  # 이미 적용됐는지 판별용


# ---------------------------------------------------------------- 삽입 내용

HELPERS = r'''
// ---------------------------------------------------------------------------
// sd_cond_io: 조건 텐서 직렬화 (텍스트 인코더 분리용)
//
// shape 벡터와 flat f32 데이터를 있는 그대로 왕복시킨다. 축의 의미를 해석하지
// 않으므로 sd::Tensor 와 ggml 의 축 규약이 어떻든 정확히 복원된다.
// FLUX(FluxCLIPEmbedder)는 c_crossattn 과 c_vector 두 필드만 채우므로 그 둘만
// 다룬다. 다른 conditioner 를 쓰는 모델에는 적용되지 않는다.
// ---------------------------------------------------------------------------

static bool sd_cond_save(const std::string& path, const SDCondition& cond) {
    struct Entry {
        const char* name;
        const sd::Tensor<float>* tensor;
    };
    const Entry entries[] = {
        {"c_crossattn", &cond.c_crossattn},
        {"c_vector", &cond.c_vector},
    };

    size_t mem = ggml_tensor_overhead() * 8 + 4096;
    for (const auto& e : entries) {
        mem += (size_t)e.tensor->numel() * sizeof(float) + 256;
    }

    struct ggml_init_params ip;
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;

    struct ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    struct gguf_context* gguf_ctx = gguf_init_empty();

    for (const auto& e : entries) {
        if (e.tensor->numel() == 0) {
            continue;
        }
        const std::vector<int64_t>& shape = e.tensor->shape();
        int n_dims                        = (int)shape.size();
        if (n_dims < 1) {
            n_dims = 1;
        }
        if (n_dims > GGML_MAX_DIMS) {
            n_dims = GGML_MAX_DIMS;
        }
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; d++) {
            ne[d] = shape[d];
        }
        struct ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(t, e.name);
        memcpy(t->data, e.tensor->data(), (size_t)e.tensor->numel() * sizeof(float));
        gguf_add_tensor(gguf_ctx, t);

        // ggml_n_dims() 는 뒤쪽 1 을 잘라내므로 원래 차원 수를 따로 기록한다.
        std::string ndims_key = std::string("sd.cond.") + e.name + ".ndims";
        gguf_set_val_i32(gguf_ctx, ndims_key.c_str(), n_dims);
    }
    gguf_set_val_i32(gguf_ctx, "sd.cond.version", 1);

    gguf_write_to_file(gguf_ctx, path.c_str(), false);

    gguf_free(gguf_ctx);
    ggml_free(ctx);
    return true;
}

static bool sd_cond_load(const std::string& path, SDCondition& cond) {
    struct ggml_context* ctx = nullptr;

    struct gguf_init_params ip;
    ip.no_alloc = false;
    ip.ctx      = &ctx;

    struct gguf_context* gguf_ctx = gguf_init_from_file(path.c_str(), ip);
    if (gguf_ctx == nullptr) {
        return false;
    }

    auto read_one = [&](const char* name, sd::Tensor<float>& dst) {
        struct ggml_tensor* t = ggml_get_tensor(ctx, name);
        if (t == nullptr) {
            return;
        }
        int n_dims            = ggml_n_dims(t);
        std::string ndims_key = std::string("sd.cond.") + name + ".ndims";
        int key_idx           = gguf_find_key(gguf_ctx, ndims_key.c_str());
        if (key_idx >= 0) {
            n_dims = gguf_get_val_i32(gguf_ctx, key_idx);
        }
        if (n_dims < 1) {
            n_dims = 1;
        }
        if (n_dims > GGML_MAX_DIMS) {
            n_dims = GGML_MAX_DIMS;
        }
        std::vector<int64_t> shape;
        shape.reserve((size_t)n_dims);
        for (int d = 0; d < n_dims; d++) {
            shape.push_back(t->ne[d]);
        }
        dst = sd::Tensor<float>(shape);
        memcpy(dst.data(), t->data, (size_t)dst.numel() * sizeof(float));
    };

    read_one("c_crossattn", cond.c_crossattn);
    read_one("c_vector", cond.c_vector);

    gguf_free(gguf_ctx);
    ggml_free(ctx);

    return !cond.c_crossattn.empty();
}

'''

BRANCH = r'''    SDCondition cond;
    const char* sd_cond_in = sd_img_gen_params->conditioning_path;
    if (sd_cond_in != nullptr && sd_cond_in[0] != '\0') {
        if (!sd_cond_load(sd_cond_in, cond)) {
            LOG_ERROR("failed to load conditioning from '%s'", sd_cond_in);
            return std::nullopt;
        }
        LOG_INFO("conditioning loaded from '%s', text encoding skipped", sd_cond_in);
    } else {
        cond = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                   condition_params);
        const char* sd_cond_out = sd_img_gen_params->save_conditioning_path;
        if (sd_cond_out != nullptr && sd_cond_out[0] != '\0') {
            if (sd_cond_save(sd_cond_out, cond)) {
                LOG_INFO("conditioning saved to '%s'", sd_cond_out);
            } else {
                LOG_ERROR("failed to save conditioning to '%s'", sd_cond_out);
            }
        }
    }
'''

CLI_OPTIONS = '''        {"",
         "--conditioning",
         "path to a precomputed conditioning gguf. skips text encoding entirely",
         0,
         &conditioning_path},
        {"",
         "--save-conditioning",
         "save the computed conditioning to this gguf path",
         0,
         &save_conditioning_path},
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
    return True


# ---------------------------------------------------------------- 각 파일 패치

def patch_sd_h(text):
    anchor = "    sd_hires_params_t hires;\n} sd_img_gen_params_t;"
    once(text, anchor, "stable-diffusion.h")
    new = ("    sd_hires_params_t hires;\n"
           "    const char* conditioning_path;       // sd_cond_io: 있으면 텍스트 인코딩을 건너뛴다\n"
           "    const char* save_conditioning_path;  // sd_cond_io: 계산한 조건을 이 경로에 저장\n"
           "} sd_img_gen_params_t;")
    return text.replace(anchor, new)


def patch_sd_cpp(text):
    # 1) gguf 헤더
    if "gguf.h" not in text:
        m = re.search(r'^#include [^\n]*\n(?![\s\S]*?^#include )', text, re.M)
        if m is None:
            raise RuntimeError("stable-diffusion.cpp: #include 블록을 찾지 못함")
        text = text[:m.end()] + '#include "gguf.h"\n' + text[m.end():]

    # 2) 헬퍼 삽입 — prepare_image_generation_embeds 정의 직전
    fn_anchor = "static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds("
    once(text, fn_anchor, "stable-diffusion.cpp / 함수 정의")
    text = text.replace(fn_anchor, HELPERS.lstrip("\n") + fn_anchor)

    # 3) cond 계산부를 분기로 교체 (공백에 의존하지 않도록 줄 단위로 처리)
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if "auto cond" in line and "get_learned_condition" in line:
            start = i
            break
    if start is None:
        raise RuntimeError("stable-diffusion.cpp: 'auto cond ... get_learned_condition' 줄을 찾지 못함")

    end = None
    for j in range(start, min(start + 6, len(lines))):
        if "condition_params);" in lines[j]:
            end = j
            break
    if end is None:
        raise RuntimeError("stable-diffusion.cpp: cond 계산의 닫는 줄을 찾지 못함")

    lines[start:end + 1] = BRANCH.rstrip("\n").split("\n")
    return "\n".join(lines)


def patch_common_h(text):
    anchor = "    std::string negative_prompt;"
    once(text, anchor, "common.h")
    new = (anchor +
           "\n    std::string conditioning_path;       // sd_cond_io"
           "\n    std::string save_conditioning_path;  // sd_cond_io")
    return text.replace(anchor, new)


def patch_common_cpp(text):
    # 1) CLI 옵션 — SDGenerationParams::get_options() 의 -p/--prompt 항목 뒤
    anchor = ('        {"-p",\n'
              '         "--prompt",\n'
              '         "the prompt to render",\n'
              '         0,\n'
              '         &prompt},\n')
    once(text, anchor, "common.cpp / string_options")
    text = text.replace(anchor, anchor + CLI_OPTIONS)

    # 2) 구조체 대입 — init 직후 (init 이 0 으로 덮으므로 뒤에 와야 한다)
    anchor2 = "    sd_img_gen_params_init(&params);"
    once(text, anchor2, "common.cpp / to_sd_img_gen_params_t")
    new2 = (anchor2 +
            "\n    params.conditioning_path      = conditioning_path.c_str();"
            "\n    params.save_conditioning_path = save_conditioning_path.c_str();")
    return text.replace(anchor2, new2)


TARGETS = [
    (SD_H, patch_sd_h),
    (SD_CPP, patch_sd_cpp),
    (COMMON_H, patch_common_h),
    (COMMON_CPP, patch_common_cpp),
]


# ---------------------------------------------------------------- main

def revert():
    n = 0
    for path, _ in TARGETS:
        backups = sorted(glob.glob(path + ".bak-*"))
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

    for path, _ in TARGETS:
        if not os.path.exists(path):
            print("[FAIL] 파일 없음: %s" % path)
            return 1
        if MARK in read(path):
            print("[FAIL] 이미 적용된 것으로 보임 (%s 발견): %s" % (MARK, path))
            print("       되돌리려면: python3 cond_serialize_patch.py --revert")
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
        shutil.copy2(path, "%s.bak-%s" % (path, stamp))
        write(path, out)
        print("[OK] %s" % path)

    print("\n백업 접미사: .bak-%s" % stamp)
    print("재빌드: cmake --build %s/build -j 8" % ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
