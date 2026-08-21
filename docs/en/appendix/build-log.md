**English** / [한국어](../../ko/appendix/build-log.md)

# Appendix - Build Log

> **The actual procedure for standing up a heterogeneous GPU server from bare metal.** Where the main documents say "why do it this way," this one says "what was typed, in what order."
> The commands are tailored to this project's environment, so **do not copy them verbatim - substitute your own paths.**

Related documents: [Common Methodology](../00-method.md) / [No.1 Role Assignment](../01-role-assignment.md) / [Pitfalls](../99-pitfalls.md) / [Design Record](design-record.md)

---

## Background - How We Got Here

This project's final configuration was not designed from the start; it is **the result of three successive changes.** What follows is a chronology, trial and error included.

### How It Started

Commercial LLMs come with restrictions of various kinds. Three were the problem here: **running out of tokens on a flat-rate plan**, **content restrictions on creative generation**, and **certain categories of code analysis being refused**. Building locally was the only option; that is what started the project, and Qwen 27B was chosen as the target.

Running that model **unquantized (BF16)** requires 54GB for the weights alone, and 65GB or more of VRAM including the KV cache.

The first candidate was an **A100 80GB**. Renting one in the cloud confirmed **it ran**, but the cost of building locally overshot the budget badly, so it was dropped.

> At the time (March 2026) an SXM4 module converted to PCIe cost about **8,000 USD**. **Every other price in this repository is as of August 2026, and the same seller's same product was 11,000 USD by August.** That means the market moved sharply in half a year, so a used-market build should assume price swings of this magnitude.
>
> The cloud test confirmed **only that the unquantized model ran**; no performance was measured. It cannot be used for comparison -> [Appendix - What Was Not Measured](../98-benchmark.md#11-what-was-not-measured)

### First Configuration - NVLink V100 32GB x2

The first physical GPUs purchased were **two SXM2 V100 32GB cards.** Two cards sat on a reverse-engineered NVLink board from China connected at **NVLink 300GB/s**, and that GPU module attached to a single x16 slot on the main system through an **external PCIe switch card.** The connecting cable was **SFF-8654**.

> The switch card appears to be from the PLX/PEX family but **its exact model was never confirmed.** Confirm and fill it in if you plan to reproduce this.

```
CPU      Ryzen 4350G
M/B      ASRock A320
RAM      DDR4 16GB UDIMM x2
GPU      NVLink board + SXM2 V100 32GB x2
PSU      800W x2
CASE     ANiX m6509 mini
```

The GPU module (board + PSU + V100 x2 + PLX8749 + cables) cost about 2,100 USD, and about 2,500 USD including the main system.

**This configuration delivered everything it was meant to.** Qwen 27B fit on one 32GB card at Q6, and FLUX D ran on the other. **It was expected to serve for years.**

There were requests for improvement.

- Image generation was slow - **261.7 seconds for eight images** at 768x768 / 15 steps (32.7s per image)
- **The NVLink was never actually used** [Notes and Caveats 1)](#notes-and-caveats)

**The direct trigger for replacement came from elsewhere, though.** An unexplained motherboard failure (the audio chipset went completely unresponsive), and **a buyer appearing who wanted to purchase just the GPU module secondhand.** The two together pushed it into a full redesign.

### Rebuild Requirements

Three conditions were set for the redesign.

```
1. Models to run   Qwen 3.6 27B Q6 x1, FLUX D x2
2. GPUs to use     V100 32GB x1, V100 16GB x2
3. Slots           three or more physical PCIe x16 slots
```

**The model was decided first, then the GPUs to hold it, then the motherboard and case and power supply to hold those, in that order.** Not inverting that order is this project's basic policy -> [Workload Analysis](../00-method.md)

**A 16GB card could appear in condition 2 only because one conclusion was already in.** In the first system FLUX used an entire 32GB card and there was no reason to separate the encoder. During the redesign, **confirming first that the model could be split-loaded** is what made a 16GB card admissible as a worker candidate -> [No.2 Encoder Separation](../02-encoder-separation.md)

### Second Configuration - Passed the Paper Test, Failed in the Flesh

The cheapest board with three or more x16 slots was chosen, and a CPU and RAM were matched to it.

```
CPU      Xeon E5-2680 v4
M/B      GIGABYTE GA-X99-UD4P
RAM      DDR4 16GB ECC x2
PSU      2000W
CASE     Micronics WIZMAX Woodrian Max
```

**Every item passed the paper test.** The problem showed up in the flesh.

The board's F22 firmware **had no Above 4G Decoding option at all.** BIOS modding enabled it, and **the mod actually worked.** With a P104 installed alone, Ubuntu showed **the BAR allocated at `0x100000000` - the 64-bit address region just above the 4GiB boundary.** With 4G Decoding off, BARs are confined to 32-bit space below 4GiB, so **this is direct evidence the mod took effect.**

> The allocated address differs per platform. This value was observed on X99; EPYC platforms sometimes use much higher regions. **Whether it lands in 64-bit space, exceeding eight digits**, is the test - not any specific value.

**And yet no configuration containing a V100 passed POST.**

| Configuration (all with 4G Decoding enabled) | Result |
|---|---|
| P104 alone | **works** |
| V100 32GB x1 + 16GB x2 | POST failure |
| V100 16GB x2 | POST failure |
| V100 32GB alone | **POST failure** |

> That is the order they were tested in. **A single V100 16GB was never tested.**

**4G Decoding was not the cause**, since the symptom persisted after modding fixed it and a P104 verified it. **The cause was never determined.** After about four hours on per-part testing and solution design, the resources still required were weighed against what they would buy, and judging the cost not worth the benefit, **the CPU and motherboard were replaced.**

At this step **the X79 and X99 platforms were dropped from consideration entirely.** The main reason is those platforms' fundamental shortage of PCIe lanes -> [No.1 7-2](../01-role-assignment.md#7-2-splitting-a-higher-tier-gpu-into-lower-tier-gpus-carries-a-pcie-slot-cost)

> **A paper test does not substitute for a physical test.** Slot count, lane count, and firmware version can all pass on the spec sheet and still stop at POST, and the cause may not be anything listed on that sheet. **Finding a case where someone actually built the same combination is the only reliable verification.**

### Third Configuration - Final

Replacing only the CPU and motherboard produced the final configuration.

```
CPU      AMD EPYC 7232P                  128 PCIe lanes
M/B      ASRock Rack EPYCD8-2T
RAM      DDR4 16GB ECC x2
PSU      2000W
CASE     Micronics WIZMAX Woodrian Max
OS       Ubuntu 24.04.1
Driver   580 branch                      <- the last branch covering Maxwell/Pascal/Volta
```

After installing the GPUs, **the 250GB NVMe used since the first configuration was transplanted** as model storage, and the OS went on a separate 480GB SATA SSD.

In this project **the first configuration's NVMe could be moved without damage.** Not having to re-download the model files and source trees saved a great deal of time. [Notes and Caveats 2)](#notes-and-caveats)

At the replacement-design stage the same budget had to buy cards again, and out of that came the principle of "allocate each task only what it needs" -> [No.1 Role Assignment](../01-role-assignment.md)

### First vs Third - What Changed

| Item | First (V100 32GB x2 NVLink) | Third (32GB x1 + 16GB x2 + P104) |
|---|---|---|
| **System build cost** | ~2,500 USD | **~2,000 USD** |
| LLM generation speed | 31.5 t/s | **33.5 t/s** |
| MTP acceptance rate | 41% | **45.5%** |
| 8 images (768x768 / 15 steps) | 261.7s | **134.0s** |

> **No measurement record survives for the first configuration's image generation time.** The value above was measured on the third system under the same conditions (a V100 32GB carrying everything, 768x768 / 15 steps / PuLID enabled) -> [No.2 Section 8, condition 1](../02-encoder-separation.md#8-results). Image generation is work that finishes on the GPU, so it stays within margin across platforms, but **strictly speaking it was not measured on the first system.**

**Card count went up while GPU tier went down.** With the NVLink custom board gone and one V100 32GB replaced by two 16GB cards, **build cost fell as well.** That all three metrics improved anyway is what this repository is about.

---

## 0. Why the Order Matters

```
0. Pin the network interface name to the MAC address   <- before installing any card
1. Finish the OS update first
2. Confirm driver detection
3. Install the CUDA toolkit
4. Only then build anything
```

**Do 3 and 4 first and you will end up rebuilding everything when the kernel and driver drift apart later.**

### Skip Step 0 and You Cannot Even Reach the Server

Linux network interface names (`enp2s0`) are **derived from the PCI slot position.** Add a card or move a slot and the name changes, and every network setting pinned to that name breaks with it.

**On a headless server this is fatal.** You installed a GPU, SSH stops working, and with no display you cannot see why. This project hit it for real.

Pin the name to the MAC address.

```
/etc/systemd/network/10-nicname.link      <- [Match] MACAddress= / [Link] Name=
```

Then use that pinned name in netplan and elsewhere.

**It is exactly the same problem as GPU UUIDs.** Position-based identifiers are all vulnerable to hardware changes -> [Pitfalls](../99-pitfalls.md#position-based-identifiers-are-all-the-same-trap)

> References
> - [Predictable Network Interface Names - systemd](https://systemd.io/PREDICTABLE_INTERFACE_NAMES/) - the naming rules
> - [Ethernet drops after replacing a PCI-E device - symptoms and recovery](https://pyys.cafe24.com/wp/?p=729) - detailed diagnosis and recovery

### Confirm Hardware Detection

```
nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version --format=csv
```

**If the name reads `Unknown` or the card is missing from the list, that driver branch has dropped the architecture.** Failing here means there is no reason to continue.

**Record the UUIDs at this point.** Adding cards shifts the indices, so every setting from here on uses UUIDs -> [No.4 Section 1](../04-orchestration.md#1-always-specify-gpus-by-uuid)

Check PCIe link width and generation at the same time. Mining cards and cards on risers are frequently stuck on narrow lanes.

```
nvidia-smi --query-gpu=index,name,pcie.link.width.max,pcie.link.gen.max --format=csv
```

---

## 1. CUDA Toolkit

> **CUDA 13.0 removed Volta (sm_70) support and cannot be used with a V100.** 12.x is required. 12.8 was chosen because it is the line officially supporting Ubuntu 24.04 + GCC 13.3 -> [No.1 6-1](../01-role-assignment.md#6-1-the-support-window-decides-whether-a-configuration-is-possible-at-all)

### 1-1. Register the Repository

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb
```

### 1-2. Block Driver Packages - Mandatory

The NVIDIA repository supplies drivers too. Leave it alone and `apt upgrade` blocks entirely on a conflict between `libnvidia-gl-580` and `libnvidia-egl-gbm1`.

```
printf 'Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* libxnvctrl*\nPin: origin developer.download.nvidia.com\nPin-Priority: -1\n' > /etc/apt/preferences.d/nvidia-no-driver
```

**Take only the CUDA toolkit and use the distribution's driver.**

### 1-3. Update

```
apt update && apt upgrade -y
```

> If the driver is registered with DKMS (`dkms status`) it rebuilds automatically when the kernel moves. Removing and reinstalling the driver is unnecessary.

### 1-4. Install the Toolkit

**It is `cuda-toolkit-12-8`, not `cuda` or `cuda-12-8`.** Those two replace the driver as well.

```
apt install -y cuda-toolkit-12-8
```

### 1-5. Environment Variables and Verification

```
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```

```
nvcc -V && nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

### 1-6. Build Tools

```
apt install -y build-essential cmake git ccache libcurl4-openssl-dev pkg-config
```

---

## 2. Obtaining Sources and Models

### 2-1. Fetch Sources at a Pinned Commit

**Do not run `git pull`.** Use the verified commit as-is. Option names may have changed or GGUF compatibility may have broken.

```
git clone https://github.com/ggml-org/llama.cpp /root/llama.cpp && cd /root/llama.cpp && git checkout 1593d5684
```

```
git clone https://github.com/leejet/stable-diffusion.cpp /root/stable-diffusion.cpp && cd /root/stable-diffusion.cpp && git submodule update --init --recursive && git checkout f440ad9c
```

The commits this project pinned are llama.cpp **`1593d5684`** (b9602, 2026-06-12) and stable-diffusion.cpp **`f440ad9c`** (2026-06-23). The anchors in [`patches/`](../../../patches/) are written against the latter.

Verify like this.

```
cd /root/stable-diffusion.cpp && git log -1 --format='%H %ad %s'
```

**Clone with `--depth 1` and `git pull` may answer "Already up to date" while sitting hundreds of commits behind.** If you are going to check out a specific commit, do not use a shallow clone.

### 2-2. Downloading the Models

```
mkdir -p /root/models/flux-sd-cpp /root/models/pulid /root/loras
```

**LLM** - what this project used is a Qwen 3.6 27B derivative quantized to Q6_K with MTP weights included, in GGUF.

| File | Size | Source |
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

> **If you use a different model, check that the GGUF includes MTP weights.** Without MTP, speculative decoding does not work and this document's tokens-per-second figures will not reproduce.

**Image generation** - what this project used is FLUX.1 [dev]. The diffusion model itself, the VAE, two text encoders, and the PuLID weights are needed.

| File | Size | Source |
|---|---|---|
| `flux1-dev-Q8_0.gguf` | 12.7 GB | [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf) |
| `ae.safetensors` | 335 MB | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `clip_l.safetensors` | 246 MB | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |
| `t5xxl_fp16.safetensors` | 9.79 GB | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |
| `pulid_flux_v0.9.1.safetensors` | 1.14 GB | [guozinan/PuLID](https://huggingface.co/guozinan/PuLID) |

> **Those are file sizes; the 19.1GB and 13.6GB in the other documents are measured VRAM at load time.** The Q8_0 weights this project measured with differ slightly in file size from the distributions above. With the same quantization the verdict (it does not fit whole on a 16GB card) does not change, but **the decimal places may differ by distribution.**

> **Only `ae.safetensors` is gated.** `black-forest-labs/FLUX.1-dev` requires signing in with an HF account and accepting the non-commercial license before the file can be downloaded. After accepting on the web, register a token with `hf auth login` for the CLI download to go through. The other four are not gated.

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

T5-XXL is used by **converting the downloaded fp16 to q8_0 yourself.** That conversion is what let this project put the encoder on an 8GB card, and the quality comparison is recorded in [Design Record 0-2](design-record.md#0-2-quantization-ab---on-the-current-stack-with-no-card).

```
/root/stable-diffusion.cpp/build/bin/sd-cli -M convert -m /root/models/flux-sd-cpp/t5xxl_fp16.safetensors -o /root/models/flux-sd-cpp/t5xxl-q8_0.gguf --type q8_0
```

**This conversion alone runs after the build in [4-1](#4-1-build---include-every-target-architecture)**, because `sd-cli` does not exist yet. Once converted, the layout is as follows.

```
/root/models/flux1-dev-Q8_0.gguf                   diffusion model (12.7 GB)
/root/models/flux-sd-cpp/ae.safetensors            VAE (335 MB)
/root/models/flux-sd-cpp/clip_l.safetensors        CLIP-L (246 MB)
/root/models/flux-sd-cpp/t5xxl-q8_0.gguf           T5-XXL q8_0 (5.2 GB)
/root/models/pulid/pulid_flux_v0.9.1.safetensors   PuLID weights (1.14 GB)
```

**`/root/loras` must be empty.** sd-server recursively scans this directory on every request, so using the default (`.`) or pointing it at a large directory will crash or slow it down.

## 3. LLM Inference Engine

### 3-1. Build

The V100 is Compute Capability 7.0, so the architecture is pinned to 70. Skipping unneeded architectures cuts build time substantially.

```
cd /root/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
```

```
cmake --build build --config Release -j 8
```

> **The prebuilt frontend assets were dead with a 404.** If you need the WebUI, install Node and build them yourself. Distribution hosts disappearing is common enough that **keeping a build path available is the safer choice.**

### 3-2. Verification

```
/root/llama.cpp/build/bin/llama-server --list-devices
```

**If no CUDA devices are listed, you built a CPU-only binary.** Catching it here saves you from wandering around "why is this so slow" later.

### 3-3. Launch - Designating the Main GPU Is Not Enough

```
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-... llama-server -m <model> --mmproj <projector> -ngl 99 -c 80000 --split-mode none -mg 0
```

Specifying only the main GPU with `-mg 1` put **the vision projector on GPU0, taking 1.4GB**, and the image generation server hit OOM later. Making **only one card visible with `CUDA_VISIBLE_DEVICES` and then passing index 0** is the reliable way.

---

## 4. Image Generation Engine

### 4-1. Build - Include Every Target Architecture

```
cd /root/stable-diffusion.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70"
```

```
cmake --build build --config Release -j 8
```

**The quotes around `"61;70"` are mandatory.** Without them the shell splits the command in two.

**Omit an architecture and it will not run - or worse, the optimal kernel is silently disabled.** The first build here used only `70`, and after installing the P104 it would not run at all, so it had to be rebuilt -> [No.1 5-1](../01-role-assignment.md#5-1-three-paths-for-quantized-matrix-multiplication)

### 4-2. What Must Be Enabled

**`--vae-tiling`** - reduces the VAE decode compute buffer from 3,744MB to 416MB. Without it, **it dies with OOM at the final VAE stage after sampling has fully completed.** The cost is about 1.4 seconds.

**`--lora-model-dir <empty directory>`** - the default is `.`, so launching the server from `/` makes it recursively scan the entire root on every request and crash on a symbolic link.

```
{"error":"server_error","message":"filesystem error: status: Too many levels of symbolic links [./run/udev/watch/22]"}
```

### 4-3. What Must Not Be Used

**`--mmap`** - enabled to save RAM, it pushed model loading **from 8 seconds to 110.** Page-fault-granularity random reads are far slower than sequential reads. Unless several processes share the same weights, it is a loss.

**`txt_cfg` defaulting to 7.0** - FLUX is a distilled model and does not use CFG. Leave the default and it encodes an empty negative prompt and computes twice per step, making it **3.3x slower.** Specify `"guidance": {"txt_cfg": 1.0}` in the request.

### 4-4. Do Not Guess Argument Names

```
sd-server --help
```

It was `--listen-ip` / `--listen-port`, not `--host` / `--port`. **Pass a wrong argument and it prints usage and exits quietly**, so without reading the log you lose time on "why won't it start."

> The five items from 4-2 through 4-4 are collected as a symptom/response table in [Pitfalls - Inference Engines](../99-pitfalls.md#3-inference-engines). They remain here as part of the launch procedure.

### 4-5. What to Check in the Startup Log

```
grep -E "total params memory size|loading tensors completed" <log>
```

**The log's memory estimate differs from actual allocation.** Always confirm with `nvidia-smi`.

```
reported estimate : 17346 MB
actually allocated:  5181 MB
```

**The port opening does not mean the weights are up.** Weights are lazily loaded, so the model is read on the first request. Mistaking the `listening` log for startup completion throws off an entire benchmark.

**Every launch command in this document runs in the foreground.** Close the terminal and the service dies, and it will not come up on reboot. For continuous operation, register it with `systemd` or at minimum wrap it in `tmux` / `nohup`.

---

## 5. Face Consistency Extension (PuLID)

To keep a character's face consistent across images during generation, [PuLID](https://github.com/ToTheBeginning/PuLID) was added. Face embedding extraction runs **once per character.**

### 5-1. Installing Dependencies - Use the CPU Build of onnxruntime

Embedding extraction is on the Python side. What is needed is torch, InsightFace, and an ONNX runtime.

```
pip install torch insightface onnxruntime
```

**It is `onnxruntime`, not `onnxruntime-gpu`.** The GPU build does not work on a V100 (see [5-3](#5-3-extract-face-embeddings-on-the-cpu)), so installing the CPU build from the start is the better move. Extraction runs once per character, so handling it on the CPU costs essentially nothing at the operational stage.

```
python3 -c "import torch, insightface; print(insightface.__version__)"
```

> If you are migrating from an older system, reinstalling may be unnecessary. **When the OS and Python minor version match, the ABI lines up and torch bundles its own CUDA runtime, so simply pointing `PYTHONPATH=/old/path/dist-packages` at it works.** That is what this project actually did.

### 5-2. Face Recognition Model (antelopev2)

InsightFace looks in `~/.insightface/models/` and, finding nothing, **downloads about 400MB automatically.** With working networking there is nothing else to do.

In an environment where the automatic download is blocked, fetch and extract it manually.

```
mkdir -p /root/.insightface/models && cd /root/.insightface/models && curl -LO https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip && unzip antelopev2.zip
```

The `.onnx` files need to end up under `antelopev2/`. **It sometimes extracts one level deeper into `antelopev2/antelopev2/`, so check.**

```
ls /root/.insightface/models/antelopev2/
```

### 5-3. Extract Face Embeddings on the CPU

**onnxruntime-gpu 1.27 does not support Volta (sm_70).** Attempting it on the GPU fails like this.

```
CUBLAS failure 8: the function requires an architectural feature absent from the device
```

Setting `CUDA_VISIBLE_DEVICES=""` switches it to the CPU automatically. **Once per character, so the CPU is plenty.**

> This too is a form of the [support window](../01-role-assignment.md#6-1-the-support-window-decides-whether-a-configuration-is-possible-at-all) problem. It is not just inference engines - **supporting libraries drop architectures as well.** When planning a heterogeneous configuration, check the dependency libraries too.

---

## 6. Final State

| GPU | Role | VRAM used |
|---|---|---|
| V100 32GB | LLM server | 30.3GB / 32.7GB |
| V100 16GB x2 | Diffusion workers | 13.6GB / 16.1GB each |
| P104-100 8GB | Text encoder | 5.2GB / 8.0GB |

The image generation side was **reorganized after this build into a structure with the text encoder separated into a single copy.** That process is in the [Design Record](design-record.md) and the results in [Appendix - Benchmarks](../98-benchmark.md).

---

## 7. Where Time Was Actually Lost in This Build

| Pitfall | Symptom | Response |
|---|---|---|
| **A board that passed only the paper test** | specs matched but **POST failed**, cause never determined | find a real build case with the same combination |
| **Network interface names** | **no SSH** after swapping a card | pin to the MAC address |
| Installing CUDA 13 | no Volta support | pin to 12.8 |
| NVIDIA repository driver conflict | `apt upgrade` fails | apt pinning |
| rsync `--exclude 'build'` | deleted third-party directories -> build failure | `--exclude "/build"` |
| Omitted architecture | would not run at all on the P104 | include every target card |
| Missing quotes on the cmake argument | the command splits in two | `"61;70"` |
| Designating only the main GPU | a secondary module occupies another card -> OOM | isolate with `CUDA_VISIBLE_DEVICES` |
| Missing `--vae-tiling` | OOM at the VAE stage after sampling completes | add the option |
| `txt_cfg` default | 3.3x slower | specify 1.0 |
| `--mmap` | load 8s -> 110s | do not use |
| LoRA directory default `.` | recursive scan per request, crashes from root | specify an empty directory |
| Guessing argument names | prints usage and exits | check `--help` |
| onnxruntime-gpu 1.27 | CUBLAS failure 8 during face extraction | extract on the CPU |
| Prebuilt assets 404 | 404 when accessing the WebUI | install Node and build |

The classification and the response principles are collected in [Pitfalls](../99-pitfalls.md).

### Characters Lost on Terminal Paste

Throughout the work, the terminal repeatedly stripped `*`, `\`, and `__` from pasted text. **Commands containing wildcards or backslash escapes are safer written with explicit filenames or in an alternative form such as `grep -E`.** Not knowing the cause, "why isn't this command working" happened several times.

---

## Notes and Caveats

**1) NVLink Was Present and Never Used**

**Looking back at the operating design of the time, the two cards were split by role from the start.**

```
GPU0    llama-server (LLM)
GPU1    sd-server (FLUX image generation) + PuLID extraction
```

llama-server was launched with `--split-mode none -mg 0` as **single-GPU only**, and sd-server was pinned to the other card with `CUDA_VISIBLE_DEVICES=1`. **It was not VRAM pooled over NVLink but a different model on each card.**

**More precisely, the method of using tensor parallelism was simply not known at the time.** The command written while attempting a multi-GPU load looked like this,

```
llama-server -m <model> --mmproj <projector> -ngl 99 -c 131072 --host 0.0.0.0 --port 8080
```

**and it has no `--split-mode row`.** llama.cpp defaults to layer splitting, so **even with two cards visible it merely divides layers across them and never runs tensor parallel.** This was only confirmed much later.

> **Having NVLink hardware and actually using its bandwidth are separate things.** This project bought an NVLink board, ran it for a month, and **never once used the link.** Giving up the NVLink cost at rebuild time was fine as one direction for the solution, but **had tensor parallelism been experienced at that stage and proven decisive, a different direction might well have been chosen** -> [No.1 1-3](../01-role-assignment.md#1-3-tensor-parallelism-is-a-separate-problem)

**2)** Backups save time cost in almost every case. The storage cost and the tooling cost of backing up are almost always cheaper than the labor cost when something goes wrong.
