**English** / [한국어](../ko/01-role-assignment.md)

# No.1 Heterogeneous GPU Role Assignment and Cost

A record of mixing GPUs of different architectures and memory sizes, giving each task **only as much as it needs**, and cutting the build cost that way.
This document stands on its own. The calculations behind the assignments are in [Workload Analysis](00-method.md).
The material was collected from a real build: EPYC 7232P / V100 32GBx1 + V100 16GBx2 + P104-100 8GB / Ubuntu 24.04.

Related documents: [Common Methodology](00-method.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.3 Pipeline Throughput](03-pipeline-throughput.md) / [No.4 Orchestration](04-orchestration.md) / [Appendix - Benchmarks](98-benchmark.md) / [Pitfalls](99-pitfalls.md)

---

## Contents

0. [Why Heterogeneous - The Trigger Is Budget](#0-why-heterogeneous---the-trigger-is-budget)
1. [When a Heterogeneous Configuration Works](#1-when-a-heterogeneous-configuration-works)
2. [Priority Order for Choosing GPUs](#2-priority-order-for-choosing-gpus)
3. [Let the Nature of the Work Choose the Card](#3-let-the-nature-of-the-work-choose-the-card)
4. [Identify the Card's Compute Units First](#4-identify-the-cards-compute-units-first)
5. [Where Performance Actually Diverges Across Heterogeneous GPUs](#5-where-performance-actually-diverges-across-heterogeneous-gpus)
6. [Software Stack Constraints](#6-software-stack-constraints---the-biggest-trap)
7. [Cost Structure and Hardware Constraints](#7-cost-structure-and-hardware-constraints)
8. [When a Heterogeneous GPU Configuration Is the Wrong Fit](#8-when-a-heterogeneous-gpu-configuration-is-the-wrong-fit)

---

## 0. Why Heterogeneous - The Trigger Is Budget

**A heterogeneous configuration comes out of a budget constraint, not a preference.** In this project, the VRAM size able to hold every required model on one card was in the 80GB class, and an A100 80GB put the whole system at about 12,000 USD. The final configuration here, which delivers comparable output, came to about 2,000 USD for the entire system. **Purchase cost is roughly one sixth.**

Something grows in exchange. The skill level required to assemble the system rises, power consumption becomes 3.6x, and heat and noise rise with it. On top of that, the whole stack gets pinned to a specific CUDA version. Per-item figures and the arithmetic are in [Section 7](#7-cost-structure-and-hardware-constraints).

So if budget is not a constraint, much of this document is moot -> [Section 8](#8-when-a-heterogeneous-gpu-configuration-is-the-wrong-fit)

**That said, the range of work where this method saves money is limited.** Which work qualifies is the subject of Section 1.

---

## 1. When a Heterogeneous Configuration Works

There are broadly two ways a heterogeneous configuration lowers initial system cost, and **they are different enough in character that they must be judged separately.**

### 1-1. Type A - One Model Can Be Split into Modules and Distributed

A single pipeline divides into several modules, each with different resource demands, and each module goes on a different card. In this project image generation (the Stable Diffusion family) was the case in point, and workloads like it share four properties.

| Condition | Description | In this project |
|---|---|---|
| **Module separation is possible** | the model divides into independent stages | text encoder / diffusion / VAE |
| **Separation costs little performance** | inter-module traffic is small enough that transfer is not the bottleneck | conditioning tensor 4.2MB per request |
| **Modules are used at different frequencies** | some run once, some run N times | encoding once -> diffusion N times |
| **Modules demand different resources** | compute, memory, and bandwidth needs differ | encoder is light on compute, diffusion is heavy |

Without the first, **the technique cannot be applied at all**; without the second, separation yields little or nothing. The third amplifies the gain and the fourth maximizes it. In this project, **separating the image model's text encoder from its diffusion module** satisfied all four -> [No.2 Encoder Separation](02-encoder-separation.md)

**The first effect visible once condition 1 holds is that the maximum VRAM any single card must carry comes down.** Whether the model loads at all is decided by condition 1, and at the initial costing stage that is the largest single change. Here, a Q8 diffusion model loaded whole needed 19.1GB; after separation it became a 13.6GB worker plus a 5.2GB encoder, and **each piece fits on a cheaper card.** That is why a 16GB + 8GB pairing replaced a single 32GB card.

Conversely, **an LLM inference model fails conditions 2, 3, and 4, so there was no reason to apply it.** An LLM can be split across cards, but inter-layer traffic is large, every layer is used equally on every token, and the resource demands are identical throughout.

### 1-2. Type B - Several Models Run Concurrently (Especially with Different Resource Profiles)

This is why cards of the same architecture but different VRAM sizes (V100 32GB x1 + 16GB x2) were used together. The starting point is that the two models have different requirements.

| Model | VRAM needed | Basis |
|---|---|---|
| Qwen 3.6 27B Q6 (LLM) | **~30GB** | 21GB weights + KV cache and overhead |
| FLUX D Q8 (image) - loaded whole | **19.1GB** | DiT 12.1 + T5 4.8 + CLIP-L/VAE/buffers |
| FLUX D Q8 - **worker after encoder separation** | **13.6GB** | DiT 12.1 + VAE + buffers |

Even after quantization the LLM's weights alone take 21GB, and with the KV cache and overhead the total is about 30GB. **There is no option other than a 32GB card.**

FLUX, by contrast, has four constituent modules, five components once buffers are counted. Applying [No.2 Encoder Separation](02-encoder-separation.md) shrinks the worker to 13.6GB, which **makes it fit a 16GB card - a large contribution to the budget, so this had high research priority.** Introducing the P104 came later; it is closer to a fix for the performance loss that split loading introduced, arrived at while refining the design.

That leaves two questions.

1. Is the loss from choosing the cheaper 16GB card justifiable?
2. Is there a way to avoid or recover that loss?

In the initial design the diffusion model's encoder was placed in CPU RAM. That choice **saved over 450 USD** [(2)](#notes-and-caveats), and the price - increased total time for eight images - was **estimated at roughly 10% or less** against loading every module on a V100 32GB [(1)](#notes-and-caveats). **The saving justified the loss.**

At this point the V100 32GB + V100 16GB heterogeneous configuration was settled. The alternative of putting two workers on a single 32GB card was later rejected by measurement [(3)](#notes-and-caveats).

The encoder's arithmetic intensity was then estimated, and while searching for a way to serve that intensity, the P104's price and ridge point were checked; the conclusion was that **introducing the P104 would improve performance** -> [Common Methodology Section 3](00-method.md#3-generalizing---the-roofline-model). When actually deployed and measured, the improvement exceeded the prediction -> [Appendix 7-3](98-benchmark.md#7-3-why-25x-faster-than-cpu---arithmetic-intensity-and-ridge-point)

In the same vein, putting a draft model for speculative decoding on a separate card to raise LLM tokens per second is also Type B. The main model and the draft model differ in size and in resource demand, so different cards may be optimal for each.

### 1-3. Tensor Parallelism Is a Separate Problem

Tensor parallelism - binding several GPUs' VRAM into what behaves as one - is a useful technique, but **its premises differ from what this document covers, so it is excluded from the discussion.** Tensor parallelism splits **the layers themselves** rather than modules, and so fails condition 2.

Only the reason the premises differ is worth noting here. Tensor parallelism splits layers so the cards exchange results at every step, making it **transfer-bound**; consequently (a) overall speed is set by the slowest card and (b) an NVLink-class interconnect is required, which changes the cost structure. **The more heterogeneous the cards, the worse it gets** - the opposite direction from this document's approach of giving each card an independent role.

**Before the rebuild, this project ran a two-card V100 32GB NVLink configuration for a month.** At rebuild time that cost was given up.

> **That history cannot be used as evidence about tensor parallelism, however.** The two cards were split by role - LLM on GPU0, image generation on GPU1 - and the command lacked `--split-mode row`, so it **only ever did layer splitting.** In other words **the NVLink link was never actually used.** Had tensor parallelism been tried at that stage and proven decisive, a different direction might well have been chosen -> [Appendix - Build Log, Notes and Caveats 1)](appendix/build-log.md#notes-and-caveats)

---

## 2. Priority Order for Choosing GPUs

Once the [workload analysis](00-method.md) exists, choose the cards. **In descending priority.**

| Rank | Axis | Reason |
|---|---|---|
| **0** | **Budget** | if you cannot buy it, reconsider building locally at all. Moving to a commercial cloud is a legitimate option |
| 1 | **Memory capacity** | if the model does not fit, the entire design premise changes. Tensor parallelism has completely different conditions from a single-card setup (1-3) |
| 2 | **Compiler and runtime support** | if CUDA and the driver have dropped that architecture, you cannot use it (Section 6) |
| 3 | Compute throughput | considered only after the three above pass |

The most common failure is starting from rank 3 and getting stopped by an earlier rank.

---

## 3. Let the Nature of the Work Choose the Card

Divide the work along two axes.

| | **High VRAM demand** | **Low VRAM demand** |
|---|---|---|
| **Compute-heavy** | video generation | **diffusion steps** |
| **Compute-light** | **LLM inference (decoding)** | text encoding, VAE decode, upscaling |

Placing the LLM under "compute-light" looks counterintuitive, but **compute is 1% of the decoding stage.** The heavy part is memory reads ([calculation](00-method.md#2-bottleneck-classification-can-be-settled-by-calculation)).

Broken out per task:

| Task | VRAM | Compute | Bandwidth | Suitable card |
|---|---|---|---|---|
| LLM decoding | large (21GB + KV) | low | **very high** | wide bandwidth, large VRAM |
| Diffusion step | medium (12GB) | **very high** | low | strong compute |
| Text encoding | small (5GB) | low | low | **the cheapest card** |

This project's assignment came out as follows.

| Card | Architecture | VRAM size | Actually allocated | Primary role | Reason |
|---|---|---|---|---|---|
| V100 32GB | Volta sm_70 | 32GB | 30.3GB | **LLM server** | memory size. It does not fit anywhere else |
| V100 16GB x2 | Volta sm_70 | 16GB | 13.6GB each | **Diffusion workers** | the stronger compute of V100 vs P104 |
| P104-100 | Pascal sm_61 | 8GB | 5.2GB | **Text encoder** | light on compute and light on transfer |

**Allocating each task only what it needs is the core of a heterogeneous configuration.** Comparing the resource demands of the text encoder and the diffusion module makes the difference obvious.

| | Text encoder (T5 Q8) | Diffusion module (DiT Q8) | Ratio |
|---|---|---|---|
| Compute time | 0.86s (P104, 512 tokens) | 26.9s (V100, one image) | 1 : 31 |
| Memory size | 5.2GB | 12.1GB | 1 : 2.3 |
| **Compute time per GB** | **0.17 s/GB** | **2.2 s/GB** | **1 : 13** |
| Memory bandwidth demand | low | low | - |

> These values come from different cards, so this is not an absolute performance comparison. **Read it as a contrast between resource profiles.**

**T5 is the kind of module that eats memory but barely computes.** Its compute time per GB is one thirteenth of diffusion's. Spending an expensive card's VRAM on a module like that is waste.

> **This is relative to diffusion, though; the model is still too heavy to assign to a CPU.** In absolute terms its arithmetic intensity of 498 makes it compute-bound on any card ([Common Methodology Section 3](00-method.md#3-generalizing---the-roofline-model)). What makes it work is that the P104's price per GB of memory is about 15% of the V100's, so putting a "needs memory, does not need compute" module there is the right call.

---

## 4. Identify the Card's Compute Units First

**Judge by "which compute units does it have," not by "old or new."** Generations differ like this.

| Generation | sm | FP16 | INT8 instructions | Tensor cores |
|---|---|---|---|---|
| Maxwell | 50 / 52 | no acceleration (handled as FP32) | none | none |
| **Pascal GP100** | **60** | **2x FP32** (packed) | **no DP4A** | none |
| **Pascal GP10x** | **61** | **1/64 FP32** | **DP4A / DP2A** | none |
| Volta | 70 | 2x FP32 | DP4A | **1st gen** (FP16 multiply -> FP32 accumulate) |
| Turing | 75 | 2x FP32 | DP4A | 2nd gen (+ INT8 / INT4) |
| Ampere | 80 / 86 | 2x FP32 | DP4A | 3rd gen (+ TF32, BF16, sparsity) |
| Ada | 89 | 2x FP32 | DP4A | 4th gen (+ FP8) |
| Hopper | 90 | 2x FP32 | DP4A | 4th gen + Transformer Engine |
| Blackwell | 100 / 120 | 2x FP32 | DP4A | 5th gen (+ FP4 / FP6) |

**Even within Pascal, GP100 (sm_60) and GP10x (sm_61) are opposites.** GP100 has fast FP16 and no DP4A; GP104 (including the P104-100) has FP16 crippled to 1/64 but does have DP4A. **"It is Pascal, so it must behave like this" is wrong.**

This is also where the reason the P104-100 cannot serve diffusion or the LLM comes from.

- **FP16 runs at 1/64 of FP32** -> inefficient for FP16-dominated workloads
- **No tensor cores**
- **8GB** does not hold a modern model

On the other hand, **INT8 DP4A runs at full speed** (about 4x FP32, roughly 22 TOPS).

### 4-1. There Are Two Directions of Attack

Besides finding work that suits the card, there is **the option of reshaping the work to suit the card.**

| Direction | Content | In this project |
|---|---|---|
| (a) assign work that suits the card | give it the operations it is good at | P104 -> text encoder |
| (b) **reshape the work to suit the card** | quantize into a format the card supports | T5 to **q8_0** -> becomes eligible for DP4A |

**Quantizing the diffusion model's T5 encoder from fp16 to q8_0 bought three things.**

1. **It fits on a card with less VRAM** - 9,084MB -> 4,826MB
2. **Memory bandwidth consumption halves** -> the memory-bound phase of the pipeline gets shorter
3. **The INT8 unit (DP4A) becomes usable** - the P104 takes that path deterministically -> [5-1](#5-1-three-paths-for-quantized-matrix-multiplication)

The third is the essence of direction (b). **What made a module that would not fit on an 8GB card fit was not the card selection but the change of data format.**

The size of the gain was settled by measurement.

| T5 encoding on a P104-100 (256 tokens) | Time |
|---|---|
| CPU, 16 threads | 8.30s |
| **quantized to q8_0** | **0.33s** |

**Quantization's first effect is fit.** T5-XXL fp16 is 9.79GB and does not go on an 8GB card. Only once q8_0 brought it to 5.2GB did the P104 become a candidate.

q8_0 also wins on speed. Measured on the actual encoding shape (GEMM), the P104 gives **4.34 TFLOPS for f16 against 9.42 TFLOPS for q8_0, a factor of 2.17** -> [Appendix 12](98-benchmark.md#12-sm-scaling-and-kernel-path-measurements)

> **Caution -** on paper the P104's FP16 arithmetic is 1/64 of FP32, but **that cap never triggers under ggml.** With no tensor cores, f16 tensors are converted to FP32 and computed there. The full account is in [5-2](#5-2-a-kernel-path-cannot-be-settled-by-timing).

Quality change from quantization was assessed by generating with the pre- and post-quantization models at **identical prompt and identical seed**.

| Metric | Value |
|---|---|
| Mean pixel difference | **0.72 / 255** |
| Maximum pixel difference | 149 / 255 |
| Semantic elements (color, composition, prop placement) | preserved |
| Where it diverged | fingers, face shape, and other inherently unstable regions |

The mean difference is below visual discrimination, and the places with a large maximum difference were concentrated in regions that change anyway from a single seed change.

**Always compare quantization A/B at the pixel level.** Byte-comparing PNGs always reports a difference because of metadata chunks. The command below does a pixel-level comparison.

```
python3 -c "from PIL import Image; a=Image.open('a.png').tobytes(); b=Image.open('b.png').tobytes(); print('IDENTICAL' if a==b else 'DIFFERENT')"
```

### 4-2. Do Not Judge a GPU by VRAM and Bandwidth Alone

Some cards have excellent memory specifications but **restricted compute units.** The mining-dedicated line (NVIDIA CMP) is the classic example.

The **CMP 170HX** (GA100-based) is one such card. Based on community reports for the converted version:

| Item | Value | Assessment |
|---|---|---|
| VRAM | 64GB | datacenter class |
| Memory bandwidth | ~1500 GB/s | faster than a V100 (900 GB/s) |
| **Tensor FP16 compute** | **~6% of a V100** | **crippled** |
| **Scalar FP16 compute** | **~90% of a V100** | **normal** |

> The figures above are **community-reported and were not verified in this project.** The 170HX's tensor throughput is understood to be limited to 1/32 of an A100 core, which produces the gap shown. The CMP line has poor official specification disclosure and circulates in memory-unlocked converted form, so **when considering a converted GPU, gather as much material with actual benchmark numbers as possible before deciding.**

| Workload | Suitability |
|---|---|
| Diffusion steps | X large models load, but compute demand is high so throughput is minimal |
| Paths that avoid tensor cores | ~ viable while scalar performance is intact |
| **Memory-bound work** (LLM decoding, KV cache residency, embedding search) | O bandwidth and capacity become the strength |

### 4-3. The Decision Order, Summarized

```
fix the budget
  +--> write the workload analysis (decompose phases -> classify bottleneck)
       +--> is VRAM capacity sufficient
            +--> is it inside the driver/CUDA support window
                 +--> compute throughput "in which format, on which kernel"
                      (FP32 / FP16 scalar / FP16 tensor / INT8, each separately)
                      +--> compare price per unit of throughput
                           +--> is there a module that can be separated
```

---

## 5. Where Performance Actually Diverges Across Heterogeneous GPUs

Same code, same model, yet cards differ by multiples. The cause is **which kernel path gets taken.** The explanation uses ggml, but the principle is the same in any framework.

### 5-1. Three Paths for Quantized Matrix Multiplication

| Path | Condition | Uses INT8 |
|---|---|---|
| `mul_mat_vec_q` | batch 1 | O |
| **MMQ** (`mul_mat_q`) | mat-mat, passes the gate | O |
| dequant + cuBLAS | the above failed | X |

The heart of the decision function (`ggml/src/ggml-cuda/mmq.cu`) is these two lines.

```c
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) return false;  // 610
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;  // 64
}
```

Two things follow.

**(a) If the compiled architecture is below the floor (610 = sm_6.1), MMQ is disabled.**
Use an sm_61 card without putting 61 in `CMAKE_CUDA_ARCHITECTURES` and it may still run, but it cannot take the INT8 path. **Build configuration determines performance.**

**(b) The presence of tensor cores inverts the decision.**

| Card | FP16 tensor cores | On 256/512-token mat-mat |
|---|---|---|
| P104 (sm_61) | none | **always MMQ -> DP4A INT8** |
| V100 (sm_70) | present | `ne11 < 64` fails -> dequant + cuBLAS FP16 |

This is the reverse of intuition. **The older card is the one that deterministically takes the INT8 path.** A card with tensor cores decides "cuBLAS is better for large batches" and skips INT8.

This analysis was the basis for deciding to use the P104 as the encoder.

> The first path (`mul_mat_vec_q`, batch 1) is what **LLM decoding** uses. That is, LLM decoding does not use tensor cores even on cards that have them. It is also [why compute came out as 1% of the total](00-method.md#2-bottleneck-classification-can-be-settled-by-calculation).

### 5-2. A Kernel Path Cannot Be Settled by Timing

**Concluding "DP4A worked" from measured time alone is dangerous.**

Encoding 256 tokens through T5-XXL (4.7B) is about **2,400 GFLOP**. Theoretical times per path on the P104:

| Path | Theoretical throughput | Theoretical time | Relation to the measured 0.33s |
|---|---|---|---|
| DP4A INT8 | 22 TOPS | 109ms | 33% efficiency - plausible |
| **FP32 fallback** | **6.1 TFLOPS** | **393ms** | **indistinguishable** |
| FP16 (crippled to 1/64) | 0.10 TFLOPS | 24s | clearly not this |

**The FP32 path produces a similar value.** FP32 never runs at 100% efficiency though, and at a realistic 60% it would be 655ms against a measured 330ms, so it is fair to say the path is **something faster than FP32** - and no more than that.

**The primary evidence for DP4A is therefore source analysis, not timing.** Reading the decision function and confirming the gate conditions is definitive; timing serves only as supporting material for checking that nothing contradicts that conclusion.

#### The Last Row of This Table Was Later Overturned by Measurement

The table above holds **values inferred from specifications.** Measuring kernel throughput directly with `test-backend-ops` afterward showed the last row was not true.

| P104, actual encoding shape (m=4096 n=512 k=14336) | Measured | Inferred from spec |
|---|---|---|
| f16 | **4.34 TFLOPS** | 0.095 TFLOPS |
| q8_0 | **9.42 TFLOPS** | 43% of 22 TOPS |

**The measured f16 figure is 45x the spec inference.** The cause is that ggml does not compute f16 tensors with FP16 arithmetic. Pascal has no tensor cores and therefore cannot take the cuBLAS FP16 path, so **f16 is converted to FP32 and processed there.** The cap never gets a chance to apply. The measured 4.34 TFLOPS is 71% of the P104's FP32 theoretical figure (6.1 TFLOPS), which shows the FP32 path plainly.

**This document itself fell into the trap this section warns about.** It multiplied the spec sheet's "FP16 at 1/64" straight through to get 24 seconds, but **that path was never selected in the first place.** Keeping the table as a record and appending the correction seemed better than deleting it, so it stays.

> **The lesson is unchanged - if anything it is stronger.** A kernel path cannot be settled from specifications any more than from timing. **You have to read the source or measure it directly.** The raw data is in [Appendix 12](98-benchmark.md#12-sm-scaling-and-kernel-path-measurements).

> **Generalizing:** which kernel a framework picks differs per card, and the decision logic is usually spelled out in the source. **If you are planning a heterogeneous configuration, reading that decision function yourself is the most reliable preliminary research there is.**
>
> The CMP 170HX case in 4-2 is the same problem wearing a different face. A card with crippled tensor cores is slow only on "kernels that use tensor cores," so **without knowing which kernel gets selected you cannot predict effective performance.**

### 5-3. Cheap, and It Only Has to Fit

**The card handling encoding had only two requirements - is there VRAM for the module, and is it cheap.**

Encoding is an operation that benefits from parallelism (Section 3), so **moving it from CPU to GPU is already a large gain by itself.** Measured, the P104 was 25x the CPU, and that gain does not depend on any particular compute path. Running the same shape in f16 still gives 4.34 TFLOPS, far ahead of the CPU's 290 GOPS.

**DP4A amplified the gain; it did not decide whether there was one.** q8_0 is 2.17x faster than f16, but even f16 would have been far faster than the CPU -> [Appendix 12](98-benchmark.md#12-sm-scaling-and-kernel-path-measurements)

**What was decisive is fit.** The encoder was 5.2GB, which is what made an 8GB card a candidate. Had that size exceeded the card's capacity, no price would have made it worth considering. **For "cheap" to mean anything, "it fits" has to come first.**

> **Quantization is not always a gain, however.** The same q8_0 is 19% slower than f16 on a V100, because on a card with tensor cores it only adds dequantization cost. **Choose cards on the assumption that quantization always makes things faster and you will be wrong** -> [5-1](#5-1-three-paths-for-quantized-matrix-multiplication)

-> The practical principle that follows is collected in [Pitfalls](99-pitfalls.md#do-not-choose-the-cheap-card-first).

---

## 6. Software Stack Constraints - The Biggest Trap

This is where the most time goes when actually building the system. **Which means it has to be checked before buying hardware.**

### 6-1. The Support Window Decides Whether a Configuration Is Possible At All

People say "you cannot mix cards that are too many generations apart," but the real constraint is sharper than that.

> **The set of architectures supported simultaneously by one driver branch and one CUDA version - every card has to fall inside that window.**

The two constraints differ in character.

| | Scope | Character |
|---|---|---|
| **Driver** | one per system | **absolute.** Miss a single card and that card is not even detected |
| **CUDA toolkit** | can differ per binary | avoidable by splitting the build (not recommended) |

As of 2026, the window that made this project possible was as follows.

| CUDA | Support floor | What it meant here |
|---|---|---|
| 13.0 | sm_75 (Turing) | **V100 (sm_70) unusable** -> could not adopt |
| **12.8** | sm_50 | both V100 and P104 work -> **adopted** |

The 580 driver branch is announced as the **last** line supporting Maxwell, Pascal, and Volta, while also supporting current Blackwell. In other words this is **both the widest window and the last one.**

CUDA 12.8's nvcc emits this warning at build time.

```
nvcc warning : Support for offline compilation for architectures prior to
'<compute/sm/lto>_75' will be removed in a future release
```

**That is advance notice.** CUDA 13 has already dropped Volta and Pascal is next. A configuration mixing previous-generation cards needs an operating principle of **pinning the CUDA and driver versions and not casually upgrading them.**

> Before mixing in a new card, always **check the supported architecture list for that specific driver branch and CUDA version yourself.** Do not infer it from generation names.

### 6-2. Confirm Driver Detection

Once a card is installed, the **first** thing to check is whether `nvidia-smi` reports its name correctly. If the name is `Unknown` or the card is missing from the list, that branch has dropped the architecture.

```
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
```

There is one more thing to check at the same time: **PCIe link width and generation.** Mining cards and cards on risers are frequently stuck on narrow lanes.

```
nvidia-smi --query-gpu=index,name,pcie.link.width.max,pcie.link.gen.max --format=csv
```

### 6-3. apt Repository Conflicts (Ubuntu)

Registering the NVIDIA CUDA apt repository conflicts with the distribution's driver packages.

```
libnvidia-gl-580 : Conflicts: libnvidia-egl-gbm1
```

**Block only the driver packages with pinning.** Create `/etc/apt/preferences.d/nvidia-no-driver` and lower the priority of driver packages from the NVIDIA repository. Take only the CUDA toolkit and use the distribution's driver.

And **finish `apt update && apt upgrade` immediately after installing the OS.** Doing it before building anything keeps the kernel and driver from drifting apart later.

### 6-4. One Binary, Multiple Architectures

Putting several architectures into a single binary beats maintaining a separate build tree per card, both operationally and for version tracking.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70"
```

| Comparison | Single binary | Per-architecture builds |
|---|---|---|
| Configuration | one path | a binary must be specified per service |
| Running the wrong binary | impossible | **possible (fails silently or runs slow)** |
| Version tracking | one commit = one state | trees can drift out of sync |
| Build time | proportional to architecture count | 1x each |

**Always quote the `;`.** Otherwise bash splits the command in two.

**Omitting an architecture means it will not run at all, or worse, the optimal kernel is silently disabled** (see 5-1).

---

## 7. Cost Structure and Hardware Constraints

### 7-1. What Goes Down and What Goes Up

**The purpose of mixing previous-generation cards is almost always to cut initial build cost.** Things grow in exchange, and leaving them out of the calculation distorts the judgment.

Below compares a **single high-cost card (A100 80GB)** against **this project's final configuration.**

| Item | This project (4 GPUs) | A100 80GB x1 | Ratio |
|---|---|---|---|
| **GPU purchase** | ~1,115 USD | 11,000 USD | **10%** |
| **System build cost** (board, CPU, PSU included) | ~2,000 USD | ~12,000 USD | **17%** |
| Power (GPU peak) | 1,080W | 300W | **3.6x** |
| Heat and noise | four server blowers' worth | one card | - |
| Space | 4 slots + a large PSU | 1 slot | - |
| **Motherboard requirement** | **server board required** (128 lanes) | an ITX board suffices | - |
| Management complexity | multi-architecture builds, PCIe distribution, airflow design | single architecture, single GPU | - |
| Build time | 2x (two architectures) | 1x | 200% |
| **Software lifespan** | **pinned to CUDA 12.8** | keeps moving to current CUDA | - |

> Build costs are estimates at August 2026 South Korean market prices. Power is the sum of GPU TDPs; actual electricity cost scales with utilization, so at equal utilization the ratio holds.

**Initial build cost drops to about one sixth while power becomes 3.6x.** So when does the electricity bill eat the savings - the arithmetic says **about 338 months (28 years).** The hardware's service life ends first. **The price saving is therefore meaningful and power cost is not a basis for the decision.** The real cost is the software lifespan below.

> **The arithmetic** - power difference (1,080 - 300) = 780W. At 50% utilization, `780W x 24h x 30 days x 0.5 = 280.8 kWh/month`; at 0.1 USD/kWh that is 28.1 USD per month. Build cost difference 9,500 USD / 28.1 = **338 months**. Even at 100% utilization it is 169 months (14 years).
>
> The rate is South Korean industrial electricity (about 130 KRW/kWh as of 2026) converted to USD, and **it varies by more than 3x between countries, so recompute with your own rate.** In a region at 0.3 USD/kWh like much of Europe it falls to 113 months (9.4 years), which could change the judgment.

**Software lifespan is the most expensive item over the long run.** The moment you use previous-generation cards the entire stack is pinned to a specific CUDA and driver version. New kernel optimizations and new quantization formats pass you by, and when a new model demands current CUDA the whole system has to be replaced - a cycle that generally arrives sooner than the hardware's usable life.

### 7-2. Splitting a Higher-Tier GPU into Lower-Tier GPUs Carries a PCIe Slot Cost

Slot cost means: first derive how many PCIe slots the whole system needs, then take the minimum cost that satisfies that requirement. If the entire system needs exactly one PCIe Gen3 x16 slot, for instance, the slot cost is the price of the cheapest CPU/motherboard/RAM combination that provides it.

At the initial design stage, between two and four heterogeneous GPUs were expected. GPUs generally require a physical x16 slot, so a board with four or more x16 slots was needed, and given that a standard ATX board has seven slot positions, having slots 1, 3, 5, and 7 be physically x16 would have been ideal.

However, the highest-volume form factor in the current consumer motherboard market is M-ATX, and among those the most common case is a single physical x16 slot. This traces back to a fundamental limit of consumer platforms: the number of PCIe lanes.

Consumer platforms generally offer 16 to 24 PCIe lanes, and even among server platforms with comparatively many lanes, previous-generation Intel platforms (X79/X99 and similar) sit at about 40. Placing many physical x16 slots, each demanding 16 lanes, is unusual design even by server standards, and in many cases a slot that is physically x16 is wired to fewer lanes. Boards that do exist are produced in small volume, which raises their unit cost.

#### Cost by Slot Count - A Rule of Thumb

| x16 slots required | (a) physical x16 only | (b) all at logical x8 or better | Platform implied |
|---|---|---|---|
| 1 | 10 USD | 10 USD | consumer board |
| 2 | 40 USD | 80 USD | consumer board (if one is available) |
| 3 | 80 USD | **200 USD** | **server board required** |
| 4 | 100 USD | **200 USD** | **server board required** |

> **This is a rule of thumb, not a derived figure.** It ignores CPU compute requirements, CPU generation, and PCIe generation entirely, estimating from physical x16 slot count alone, and reflects **the South Korean used market as of August 2026.** The 10 USD for one slot is based on actual transaction prices for an H110-class board + Celeron CPU + 2GB RAM. It shifts with region and date, so read the **multiplier as slot count rises**, not the absolute values.

**Do not read the table across; read it broken at the platform boundary.** The +120 USD from 2 to 3 slots is not the price of two slots but **the threshold for crossing from consumer boards to server boards.** Once crossed, additional slots are nearly free - within the server platform, the price difference between 2 and 4 slots is under 15%. **That is why this project chose four slots.**

If the requirement goes beyond physical x16 to include transfer rates dependent on generation and logical width, such as Gen3 x16, the cost climbs far more steeply. A requirement like "six slots at physical and logical x16, spaced three slots apart" could push the per-slot price close to 100 USD.

**Leave this slot cost out at the system design stage and the total system price is distorted - or worse, the configuration turns out to be impossible.**

#### Why Three Slots Means a Server Platform

| Platform | PCIe lanes | Three or more x16 | Used market |
|---|---|---|---|
| LGA 11xx family | 16~20 | almost none | almost none |
| AM4 | 24 | almost none | almost none |
| LGA2011 (X79/X99) | 40 | reasonably common | cheap |
| EPYC (server) | 128 | mostly possible | **entry price is far higher** |

**If you need three or more x16 slots, you effectively have to start from a server platform.** On consumer platforms even a two-x16-slot configuration is rare. So for this project, premised on cutting initial build cost, the realistic options were X99 and previous-generation EPYC, and nothing else.

> **An extreme case** - an ASRock H110 Pro BTC+ board with twelve P104-100 8GB cards gives **96GB of total VRAM for about 350 USD** including power supply and storage. But that board allocates **a single PCIe lane** per GPU slot, and the topology routes some of them through a PCIe expansion slot. Outside very particular circumstances it is hard to call such a machine appropriate for AI work.

#### What Actually Happened in This Project

```
1) A100 vs V100 -> the A100's 2.5x performance does not justify a 15.7x price, so V100
2) 30GB of LLM and diffusion will not fit on one card -> two or more V100s -> two or more x16 slots
3) A consumer board cannot provide two x16 -> move to a server board
4) On server boards the 2 <-> 4 slot price difference is small -> take four slots
5) Lowest per-slot cost among four-slot candidates -> GIGABYTE GA-X99-UD4P
6) Physical test failed -> drop the entire X99 platform from consideration
7) Lowest of the remaining candidates -> ASRock Rack EPYCD8-2T
```

The arithmetic behind 1) is in [Common Methodology Section 5](00-method.md#5-candidate-gpu-reference-table).

**The slot cost appeared at 2).** The moment one A100 80GB became two V100s, the x16 slot requirement went from one to two.

By the rule-of-thumb table, 1 slot -> 2 slots is **+70 USD**. But **as of August 2026 the only two-slot motherboards available under 50 USD were X99-based.** The moment you use X99 you have already given up the real advantage of a consumer retail board - manageability - and within the same server platform four slots costs only 15% more. **So the reason to pick two slots evaporated, and the jump went straight to a four-slot server board (200 USD). The effective slot cost was +190 USD.**

**Splitting a higher-tier card into several lower-tier cards does not only reduce the card price. It adds slots, and adding slots changes the platform tier itself.** The saving against the A100 ran to thousands of USD so 190 USD disappeared into it, but where the saving is smaller this transition cost can flip the decision.

> **The table is a floor, not a budget.** In a used-market build, "what is actually for sale at that price" is a separate variable.

#### Crossing the Threshold Brings Costs That Are Not on the Price Tag

**Moving to a server platform increases costs that never appear as a number.** It sounds trivial, but a consumer board reaches POST in under 10 seconds while a server board takes closer to 30. Repeat that across dozens of reboots during development and verification and the difference becomes real time cost.

This is **why cost must not be counted as purchase price alone** -> [the definition of cost](../../README.md#notes-and-caveats)

#### A Board That Passed the Paper Test Failed in the Flesh

The GIGABYTE GA-X99-UD4P chosen at 5) above passed the paper test - review of published specifications only - but **failed the physical test.** Above 4G Decoding was enabled through BIOS modding and confirmed working with a P104, and yet **not even a single V100 32GB would pass POST.** The cause was never determined. Details are in [Appendix - Build Log](appendix/build-log.md#second-configuration---passed-the-paper-test-failed-in-the-flesh).

The X99 platform has only 40 PCIe lanes. Few lanes not only reduces the number of **physical x16 slots** a manufacturer places on the board but **affects nearly every kind of expansion - storage controllers, NICs, high-speed M.2 storage.** Judging this unsolvable within X99, the entire platform was dropped from consideration.

What was chosen at 7) and used through to the final system is the **EPYC server board (ASRock Rack EPYCD8-2T).** The EPYC platform has 128 lanes, and this board allocates **four x16 slots + three x8 slots = 88 lanes** to physical slots with room to spare, alongside two 10G NICs, two M.2 slots, and nine SATA ports. It also **recognized all four GPUs immediately, with no high-risk work such as firmware modding**, cutting a great deal of time out of the hardware test and assembly stages. Whether it would detect multiple V100s was confirmed in the paper test after 7).

**Multi-GPU support and Above 4G Decoding support must be confirmed at the planning stage.** They are usually not on the spec sheet, so **collecting as many cases as possible of someone actually building and verifying the same combination, and using those as reference, is the only reliable method.** If a board bought on slot count alone cannot even POST, there is no realistic remedy short of replacing hardware. **The time and effort spent in that process is also a cost.**

### 7-3. When Lanes Are Short - Some Work Runs Fine on Narrow Lanes

Not every case needs wide PCIe bandwidth. It depends on the nature of the work, but for much of it the PCIe lanes go almost unused except at the moment the model is loaded onto the GPU - and even then the bottleneck is frequently storage I/O rather than the lanes.

**There is a single test - is the traffic "one-time" or "every step / every token."** Loading a model, however many GB, happens once at startup, so narrow lanes add little; traffic that occurs every step makes bandwidth the throughput ceiling directly.

| Work | When traffic occurs | Cost of narrow lanes |
|---|---|---|
| **Tensor parallelism** (llama.cpp `--split-mode row` etc.) | all-reduce per layer, **per token** | X **high cost** (1-3) |
| **Multi-GPU training** (data parallel) | gradient all-reduce sized to the parameters, **per step** | X **high cost** |
| **Weight offload / layer streaming** | weight transfer **per token** | X **high cost** - bandwidth is speed |
| **KV cache held outside the GPU** | read and written per token | X **high cost** |
| **MoE experts scattered across cards** | routing per token | X **high cost** |
| Pipeline parallel / layer split (`--split-mode layer`) | boundary activations only, per step | ~ medium cost - traffic is small enough to usually tolerate |
| Large per-request I/O (high resolution, video) | hundreds of MB per request | ~ medium cost - judge against the transfer volume |
| Swapping models or LoRAs per request | GB-scale loads per request | ~ medium cost - the case where it stops being one-time |
| **Resident modules** (this project's encoder and VAE) | once at startup + a few MB per request | O **low cost** |

**"High cost" here does not mean it will not work.** It runs on narrow lanes too; what grows is time. The upper entries simply grow it enough to make the work pointless, which removes them from consideration in practice.

**Calculate per-module traffic first, then allocate slots.**

For work near the bottom of the table, a narrow physical slot can be overcome with riser cables and adapters. Adapters converting an M.2 slot into a PCIe slot, external connection standards such as OCuLink, and PCIe expansion cards in the PLX PEX8749 family all exist. **This project actually used an external PCIe switch card in its first configuration to connect the GPU module to the main system** (SFF-8654 cable, model number unconfirmed) -> [Appendix - Build Log](appendix/build-log.md#first-configuration---nvlink-v100-32gb-x2)

This project's diffusion encoder module falls here. Outside the loading moment, the data exchanged for encoding is on the order of a few MB, so even under the worst conditions (PCIe Gen1 x1) **about 20ms** - under 50ms with margin - was enough for the I/O. Putting the encoder on a P104-100 at PCIe Gen1 x4 was therefore expected to leave bandwidth irrelevant given that one image takes tens of seconds, and that is what happened.

| Transfer | Frequency | V100 (x16 Gen3) | P104 (x4 Gen1) |
|---|---|---|---|
| T5 weights 5.06GB | **once at service startup** | 0.58s | **5.88s** |
| Effective transfer rate | - | 8.7 GB/s (55% of theoretical) | **0.86 GB/s (86% of theoretical)** |
| Conditioning tensor 4.2MB | per request | 0.5ms | ~5ms |

And yet **the window where that gap actually matters is extremely narrow.**

| Phase | Loss vs V100 | Character |
|---|---|---|
| Service startup | **+5.3s** | **one-time.** Occurs only on restart |
| Conditioning transfer per request | +4.5ms | **about 0.02%** of the ~100s for eight images |

**In exchange for spending 5.3 seconds more once at startup, the running cost is effectively zero.** So there is no need to dedicate an x16 slot to the P104.

> It holds even at the extreme of a PCIe x1 riser. **At x1 Gen1 the startup load grows to about 20 seconds, but the running cost is still on the order of 20ms.**

---

## 8. When a Heterogeneous GPU Configuration Is the Wrong Fit

- **If budget is not a constraint**, much of this configuration is moot. The reason is not the electricity bill, though (7-1). The real price is **complexity and software lifespan.** Multi-architecture builds, PCIe distribution, and airflow design stay attached permanently, and the whole stack is pinned to CUDA 12.8, cut off from new optimizations and new quantization formats. If you can buy identical current-generation cards, do that instead
- **If the model does not fit on one card and tensor parallelism is required**, the story changes entirely (1-3). Tensor parallelism across heterogeneous cards runs at the slowest card's pace and needs an NVLink-class interconnect, which drives cost up
- **If you cannot or will not modify the inference engine**, module separation is impossible (see [No.2 Encoder Separation](02-encoder-separation.md))
- **If the workload does not satisfy the four conditions in 1-1**, there is no reason to split it
- **If you lack hardware troubleshooting skill**, you will stall at diagnosing a machine that will not boot, as the [GA-X99-UD4P case in 7-2](#a-board-that-passed-the-paper-test-failed-in-the-flesh) shows. Assess your own capability honestly, and if it is not there, **starting with a consultation at a specialist shop will cut cost substantially in the end**

> That said, hitting one of the items above is not a reason to abandon cheap cards outright. **There is usually some place in the pipeline where a low-end card can contribute.** Even here, the P104-100 was useless for diffusion and useless for the LLM, but once the text-encoder position was found it earned its keep for 15 USD. Before giving up, work carefully through the table that decomposes the work into phases, find a task it can serve, and search thoroughly for cases and references.

## Notes and Caveats

> **(1), (2), and (3)** are all cited from [1-2 Type B](#1-2-type-b---several-models-run-concurrently-especially-with-different-resource-profiles).

**(1)** A prediction from background research. In the measurements, **per-image generation time barely changed even with the encoder in CPU RAM** - encoder placement affects only the first image, not the second and subsequent ones -> [Appendix 7](98-benchmark.md#7-interpretation---the-effect-of-encoder-placement)

**(2)** A figure derived at the planning stage. `500 USD saved by using 16GB instead of V100 32GB - 50 USD for 8GB of encoder RAM = 450 USD`.

> The 8GB RAM price confirmed at build time was closer to **30 USD than 50, so the real gain was closer to 470 USD.** But the 450 USD above is **the figure derived at planning time from the information available then**, and this section records the decision process, so 450 is the appropriate number here.

> **Slot cost is not part of this calculation.** Whether you use 32GB or 16GB, **it occupies one x16 slot either way.** The slot cost appeared earlier, **at the step from one A100 to two V100s** -> [7-2](#7-2-splitting-a-higher-tier-gpu-into-lower-tier-gpus-carries-a-pcie-slot-cost)

**(3)** **Why not spend the same money on a single 32GB card?** Two separated workers (13.6 x 2 = 27.2GB) do fit on a 32GB card, so it is formally possible. But **measurement during initial testing after assembly showed that it fits and yet runs slower.**

| 8 images | Workers | Time |
|---|---|---|
| **one** 32GB card, one worker | 1 | 265.1s |
| **one** 32GB card, two workers | 2 | **296.4s** |
| **two** 16GB cards, two workers | 2 | **134.0s** |

**Doubling the workers made it 11.8% slower.** They time-share the same compute units and contend for the disk at startup on top of that. Splitting across physically separate cards, by contrast, is 2.21x faster.

**Fitting in VRAM and performing are separate things.** Raising throughput required physically more compute units, and **the cheaper those units, the better.** That is why two 16GB cards beat one 32GB card -> [No.2 Section 8, Conclusion No.3](02-encoder-separation.md#8-results)
