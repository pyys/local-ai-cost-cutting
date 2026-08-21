**English** / [한국어](../ko/98-benchmark.md)

# Appendix - Cost and Performance Benchmark Raw Data

> **The source of every measured value cited by No.1, No.2, and No.3, along with its interpretation**, is collected here.
> The main documents carry only conclusions; measurement conditions, raw records, derived calculations, and the disconfirmation process are checked in this one.
> Reproduction scripts are in [`bench/`](../../bench/).

Related documents: [Common Methodology](00-method.md) / [No.1 Role Assignment](01-role-assignment.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.3 Pipeline Throughput](03-pipeline-throughput.md) / [No.4 Orchestration](04-orchestration.md) / [Pitfalls](99-pitfalls.md)

---

## Contents

1. [Experiment Design](#1-experiment-design)
2. [Comparison Conditions](#2-comparison-conditions)
3. [Raw Records](#3-raw-records)
4. [Derived Calculations](#4-derived-calculations)
5. [Interpretation - Axis A: Cost Reduction at Equal Throughput](#5-interpretation---axis-a-cost-reduction-at-equal-throughput)
6. [Interpretation - Axis B: Throughput Increase at Equal Cost](#6-interpretation---axis-b-throughput-increase-at-equal-cost)
7. [Interpretation - The Effect of Encoder Placement](#7-interpretation---the-effect-of-encoder-placement)
8. [Interpretation - The Cost of Splitting Work](#8-interpretation---the-cost-of-splitting-work)
9. [Disconfirmation - Why the 32GB Card Was Slower](#9-disconfirmation---why-the-32gb-card-was-slower)
10. [Phase-Separated Logs](#10-phase-separated-logs)
11. [What Was Not Measured](#11-what-was-not-measured)
12. [SM Scaling and Kernel Path Measurements](#12-sm-scaling-and-kernel-path-measurements)
13. [Reproduction](#13-reproduction)

---

## 1. Experiment Design

### The Question Being Answered

> **The 32GB card for the LLM already exists. What should be bought in addition, for image generation?**

Allocating one 32GB card to the LLM is a fixed condition. On top of that, the question is whether **buying a second 32GB card** or **combining cheaper cards** is better for image generation.

### The Claim Being Tested

> A mix of low-cost cards can **either raise throughput at equal cost or cut cost at equal throughput**, compared with a single high-cost card.

### Measurement Conditions

```
Model        FLUX D Q8 (DiT) + T5-XXL q8_0 + CLIP-L + VAE
Resolution   768 x 768
Steps        15, euler, txt_cfg 1.0, distilled_guidance 3.5
Extension    PuLID enabled (id_weight 0.5)
Prompt       production length, 512 tokens (2 chunks)   <- verified by conditioning file size 8,391,936 B
Seed         fixed (condition 1 uses a single seed + batch_count, the rest use SEED+i)
VAE          tiling enabled
```

```
Cold start   full sd-server shutdown -> sync -> drop_caches=3 -> wait 1s
Timer start  at the first request, not at service launch
             (sd-server lazy-loads weights, so model loading is included in the request time)
Repeats      three per combination, median used
Concurrent load  none. Every other service including the LLM server stopped
```

### Measured Quantities

| Quantity | Definition |
|---|---|
| **1 image** | one "generate 1 image" command from cold -> until complete |
| **8 images** | one "generate 8 images" command from cold -> until all complete |
| **First image** | the moment the first image of the 8-image command finishes |

> **"8 images" is not "one image plus seven more."** It is the total for requesting eight with a single command, chosen to match the actual usage pattern.

### Why the Baseline's Conditions Are Stated Explicitly

This project cited an early figure of "8 images in 124 seconds" for some time before discovering it was a **pre-PuLID** value. PuLID adds 22% per step (1.39 -> 1.70 s/it). **A wrong baseline either understates the improvement or invents a regression that never happened.**

Every measurement above is unified on PuLID enabled. The per-image time quoted in the main documents (26.9s) uses the same baseline.

> Values observed in the production configuration before this experiment (LLM running concurrently, encoder in fp16) were 145s -> 132s -> 108s for eight images. **Those are not isolated conditions and must not be placed on the same axis as the tables below.**

---

## 2. Comparison Conditions

| Condition | Worker card | Encoder location | Unit of work | Added cost | Workers |
|---|---|---|---|---|---|
| **1** | V100 32GB x1 | **same card (loaded whole)** | batch of 8 | 700 USD | 1 |
| **2** | V100 16GB x1 | RAM (`te=cpu`) | individual | 230 USD | 1 |
| **3** | V100 16GB x1 | P104-100 | individual | **215 USD** | 1 |
| **4** | V100 16GB x2 | P104-100 | individual | 415 USD | 2 |
| *5* | *V100 16GB x3* | *P104-100* | *individual* | *615 USD* | *3* |
| **6** | V100 32GB x1 | P104-100 | individual | 715 USD | **2 (same card)** |
| **7** | V100 32GB x1 | P104-100 | individual | - | 1 |
| **8** | V100 32GB x1 | **same card (loaded whole)** | **individual (re-encode per image)** | - | 1 |

> **The 230 USD for condition 2** is 200 USD of GPU plus 30 USD of RAM. Putting the separated encoder in RAM requires that much physical memory, and an 8GB DDR4 module is about 30 USD. **RAM is a cost too and belongs in the calculation.**

**Condition 5 is not a measurement.** Three V100 16GB cards were not available, so it is a value back-calculated from conditions 3 and 4 (Section 4).

**Condition 7 is a diagnostic, not a purchase option.** Its worker card matches condition 1 and its operation matches condition 3, which separates whether condition 1's per-image latency comes from the card or from the structure (Section 9).

### Price Basis

South Korean used market, August 2026. **These vary widely by region and date, so refill them with your own market's prices.**

| Part | Price |
|---|---|
| V100 SXM2 32GB | 700 USD |
| V100 SXM2 16GB | 200 USD |
| P104-100 8GB | 15 USD |
| DDR4 8GB | 30 USD |

### Hardware

```
CPU     AMD EPYC 7232P
Board   ASRock Rack EPYCD8-2T
OS      Ubuntu 24.04 / CUDA 12.8
GPU0    Tesla V100-SXM2-16GB
GPU1    NVIDIA P104-100          (PCIe x4 Gen1)
GPU2    Tesla V100-SXM2-32GB
GPU3    Tesla V100-SXM2-16GB
Engine  stable-diffusion.cpp (commit f440ad9c + local patches)
```

---

## 3. Raw Records

The full `results.csv`. 36 rows (12 combinations x 3 runs).

```
time,condition,price_usd,workers,count,run,total_s,first_image_s,cond_bytes,tokens,pulid
2026-08-12 04:17:39,1,700,1,1,1,46.29,46.09,,,on
2026-08-12 04:18:38,1,700,1,1,2,46.54,46.34,,,on
2026-08-12 04:19:37,1,700,1,1,3,46.51,46.31,,,on
2026-08-12 04:20:57,2,230,1,1,1,65.09,64.89,8391936,512,on
2026-08-12 04:22:15,2,230,1,1,2,65.11,64.91,8391936,512,on
2026-08-12 04:23:32,2,230,1,1,3,64.77,64.57,8391936,512,on
2026-08-12 04:24:35,3,215,1,1,1,48.18,47.98,8391936,512,on
2026-08-12 04:25:36,3,215,1,1,2,48.13,47.93,8391936,512,on
2026-08-12 04:26:37,3,215,1,1,3,48.23,48.03,8391936,512,on
2026-08-12 04:27:40,4,415,2,1,1,48.22,48.02,8391936,512,on
2026-08-12 04:28:41,4,415,2,1,2,48.15,47.95,8391936,512,on
2026-08-12 04:29:41,4,415,2,1,3,48.18,47.98,8391936,512,on
2026-08-12 04:34:13,1,700,1,8,1,256.68,256.48,,,on
2026-08-12 04:38:47,1,700,1,8,2,261.76,261.55,,,on
2026-08-12 04:43:21,1,700,1,8,3,261.74,261.54,,,on
2026-08-12 04:47:52,2,230,1,8,1,254.92,64.80,8391936,512,on
2026-08-12 04:52:20,2,230,1,8,2,256.08,64.61,8391936,512,on
2026-08-12 04:56:50,2,230,1,8,3,256.64,64.93,8391936,512,on
2026-08-12 05:01:06,3,215,1,8,1,240.32,48.13,8391936,512,on
2026-08-12 05:05:19,3,215,1,8,2,240.70,48.00,8391936,512,on
2026-08-12 05:09:32,3,215,1,8,3,240.78,48.00,8391936,512,on
2026-08-12 05:11:59,4,415,2,8,1,131.56,48.11,8391936,512,on
2026-08-12 05:14:25,4,415,2,8,2,134.01,48.09,8391936,512,on
2026-08-12 05:16:52,4,415,2,8,3,134.85,47.97,8391936,512,on
2026-08-12 05:49:52,7,715,1,1,1,50.05,49.85,8391936,512,on
2026-08-12 05:50:54,7,715,1,1,2,49.66,49.45,8391936,512,on
2026-08-12 05:51:56,7,715,1,1,3,49.69,49.49,8391936,512,on
2026-08-12 05:56:33,7,715,1,8,1,261.63,49.48,8391936,512,on
2026-08-12 06:01:10,7,715,1,8,2,265.12,49.91,8391936,512,on
2026-08-12 06:05:48,7,715,1,8,3,265.72,50.02,8391936,512,on
2026-08-12 06:06:53,6,715,2,1,1,49.93,49.73,8391936,512,on
2026-08-12 06:07:56,6,715,2,1,2,49.91,49.71,8391936,512,on
2026-08-12 06:08:58,6,715,2,1,3,49.99,49.79,8391936,512,on
2026-08-12 06:14:08,6,715,2,8,1,294.31,86.81,8391936,512,on
2026-08-12 06:19:16,6,715,2,8,2,296.36,88.02,8391936,512,on
2026-08-12 06:24:26,6,715,2,8,3,296.98,88.09,8391936,512,on
```

**All 36 runs succeeded; there were no failures.** Spread within a combination is at most 1.3% (condition 1's 8-image runs) and mostly under 0.5%.

`cond_bytes` is 8,391,936 B on every row, which **confirms every measurement used the same 512-token prompt.**
`tokens = (file size - 3328) / 16384` -> `(8391936 - 3328) / 16384 = 512`

---

## 4. Derived Calculations

### 4-1. Median Summary

| Condition | Cost | Workers | 1 image | 8 images | First image | Per round |
|---|---|---|---|---|---|---|
| 1 | 700 | 1 | 46.51s | 261.74s | 261.54s | 30.75s |
| 2 | 230 | 1 | 65.09s | 256.08s | 64.80s | 27.28s |
| 3 | 215 | 1 | 48.18s | 240.70s | 48.00s | 27.50s |
| 4 | 415 | 2 | 48.18s | 134.01s | 48.09s | 28.61s |
| *5* | *615* | *3* | - | *105~107s* | - | *derived* |
| 6 | 715 | 2 | 49.93s | 296.36s | 88.02s | 82.14s |
| 7 | - | 1 | 49.69s | 265.12s | 49.91s | 30.78s |

```
per round = (8 images - 1 image) / (ceil(8 / workers) - 1)
```

With one worker this equals per-image generation time; with two workers it is the time for one round producing two images simultaneously.

### 4-2. Deriving Condition 5

Conditions 3 and 4 differ only in worker count, so the fixed phase and the per-image generation time can be back-calculated from them.

```
t(N) = S + ceil(8/N) x a        S = startup + encoding (fixed),  a = per-image generation

Condition 3 (1 worker) : t3 = S + 8a = 240.70
Condition 4 (2 workers): t4 = S + 4a = 134.01
  -> a = (t3 - t4) / 4 = 26.67s
     S = t4 - 4a        = 27.33s
Condition 5 (3 workers): t5 = S + 3a = 107.3s        (ceil(8/3) = 3 rounds)
```

Taking the measured first image (48.1s) as the fixed point instead gives:

```
t5 = 48.1 + 2 x 28.61 = 105.4s
```

**The two methods converge on 105~107s, so that range is used.**

**This value is an optimistic upper bound.** In practice it is likely to be slower, for two reasons.

- **Disk I/O contention at startup** - three processes reading a 12GB model at once increases S. In condition 6, going to two workers pushed the first image from 49.9s to 88.0s, a 76% increase
- **Thermal interaction between cards** - cards that were idle during condition 4 heat up together in condition 5 (Section 9)

---

## 5. Interpretation - Axis A: Cost Reduction at Equal Throughput

```
Condition 1   700 USD   261.7s
Condition 3   215 USD   240.7s
```

**Condition 3 produced throughput equivalent to condition 1 at 69% lower cost.**

In the measurements condition 3 was 21 seconds (8%) faster. But **that difference is not an effect of encoder separation; it comes from thermal throttling on the 32GB card**, so it is not counted as a gain. The evidence is in Section 9.

With equal cooling, the two conditions' 8-image times would have been nearly identical. **The claim is not "faster" but "the same job for a third of the price."**

### The Precondition for This Result

A 200 USD 16GB card **was never a candidate, because the model would not fit.**

| | VRAM required |
|---|---|
| FLUX D Q8 loaded whole | 19.1GB -> will not fit a 16GB card |
| Worker after encoder separation | **13.6GB** -> fits a 16GB card |

Only after [module separation](02-encoder-separation.md) lowered the minimum VRAM requirement did this comparison become possible at all. **It looks like a card-selection problem but was really a task-partitioning problem.**

---

## 6. Interpretation - Axis B: Throughput Increase at Equal Cost

```
Condition 1   700 USD   261.7s
Condition 4   415 USD   134.0s    <- 41% cheaper, 1.95x            (measured)
Condition 5   615 USD   ~106s     <- 12% cheaper, 2.47x            (derived)
```

**Condition 4 reaches its conclusion from measurement alone.** Replacing one 700 USD card with two 200 USD cards plus a 15 USD encoder drops cost by 41% while doubling throughput.

### Testing the Counterargument - Is It Just More Workers?

One can object: "the gain came from doubling the workers, not from splitting the cards." **Condition 6 tests exactly that.** It puts two workers (13.6 x 2 = 27.2GB) on **one** 32GB card.

| | Card | Workers | 8 images | First image |
|---|---|---|---|---|
| Condition 6 | **one** 32GB | 2 | **296.4s** | 88.0s |
| Condition 4 | **two** 16GB | 2 | **134.0s** | 48.1s |

**With the same worker count, the 42% cheaper configuration is 2.21x faster.** The only difference is split cards versus one card, so **the gain comes from physically separate compute units, not from worker count.**

### Condition 6 Is Slower Even Than One Worker

```
Condition 7   one 32GB card, 1 worker    265.1s
Condition 6   one 32GB card, 2 workers   296.4s    <- 11.8% slower
```

There was no OOM. 27.2GB fit inside 32GB and the worker logs show no allocation failures. And yet it is **not merely no gain but a net loss.** There are two causes.

| Cause | Evidence |
|---|---|
| **Time-sharing the compute units** | two processes divide the same SMs |
| **Disk contention at startup** | first image 49.9 -> 88.0s, a **76% increase**. They read a 12GB model simultaneously |

Per round it is 82.14s, whereas simple time-sharing would give `2 x 30.78 = 61.6s`. **The extra 20 seconds is the real cost of contention.**

> **Fitting in VRAM and performing are separate things.** Raising throughput requires physically distinct compute units, and the cheaper those units, the better.

---

## 7. Interpretation - The Effect of Encoder Placement

Conditions 2 and 3 differ **only in whether the encoder sits in RAM or on the P104.** Worker card and unit of work are the same.

| | Condition 2 (RAM) | Condition 3 (P104) | Difference |
|---|---|---|---|
| First image | 64.80s | 48.00s | **16.80s** |
| 8 images | 256.08s | 240.70s | **15.38s** |
| **Per round** | **27.28s** | **27.50s** | **~= 0** |

**The 16.8-second gap opens on the first image and then simply persists; it is not reflected in per-image cost at all.**

This is direct evidence that the core premise of [the encoder separation design](02-encoder-separation.md) - **encode once and reuse the result** - actually works. A slow encoder delays only the first image and has no effect on subsequent ones.

### A Corollary - Encoding Time Is Perfectly Linear in Chunk Count

A dedicated measurement (7-3) confirmed linearity on both sides.

| | 256 tokens | 512 tokens | Ratio |
|---|---|---|---|
| CPU (16 threads, isolated) | 8.30s | 16.51s | **1.99** |
| P104-100 (isolated) | 0.33s | **0.66s** | **2.00** |

sd.cpp processes in 256-token chunks, so **each chunk re-reads the weights and the computation grows accordingly.** There is no per-chunk fixed cost.

> P104 encoding in production logs was 0.86s. The isolated measurement is exactly 0.66s, so **the 0.2s difference is system contention, not per-chunk fixed cost.** Recording it as "per-chunk overhead" in early documents was an error.

### A Corollary - First-Image Latency Differs from Actual Encoding Time

The first-image difference between conditions 2 and 3 is 16.8s, while the actual encoding difference is `16.51 - 0.66 = 15.85s`. The remainder is explained like this.

```
The encoder and the worker start at the same time
  Condition 2   while the CPU encodes, the worker loads 12GB of DiT
  Condition 3   even though P104 encoding ends in 0.66s, it still waits for the worker load (11.3s)

-> part of the CPU encoding hides behind the worker load
```

**The latency the user perceives is shorter than the encoding time**, because concurrent startup absorbs part of it.

### 7-2. Encoding Performance by Card - Isolated Measurement

Separate from conditions 1 through 7, this measurement **launched only the text encoder to compare cards against each other.** Card comparison is meaningful only through isolated measurement with matched conditions.

```
1. Stop every other service (workers and LLM included)
2. Launch only the encoder, standalone
3. Three requests with the same prompt
4. Discard the first, use the second and third   <- the first includes weight loading
5. Swap the card and repeat
```

**Results** (T5 Q8, identical 256-token prompt)

| Phase | V100 16GB | P104-100 | Ratio |
|---|---|---|---|
| T5 load (disk read) | 1.81s | 0.15s | note: page cache effect |
| T5 load (VRAM transfer) | 0.58s | **5.88s** | 10.1x |
| **First-run encoding** | 3.22s | 7.31s | 2.3x |
| **Warm encoding (runs 2 and 3)** | **0.085s** | **0.33s** | **3.9x** |

These values are the basis for the [price per unit of throughput](00-method.md#4-price-per-unit-of-throughput) calculation.

Two things had to be filtered out:

- **The inversion in disk read time has nothing to do with the cards.** The V100 test ran first and read from disk; the P104 simply read a file already in page cache
- **Comparing first runs distorts the ratio.** It looks like 2.3x, but the actual compute ratio is 3.9x

> **The P104 is 3.9x slower per module, yet across the whole-system time for generating eight images that difference is under 1%.** Spending 185 USD more on the strength of "3.9x" alone is a decision made without doing the arithmetic.

### 7-3. Why 25x Faster Than CPU - Arithmetic Intensity and Ridge Point

**Stopping at "moving the encoder to a GPU made it faster" leaves nobody else able to judge it for their own hardware.** The mechanism was decomposed by calculation.

#### Measurement - CPU Thread Scaling

Measured across thread counts with a dedicated script ([`bench/thread_scaling.py`](../../bench/thread_scaling.py)). The conditioning file cache was dropped each time, and the first run, which mixes in model loading, was discarded.

| Configuration | 256 tokens | 512 tokens | GOPS | vs 1 thread | Parallel efficiency |
|---|---|---|---|---|---|
| CPU 1 thread | 69.52s | 138.16s | 35 | 1.00x | 100% |
| CPU 2 threads | 37.53s | 74.26s | 64 | 1.85x | 92.6% |
| CPU 4 threads | 19.79s | 39.33s | 122 | 3.51x | 87.8% |
| CPU 8 threads | 10.14s | 20.12s | 237 | 6.86x | 85.7% |
| **CPU 16 threads (SMT)** | **8.30s** | **16.51s** | **290** | **8.38x** | - |
| CPU default (`-t` unspecified) | 10.09s | 20.05s | 238 | 6.89x | - |
| **P104-100** | **0.33s** | **0.66s** | **7,292** | **210.7x** | - |

**The CPU side uses its best configuration (16 threads) as the reference.** The argument is "even the best CPU cannot catch up," so the value favorable to the CPU has to be used.

```
Gap    256 tokens   8.30 / 0.33  = 25.2x
       512 tokens  16.51 / 0.66  = 25.0x      <- independent of prompt length
```

#### Arithmetic Intensity

```
Compute   = 2 x 4.7e9 (T5-XXL encoder parameters) x 256 tokens = 2,406 GFLOP
Memory    = the Q8 weights, 4.83 GB, read once

Arithmetic intensity = 2,406 / 4.83 = 498 FLOP/byte
```

Because processing is chunked, **intensity stays at 498 as token count grows.** The perfect linearity in the table above supports this.

> Taking the parameter count as 4.55e9, or including activation traffic, moves it into the 400~500 range, but the classification below does not change.

#### Ridge Point

`ridge point = compute throughput / memory bandwidth`. The P104-100's bandwidth is 320 GB/s.

| Compute path | Throughput | Ridge point | |
|---|---|---|---|
| **INT8 DP4A** | 22 TOPS | **68.8** | the path q8_0 takes |
| FP32 | 6.1 TFLOPS | 19.1 | |
| FP16 | 0.095 TFLOPS | 0.30 | **crippled to 1/64 - but this path was never selected (Section 12)** |

```
arithmetic intensity 498  vs  ridge point 68.8   ->  7.2x   ->  unambiguously compute-bound
```

Recomputed with the measured effective throughput (7,292 GOPS), the effective ridge point of 22.8 still gives 21.8x. **Compute-bound by any measure**, which makes the conclusion robust.

#### Why This Is Decisive

The P104's 320 GB/s is **about a third of a V100's.** Had this been memory-bound work, the card would have been crushed. In fact reading the weights takes 15ms, just 4.6% of the 330ms total.

> **The P104 fits this position because two conditions hold at once.**
> No.1 the work is one where weak bandwidth is not the bottleneck, and No.2 that card's INT8 compute is intact.
> **Break either one and it does not hold.**

**What made this card usable was not the card selection but q8_0 quantization.** The reason, though, was fit rather than speed - T5-XXL at fp16 is 9.79GB and simply does not go on an 8GB card. On speed, q8_0 is 2.17x f16, a far narrower margin than the specification suggested (Section 12) -> [No.1 4-1](01-role-assignment.md#4-1-there-are-two-directions-of-attack)

#### Cross-Verification

Two figures obtained by independent routes agree.

```
V100 warm encoding 0.085s  ->  28,306 GOPS   (22.6% of 125 TFLOPS tensor FP16)
P104 warm encoding 0.33s   ->   7,292 GOPS   (33.1% of 22 TOPS INT8)

Ratio  28,306 / 7,292 = 3.88x
```

This matches the **"3.9x versus V100"** from the isolated measurement in 7-2.

#### SMT Gave 22% - The Opposite of Expectation

On compute-bound SIMD work, SMT is normally a loss, since it only divides the execution units. **And yet 16 logical threads (8.30s) beat 8 physical cores (10.14s) by 22%.**

Zen 2 has no AVX512-VNNI, so ggml's int8 path runs a `maddubs -> madd -> accumulate` dependency chain. **SMT appears to be filling the pipeline bubbles in between.**

> **"Compute-bound" does not mean the execution units are saturated.** A roofline classification says which resource is the ceiling; it guarantees nothing about that resource's utilization. -> [Pitfalls - Do not disable SMT just because the work is compute-bound](99-pitfalls.md#do-not-disable-smt-just-because-the-work-is-compute-bound)

For reference, sd-server's `-t` default was **the physical core count (8).** Leaving the default in place costs you the 22%.

#### Can More CPU Catch Up?

Per physical core it is `290 / 8 = 36.2 GOPS`. The whole same-generation lineup (EPYC Rome) converted by core count and base clock:

| Model | Cores | Base | Estimated GOPS | vs P104 | Used price |
|---|---|---|---|---|---|
| 7232P *(this server)* | 8 | 3.1 | 290 | 4.0% | $30-60 |
| 7402P | 24 | 2.8 | 786 | 10.8% | $48-90 |
| 7702P | 64 | 2.0 | 1,497 | 20.5% | $600-1,200 |
| 7H12 | 64 | 2.6 | 1,946 | 26.7% | rare on the used market |
| **2 x 7H12 (dual socket, 128 cores)** | 128 | 2.6 | **3,892** | **53.4%** | - |

**Even filling both sockets with Rome's top part gets you barely half of one 15 USD card.** Rome caps at two sockets, so **a single machine cannot reach it.**

To match it anyway, the cheapest option per GOPS is the 7402P, requiring **ten of them - five dual-socket machines.**

| | Cost | Power |
|---|---|---|
| CPU silicon only | ~$480 | - |
| Whole systems (five, boards/RAM/PSUs included) | **~$3,500** | **1,800W** |
| **P104-100** | **$15** | **180W** |

**About 230x the cost and 10x the power** - and even that **assumes software exists that supports distributed encoding.** It does not.

The conversion above linearly extrapolates an 8-core measurement by core count and clock; it is an **estimate**, and it ignores losses at NUMA and CCD boundaries, so it **favors the CPU side.** The real gap is larger. Board and RAM prices are approximate too.

#### Discrepancy with the Early Measurement

Early documents in this project recorded CPU encoding as **13.97 seconds** (256 tokens). Controlled re-measurement gave **10.09 seconds** with the same default settings.

**The cause of the difference was not identified.** The re-measurement protocol differed in three ways - discarding one warm-up run, dropping the conditioning file cache, and fully stopping other services. **The first is the most likely, but it was not confirmed.**

The value measured under the stricter conditions was adopted.

---

### 7-4. Running N Encoders - Operational Observations

Observations from before encoder separation was introduced, when adding a worker also added an encoder.

| | 1 encoder | 2 encoders concurrently | Ratio |
|---|---|---|---|
| CPU encoding time | 17.7s | **33~37s** | 1.9~2.1x |
| RAM consumption | 9.2GB | **18.4GB** | 2.0x (linear) |
| **Model load at startup** | 7.32s | **21.03s** | **2.9x** |

**This table is not a controlled A/B.** These are production observations from the period when T5 sat in RAM at fp16 (production prompts, other services running), and they must not be placed on the same axis as conditions 1 through 7. **Use them only to read the direction of the ratios.**

The third item is easy to overlook. **It is not only the CPU that contends - storage does too.** Two processes reading the same weight file at once made load time 2.9x. This server has NVMe, hence that figure; **on a SATA SSD or HDD with weaker random 4K read the gap would be larger.**

The same phenomenon reproduced under controlled conditions - in condition 6, going to two workers pushed **the first image from 49.9s to 88.0s, a 76% increase** (Section 6). Anyone planning a configuration with many workers must account for it.

---

## 8. Interpretation - The Cost of Splitting Work

Running N images as N independent jobs rather than one batch means **the per-job fixed overhead occurs N times.** Saying "there is no loss" without knowing how much that is would be irresponsible, so it was measured.

Two conditions are compared on **the same GPU (V100 32GB) with the same single worker**, changing only batch versus individual jobs.

| | Unit of work | Encoder | 8 images | **Per round** |
|---|---|---|---|---|
| Condition 1 | `batch_count=8` as one block | same card | 261.74s | 30.75s |
| Condition 7 | 1 image = 1 job x 8 | P104 | 265.12s | **30.78s** |

**The per-image difference is 0.03 seconds. Effectively zero.**

The 3.4-second difference in the 8-image total is not the cost of splitting but **the startup cost of putting the encoder on a separate card.** Condition 7's encoder is a P104 on PCIe x4 Gen1, which takes 5.88 seconds to upload the T5 weights. The 1-image measurement already shows a 3.2-second gap (46.51 -> 49.69), and that gap merely persists into the 8-image run without appearing in per-image cost.

> This result supports the premise of [No.3 Incremental Display](03-pipeline-throughput.md). **Because splitting the work costs essentially nothing, the human evaluation time `(N-1) x T` is recovered for free.** The same held even without sharing the conditioning tensor, re-encoding per image (condition 8 below, 0.17s per image). **With encoding on the CPU, however, it becomes 8.30s per image and the argument no longer holds.**

### First-Image Display - A Controlled Experiment (Condition 1 vs Condition 8)

The earlier comparison (conditions 1 and 4) varied **worker count and encoder placement alongside the display method**, so the effect could not be isolated. **Condition 8 was therefore created and measured with display method as the sole independent variable.**

| | Condition 1 (batch) | **Condition 8 (one at a time)** | |
|---|---|---|---|
| Card / encoder / workers | 32GB / co-located / 1 | **identical** | - |
| **First image (F)** | **261.54s** | **46.68s** | **5.6x** |
| 8 images (G) | 261.74s | **262.42s** | +0.3% |

**For condition 1, `F ~= G`.** A batch emits nothing until all eight are done, so the first-image wait is the entire time.

**Splitting into single images increased the 8-image total by only 0.3%** - and that is with re-encoding on every image.

```
Phase log for condition 8, run 1

First image  get_learned_condition   6.10s   <- cold. Includes T5 load
             sampling               38.18s
             decode_first_stage      1.70s   -> 45.99s

2nd onward   get_learned_condition   0.16s   <- the re-encoding cost
             sampling            26.99 ~ 28.90s
             decode_first_stage      1.5s     -> 28.62 ~ 30.60s
```

**Re-encoding costs 0.16~0.18 seconds per image.** Seven more times is 1.2 seconds, which disappears into the total.

> Per-image generation time climbs monotonically from 28.62s to 30.60s. That is Section 9's thermal throttling reproducing itself.

**Measurement conditions** - condition 1 was measured 2026-08-12 and condition 8 on 2026-08-19, with no hardware or stack changes in between. Condition 1 was not re-measured because its three-run spread of 0.02s (261.74 / 261.54 / 261.76) confirmed reproducibility. Condition 8's three runs were 256.58 / 262.42 / 263.17s, and the median was used.

> **For reference, the actual final configuration (condition 4) has a first image at 48.09 seconds.** Similar to condition 8's 46.68s but different in kind. Condition 8 is a diagnostic built to isolate the display-method effect; condition 4 is a real purchase option using two workers and a P104 encoder together.

---

## 9. Disconfirmation - Why the 32GB Card Was Slower

Condition 1's per-image generation was 30.75s against condition 3's 27.50s, **12% slower.** Four causes were suspected and eliminated one at a time.

| Candidate | Verification | Result |
|---|---|---|
| Memory bandwidth difference | check the specification | X V100 is 900 GB/s at both 16 and 32GB. Only stack height (4-Hi/8-Hi) differs |
| Form factor clock difference | card names via `nvidia-smi` | X every card is **SXM2** (`clocks.max.sm` 1530 MHz) |
| Incidental cost of the batch path | `sampling completed` in the log | X the entire gap sat **inside** sampling |
| VRAM pressure from co-locating the encoder | **added condition 7** | X see below |

### Condition 7 Eliminated the Last Candidate

The worker card stays as it is and **only the encoder moves to another card.**

| | Worker card | Encoder | Unit of work | **Warm sampling** |
|---|---|---|---|---|
| Condition 1 | 32GB | **same card** (19.1GB) | batch | **29.15s** |
| Condition 7 | 32GB | **separated** (13.6GB) | individual | **29.20s** |
| Condition 3 | 16GB | separated (13.6GB) | individual | **26.00s** |

**Encoder co-location and batching were both varied, yet sampling tracked only the worker card.**

### The Cause - Thermal Throttling

The cause only emerged after measuring clocks and temperature under load.

```
index, name,                    clocks.sm, clocks.max.sm, power.draw, power.limit, temp
0,     Tesla V100-SXM2-16GB,    135 MHz,   1530 MHz,      26.65 W,    300.00 W,    39
1,     NVIDIA P104-100,         139 MHz,   1911 MHz,       5.32 W,    180.00 W,    32
2,     Tesla V100-SXM2-32GB,   1200 MHz,   1530 MHz,     201.66 W,    300.00 W,    82
3,     Tesla V100-SXM2-16GB,    135 MHz,   1530 MHz,      26.63 W,    300.00 W,    39
```

Under load the 32GB card moved between **1155~1207 MHz**. Observed temperature was **82~83C**.

| Verdict | Evidence |
|---|---|
| **Thermal throttling** O | **78%** of boost. Below even the base clock (1290 MHz) |
| Power cap X | **65~100W of headroom** remained against the 300W limit |
| Specification difference X | `clocks.max.sm` is 1530 MHz on every card |

`1200 / 1530 = 78%`. Under sustained load this card uses only three quarters of its boost clock. It simply sat in a worse position inside the chassis.

### Why It Was Invisible When Cold

| | Condition 3 (16GB) | Condition 7 (32GB) | Difference |
|---|---|---|---|
| Cold first sampling | 37.75s | 39.50s | +4.6% |
| **Warm sampling** | 26.00s | 29.20s | **+12.3%** |

**Because heat takes time to accumulate.** A static specification difference would show 12% from the first image. **Widening as load accumulates is the classic signature of thermal throttling.**

### What This Disconfirmation Changed in the Documents

Had the 12% gap been written up as "thanks to module separation," that sentence would have stood unverified, and anyone else building the same structure would have failed to reproduce it. **Without designing an additional condition (condition 7) to eliminate hypotheses, it would never have come to light.**

The lesson is collected in [Pitfalls](99-pitfalls.md#cards-of-identical-specification-differ-in-sustained-performance).

---

## 10. Phase-Separated Logs

Launching `sd-server` with `-v` prints time broken out by phase. **Had only total time been recorded, the disconfirmation above would have been impossible.**

### Warm Sampling Comparison (8-image run, third repeat)

```
Condition 1 (32GB, encoder co-located, batch)
  38.75  28.45  29.01  29.18  29.15  29.22  29.14  29.23     -> warm mean 29.15

Condition 7 (32GB, encoder separated, individual)
  39.50  28.44  29.12  29.12  29.32  29.26  29.30  29.32     -> warm mean 29.20

Condition 3 (16GB, encoder separated, individual)
  37.75  25.71  25.93  26.02  26.05  26.12  26.08  26.10     -> warm mean 26.00
```

**The first image is more than 10 seconds slower under every condition.** That is GPU clock ramp-up, and it appears identically regardless of card.

### Full Log of Condition 7's 1-Image Run

```
--- Encoder (P104) ---
loading tensors completed, 0.40s (read 0.12, copy_to_backend 0.18)      <- CLIP-L
loading tensors completed, 6.47s (read 0.60, copy_to_backend 5.50)      <- T5 Q8
sd_encode_conditioning completed, 8.97s

--- Worker (V100 32GB) ---
get_learned_condition completed, 0.02s                                  <- conditioning injection
loading tensors completed, 11.31s (read 10.21, copy_to_backend 0.46)    <- DiT 12GB
sampling completed, 38.92s
loading tensors completed, 0.20s                                        <- VAE
decode_first_stage completed, 1.59s
generate_image completed in 40.55s
```

Three things to read here:

| Value | Meaning |
|---|---|
| `get_learned_condition 0.02s` | **injecting the conditioning tensor costs essentially nothing.** The premise of the encoder separation design |
| T5 `copy_to_backend 5.50s` | the price of PCIe x4 Gen1. **Occurs once, at startup** |
| DiT `read 10.21s` | the cold start dropped the page cache, so it reads from disk |

### PCIe Bandwidth Reference

| Transfer | Frequency | V100 (x16 Gen3) | P104 (x4 Gen1) |
|---|---|---|---|
| T5 weights 5.06GB | once at startup | 0.58s | **5.88s** |
| Effective transfer rate | - | 8.7 GB/s (55% of theoretical) | **0.86 GB/s (86% of theoretical)** |
| Conditioning tensor 4.2MB | per request | 0.5ms | ~5ms |

**In exchange for spending 5.3 seconds more once at startup, the running cost is effectively zero.** Against the 240 seconds for eight images, conditioning transfer is about 5ms per request, roughly 0.02% even summed across all eight.

---

## 11. What Was Not Measured

Recorded honestly. The following cannot be answered by this experiment.

| Item | Reason | Effect |
|---|---|---|
| **Condition 5 (3 workers)** | three V100 16GB cards were not available | a derived value and an optimistic upper bound (4-2) |
| **Human evaluation time T** | varies widely with task, skill, and fatigue | [No.3](03-pipeline-throughput.md)'s time model is a structural comparison with T as an unknown |
| **Warm-state performance** | no separate measurement was made | estimable only by subtracting load and ramp-up phases from the cold logs |
| **Concurrent multi-user requests** | measured for a single user | encoder serialization could become the bottleneck -> [No.4 Section 3](04-orchestration.md#3-the-order-in-which-bottlenecks-move-as-n-grows) |
| **Other resolutions and step counts** | fixed at 768x768 / 15 steps | higher resolution raises latent patch count, so per-image time grows nonlinearly |
| **An environment with even cooling** | this server's 32GB card was thermally constrained | conditions 1, 6, and 7 have room to improve with better cooling (Section 9) |
| **Direct comparison against an A100 or other higher-tier card** | no published data measured with the same engine | see below |

### 11-1. Why No Comparison Against Higher-Tier Cards

An attempt was made to put A100 measurements alongside this table, and **abandoned.** Published A100 figures come from conditions that are too different.

| Variable | This project | Published A100 figures |
|---|---|---|
| Framework | ggml / stable-diffusion.cpp | PyTorch + diffusers, frequently with `torch.compile` |
| Precision | Q8 | bf16 |
| Resolution | 768^2 (2,304 latent patches) | 1024^2 (4,096) |
| Steps | 15 | 28 (dev) or 4 (schnell) |
| Extension | PuLID enabled (+22%) | none |

Force a normalization and the apparent gap comes out at **8x or more**, while **the A100's compute ratio is only 2.5x.** The remaining 3x-plus is framework, optimization, and precision - not hardware. **Put side by side in a table, it would overstate the A100's performance by more than 3x.**

> **This project did rent an A100 in the cloud and try it.** But only **whether the unquantized model ran** was confirmed; no performance was measured, so there is nothing usable for comparison -> [Appendix - Build Log](appendix/build-log.md#how-it-started)

> **No A100 data measured with the same engine (ggml) could be found.** This looks structural rather than a failure of searching - A100 users have VRAM to spare and therefore no reason to quantize, so they use PyTorch rather than ggml. **Quantization plus ggml is the combination of those who are short on VRAM.**

Comparison against higher-tier cards was done **from specifications instead of measurement.** Image generation is compute-bound, so the compute ratio (2.5x) and the price ratio (15.7x) are enough to settle it -> [price per unit of throughput](00-method.md#4-price-per-unit-of-throughput)

**Not pretending to measure what cannot be measured is better than putting numbers from different conditions side by side.**

---

## 12. SM Scaling and Kernel Path Measurements

This section records **the verification of two questions raised while writing the documents.** One was "is a total-throughput-divided-by-core-count average meaningful," the other "does the P104's FP16 cap actually apply." **The second answer overturned statements in earlier sections.**

At measurement time every service was down and all four GPUs were idle. Temperatures were 37~44C with SM clocks pinned at 1,530MHz, so no throttling was involved.

### 12-1. SM Scaling - Is Performance Linear in Core Count?

On the V100 32GB, 512-token encoding was measured across MPS SM allocation ratios. sd-server was relaunched at each point (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is read only when the client starts) and the first run, which mixes in model loading, was discarded.

| Setting | Effective SMs | Encoding | vs 100% | If linear |
|---|---|---|---|---|
| No MPS | 80 | 0.199s | 0.99x | - |
| MPS 100% | 80 | 0.200s | 1.00x | - |
| MPS 50% | 40 | 0.245s | 1.22x | 2.00x |
| MPS 25% | 20 | 0.398s | 1.99x | 4.00x |
| MPS 12.5% | 10 | 0.686s | 3.43x | 8.00x |

**Read as-is, it is nonlinear.** SMs were cut to one eighth while time grew only 3.43x.

#### Separate the Fixed Cost and It Is Linear

A least-squares regression on `time = fixed cost + k/SM` gives:

```
fixed cost c = 0.117s
coefficient k = 5.67
```

| SMs | Measured | Model prediction | Error |
|---|---|---|---|
| 80 | 0.200s | 0.188s | +6% |
| 40 | 0.245s | 0.259s | -5% |
| 20 | 0.398s | 0.400s | **-0.5%** |
| 10 | 0.686s | 0.684s | **+0.3%** |

**The compute portion is exactly inversely proportional to SM count.** What looked nonlinear was a fixed cost of 0.117s, independent of SMs, sitting on top of every measurement.

The identity of that fixed cost is confirmable too. This measurement includes the HTTP round trip and **the time to write the 8.39MB conditioning gguf to `/dev/shm`.**

```
80 SM compute portion (regression model)   0.071 ~ 0.083s
V100 warm encoding from 7-2                   0.085s    <- the pure encoding phase
difference                                 0.115s  ~=  the regression's fixed cost of 0.117s
```

**Values from two independent routes agree.**

> **So "total throughput / core count = per-core average" is a valid metric.** Efficiency did not collapse while cores were cut to one eighth.
>
> That said, **CUDA cores and CPU cores differ structurally and do not admit a 1:1 comparison.** This average is used **only to explain the gap between 240x the core count and 25x measured** in [No.2 Section 3](02-encoder-separation.md#why-25x).
>
> **Limitation -** this check was done on a V100 (Volta). **The P104 (Pascal) does not support MPS SM ratio limiting, so the same verification could not be run there.** Also note that MPS ratio limiting does not physically disable SMs but restricts allocation, so clocks and bandwidth remain intact.

Two incidental observations. **MPS's own overhead was effectively zero** (no MPS 0.199s vs MPS 100% 0.200s). And cold start was faster with MPS (6.12s vs 2.96s).

### 12-2. Kernel Path - The FP16 Cap Never Applied

ggml kernel throughput was measured directly with `test-backend-ops perf -o MUL_MAT`. To avoid mixing with the production build (`build/`, sm_70), a separate `build-bench/` was built with `"61;70"`.

#### The Actual Encoding Shape (GEMM, m=4096 n=512 k=14336)

| | f16 | q8_0 | Winner |
|---|---|---|---|
| **P104** (Pascal, no tensor cores) | 4.34 TFLOPS | **9.42 TFLOPS** | **q8_0 by 2.17x** |
| **V100** (Volta, tensor cores) | **65.71 TFLOPS** | 53.18 TFLOPS | **f16 by 1.24x** |

Set against theoretical figures, each value reveals which path it took.

```
P104 f16   4.34 / 6.1 TFLOPS(FP32)     = 71%   -> FP32 path
P104 q8_0  9.42 / 22 TOPS(INT8)        = 43%   -> DP4A working
V100 f16   65.71 / 125 TFLOPS(FP16 TC) = 53%   -> FP16 tensor cores
V100 q8_0  53.18                               -> dequantize then FP16. That is the loss
```

#### The LLM Decoding Shape (GEMV, n=1)

| | f16 | q8_0 | Winner |
|---|---|---|---|
| P104 | 294.78 GFLOPS | 432.04 GFLOPS | q8_0 by 1.47x |
| V100 | 840.71 GFLOPS | 1.41 TFLOPS | q8_0 by 1.68x |

**This phase is memory-bound, so whichever reads half the bytes wins.** The compute path is irrelevant.

#### Measured End-to-End, the Direction Is the Same

A kernel benchmark measures only a single GEMM shape. It was confirmed with **total elapsed time for pushing 512 tokens through the encoder** as well. T5 was loaded alternately onto the V100 32GB, measured five times each, with the first run discarded.

| V100, 512 tokens | Encoding | Cold start |
|---|---|---|
| q8_0 (5.20 GB) | 0.199s | **2.98s** |
| **fp16 (9.79 GB)** | **0.169s** | 8.46s |

**On compute, fp16 is 1.18x faster.** Removing the 0.117s fixed cost (12-1) leaves 0.082s against 0.052s for the compute portion alone, **a factor of 1.58** - a wider gap than the kernel benchmark's 1.24x.

**The reason reading half the weights does not pay off in compute is that encoding is compute-bound** (arithmetic intensity 498). What it does buy is **startup: q8_0 is 2.8x faster** - the difference of having to read 9.79GB shows up right there.

> **The point of this table is that quantization's benefit lies in loading, not in compute.** The reason q8_0 was mandatory on the P104 was likewise not speed but **whether it fit in 8GB.**

#### Whether Quantization Wins Depends on Card and Shape Together

| | GEMV (memory-bound) | GEMM (compute-bound) |
|---|---|---|
| **P104** | q8_0 by 1.47x | q8_0 by 2.17x |
| **V100** | q8_0 by 1.68x | **f16 by 1.24x** |

**Only one of the four favors f16.** The three paths that [No.1 5-1](01-role-assignment.md#5-1-three-paths-for-quantized-matrix-multiplication) inferred from source analysis are confirmed here by measurement.

One incidental value: the gap between the two cards on the same shape.

```
by f16    65.71 / 4.34  = 15.1x
by q8_0   53.18 / 9.42  =  5.6x
```

**Quantization narrows the gap between heterogeneous cards from 15.1x to 5.6x.** This is the quantitative reason work could be handed to the weaker card.

### 12-3. What This Measurement Overturned

**This document once stated that "using fp16 on a P104 takes about 24 seconds."** That was the spec sheet's "FP16 is 1/64 of FP32" multiplied straight through.

| P104, GEMM shape | Inferred from spec | Measured |
|---|---|---|
| f16 | 0.095 TFLOPS | **4.34 TFLOPS** |
| Ratio | | **45x** |

**The cap never applied.** ggml does not compute f16 tensors with FP16 arithmetic. Pascal has no tensor cores and cannot take the cuBLAS FP16 path, so **f16 is converted to FP32 and processed there.** The measured 4.34 TFLOPS being 71% of the FP32 theoretical figure is the evidence.

**The 24-second figure had no basis and was deleted throughout the documents.** The reason q8_0 was needed was corrected to **fit** rather than speed - T5-XXL at fp16 is 9.79GB and does not go on an 8GB card.

> **This error is precisely the trap [No.1 5-2](01-role-assignment.md#5-2-a-kernel-path-cannot-be-settled-by-timing) warned about.** That section said *"a kernel path cannot be settled by timing"* - and **it could not be settled from specifications either.** There is no route other than reading the source or measuring it directly.

### 12-4. Reproduction

MPS scaling and end-to-end encoding are measured with [`bench/mps_scaling.py`](../../bench/mps_scaling.py); kernel throughput with llama.cpp's `test-backend-ops`.

```
python3 mps_scaling.py --no-mps --repeat 5 --t5 <T5 path>
```

```
cd /root/llama.cpp && cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70" -DLLAMA_BUILD_TESTS=ON
```

```
CUDA_VISIBLE_DEVICES=<card UUID> ./build-bench/bin/test-backend-ops perf -o MUL_MAT
```

---

## 13. Reproduction

The scripts are in [`bench/`](../../bench/). Only the GPU UUIDs and model paths need changing.

```
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

Put your own UUIDs into `GPU32` / `GPU16A` / `GPU16B` / `GPUP104` at the top of `bench.py`. **For why UUIDs rather than indices, see [No.4 Section 1](04-orchestration.md#1-always-specify-gpus-by-uuid).**

```
python3 bench.py --condition 3 --count 8 --repeat 3
```

Unattended run of every combination:

```
nohup python3 bench_all.py > bench/console.log 2>&1 &
```

Reprint the summary (recalculated from `results.csv`, so it can be run any number of times):

```
python3 bench_all.py --summary
```

**Root privileges are required, because the cold start writes to `/proc/sys/vm/drop_caches`.**

**Always log clocks and temperature during the run.** In a separate terminal:

```
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv -l 5
```

As Section 9 shows, failing to measure this **leads to attributing the cause to the wrong thing.**
