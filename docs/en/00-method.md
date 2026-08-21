**English** / [한국어](../ko/00-method.md)

# Workload Analysis - The Math to Do Before Choosing a Card

> This document is the **common methodology** shared by the other three.
> It covers how to settle "which GPU should I buy" on paper, without benchmarking.
> The basis is a real build - EPYC 7232P / V100 32GBx1 + V100 16GBx2 + P104-100 8GB / Ubuntu 24.04 -
> and every figure without a note attached is measured on that machine.

Related documents: [No.1 Role Assignment](01-role-assignment.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.3 Pipeline Throughput](03-pipeline-throughput.md) / [No.4 Orchestration](04-orchestration.md) / [Appendix - Benchmarks](98-benchmark.md) / [Pitfalls](99-pitfalls.md)

---

## Contents

1. [Break the Workload Down in Detail](#1-break-the-workload-down-in-detail)
2. [Bottleneck Classification Can Be Settled by Calculation](#2-bottleneck-classification-can-be-settled-by-calculation)
3. [Generalizing - The Roofline Model](#3-generalizing---the-roofline-model)
4. [Price per Unit of Throughput](#4-price-per-unit-of-throughput)
5. [Candidate GPU Reference Table](#5-candidate-gpu-reference-table)
6. [Full Procedure Checklist](#6-full-procedure-checklist)

---

## 1. Break the Workload Down in Detail

The very first thing to do is decompose the work you intend to run into phases and identify what resource each phase consumes.

```
1. List every task you have to run           (LLM inference / image generation / encoding ...)
2. Decompose each task into phases            (compute / memory bandwidth / interconnect / storage I/O)
3. Compute each phase's share and classify the bottleneck
4. Substitute candidate GPUs and derive "price per unit of throughput"
5. Check whether any module can be separated
```

Only after this can you derive the hardware specification you need. Buying hardware before doing this leaves you with either **"cheap hardware with nothing to run on it"** or **"expensive, over-specified hardware."**

---

## 2. Bottleneck Classification Can Be Settled by Calculation

> Note: the figures below are measurements and estimates for this project's models (Qwen 3.6 27B Q6 / FLUX D Q8) on a V100 32GB.
> **Different hardware or different work changes both the ratios and the numbers.** Take the procedure, then refill it with your own machine's values.

**LLM inference (decoding)** - at 30 t/s, 33.3ms per token

```
33.3 ms/token ~= memory reads 27.8ms (84%) + compute 0.4ms (1%) + overhead 5ms (15%)
```

**Image generation (FLUX D)** - 26.9s per image once prompt encoding is done (768x768 / 15 steps / PuLID enabled)

```
26,900 ms/img ~= memory reads 1,200ms (4%) + compute 24,300ms (90%) + VAE 1,400ms (5%)
```

> **Every per-image time in this document assumes PuLID is enabled.** PuLID (the face-consistency extension) adds 22% per step (1.39 -> 1.70 s/it). With it disabled, the same conditions give 22.2s. Mixing baselines leads to over- or under-stating the improvement -> [Pitfalls - Measurement and Verification](99-pitfalls.md#5-measurement-and-verification)

**Comparing the two: LLM inference spends 84% of its time on memory reads, image generation spends 90% on compute.** Skip this calculation up front and you can end up paying for a compute-heavy GPU to serve an LLM that is actually bandwidth-bound.

The underlying arithmetic:

| | LLM decoding (one token) | Diffusion (one step) |
|---|---|---|
| Compute | 2 x 27B = **54 GFLOP** | 2 x 12B x 2,560 tokens + attention = **66,000 GFLOP** |
| Memory traffic | 21 GB (all weights, once) | 60.5 GB (Q8 read + fp16 conversion/re-read) |
| Compute / data size (= arithmetic intensity) | **2.6 FLOP/byte** | **1,091 FLOP/byte** |

> Conditions apply to this table -> [Notes and Caveats 1)](#notes-and-caveats)

LLM decoding reads the weights once to produce **one** token; diffusion reads them once to process **2,304 latent patches**. The result is a **420x spread in arithmetic intensity** (2.6 -> 1,091), which puts the bottleneck in an entirely different place.

---

## 3. Generalizing - The Roofline Model

That classification generalizes using **two card specs and two workload properties**. The method is called the [**roofline model**](https://en.wikipedia.org/wiki/Roofline_model), and it compares two values - **arithmetic intensity** and the **ridge point** - to determine which resource is the ceiling.

```
arithmetic intensity = workload FLOP / workload memory traffic
ridge point          = card compute throughput / card memory bandwidth

arithmetic intensity < ridge point  ->  memory-bound  ->  needs a card with wide bandwidth
arithmetic intensity > ridge point  ->  compute-bound ->  needs a card with strong compute
```

Both come out in `FLOP/byte`. **On the workload side, a larger number means the work leans on compute; on the card side, a larger number means the card's specs lean toward compute over memory.**

**The closer the two values, the better that GPU fits that work.** When they diverge, whichever resource is in surplus sits idle.

| Card | Compute | Bandwidth | **Ridge point** |
|---|---|---|---|
| V100 | 125 TFLOPS (tensor FP16) | 900 GB/s | **139** |
| **P104-100** | **22 TOPS (INT8 DP4A)** | **320 GB/s** | **68.8** |
| CMP 170HX (converted) | 7.7 TFLOPS (tensor FP16, crippled) | 1500 GB/s | **5.1** |

| Workload | Arithmetic intensity | vs V100 ridge point | vs 170HX ridge point | Classification |
|---|---|---|---|---|
| LLM decoding | 2.6 | 0.019x | **0.51x** | **170HX is the closer fit** |
| **Text encoding (T5)** | **498** | 3.6x | 98x | compute-bound |
| Diffusion step | 1,091 | **7.8x** | 214x | **V100 is the closer fit** |


> **Caveats attach to this classification. Do not draw conclusions from the table alone** -> [Notes and Caveats 2)](#notes-and-caveats)

**This calculation is possible before you own any hardware.** Cards with completely different characteristics can sit at similar prices, so it belongs in the planning stage.

Note: for example, a V100 32GB runs about 700 USD and a CMP 170HX about 1,000 USD - **a price difference of roughly 30% measured against the 170HX** - yet **their ridge points differ by more than 20x on paper.**


---

## 4. Price per Unit of Throughput

```
price per unit of throughput = card price / throughput on that card
```

Measured examples from this project (image generation with PuLID enabled; encoding measured [in isolation](99-pitfalls.md#5-measurement-and-verification) at 256 tokens standalone):

| Task | Card | Throughput | Price | Per unit |
|---|---|---|---|---|
| Image generation **(1)** | V100 16GB | 26.9 s/img = 0.037 img/s | 200 USD | ~5,400 USD per (img/s) |
| Image generation | V100 32GB | 26.9 s/img = 0.037 img/s | 700 USD | ~18,900 USD per (img/s) |
| Text encoding | V100 16GB | 0.085 s/call = 11.7 calls/s | 200 USD | ~17 USD per (call/s) |
| Text encoding | **P104-100 8GB** | 0.33 s/call = 3.0 calls/s | **15 USD** | **~5 USD per (call/s)** |

> **(1)** This row presumes [No.2 Encoder Separation](02-encoder-separation.md) -> [Notes and Caveats 3)](#notes-and-caveats)

### Do Not Stop at Per-Module Comparison

The table above looks at a single module. **Converted to whole-system terms, the gap widens considerably.**

| | Encoding time | 8-image total | Saved vs previous | Added cost | **Saved per USD** |
|---|---|---|---|---|---|
| CPU (under contention) | 24.5s | 132s | - | - | - |
| **P104-100 added** | **0.86s** | **108s** | **23.6s** | **15 USD** | **1.58 s/USD** |
| V100 added (hypothetical) | 0.17s | 107.3s | 0.69s | 185 USD more | **0.0037 s/USD** |

**The cost efficiency differs by roughly 420x.** Spending 15 USD to save 23.6 seconds and spending another 185 USD to save 0.69 seconds are not the same kind of decision.

### There Are Two Axes of Comparison

Price per unit of throughput only looks at getting **the same output more cheaply**. A real purchase decision has a second axis running the other way.

```
Axis A  same throughput -> how much does cost drop?
Axis B  same cost       -> how much does throughput rise?
```

In this project, **the mix of low-cost cards beat the single high-cost card on both axes.** The baseline is a single V100 32GB (700 USD) with the image model loaded whole, which took 261.7 seconds for 8 images.

| | Configuration | Cost | 8 images | Per image |
|---|---|---|---|---|
| *estimate* | A100 80GB x1 | *11,000 USD* | *~110s* | *13.8s* |
| **baseline** | V100 32GB x1 | 700 USD | 261.7s | 32.7s |
| **A** | V100 16GB x1 + P104 | **215 USD** | 240.7s | 30.1s |
| **B** | V100 16GB x2 + P104 | **415 USD** | **134.0s** | 16.8s |
| *C* | V100 16GB x3 + P104 | *615 USD* | *105~107s* | *13.2s* |

**A is axis A (same throughput); B and C are axis B (same cost).** The A100 and C rows are estimates; the derivation and the thermal-throttling caveat are in [Appendix - Benchmarks](98-benchmark.md).

**One prerequisite had to hold for this result.** A 200 USD 16GB card was not a candidate at all because the model would not fit; only after [module separation](02-encoder-separation.md) lowered the minimum VRAM requirement from 19.1GB to 13.6GB did it become an option.

**There is a definite order between choosing cards and partitioning work - you can only choose after you have partitioned.** That said, the procedure does not finish in a single pass, and **there are good reasons to run the whole loop several times** ([Section 6](#6-full-procedure-checklist)).

---

## 5. Candidate GPU Reference Table

**The values below are reference figures from published specifications.** Converted and used cards vary widely, and **GPU prices reflect the author's region and the date of writing, so treat them as reference only and refill the column with your own market's prices.**

(As of 2026-08-11, South Korea)

| GPU | Architecture | VRAM | Memory bandwidth | FP16 | INT8 DP4A | Price (USD) |
|---|---|---|---|---|---|---|
| GTX 1080 (retail) | Pascal sm_61 | 8GB | ~=320 GB/s | **crippled to 1/64** | O ~=22 TOPS | 80 |
| **P104-100** | Pascal sm_61 | 8GB | ~=320 GB/s | **crippled to 1/64** | O ~=22 TOPS | **15** |
| **V100 16GB** | Volta sm_70 | 16GB | 900 GB/s | 1st-gen tensor ~=125 TFLOPS | O | **200** |
| **V100 32GB** | Volta sm_70 | 32GB | 900 GB/s | 1st-gen tensor ~=125 TFLOPS | O | **700** |
| RTX 2080 Ti 22GB (converted) | Turing sm_75 | 22GB | 616 GB/s | 2nd-gen tensor | O | 350 |
| RTX 3090 | Ampere sm_86 | 24GB | 936 GB/s | 3rd-gen tensor ~=142 TFLOPS | O | 800 |
| CMP 170HX (converted) | Ampere sm_80 | 64GB (reported) | ~=1500 GB/s | **tensor crippled** | ? | 1,000 |
| A100 80GB | Ampere sm_80 | 80GB | ~=2000 GB/s | 3rd-gen tensor ~=312 TFLOPS | O | 11,000 |

> **"Crippled to 1/64" in the table is a published specification, not effective performance.** In this project's measurements the P104's f16 ran at 4.34 TFLOPS - **45x** the 0.095 TFLOPS the spec implies. With no tensor cores, f16 is converted to FP32 and processed there, so **the cap never gets a chance to apply.** When screening cards by spec sheet, check whether that path is actually the one taken -> [Appendix 12-2](98-benchmark.md#12-2-kernel-path---the-fp16-cap-never-applied)

Build a table like this one, **first eliminating cards that cannot meet the workload's minimum requirement** (a pass/fail test, not a ratio), then fill in your own market's prices and cross-check against the bottleneck calculation above to get **the best card per task**. [Notes and Caveats 4)](#notes-and-caveats)

Running the same calculation for image generation gives this:

| | Compute ratio | Price ratio | Verdict |
|---|---|---|---|
| A100 80GB vs V100 32GB | **2.5x** (312 vs 125 TFLOPS) | **15.7x** | inefficient |

**On both workloads the A100 offers 2-3x the performance at 15x the price.** This calculation was the basis for passing on the higher-tier card, and **the conclusion was reached from specifications alone, with no hardware on hand.**

> Why this calculation could not be cross-checked against published A100 measurements is in [Notes and Caveats 5)](#notes-and-caveats).

---

## 6. Full Procedure Checklist

```
0. Workload analysis
   - List every task to be run
   - Decompose each task into phases (compute / memory bandwidth / interconnect / storage I/O)
   - Compare arithmetic intensity against card ridge points to classify the bottleneck
       note: calculate using the compute path the work actually takes
   - Check whether any module can be separated
   - Collect related material and prior cases

1. Fix the budget and the hardware
   - Set a budget ceiling. If exceeded, reconsider as far as moving to the cloud
   - Derive "price per unit of throughput" for each candidate GPU -> decide the best card per task
       note: judge by contribution to whole-system time, not per module
   - Confirm a driver branch exists that covers every target card (an absolute requirement)
   - Confirm how long that branch and its CUDA version will be maintained
   - Check FP32 / FP16 scalar / FP16 tensor / INT8 throughput for each card separately
   - Check PCIe slots and lanes, power capacity, and board BIOS support

2. OS and drivers
   - Finish apt update && apt upgrade first
   - Install the driver, confirm every card is detected and named via nvidia-smi
   - Record PCIe link width and generation at the same time
   - Pin driver packages when registering the CUDA apt repository
   - Install the CUDA toolkit

3. Build
   - Put every target architecture in CMAKE_CUDA_ARCHITECTURES (quote the value)
   - Verify kernel selection conditions such as INT8/MMQ in the source

4. Role assignment
   - Collect UUIDs and record them in configuration as a dictionary
   - Assign cards to tasks according to the bottleneck classification from the analysis
   - Choose the quantization format that matches what each card is good at

5. Module separation (where applicable)
   - Clear the investigation gate
   - Implement serialization and injection, then verify pixel-identical output via CLI
   - Wrap it in an HTTP endpoint
   - Convert the orchestrator to the smallest unit of work

6. Measurement
   - Stop other services and measure in isolation
   - Three runs on identical input; discard the first, use the second and third
   - Explicitly record the baseline's conditions (which features were enabled)
   - Measure time-to-first-result separately from total time
   - If human evaluation is part of the workflow, account for the overlap effect as well
```

Pitfalls specific to the measurement stage are collected in [Pitfalls - Measurement and Verification](99-pitfalls.md#5-measurement-and-verification). **They bear directly on the reliability of every figure in this document, so read them alongside it.**

### It Does Not Finish in One Pass

**Running the procedure once is rarely enough.** While working through steps 1 and 2 and collecting material and prior cases, new GPUs or new solutions turn up almost without exception, and **choices later in the flow do affect earlier ones.**

That said, **repeating the procedure is itself a cost.** Repeat it only to a reasonable depth.

---

## Notes and Caveats

1) **The arithmetic intensity table in Section 2 is for the model itself, without PuLID.** Enabling PuLID adds 22% compute only, which raises arithmetic intensity further, so the classification (compute-bound) does not change. The 2,560 tokens for diffusion is the sequence length: 2,304 latent patches (768^2 / 16^2) plus 256 tokens of text conditioning.

2) **A ridge point must be calculated using the compute path the work actually takes.** The ridge point table in Section 3 uses tensor FP16, but LLM decoding runs at batch 1, making it a matrix-vector product (GEMV) that **barely touches the tensor cores.** The 170HX's real suitability may therefore be better than the 0.51x shown. The 170HX figures are community-reported and need verification in their own right. For kernel path determination see [No.1 Role Assignment - Where Performance Actually Diverges](01-role-assignment.md#5-where-performance-actually-diverges-across-heterogeneous-gpus).

3) **(1)** FLUX D Q8 loaded whole needs 19.1GB and simply does not fit on a 16GB card. Only after the encoder is split off and the worker drops to 13.6GB does a 16GB card become a candidate. **The conclusion "the same work on a cheaper card" rests on module separation as a prerequisite.**

4) At the time of writing, the 170HX can be judged better suited to LLM inference than a V100 32GB. **At the time the server was built, however, prices moved too violently for that judgment to be made, the success rate of the conversion was not 100%, and factoring in planned future training work, it was not adopted in the final configuration.**

5) An attempt was made to cross-check the calculations in Sections 4 and 5 against published A100 measurements, but **framework, model variant, resolution, and precision all differed, making it impossible to isolate the hardware difference** -> [Appendix 11](98-benchmark.md#11-what-was-not-measured)
