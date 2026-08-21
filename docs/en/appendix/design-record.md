**English** / [한국어](../../ko/appendix/design-record.md)

# Appendix - Design Record

> A record of **the order in which [No.2 Encoder Separation](../02-encoder-separation.md) was carried out.**
> Where the main document explains the finished structure, this one covers **the stages leading to it and the stop criterion at each stage.**
> Work that touches the inside of an inference engine has a high chance of failing, so **deciding in advance where to stop** was half the design.

Related documents: [No.2 Encoder Separation](../02-encoder-separation.md) / [No.4 Orchestration](../04-orchestration.md) / [Appendix - Benchmarks](../98-benchmark.md) / [Build Log](build-log.md)

---

## Why It Was Split into Phases

The nature of this work was as follows.

```
modify the inside of an inference engine      <- may hit a wall
buy new hardware                              <- hard to return
change a service that is in production        <- breakage is immediately visible
```

Three risks overlapping, so **the cheapest and most reversible items went first.**

| Phase | Content | Cost | Loss on failure |
|---|---|---|---|
| **0** | investigation / quantization A/B | **0** (no card needed, no service impact) | time only |
| 1 | hardware preparation | card purchase | 15 USD |
| 2 | engine modification | time | recoverable from backup |
| 3 | orchestrator integration | time | recoverable from backup |
| 4 | verification and cutover | - | rollback |

**The existing service was left untouched until the final stage.** The old structure does not come down before the new one is verified.

---

## Phase 0 - Investigation (No Card Needed, No Service Impact)

### 0-1. Investigate the Engine's Internals [important] First of All

**Hit a wall here and the entire plan collapses, so this came first.** Do it before buying a card, before editing a line of code.

What was checked, and the **stop criteria**:

| Check | Why it matters | Stop criterion |
|---|---|---|
| Is the conditioning a value type | pointers or handles cannot be serialized | a handle means **abandon the design** |
| Is conditioning produced in one place | several places means the modification scope explodes | if scattered, reconsider |
| Is the injection point unambiguous | determines the amount of plumbing | - |
| Is conditioning reused across the whole batch | otherwise separation gains nothing | recomputed per image means **no gain** |
| Do other extensions touch the conditioning | if they do, separation is impossible | if they do, reconsider |
| How many fields are actually populated | fixes what has to be serialized | - |

**All items passed.** The last one mattered most: the conditioning struct had more than ten fields, but only **two were actually used - the hidden states and the pooled vector.** The rest were for other model families.

> This investigation came back later as **after-the-fact verification.** The serialized file size matched `4096 x tokens x 4 + 768 x 4 + header` to the byte, which **confirms arithmetically that the file contains exactly two tensors.**

### 0-2. Quantization A/B - On the Current Stack, With No Card

Putting the encoder on an 8GB card requires quantization. **But if quantization ruins quality, the whole plan is pointless.** That was checked on the existing stack before buying a card.

```
sd-cli -M convert -m t5xxl_fp16.safetensors -o t5xxl-q8_0.gguf --type q8_0
```

fp16 and q8_0 were compared at identical prompt, seed, and step count.

> **The values below are exactly as measured at the time.** q8_0's 13.97 seconds later came out as **10.09 seconds** in a controlled re-measurement (same default settings). This document records the design process, so the original values stay, but **for currently valid figures see [Appendix 7-3](../98-benchmark.md#7-3-why-25x-faster-than-cpu---arithmetic-intensity-and-ridge-point).**

| Item | fp16 | q8_0 | Change |
|---|---|---|---|
| CPU encoding | 25.09s | **13.97s** | -44% |
| Model load | 7.32s | 3.83s | -48% |
| RAM | 9,084MB | **4,826MB** | **-4.26GB** |
| Sampling | 1.39 s/it | 1.39 s/it | **unchanged** |

On quality, the **mean pixel difference was 0.72/255** with a maximum of 149. Semantic elements such as color, composition, and prop placement were preserved, and only **inherently unstable regions** like fingers and face shape diverged. The embeddings shift slightly, which forks the diffusion trajectory.

**Always compare A/B at the pixel level.** Byte-comparing PNGs always reports a difference because of metadata chunks.

```
python3 -c "from PIL import Image; a=Image.open('a.png').tobytes(); b=Image.open('b.png').tobytes(); print('IDENTICAL' if a==b else 'DIFFERENT')"
```

**There was a side benefit.** Quantization paid off immediately on the stack of the time, with no new card - RAM halved and CPU encoding got 44% faster. **Even if Phase 1 had failed, Phase 0's output would have remained.** Arranging the phases this way lowers risk.

### 0-3. Confirm the Kernel Path from the Source [important] No Card Needed

The basis for handing work to a 15 USD card was **"INT8 operations run at full speed."** But the hardware having a feature and the framework taking that path are **different questions.**

The decision function was read directly.

```c
if (turing_mma_available(cc)) return true;
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) return false;   // 610
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;   // 64
}
```

- **P104 (sm_61)** - no FP16 tensor cores, so `!false = true`. **It takes the INT8 path always, regardless of batch size**
- **V100 (sm_70)** - has tensor cores, so the `ne11 < 64` condition applies. At 512 tokens it is false, giving the dequant + cuBLAS FP16 path

**This is the reverse of intuition.** The older card is the one that deterministically takes the INT8 path.

**One mandatory condition came out of this.** If `ggml_cuda_highest_compiled_arch(cc) < 610`, the INT8 path is disabled. The build at the time was `CMAKE_CUDA_ARCHITECTURES=70`, meaning **it would not even run on a P104.**

**This was not found in advance, though.** It was confirmed only after installing the card and finding it would not run, and the build was redone -> [Build Log 4-1](build-log.md#4-1-build---include-every-target-architecture). **The condition was already in hand from reading the source, but it was never connected to the build configuration.**

> **Timing could not have settled this.** The measured 0.33 seconds sits between the INT8 theoretical figure (109ms) and the FP32 fallback theoretical figure (393ms), which makes them indistinguishable. **The primary evidence is source analysis; timing is supporting material for checking consistency** -> [No.1 5-2](../01-role-assignment.md#5-2-a-kernel-path-cannot-be-settled-by-timing)

---

## Phase 1 - Hardware Preparation

### 1-1. Switch GPU Designation to UUIDs [important] **Before** Installing Cards

**Order matters.** Installing a card shifts the indices, so the switch to UUIDs has to happen before they shift. They shifted like this in practice.

| | Before install | After install |
|---|---|---|
| V100 32GB (LLM) | index 1 | **index 2** |
| V100 16GB | index 0 | index 3 |
| V100 16GB | index 2 | index 0 |

**Had it been installed with indices in the configuration, the LLM would have tried to load onto a 16GB card and hit OOM.**

### 1-2. Multi-Architecture Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70"
```

The draft design **intended two separate builds** - `build-p104` (sm_61) for the encoder and `build` (sm_70) for the worker - on the grounds that compile time and library size would halve.

**In practice a single binary was used.** The advantages in operation and version tracking were larger.

| | Single binary | Per-architecture builds |
|---|---|---|
| Configuration | one path | a binary specified per service |
| Running the wrong binary | impossible | **possible (fails silently or runs slow)** |
| Version tracking | one commit = one state | trees can drift out of sync |

**"Half the build time" is a cost paid once; "you can run the wrong binary" is a risk that stays forever.**

### 1-3. The Measurement Gate [important]

With the card installed, **encoding time was measured first of all.** Falling short of expectations here was the point to turn back.

Warm, 256 tokens took **0.33 seconds.** Against the CPU value of the time (13.97s) that was judged as 42x and passed. **A later controlled re-measurement put the CPU's best at 8.30 seconds, making the real gap 25x** - which did not overturn the verdict.

---

## Phase 2 - Engine Modification

### 2-1. Serialization - Not Interpreting Is the Key

The decision was **not to interpret** the axis convention (row-major versus the framework's own ordering).

```
round-trip the shape vector and the flat data exactly as they are
```

Assign no meaning and the data is restored exactly whichever convention applies. **If the axes get transposed, images break silently and debugging is extremely tedious.** Attempting to interpret would have cost a great deal of time here.

There was one trap. `ggml_n_dims()` **truncates trailing dimensions of size 1.** The original dimension count has to be recorded separately in the metadata for the round trip to be exact.

### 2-2. The Verification Gate [important] - CLI Before HTTP

**This ordering was the most important decision at this stage.**

Build HTTP first and when an image comes out wrong, **you cannot tell whether serialization, injection, or the network is at fault.**

```
1) normal generation + save the conditioning     -> image A
2) drop the text encoder arguments, inject only the conditioning  -> image B
3) are A and B identical pixel for pixel
```

That single test proves three things at once.

- the serialization round trip is correct
- the injection point is correct
- **the worker starts without a text encoder at all**

The third had a prior concern - "model metadata validation might reject a missing encoder." **It passed.** Weights are lazily loaded, so what is never called is never loaded.

> This gate actually caught a bug. The first attempt produced different pixels, and the cause was a floating-point difference between the CPU and GPU backends. **Matching the comparison to the same backend produced an exact match.** Had HTTP been built first, reaching that cause would have taken far longer.

### 2-3. An Unexpected Find - It Was Not a Startup Argument

The face consistency extension's ID embedding was **assumed to require restarting every worker whenever the character changed**, because structurally it was passed as a startup argument.

On inspection it turned out to have been **a per-request parameter inside the engine all along**, with only the channel to pass it over HTTP missing. **Two lines of code** removed the restarts.

```c
load_if_exists("pulid_id_embedding_path", pulid_id_embedding_path);
load_if_exists("pulid_id_weight", pulid_id_weight);
```

**The larger the worker count N, the more the restart cost multiplies by N.** In a design that presumes scaling, finding items like this early is exactly where the value lies -> [No.4 Section 4](../04-orchestration.md#4-question-whether-a-startup-argument-is-really-global-state)

---

## Phase 3 - Orchestrator Integration

### 3-1. Design Principle - Always Presume Worker Count N

Every decision at this stage presumed **"more cards will be installed later."**

| Principle | Reason |
|---|---|
| Do not assume worker count is a constant | derive it from list length only |
| Do not assume workers are homogeneous | performance, VRAM, and architecture may differ |
| Exactly one encoder, always | the premise of separation |
| **Minimize the unit of work** | imbalance is capped at one job |

**Split into one image = one job and static round-robin is sufficient.** Pre-dividing into batches lets a slow worker delay everything; at one-image granularity the imbalance is at most one image. **A dynamic dispatcher is unnecessary until genuinely heterogeneous workers actually appear.**

The implementation is generalized in [`reference/orchestrator.py`](../../../reference/orchestrator.py).

### 3-2. This Is Where the Most Time Was Lost

Right after integration, **encoding grew from 14 seconds to 56.** The cause was three things chaining together.

```
No.1 the orchestrator requests a service launch just before generation
No.2 if the model is still loading at that moment, it judges "not running" and brings up a second set
No.3 the first set, still loading, handles SIGTERM late and does not die
      -> two processes per port, fighting over CPU and disk
```

There were two fixes.

- **Serialize launches with a lock** - block duplicate requests during the 20 seconds of loading
- **A `kill()` fallback on termination** - kill for certain if `terminate()` times out

> The symptom presented as **"why did this suddenly get slow,"** so reaching process management as the cause took a long time. It only became clear after counting processes with `ps -ef`. **When a performance anomaly appears, checking the process count is cheap.**

---

## Phase 4 - Verification and Cutover

Production measurements at cutover time. **These are not isolated conditions** (the LLM was running concurrently), so they must not be placed on the same axis as [Appendix - Benchmarks](../98-benchmark.md).

| | Before the rework | CPU encoder | Final (P104) |
|---|---|---|---|
| Encoding | ~37s | 24.5s | **0.86s** |
| **First image displayed** | 145s | 52s | **~28s** |
| 8 images total | 145s | 132s | **~108s** |
| RAM available | 10GB | 18GB | **18GB** |
| Minimum VRAM requirement | 19.1GB | ~13GB (+RAM) | **13.6GB** |
| Display one at a time | X | O | O |

Note: **whether the face consistency extension was enabled was not recorded for this table.** Enabling it adds 22% per step. The figures in [No.2](../02-encoder-separation.md#8-results) and [Appendix - Benchmarks](../98-benchmark.md) are all with the extension enabled, so **they must not be compared directly with the values above.** The table itself is caught by the lesson recorded just below it.

**Check the conditions when quoting a baseline.** This project cited an early figure of "8 images in 124 seconds" for some time before discovering it was a value from **before the face consistency extension was enabled.** That extension adds 22% per step. **A wrong baseline either understates the improvement or invents a regression that never happened.**

---

## In Retrospect - What Worked

### No.1 Putting the Investigation First

Without "is the conditioning a value type" in Phase 0-1, the whole plan would have collapsed. **Learning it after buying a card would have cost not 15 USD but time.**

Generalized: **place the least reversible decision (buying hardware) after the cheapest check (reading source).**

### No.2 Writing the Stop Criteria in Advance

Writing "if this happens here, stop" for each stage means that when you do get stuck, **whether to keep digging is decided by a criterion rather than by mood.**

### No.3 Arranging Phases So Partial Output Survives

Phase 0-2's quantization was a gain in itself even if Phase 1 had failed (half the RAM, 44% faster encoding). **Arrange the phases this way and stopping early still leaves the loss at more than zero.**

### No.4 Verifying Through the CLI Before HTTP

Verifying while stacking one layer at a time keeps the failure point narrowed to **the layer added last.** Build everything and then test, and the candidate causes multiply.

### And What Was Missed

**The draft design's target of "8 images in 92 seconds" was off against the actual 108.** The cause was a baseline missing the face consistency extension. **Conditions on the baseline should have been stated when setting the target, too.**

And process management (3-2) had no entry in the design document at all. **"Starting and stopping things" was treated as trivial, and it became the longest-running bug in this project.**
