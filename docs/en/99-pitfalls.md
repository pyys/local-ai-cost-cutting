**English** / [한국어](../ko/99-pitfalls.md)

# Pitfalls

> A classified collection of **the places where failures and errors cost time** during the build. You will walk into them yourself when reproducing this, so they are worth reading in advance.
> The basis is a real build: EPYC 7232P / V100 32GBx1 + V100 16GBx2 + P104-100 8GB / Ubuntu 24.04.

Related documents: [Common Methodology](00-method.md) / [No.1 Role Assignment](01-role-assignment.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.3 Pipeline Throughput](03-pipeline-throughput.md) / [No.4 Orchestration](04-orchestration.md) / [Appendix - Benchmarks](98-benchmark.md)

Appendices: [Build Log](appendix/build-log.md) / [Design Record](appendix/design-record.md)

---

## Contents

1. [Environment and Build](#1-environment-and-build)
2. [GPU Placement](#2-gpu-placement)
   - [Position-based identifiers are all the same trap](#position-based-identifiers-are-all-the-same-trap)
3. [Inference Engines](#3-inference-engines)
4. [Process Management](#4-process-management)
   - [`pgrep -f` matches itself](#pgrep--f-matches-itself)
5. [Measurement and Verification](#5-measurement-and-verification)
   - [Isolated measurement - the only way to compare cards](#isolated-measurement---the-only-way-to-compare-cards)
   - [Cases that nearly produced a wrong conclusion](#cases-that-nearly-produced-a-wrong-conclusion)
   - [Cards of identical specification differ in sustained performance](#cards-of-identical-specification-differ-in-sustained-performance)
6. [Measurement Checklist](#6-measurement-checklist)
7. [Where Common Assumptions Fail](#7-where-common-assumptions-fail)
   - [Do not choose the cheap card first](#do-not-choose-the-cheap-card-first)
   - [Do not disable SMT just because the work is compute-bound](#do-not-disable-smt-just-because-the-work-is-compute-bound)
   - [Buying the hardware does not mean you are using the feature](#buying-the-hardware-does-not-mean-you-are-using-the-feature)
8. [CPU-Bound Work Contaminates GPU Measurements](#8-cpu-bound-work-contaminates-gpu-measurements)
9. [When You Hand Work to an AI Coding Tool](#9-when-you-hand-work-to-an-ai-coding-tool)

---

## 1. Environment and Build

| Pitfall | Symptom | Response |
|---|---|---|
| Installing CUDA 13 | no Volta support | **pin to 12.8** |
| NVIDIA apt repository | `apt upgrade` fails on a `libnvidia-gl-580` conflict | block driver packages with pinning |
| cmake architecture argument | unquoted `61;70` splits the command | quotes are mandatory |
| Omitted architecture | will not run, or the INT8 kernel is silently disabled | include every target card |
| rsync `--exclude 'build'` | matches at any depth, deleting third-party directories too | anchor it as `--exclude "/build"` |
| Prebuilt frontend assets | the distribution host is dead, 404 | install Node and build them yourself |

**The omitted architecture is the quietest trap.** Even when it runs, the optimal kernel may be off, so the only symptom is "somewhat slow." -> [kernel path determination](01-role-assignment.md#5-where-performance-actually-diverges-across-heterogeneous-gpus)

---

## 2. GPU Placement

| Pitfall | Symptom | Response |
|---|---|---|
| Specifying GPUs by index | everything shifts when a card is added | **use UUIDs** |
| Designating a main GPU is not enough | a secondary module lands on another GPU and causes OOM | isolate with `CUDA_VISIBLE_DEVICES`, then specify index 0 |
| Skipping the PCIe lane check | model loading at startup takes 10x the expectation | check `pcie.link.width/gen` in advance |
| **Network interface names** | **no SSH** after swapping a card | **pin to the MAC address** (below) |

A real case of the second item: specifying only the main GPU with `-mg 1` in llama.cpp put **the vision projector (mmproj) on GPU0 taking 1.4GB**, and the image generation server hit OOM later. Making a single card visible with `CUDA_VISIBLE_DEVICES` and then passing `-mg 0` is the reliable way.

### Position-Based Identifiers Are All the Same Trap

The fourth item is the most dangerous. **You installed a GPU and now the server is unreachable.**

The cause is exactly the same as the first item - **attach identifiers based on PCI position and they change when the hardware layout changes.** Linux network interface names (`enp2s0`) are derived from the PCI slot position, so adding a card changes the name and every setting pinned to it breaks.

| | Position-based (changes) | Unique-value-based (stable) |
|---|---|---|
| GPU | `CUDA_VISIBLE_DEVICES=0` | **UUID** |
| Network interface | `enp2s0` | **MAC address** |
| Block device | `/dev/sda` | **UUID / PARTUUID** |

**On a headless server the order matters.** A wrong GPU setting merely stops a service from starting, but **a mismatched network name may force you to undo the hardware change, or in the worst case reinstall the OS.** Convert all three to unique-value identifiers before changing hardware, and do the network one first -> [No.4 1-1](04-orchestration.md#1-1-changing-the-pci-device-layout-causes-problems-outside-the-gpus-too)

---

## 3. Inference Engines

| Pitfall | Symptom | Response |
|---|---|---|
| VAE decode compute buffer | OOM from a 3744MB request | tiling (416MB, costs 1.4s) |
| Applying CFG to a distilled model | 3.3x slower | CFG scale 1.0 |
| `--mmap` | model load 8s -> 110s | do not use it. The RAM saved costs more than it gains |
| Guessing argument names | the server prints usage and exits | check `--help` (`--listen-ip`, not `--host`, and so on) |
| LoRA directory defaulting to `.` | recursive scan per request; crashes when launched from root | specify an empty directory |
| Face embedding extraction (InsightFace) | CUBLAS failure from unsupported older GPU | extract on the CPU |

**`--mmap` is the opposite of intuition.** Enabled to save RAM, it made model loading 14x slower. Unless several processes share the same weights, it is a loss.

**The LoRA directory default** blew up when the server was launched from `/`: it recursively scanned the entire root and died on a symbolic link. The scan happens per request, so an empty directory must be specified.

```
{"error":"server_error","message":"filesystem error: status: Too many levels of symbolic links [./run/udev/watch/22]"}
```

---

## 4. Process Management

| Pitfall | Symptom | Response |
|---|---|---|
| Short wait after `terminate()` | SIGTERM is handled late during model loading (20s+), so **the process survives** | `kill()` fallback after a timeout |
| Restart requested while starting | judged as "not running" and **a duplicate set launches** | serialize with a lock |
| Duplicate encoders | CPU and disk contention: encoding 14s -> 56s | resolved by the two above |
| **Running `pgrep -f` through a shell** | always reports "alive" | **run it as a list** (below) |

Three items chained together. The orchestrator requests a service launch just before generation; if the model is still loading at that moment it judges "not running" and brings up a second set. The first set, still loading, handles SIGTERM late and does not die. The result was **two processes per port**, fighting over the CPU.

### `pgrep -f` Matches Itself

Run process cleanup logic through a shell and it breaks silently.

```python
subprocess.run("pgrep -f sd-server", shell=True)   # X always finds something
subprocess.run(["pgrep", "-f", "sd-server"])       # O
```

`shell=True` spawns `/bin/sh -c pgrep -f sd-server`, and **the wrapper shell's command line contains the search term.** `pgrep` excludes only its own PID, not the parent shell, so it reports "alive" even when no target process exists at all.

The same reason makes one-off verification commands produce false positives.

```
python3 -c "... 'sd-server' ..."     # this python3's command line contains the term too
```

**In this project that single line made 24 benchmark runs fail instantly.** A derived principle follows - **exercise cleanup and defensive logic at least once for real before putting it in an unattended run (nohup).** Unverified defensive code does not defend.

---

## 5. Measurement and Verification

**This category bears directly on the reliability of figures in the other documents.** Reading it alongside [bottleneck classification](00-method.md#2-bottleneck-classification-can-be-settled-by-calculation) and [Appendix - Benchmarks](98-benchmark.md) is recommended.

### Isolated Measurement - The Only Way to Compare Cards

Comparing cards is meaningful only through **isolated measurement with matched conditions.** Values pulled from production logs mix prompt length and system load; they serve as a rough reference but are far from a precise comparison.

```
1. Stop every other service (workers and LLM included)
2. Launch only the target, standalone
3. Three requests with identical input
4. Discard the first, use the second and third   <- the first includes weight loading
5. Swap the card and repeat
```

Figures obtained this way are in [Appendix 7-2](98-benchmark.md#7-2-encoding-performance-by-card---isolated-measurement). The two cases below are why the procedure is necessary.

| Pitfall | Symptom | Response |
|---|---|---|
| Byte-comparing PNGs | always differs because of metadata chunks | compare pixels only |
| Comparing CPU against GPU output | floating-point kernel differences alter the image slightly | not a bug. About 0.72/255 on average. **Changing backends breaks reproducibility even at the same seed** |
| The log's memory estimate | differs from actual allocation | confirm with `nvidia-smi` |
| **Comparing cold measurements** | load time's share differs per card, **distorting the ratio** (2.3x vs an actual 3.9x) | **measure warm, at least twice** |
| **Page cache** | whichever ran first reads from disk and looks slower | use phase-separated logs; re-measure in the opposite order |
| **Concluding a kernel path from timing** | other paths can produce similar values | **read the decision function in the source** |
| Comparing cards from production logs | prompt length and system load are mixed in (0.33s vs 0.86s) | **isolated measurement with matched conditions** |
| **Assuming identically specified cards are identical** | sustained performance differs 12% from cooling conditions | **log `clocks.sm` and temperature during load** |

### Cases That Nearly Produced a Wrong Conclusion

**Comparing cold runs** - dividing two cards' first-run times gave an estimate of "3.9x," which happened to be right for the wrong reasons. Both values included weight loading, and its share differed completely between the cards (x16 Gen3 vs x4 Gen1). Re-measured warm, the cold-to-cold ratio came out at 2.3x - **which contradicts the actual compute ratio of 3.9x.**

**Page cache** - disk read time came out at 1.81s for the V100 and 0.15s for the P104, making the P104 look 12x faster. The V100 test simply ran first and read from disk while the P104 read a file already in cache. **That is an artifact of execution order, not a card characteristic.**

**Phase-separated logging caught both.** Because `read` and `copy_to_backend` were logged separately, disk could be separated from PCIe. Had the log recorded only total time, both errors would have gone unnoticed.

### Cards of Identical Specification Differ in Sustained Performance

In the cost benchmark, **a V100 32GB came out 12% slower per image than a V100 16GB.** Same bandwidth, same form factor, same maximum clock.

The cause was **thermal throttling.** Under load the 32GB card ran at **1,155~1,207 MHz** at 82~83C - 78% of its boost (1,530), below even its base clock (1,290). Power had nearly 100W of headroom against the 300W limit, so it was not a power cap. **That card simply sat in a worse position in the chassis.**

| Symptom | Meaning |
|---|---|
| Small difference when cold, **widening under sustained load** | the classic signature of thermal throttling |
| Only total elapsed time was recorded | cause cannot be attributed |

```
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv -l 5
```

> **In a multi-GPU benchmark, log `clocks.sm` and `temperature.gpu` during load.** Total elapsed time will never show it. **Do not assume that the same model name and form factor means the same performance.**

**What makes this trap dangerous is that the wrong conclusion is plausible.** Had the 12% gap been written up as "thanks to module separation," that sentence would have stood unverified, and anyone else building the same structure would have failed to reproduce it. **Without designing an additional condition to eliminate hypotheses one at a time, it would never have come to light.** The full elimination process is in [Appendix 9](98-benchmark.md#9-disconfirmation---why-the-32gb-card-was-slower).

#### A Corollary - More Cards Heat Each Other

During that measurement, while the 32GB card was at 82C the 16GB cards were **idle at 39C.** So each condition's measured value is a value from a situation where **only that card is hot.**

Scale the workers to three or four cards and they heat each other, so **each card's sustained clock will be lower than in this measurement.** That is one reason [the condition 5 derived value](98-benchmark.md#4-2-deriving-condition-5) must be treated as an optimistic upper bound.

**When planning to scale, treat airflow as a performance variable.** Adding a card does not only add that card's performance - **it also lowers the performance of the cards already there.**

---

## 6. Measurement Checklist

```
[ ] Were all other services stopped?              (contention removed)
[ ] Was identical input repeated three or more times?  (reproducibility)
[ ] Was the first run discarded?                  (load time removed)
[ ] Was it re-measured in the opposite order?     (page cache check)
[ ] Were the baseline's conditions recorded?      (which features were on)
[ ] Are phase-separated logs available?           (cause can be attributed)
[ ] Were clocks and temperature logged under load?  (sustained performance of identically specified cards)
[ ] Was it converted to whole-system time rather than module time?
```

The last item matters most. In the table above the P104 is **3.9x slower than a V100 per module**, yet across the whole-system time for generating eight images that difference is **under 1%.** Spending 185 USD more on the strength of "3.9x" alone is a decision made without doing the [price per unit of throughput](00-method.md#4-price-per-unit-of-throughput) calculation.

---

## 7. Where Common Assumptions Fail

Cases where a plausible premise was overturned by measurement.

### Do Not Choose the Cheap Card First

**The order is to find the position for a card first, not to find a cheap card.** Buy the card first and then look for a use, and you fail.

The 15 USD card worked here **because the text encoding position fit it.** There were two requirements - **is there VRAM for the module, and is it cheap.** Encoding gains substantially just from moving off the CPU onto a GPU, so it was not the kind of choice that depended on a particular compute path surviving.

**Put the other way, if it does not fit, no price makes it a candidate.** The encoder was 5.2GB, which is what put an 8GB card up for consideration -> [No.1 5-3](01-role-assignment.md#5-3-cheap-and-it-only-has-to-fit)

### Do Not Disable SMT Just Because the Work Is Compute-Bound

Assuming the execution units are saturated and using only the physical core count can cost you. Here, running CPU encoding on all logical threads of an 8-physical-core CPU made it **22% faster.**

| Setting | 256-token encoding time |
|---|---|
| 8 physical cores | 10.14s |
| **16 logical threads (SMT)** | **8.30s** |
| Engine default (`-t` unspecified) | 10.09s - **it was the physical core count** |

The cause appears to be an instruction dependency chain. This CPU (Zen 2) has no AVX512-VNNI, so ggml's INT8 path goes through `maddubs -> madd -> accumulate`, and **SMT fills the pipeline bubbles in between.**

> **"Compute-bound" says only which resource is the ceiling; it guarantees nothing about that resource's utilization.** A roofline classification and actual unit saturation are separate things.

**Since engines sometimes default to the physical core count, it is worth setting `-t` explicitly. Do not guess - measure.** The arithmetic is in [Appendix 7-3](98-benchmark.md#7-3-why-25x-faster-than-cpu---arithmetic-intensity-and-ridge-point).

### Buying the Hardware Does Not Mean You Are Using the Feature

Before the rebuild, this project **bought an NVLink board and ran two V100 32GB cards for a month.** And **never once used the link.**

A multi-GPU load command was written, but `--split-mode row` was missing, and llama.cpp defaults to layer splitting, so **even with two cards visible it never ran tensor parallel.** This was only confirmed much later.

**Whether a feature is on must be verified from launch arguments and logs, not from specifications.** Having the hardware and actually using its bandwidth are separate things -> [Appendix - Build Log, Notes and Caveats 1)](appendix/build-log.md#notes-and-caveats)

> The same grain as the kernel path problem in [No.1 5-2](01-role-assignment.md#5-2-a-kernel-path-cannot-be-settled-by-timing). **Which path was actually selected is knowable only by checking.**

---

## 8. CPU-Bound Work Contaminates GPU Measurements

The orchestrator runs on the CPU, but **the CPU is a shared resource, so it affects GPU measurements.**

| Case | Effect |
|---|---|
| Encoding alone 16.51s -> **24.5s** under contention | CPU contention with the LLM server and workers (512 tokens) |
| Duplicate encoders launched -> 14s -> **56s** | CPU and disk contention |
| Warm encoding 0.33s (isolated) -> **0.86s** (production) | partly CPU overhead, partly chunk count |

Treat it as **"CPU contention contaminates GPU measurements,"** not as "it is CPU work, so it can be ignored." The isolated measurement methodology in Section 5 exists for this reason.

---

## 9. When You Hand Work to an AI Coding Tool

Most of this project's build and documentation was done through Claude Code. The productivity gain was large, but **the tool did in fact touch a file it was not told to.**

On 2026-08-21, while making a backup before editing a document, **the tool overwrote an already existing backup file without checking the numbering of prior revisions.** One intermediate revision from that point vanished, and with no version control on the repository it could not be recovered.

**The damage stopped at one file because a history-style backup was in use.** This repository follows a rule of leaving the previous revision as `_old_N` on every edit, and what got overwritten was one of those copies. **The working document was untouched, and copies from other points in time survived.**

**Had the same accident hit the working document, the outcome would have been entirely different.** With no point to roll back to, that document is simply gone. In any environment where a tool can write files, at least one of the following is mandatory.

- **Version control** - a git repository makes any overwrite reversible. The most reliable option
- **History backups** - leave a copy at every edit. What this project used
- **Restricted write access** - narrow the paths the tool can touch

**The trap is reviewing the tool's output but not the tool's actions.** A person reads what was written; almost nobody looks at what was overwritten.
