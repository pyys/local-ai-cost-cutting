**English** / [한국어](../ko/04-orchestration.md)

# No.4 Orchestration - Designing for Worker Scaling

> Rules learned from actually operating services spread across several GPUs.
> Separated out because these are **the practical items needed to actually run the structures from No.1 and No.2.**
> The basis is a real build: EPYC 7232P / V100 32GBx1 + V100 16GBx2 + P104-100 8GB / Ubuntu 24.04.

Related documents: [Common Methodology](00-method.md) / [No.1 Role Assignment](01-role-assignment.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.3 Pipeline Throughput](03-pipeline-throughput.md) / [Appendix - Benchmarks](98-benchmark.md) / [Pitfalls](99-pitfalls.md)

---

## 1. Always Specify GPUs by UUID

**In a system with several PCIe devices, indices shifting when the device count changes is extremely common.** It happened here when the encoder card was installed.

| | Before install | After install |
|---|---|---|
| V100 32GB (LLM) | index 1 | **index 2** |
| V100 16GB | index 0 | index 3 |
| V100 16GB | index 2 | index 0 |
| P104 8GB | - | index 1 |

Had the configuration used indices, the launch after installing the P104 would have tried to put the LLM on a 16GB card and hit OOM. **A UUID is unique to the card and is unaffected by physical rearrangement.**

```
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Keep a UUID dictionary in the configuration file and **reference cards by name**. Adding a card then means adding an entry, nothing more.

**An option that designates the main GPU is not enough by itself.** Specifying `-mg 1` to llama.cpp here put **the vision projector (mmproj) on GPU0, taking 1.4GB**, and the image generation server later hit OOM because of it. Making only one card visible with `CUDA_VISIBLE_DEVICES` and then passing index 0 is the reliable approach.

### 1-1. Changing the PCI Device Layout Causes Problems Outside the GPUs Too

**This is not a GPU-specific problem.** The root cause generalizes like this.

> **Attach identifiers based on PCI position and those identifiers change when the hardware layout changes.**

The classic example of the same cause producing an entirely different symptom is **network interface names.** Linux's Predictable Network Interface Names policy **derives the interface name from the PCI slot position** (`p2s0` in `enp2s0` is exactly that). So adding a card or moving a slot **changes the name, and every network setting pinned to that name breaks with it.**

**On a headless server this is fatal.** You installed a GPU and now SSH does not work, and with no display you cannot see why. This project hit it for real after swapping cards.

| | Position-based (changes) | Unique-value-based (stable) |
|---|---|---|
| GPU | `CUDA_VISIBLE_DEVICES=0` | **UUID** |
| Network interface | `enp2s0` | **MAC address** |
| Block device | `/dev/sda` | **UUID / PARTUUID** |

The fix on the network side is a `.link` file pinning the name to the MAC address.

```
/etc/systemd/network/10-nicname.link      <- [Match] MACAddress= / [Link] Name=
```

**Convert all three to unique-value identifiers before changing any hardware.** On a server reachable only remotely, do the network one first - a wrong GPU setting merely stops a service from starting, but **a mismatched network name may force you to undo the hardware change just to reach the server, or in the worst case to reinstall the OS.**

Note: some motherboard and CPU combinations have no video output at all, neither D-Sub nor HDMI. **In such a configuration the following can happen.**

You install a graphics card, install the OS, and networking works. You finish the installation and remove the graphics card - **SSH drops because of the problem above, and with no video output you cannot diagnose it.** Then you put the graphics card back in to investigate, and **PCI enumeration order returns to what it was, so everything works again.**

**Because the symptom appears only while the card is out, it is very hard to track down.** Without IPMI or prior experience of the same thing, it can take a long time to resolve.

> References
> - [Predictable Network Interface Names - systemd](https://systemd.io/PREDICTABLE_INTERFACE_NAMES/) - the priority order by which names are decided
> - [CUDA - GPU designation](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html) - NVIDIA's own documentation also recommends UUIDs over indices to avoid ambiguity
> - [Ethernet drops after replacing a PCI-E device - symptoms and recovery](https://pyys.cafe24.com/wp/?p=729) - the full diagnosis and recovery process on a headless server

---

## 2. Write It Assuming Workers Will Scale

Beyond the physical GPU count changing, there are times when **you simply want to change how many GPUs a task gets.** To avoid rewriting code each time, modularize in advance.

| Principle | Reason |
|---|---|
| Do not assume worker count is a constant | derive it from list length only |
| Do not assume workers are homogeneous | performance, VRAM, and architecture may differ |
| Exactly one encoder, always | the premise of separation |
| **Minimize the unit of work** | imbalance is capped at one job + [overlap with human evaluation](03-pipeline-throughput.md) |

The last item is the most effective in practice. **Split into one image = one job and static round-robin is sufficient.** Pre-dividing into batches lets a slow worker delay everything; at one-image granularity the imbalance is at most one image (about 27 seconds). A dynamic dispatcher is unnecessary until genuinely heterogeneous workers actually appear.

---

## 3. The Order in Which Bottlenecks Move as N Grows

| Rank | Bottleneck | When | Response |
|---|---|---|---|
| 1 | Disk I/O at worker startup | N>=4 | N processes read a 12GB model at once. Stagger the launches |
| 2 | Encoder serialization | multiple users | irrelevant with one user, where it is once per prompt |
| 3 | Conditioning tensor transfer | remote workers | ID caching |

The first has the same cause as load time becoming 2.9x with just two encoder copies in [No.2 Section 4](02-encoder-separation.md#4-gain-3---adding-workers-does-not-add-encoders). **The first wall you hit when adding workers is storage, not compute.**

---

## 4. Question Whether a Startup Argument Is Really Global State

The **[PuLID](https://github.com/ToTheBeginning/PuLID) ID embedding** used for image generation here was assumed to require restarting every worker whenever the character changed, because structurally it was passed as a startup argument.

Follow-up investigation showed that **inside the engine it had been a per-request parameter all along**, and only the channel to pass it over HTTP was missing. Two lines of code removed the need to restart anything.

**The larger the worker count N, the more the restart cost multiplies by N.** In a design that presumes scaling, finding items like this early matters.

---

## 5. Do Not Trust Process Termination

A problem encountered repeatedly here. Details are in [Pitfalls - Process Management](99-pitfalls.md#4-process-management). Only the essentials follow.

- **Do not call `terminate()`, wait briefly, and move on.** A process in the middle of loading a model handles SIGTERM late. A `kill()` fallback after a timeout is required
- **Serialize launch requests.** Judging a process that is still starting as "not running" brings up a duplicate set
- **Do not run `pgrep -f` / `pkill -f` through a shell.** The wrapper shell's command line contains the search term, so it matches itself
