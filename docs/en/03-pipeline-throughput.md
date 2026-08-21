**English** / [한국어](../ko/03-pipeline-throughput.md)

# No.3 Incremental Display - Throughput in Human-in-the-Loop Pipelines

> Emitting generated results **one at a time as each completes** overlaps the human's evaluation time with the machine's generation time,
> reducing total elapsed time from `N x T` to `T`.
> **It is independent of GPU configuration.** It applies to any generative workflow where a person has to pick from the results.

Related documents: [Common Methodology](00-method.md) / [No.1 Role Assignment](01-role-assignment.md) / [No.2 Encoder Separation](02-encoder-separation.md) / [No.4 Orchestration](04-orchestration.md) / [Appendix - Benchmarks](98-benchmark.md) / [Pitfalls](99-pitfalls.md)

---

## 1. In a Generative Workflow, the Human Is the Bottleneck

In this project, generated images almost always contain a defect. Picking a usable one requires **a person to evaluate them.**

Choosing the better of several candidates and using it as the seed for the next generation is a method known as **interactive evolutionary computation (IEC)**. Ordinary evolutionary algorithms compute fitness with a formula, but in domains such as aesthetic judgment where a formula is hard to write, **a human serves as the fitness function.**

| IEC concept | Its counterpart in image generation |
|---|---|
| population | the N images generated at once |
| selection | picking the image you like |
| mutation | img2img variation strength |
| generation | regenerating from the selected image |

In this structure, **the human's evaluation time is a cost that cannot be ignored.** If evaluating one image takes T seconds, receiving all N at once costs `N x T`.

Yet a poorly designed implementation makes you pay that cost **only after generation has entirely finished.**

---

## 2. The Time Model

**Hold generation time constant and change only the display method.** Let G be total generation time, F the time to the first image, and T the evaluation time per image:

```
Batch display : [generate G] ----------------------------> [evaluate N x T]
                total = G + (N x T)

One at a time : [first image F] [ generation continues --------------- ]
                                [eval][eval][eval]...
                total ~= G + T          (when evaluation is faster than the generation interval)
```

**Completion moves from `G + (N x T)` to `G + T`.** Generation time G is unchanged; **what shrinks is the space the human's evaluation time used to occupy.**

This works because the person evaluates the first image while the machine builds the second. Only the last image's evaluation has nothing left to overlap with.

It is the same principle as hyper-threading filling idle execution units, except the idle resource being filled here is **human waiting time.**

> If evaluation is slower than generation (T larger than the generation interval), the relationship inverts and evaluation dominates. Even then the total is `F + (N x T)`, which is less than batch display's `G + (N x T)`. **As long as F < G, neither case loses.**

> **T was not measured in this project.** Human evaluation time varies widely with the task, the person's skill, and fatigue, so read the model above as **a structural comparison with T left as an unknown.** The measurements in Section 4 below cover only G and F.

---

## 3. The Implementation Requirement - The Work Must Be Divisible

For this design to hold, the overall work structure must **divide into N independent jobs.** If result production is bound into a batch, nothing can be emitted until all of it finishes.

The usual reason to batch in image generation is **to pay the prompt encoding cost only once.** Normally, emitting one image at a time means not batching and re-encoding for every image.

In this project that constraint does not exist, because [the encoder was separated](02-encoder-separation.md) so that the workers share the conditioning tensor.

**It is possible without encoder separation too.** You just accept the per-image re-encoding cost. At this project's P104 encoding rate that cost is 0.86s per image, so eight images add **about 6 seconds** - within **6%** of the total.

But **in a structure where encoding is resource-heavy, that price is hard to pay.** Encoding on the CPU here is exactly such a case. At 8.30s per image, seven more times adds **58 seconds**, growing the total by nearly a quarter. Accepting per-image encoding **only pays off where encoding is cheap.**

As a side effect, **once the encoder is separated, load balancing across two or more workers also gets simpler.** With one image = one job, worker imbalance is capped at one job, so static round-robin is enough. Pre-dividing into batches lets a slow worker delay everything.

---

## 4. Measurements

**Only the work allocation method was changed, on identical hardware.** Cards, encoder placement, and worker count are all the same; only splitting the batch into single images differs.

| | Batch display | One at a time |
|---|---|---|
| **Time until the first image is shown (F)** | **261.5s** | **46.7s** |
| Time to complete all 8 (G) | 261.7s | 262.4s |

**For batched work, `F = G`.** A batch emits nothing until all eight are finished, so review cannot begin until the entire job ends.

Split into single jobs, the first image was produced in 46.7 seconds, and review starts there.

Note: in this experiment the encoder was in V100 VRAM, so **one encoding took 0.17 seconds**, meaning the seven extra encodings the single-image run performed stayed **inside the margin of error of the total.** This is the measured value of the "affordable cost" discussed in Section 3. Had the encoder been on a slower GPU (a P104, where seven times is about 6 seconds) or on the CPU (an EPYC 7232P, where it would be about 58 seconds), that cost would have looked very different.

Measurement conditions and raw records are in [Appendix 8](98-benchmark.md#8-interpretation---the-cost-of-splitting-work).

**What the user experiences is not total time but the wait until the first result.** And as Section 2 showed, this is not only a matter of perception - it genuinely reduces total elapsed time.

---

## 5. Applicability and Limits

**Where it applies**

| Condition | Description |
|---|---|
| Multiple results are generated | N >= 2 |
| **A person has to choose** | if automatic scoring is possible, this discussion is unnecessary |
| Result order does not matter | it must be possible to evaluate them in completion order |
| The work can be split | the requirement from Section 3 |



**Where it does not apply**

- **Workflows with automatic evaluation** - with no human involved there is no idle time to overlap
- **Cases requiring the full set to judge** - for example, if comparison across all N is mandatory, everything has to be present
- **Cases where generation is far faster than human evaluation** - little room to overlap. There is no loss either, though
- **Cases producing a single result** - N = 1

**The scope is not limited to image generation.** Any structure where a person picks among candidates - code generation, draft text, design mockups, speech synthesis - follows the same logic.

---

## 6. As a Design Principle

> **If a person is part of the pipeline, that person's waiting time is a resource.**
> Split the unit of work so machine time and human time overlap.

This is not a UX improvement but a **throughput improvement.** It sits at a different level from "a progress bar makes the wait less annoying" - total elapsed time actually falls.

And this principle arrives at the same conclusion as the orchestration principle [minimize the unit of work](04-orchestration.md#2-write-it-assuming-workers-will-scale). Load balancing and human-time overlap are different reasons pointing at **the same design.**
