# Breaking SAM 2

## An Evaluative Benchmark on Extreme Object Evolution

Breaking SAM 2 is a stress-test benchmark for zero-shot video object segmentation (VOS). It studies when and why temporal mask propagation fails as an object's appearance, shape, topology, scale, viewpoint, or illumination changes.

The benchmark will evaluate both manually initialized SAM 2 and a Grounded-SAM-2 pipeline. Its central hypothesis is that temporal propagation becomes unreliable when object identity is no longer represented by stable appearance and shape cues.

> **Project status:** Early-stage implementation. The repository currently contains the project scaffold; the data pipeline, model integrations, annotations, and evaluation code are planned work.

## Research Questions

The project focuses on three classes of disruption:

- **Topology and state change:** pouring liquids, flames, smoke, melting material, kneaded dough, and other targets that split, merge, or change their boundaries.
- **Extreme scale and viewpoint:** objects rapidly approaching or receding from the camera, with large changes in visible area, texture resolution, or partial visibility.
- **Abrupt illumination:** sudden shadows, lights switching off, tunnel entry, glare, or exposure changes that alter appearance without changing semantic identity.

The benchmark will investigate:

1. Which disruption produces the earliest and most severe degradation?
2. Whether failures are abrupt, gradual, or recoverable.
3. How sensitive results are to the initial prompt.
4. Whether text-guided initialization improves localization or introduces an additional first-frame detection failure.

## Planned Benchmark Design

### Dataset

The initial benchmark target is **15-30 short clips**, approximately **5-10 seconds** each, collected from reusable or appropriately licensed sources such as Pexels. Every clip will include:

- A primary disruption label.
- A target description and initialization prompt.
- Frame rate and resolution.
- The disruption event timestamp.
- Source and licensing metadata.

Clips with ambiguous ownership, severe compression, or unclear target identity will be excluded. The dataset will include ordinary-motion control clips so that failures can be distinguished from pipeline configuration problems.

### Ground Truth

Target masks will be annotated with CVAT on every 5th-10th frame, with denser annotations around the transformation event. A smaller subset may be densely annotated to check whether sparse sampling hides short-lived failures.

### Evaluation Pipelines

The implementation will support the following variants:

- **Pipeline A - Manual initialization:** SAM 2 initialized with a manually verified box or click on the first frame.
- **Pipeline B - Text-guided initialization:** Grounding DINO receives a fixed text prompt, and its selected box initializes SAM 2.
- **Optional oracle-box variant:** Grounding DINO selects the object, then the box is manually verified before it is passed to SAM 2. This isolates propagation errors from detector errors.

The evaluation will report initialization quality separately from temporal propagation quality rather than treating Grounded-SAM-2 as a single black box.

## Metrics

The planned evaluation reports:

- Region similarity, **J** (Jaccard/IoU).
- Boundary accuracy, **F** (boundary F-measure).
- Combined **J&F**.
- Per-disruption and per-pipeline performance.
- Worst-frame accuracy.
- Temporal stability and mask-area drift.
- Fragmentation and false-positive persistence after target disappearance.
- First failure and recovery statistics.

The default first-failure definition will be pre-registered as the first annotated frame with $J < 0.5$ for two consecutive observations. The threshold and rule should remain configurable so sensitivity analyses can be reported alongside the primary result.

Qualitative failure labels will distinguish:

- Target loss.
- Background takeover.
- Identity switch.
- Mask fragmentation.
- Boundary lag.
- Over-segmentation.

Results will be plotted against normalized video time and aligned by disruption event to compare failure behavior across clips.

## Intended Workflow

The repository is expected to grow around this workflow:

1. Curate clips and record metadata.
2. Extract frames and register annotation keyframes.
3. Run SAM 2 and Grounded-SAM-2 inference.
4. Convert predictions and annotations to a common mask format.
5. Compute frame-level and temporal metrics.
6. Classify failure modes and generate tables, plots, and visual examples.
7. Review borderline cases and publish the benchmark documentation.

## Repository Layout

```text
.
├── docs/          # Benchmark protocol, annotation guidance, and experiment notes
├── src/           # Inference, data processing, metrics, and analysis code
├── requirements.txt
└── README.md
```

As implementation proceeds, generated data and model weights should remain outside version control. See `.gitignore` for the repository's ignored paths.

## Computing Requirements

The project is inference-only and does not require fine-tuning. Development can be performed locally for pipeline integration and small tests. Full evaluation is expected to run on a single consumer or cloud GPU, such as an RTX-class GPU or a Colab T4/L4.

The planned software stack includes:

- Python and PyTorch.
- The official SAM 2 implementation.
- Grounding DINO for text-guided localization.
- OpenCV and FFmpeg for video processing.
- CVAT for annotation.
- Python tooling for metric computation and visualization.

Expected storage for clips, annotations, predictions, and model weights is below approximately 15 GB, depending on the selected model checkpoints and retained intermediate files.

## Reproducibility and Evaluation Principles

To keep the benchmark interpretable:

- Pre-register the IoU threshold and consecutive-frame failure rule.
- Report continuous J/F curves alongside binary failure-frame summaries.
- Keep initialization and propagation errors separately measurable.
- Retain clips according to explicit inclusion criteria rather than selecting only visible failures.
- Use a second reviewer for borderline qualitative labels.
- Include control clips to verify that the pipelines are configured correctly.
- Record model versions, checkpoint names, prompts, video metadata, and evaluation settings for every run.

## Planned Deliverables

1. A documented Evolution Benchmark with annotations and metadata.
2. Reproducible inference and evaluation code.
3. Quantitative tables and temporal failure plots.
4. A concise taxonomy of failure modes with representative visual examples.

## References

1. Ravi et al., ["SAM 2: Segment Anything in Images and Videos"](https://arxiv.org/abs/2408.00714), 2024.
2. Liu et al., ["Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"](https://arxiv.org/abs/2303.05499), 2023.
3. Perazzi et al., ["A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation"](https://graphics.stanford.edu/~niessner/papers/2016/Perazzi_Benchmark_VOS_CVPR16.pdf), 2016.
4. Xu et al., ["YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark"](https://arxiv.org/abs/1809.03327), 2018.
