# Benchmark data

The benchmark has 52 clips. The single source of truth is
[`configs/clips.csv`](../configs/clips.csv): dataset, split, category, object
ID, verified frame-0 box, text prompt, frame rate, event frame `t_ev`, number
of frames used (`frames`, empty = all), and license for every clip.

| Category | Clips | Dataset |
| --- | --- | --- |
| Topology / state | 14 | 8 M3-VOS + 6 VOST (val) |
| Scale / viewpoint | 14 | 11 DAVIS 2017 + 3 VOT-ST2022 |
| Illumination | 10 | 3 DAVIS 2017 + 7 VOT-ST2022 |
| Control | 14 | 11 DAVIS 2017 + 3 VOT-ST2022 |

The first 26 clips (14 / 5 / 3 / 4) were fixed before any model was run. The
other 26 were added later to balance the categories. The results of the first
26 were known at that time, so the new clips were chosen only with the same
GT statistics and eye checks, before any model was run on them. Illumination has
only 10 clips because DAVIS and VOT have no more clips with a clear light
change that pass the rules below.

All commands run from the repository root with the virtual environment
active. Everything is written under `data/`, which Git ignores.

```bash
cd breaking-sam2
source .venv/bin/activate
```

## Download

```bash
python reproduce_results.py --download-only
```

This fetches only the clips that are missing or incomplete: the full DAVIS
2017 480p archive (~800 MB), and the M3-VOS, VOST, and VOT clips one by one
(~1 GB, ~235 MB, and ~500 MB). Re-running it continues where it stopped. All download
code is in `download_missing` in `reproduce_results.py`; "How the download
works" below explains each dataset.

## How the topology clips were chosen

Candidates came from each dataset's metadata (M3-VOS phase-transition labels,
VOST val split) and GT mask statistics (area change, number of fragments,
empty GT frames), and every clip was checked by eye on a contact sheet with
the GT outline.

- `Box` is the tight box around the frame-0 GT mask, checked by eye.
- `t_ev` is the first lasting split or area change of the GT mask, checked by
  eye (frame index at the clip's own frame rate).

### M3-VOS (Apache-2.0, 30 fps, every frame annotated)

| Clip | Sub-type | Frames | Why |
| --- | --- | --- | --- |
| `0018_cut_pepper_5` | split | 333 | One pepper is taken from a bunch and cut into rings. |
| `0030_cut_celery_2` | split → pieces | 390 | Cut into small pieces, area drops to 0.03%. |
| `0109_break_glass_1` | split | 300 | Bottle shatters into up to 41 fragments. |
| `0113_tearing_cabbage_1` | split | 450 | Leaf torn apart, area 0.7% → 9.7% of the image. |
| `0060_pour_water_1` | liquid | 300 | Water is poured into a bowl and merges with the water there. |
| `0428_pour_tea_8` | liquid | 332 | Tea is poured from a pitcher into three cups, area 2.7% → 16%. |
| `0158_melt_glue_block_1` | melt | 528 | Solid glue block turns into viscous liquid. |
| `0239_exhaust_smoke_1` | smoke | 310 | Smoke plume from a chimney that keeps changing shape. |

Rejected: `0016_tear_pepper_1` (the target is hidden under the hand at frame 0
and its GT is two thin slivers), `0186_burn_plastic_3` (scene cuts),
`0035_peel_garlic_1` (the garlic is about 16×20 px and has no GT in about half
of the frames). Backup: `0080_pour_water_4`.

### VOST (CC BY-NC-SA 4.0, val split, 5 fps, every frame annotated)

| Clip | Sub-type | Frames | Why |
| --- | --- | --- | --- |
| `8013_divide_dough` | dough split | 170 | Up to 14 pieces, area 2.3% → 13%. |
| `3559_break_egg` | shell break + liquid | 174 | Egg is cracked open over a pan. |
| `8267_pour_flour` | powder flow | 65 | Flour is poured from a bag, area 7% → 30%. |
| `1207_break_garlic` | split | 140 | Bulb broken into cloves, 7 fragments. |
| `3316_peel_onion` | peel | 110 | Up to 10 fragments. |
| `555_tear_aluminium_foil` | thin-sheet tear | 51 | Foil pulled from the roll and torn, area 1.9% → 14.5%. |

Rejected: `10625_knead_dough` (target has no GT in 25% of frames),
`1186_cut_chilli` and `1203_cut_mango` (4–7 tiny objects). Backups:
`4021_cut_broccoli`, `6922_split_paper`, `996_cut_bacon`.

## How the DAVIS clips were chosen

For all 205 objects in DAVIS 2017 train/val (480p, 24 fps, every frame
annotated) we computed the change of GT area (95th/5th percentile ratio),
global and per-object brightness curves, GT fragment count, and empty-GT
frames, then checked the best candidates by eye on contact sheets with the GT
outline. For scale clips `t_ev` is the first frame where the GT area is at
least 5× (or at most 1/5 of) the frame-0 area; for illumination clips it comes
from the brightness curves.

| Clip | Category | Split | Frames | Why |
| --- | --- | --- | --- | --- |
| `rallye` | scale | train | 50 | Car rushes at a low camera: 0.19% → 33% of the image, then leaves in a dust cloud. |
| `drift-turn` | scale | train | 64 | 2% → 73% of the image, tire smoke, the car turns while it approaches. |
| `night-race` | scale | train | 46 | 1.3% → 52% in a dark hall. A second, unannotated car is in view (object 2 is the target). |
| `motocross-jump` | scale | val | 40 | Jump toward the camera: 1.3% → 47% in 17 frames, then away again. |
| `upside-down` | scale | train | 65 | Roller coaster: the camera rotates upside-down; viewpoint change without much scale change. |
| `car-shadow` | illumination | val | 40 | The car drives out of a shadow into full sun (target brightness 113 → 181); gradual. |
| `mallard-fly` | illumination | train | 70 | The duck takes off from dark grass over bright water (target 58 → 164), plus motion blur. |
| `shooting` | illumination | val | 40 | Muzzle flashes cause abrupt brightness jumps; the gun also rotates. |
| `bear`, `blackswan`, `elephant`, `rhino` | control | train/val | 50–90 | Stable area (max/min ≤ 1.3), stable brightness, one clear instance of the class. |

The table above lists the DAVIS clips of the first 26. The later additions from DAVIS use the
same statistics:

| Clip | Category | Split | Frames | Why |
| --- | --- | --- | --- | --- |
| `car-turn` | scale | train | 80 | SUV comes around a bend toward the camera (×8.9) and turns sideways. |
| `drift-chicane` | scale | val | 52 | Drifting car approaches in tire smoke (×10). |
| `drift-straight` | scale | val | 50 | Car drives at and past the camera (×27), strong motion blur. |
| `motocross-bumps` | scale | train | 60 | Dirt bike (object 2; the rider is object 1) comes from far away (×81). |
| `mbike-trick` | scale | val | 79 | Motorcycle (object 2) comes out of a smoke cloud toward the camera (×32). |
| `soapbox` | scale | val | 99 | Soapbox cart comes down a ramp toward the camera (×16), thin parts. |
| `boat`, `hike`, `lucia`, `rollerblade` | control | train | 35–80 | Area max/min ≤ 1.3, stable brightness, one instance of the class. |
| `mallard-water`, `goat`, `car-roundabout` | control | train/val | 75–90 | Area max/min 1.6–2.2 (pose change only), stable brightness, one instance. |

DAVIS has no clip where the lights go off, so the DAVIS illumination group is
small and partly gradual. Rejected: `cows` and `camel` (unannotated animals of the
same class, unfair to Grounding DINO in a control), `color-run` and
`bmx-bumps` (brightness swings come from occlusion of the lens), `dogs-scale`
(the object with the largest scale change is a hand), `classic-car` and
`cat-girl` (brightness changes come from pose, not light), `drone` (5 objects,
unclear target), `paragliding-launch` (canopy not annotated), `flamingo`
(other flamingos not annotated, unfair in a control), and `scooter-black`
(many parked scooters, unclear prompt). DAVIS has no
real topology change, so no DAVIS clip is in the topology group.

## How the VOT clips were chosen

VOT-ST2022 (the same 62 sequences as VOT-ST2020/2021) gives a segmentation
mask for every frame at 30 fps and per-frame tags such as `illum_change` and
`size_change`. It is not part of SAM 2's training data (SA-V). It was added
because DAVIS has too few illumination clips. For all 62 sequences we
computed the GT area change and the global and per-target brightness, read
the VOT tags, and checked the candidates by eye on contact sheets with the GT
outline and around every large frame-to-frame change (to find shot cuts).
For illumination clips `t_ev` is the first frame with the VOT
`illum_change` tag, checked by eye.

| Clip | Category | Frames | Why |
| --- | --- | --- | --- |
| `helicopter` | scale | 708 | Helicopter takes off and flies away: the body shrinks to ~1% of its frame-0 area. |
| `graduate` | scale | 844 | Man runs toward the camera (×24), then away again; one long panning shot. |
| `singer3` | scale | 117 of 131 | Camera zooms from a wide stage shot to the singer's face (×97). Cut before the shot cut at frame 117. |
| `bag` | illumination | 311 | Plastic bag blows from a lit crosswalk into near darkness; also shrinks. |
| `shaking` | illumination | 365 | Stage lights flash straight into the camera; target is the guitarist's head. |
| `singer2` | illumination | 323 of 366 | Stage lighting changes from warm to bright white. Cut before a one-frame insert at frame 323. |
| `fernando` | illumination | 292 | Cat moves between strong sunlight and shadow; a second (black) cat is not the target. |
| `birds2` | illumination | 80 | Crane flies from backlight into front light; also grows ×26, other cranes in view. |
| `gymnastics2` | illumination | 240 | The camera turns toward the hall's ceiling lights during the routine; also grows ×11. |
| `gymnastics3` | illumination | 118 | Vault in a dark, blurred arena seen against bright lights; also rotates. |
| `car1` | control | 742 | White SUV followed from a car; area p95/p5 1.7, no light change. |
| `rabbit2` | control | 100 | Hare in a field; area p95/p5 2.1. |
| `snake` | control | 101 | Sea snake on the sea floor; area p95/p5 2.6. |

VOT clips often combine several changes. Each clip is in the category of its
main event, and the table names the other changes. Rejected: `matrix` and
`kangaroo` (shot cuts), `soldier` and `tiger` (occlusion and camouflage
dominate), `drone1`, `marathon`, and `flamingo1` (target is tiny or one of
many look-alike objects), `polo`, `lamb`, `monkey`, and `surfing` (several
objects of the same class, unfair in a control), `wheel` (target is only part
of the bike), and `bubble` (transparent target). VOT is released for research
use; see <https://www.votchallenge.net>.

## How the download works

### DAVIS 2017

The official `DAVIS-2017-trainval-480p.zip` is saved to `data/davis/` and
extracted to `data/davis/extracted/DAVIS/JPEGImages/480p/<clip>/00000.jpg …`
and `…/Annotations/480p/<clip>/00000.png …`.

### M3-VOS clips

Only the selected clip folders are downloaded from the HuggingFace dataset
`Lijiaxin0111/M3_VOS`. Each frame is one request, and anonymous users are
limited to 1000 requests per 5 minutes, so the script waits 5 minutes and
retries when it hits the limit; files that are already there are skipped.
`hf auth login` with a free account gives a higher limit.

Result: `data/m3vos/data/JPEGImages/<clip>/0000000.jpg …` and
`data/m3vos/data/Annotations/<clip>/0000000.png …`. These names are already
numeric, so no renaming is needed.

### VOST clips (about 20 minutes)

`remotezip` (in `requirements.txt`) reads single files from the 54 GB remote
`VOST.zip` with HTTP range requests, so the full archive is never downloaded.
VOST names frames by their original 30 fps index (`frame00000`, `frame00006`,
…), but the benchmark reads frame numbers with `int(path.stem)` and matches
predictions to GT by numeric name, so the files are renamed to `00000`,
`00001`, …. One connection is used on purpose: parallel SSL connections
failed on our server. Files that already exist are skipped.

Result: `data/vost_clips/JPEGImages/<clip>/00000.jpg …` and
`data/vost_clips/Annotations/<clip>/00000.png …`.

### VOT clips

The sequence list `description.json` of VOT-ST2022 gives, per sequence, one
zip with the frames and one with the annotations. The frames
(`00000001.jpg`, …) are renamed to `00000.jpg`, `00001.jpg`, … like VOST.
Each line of `groundtruth.txt` is one run-length-encoded mask; it is decoded
to a PNG with value 1 (object ID 1). If `frames` is set in
`configs/clips.csv`, only that many frames and masks are written.

Result: `data/vot/JPEGImages/<clip>/00000.jpg …` and
`data/vot/Annotations/<clip>/00000.png …`.

## Things to report

- VOST is sampled at 5 fps, so SAM 2 sees larger motion between frames than
  on DAVIS or M3-VOS.
- Clip lengths differ: DAVIS 1.5–4.1 s, VOT 2.7–28 s, M3-VOS 10–18 s,
  VOST 10–35 s.
- M3-VOS, VOST, and VOT are downloaded clip by clip, not as full archives, so a
  new clip must be added to `configs/clips.csv` before it can be downloaded.
