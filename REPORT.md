# Task 4 — what I did to make the leaf classifier better

I gave the agent this instruction: start from the Task 2 pipeline, change **one thing at a
time**, always against the same baseline and the same training images with a fixed seed,
train every variant with three seeds, and measure not only on the validation set but also
on a camera-like set and on last year's photographs. Keep the code simple and readable, no
rewriting, no extra layers I would have to explain.

## How I measured

* **Two baselines.** `B1` is the small network from Task 3, `B2` is the MobileNetV3-Small
  of Task 2 (15 epochs, as in the notebook). Everything else is one change away from one
  of them.
* **Three seeds** (0, 1, 2) for every row. The group validation set has 29 pictures, so one
  picture is 3.4 percentage points and one run means nothing by itself.
* **Three test sets.** `group val 29` are the pictures the other students in my group took,
  the same set Task 2 reports. `camera-like 29` is those same pictures randomly cropped,
  rotated and recoloured **once** with a fixed seed, so every variant is measured on exactly
  the same images. `last-year 120` is the historical validation set, other people, other
  phones, 175 pictures per species added to the training set in the variants that say `+hist`.
* **Two weights per run**, the last epoch and the epoch with the lowest validation loss. The
  second one is chosen using the group validation set, so the accuracy reported on that same
  set flatters it a bit. On the camera set and on last year's pictures the choice was made
  on other images, so those two columns are the fair ones.

One thing to keep in mind: in the `+hist` rows the historical **training** pictures were used
for training. The 120 validation pictures were never trained on, but they are no longer
foreign data, so I read that column as "how well does it know these four species" and I
compare variants on the two columns with 29 pictures.

## The table

Accuracy in %, mean ± spread over the three seeds, weights of the best epoch.

| variant | train images | group val 29 | camera-like 29 | last-year 120 |
|---|---:|---:|---:|---:|
| B1 leafnet (Task 3 baseline) | 107 | 60.9 ± 10.7 | 40.2 ± 4.3 | 33.9 ± 1.4 |
| S1 + batch normalization | 107 | 66.7 ± 3.3 | 44.8 ± 4.9 | 39.7 ± 0.4 |
| S2 + augmentation | 107 | 58.6 ± 2.8 | 52.9 ± 8.6 | 39.2 ± 2.5 |
| S3 + BN + augmentation | 107 | 69.0 ± 0.0 | 64.4 ± 3.3 | 40.8 ± 2.0 |
| S4 + BN + aug + global avg pool | 107 | 52.9 ± 9.1 | 50.6 ± 5.9 | 38.1 ± 4.9 |
| S5 + BN + aug + GAP + dropout 0.3 | 107 | 51.7 ± 8.4 | 59.8 ± 4.3 | 36.1 ± 3.9 |
| B2 mobilenet 15 ep (Task 2 baseline) | 107 | 63.2 ± 8.1 | 58.6 ± 4.9 | 44.7 ± 3.7 |
| T1 frozen backbone, last layer only | 107 | 66.7 ± 3.3 | 64.4 ± 1.6 | 43.1 ± 1.6 |
| T2 + augmentation | 107 | 69.0 ± 7.4 | 62.1 ± 8.4 | 46.9 ± 5.0 |
| T3 + aug + cosine, 40 epochs | 107 | 70.1 ± 5.9 | 60.9 ± 7.1 | 48.6 ± 1.0 |
| T4 + AdamW wd 0.05 | 107 | 70.1 ± 8.6 | 73.6 ± 3.2 | 48.1 ± 5.7 |
| T5 + label smoothing 0.1 | 107 | 78.2 ± 8.1 | 75.9 ± 9.8 | 53.1 ± 2.2 |
| T6 + last year's 700 photos added | 807 | 81.6 ± 5.9 | 70.1 ± 3.2 | 91.9 ± 0.8 |
| T7 last year's photos only | 700 | 70.1 ± 7.1 | 70.1 ± 8.6 | 85.3 ± 2.7 |
| C1 T6 + AdamW wd 0.05 | 807 | 82.8 ± 2.8 | 74.7 ± 9.1 | 88.9 ± 4.0 |
| C2 T6 + label smoothing | 807 | 78.2 ± 4.3 | 70.1 ± 1.6 | 90.6 ± 0.4 |
| C3 T6 + AdamW + smoothing | 807 | 77.0 ± 3.3 | 75.9 ± 8.4 | 90.3 ± 1.0 |

102 runs in total. The raw numbers of every run are in this repository, in
`results/results_task4_raw.csv` and `results/results_combos_raw.csv`, and the table above is
what `results/summarize.py` prints from those two files, so no number here is typed by hand.
The code that trained them is `leaves_lib.py`, `run_task4.py` and `export_phone_model.py` in my
course folder.

## What each change brought, and whether it is more than the noise

**Batch normalization alone: about +6 points, and steadier runs.** On the from-scratch line BN
makes the difference between training and not training (60.9 → 66.7 on group val, and the spread
over seeds drops from 10.7 to 3.3). It is the first change whose effect I would trust on a set
of 29 pictures, and it shows on last year's pictures too (33.9 → 39.7).

**Augmentation alone: nothing on the clean validation set, +13 points on the camera set.**
58.6 against 60.9 is well inside the noise, but 40.2 → 52.9 on the camera-like pictures is
not. That makes sense: the validation pictures are straightened and centred, so augmentation
buys nothing there, and it pays exactly where it was designed to pay.

**BN and augmentation together: the best from-scratch row, 69.0 / 64.4.** Both changes help
alone and they do not cancel each other. This is 8 points over B1 on group val and 24 points
on the camera set. I would call the camera number real, the group val number likely real.

**Global average pooling and dropout: worse, and clearly so.** GAP drops the group val from
69.0 to 52.9 and the spread grows to 9.1. Adding dropout 0.3 on top does not recover it. With
107 training pictures I expected the opposite, so this is the result that surprised me most.
My reading is that the flattened head of my small net is not what overfits here; with
6 × 6 × 64 features and 107 pictures, GAP throws away the position information that separates
a maple leaf from a chestnut leaf, and the seeds then disagree a lot.

**Frozen backbone: no.** Only the last layer trained gives 66.7 / 64.4 against 63.2 / 58.6 for
the fine-tuned baseline: better, but the difference is about one picture and its spread
overlaps. The pretrained features are already good on 107 pictures. Fine-tuning does not hurt
either, and the whole network is still only 1.5 M parameters.

**Pretrained beats from-scratch, but the interesting part is the last-year column.** On our own
validation pictures the two lines end at the same place once both have augmentation (S3 69.0,
T2 69.0). The two baselines without any augmentation show where the pretrained features come
from: 63.2 / 58.6 / 44.7 for MobileNet against 60.9 / 40.2 / 33.9 for my own network. On the
camera set and on the 120 pictures from other people and other phones the pretrained network is
clearly ahead, and that is where my own small net has no chance.

**Longer training with a cosine schedule: +1 point on group val, +2 on last year.** T3 against
T2 is not a real gain, but it makes the rest possible: with 40 epochs the last epoch and the
best epoch stop disagreeing (71.3 last against 70.1 best for T3, while the 15-epoch baseline
is 57.5 last against 63.2 best).

**Label smoothing 0.1: the biggest single-knob gain, +8 points on group val and +5 on last
year.** 78.2 / 75.9 / 53.1 is the best row of the small-data line. The spread stays around 8
points, so the group val gain is only about one seed-to-seed step, but the same direction on
the camera set (60.9 → 75.9 against T3, the row smoothing is added to) and on last year's
pictures makes it believable to me.

**AdamW with weight decay 0.05: helps the camera, not the validation set.** 70.1 group val
(same as T3, inside the noise), 73.6 camera (up from 60.9, larger than the spread).

**Last year's 700 pictures: the only change that moves the numbers by a lot.** 81.6 / 70.1 /
91.9 against 78.2 / 75.9 / 53.1 for the best small-data row. The last-year column jumps by
almost 39 points, which is far more than any spread in the table. Our own 107 pictures are the
bottleneck, not the network and not the optimizer.

**Last year's pictures alone (no group pictures): worse on our pictures.** 70.1 on group val
against 81.6 when our pictures are added, and it does not win anywhere else either. So the
700 pictures are worth keeping **next to** ours, not instead of ours.

**The best combinations did not beat T6.** C1 (T6 plus AdamW) is 1.2 points higher on group val
with a smaller spread but 3 points lower on last year's pictures, C2 and C3 are lower
everywhere or equal. All these differences are smaller than the seed-to-seed spread, so I take
them as "no difference" and keep the simpler variant.

## The model I put on the phone

**T6: MobileNetV3-Small, our 107 pictures plus last year's 700, augmentation, cosine schedule,
40 epochs, Adam, keeping the best epoch.** Chosen because it is the row that wins on the two
columns where the epoch choice cannot flatter it, and the combinations I tried on top of it
are not better.

The exported file is one run with seed 0, and it kept epoch 13 of 40. Its own numbers, measured
again after the export (`report_numbers.py`):

| set | accuracy |
|---|---|
| group val 29 (other students in my group) | 75.9 % (22/29) |
| camera-like 29 (fixed seed) | 72.4 % (21/29) |
| last-year val 120 (never trained on) | 90.8 % (109/120) |

The three seeds of the T6 row give 75.9, 79.3 and 89.7 % on group val, so the exported run is
the weakest of the three and 5.7 points below their mean. That is the spread the table warns
about, and it is the reason I do not trust any single run on 29 pictures. (Retraining with seed
0 kept epoch 13 while the run behind the table kept epoch 17; the runs on the GPU are not
identical to the last bit, which says the same thing.) The ONNX file is 6.1 MB and one
96 × 96 inference takes 0.8 ms on my laptop's CPU, so the phone browser keeps up with the
camera. In PyTorch and in the browser the exported file gives probabilities that differ by
4e-7 at most, so what the page shows is what the network computed.

What it still gets wrong on our validation pictures is mostly apple against cherry: 2 apples
read as cherry, 2 cherries read as apple, 1 chestnut read as apple, 1 chestnut as cherry, and
1 maple as chestnut. Chestnut is the weakest class, but our validation set has only four
chestnut pictures, so I would not read much into the per-class percentages.

I exported it exactly as in 2e (`leaf_model.onnx` with the classes and the preprocessing in its
metadata, plus `model_info.json`, `check_image.jpg` and `check_expected.json`) and it is the
model running on <https://Matus-smolka.github.io/mlr-ex08/>, which is the same four files the
course camera app reads.

## What I would try next

More colour augmentation, since the camera column is still 8 points behind the clean one and
the light in the room changes far more than the crop. More chestnut pictures, we have the
fewest of that species. And a bigger validation set: three seeds on 29 pictures still leaves
a spread of 3 to 10 points, which is larger than every difference in the table except the
700 extra photographs and the two that did not help.
