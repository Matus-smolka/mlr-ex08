# Leaf classifier in the browser — MLR exercise 8

**Page: <https://Matus-smolka.github.io/mlr-ex08/>**

Point your phone camera at a leaf and the page says whether it is **apple, cherry,
chestnut or maple**. The network runs in the browser of the phone; the picture is not
sent anywhere. Open it from the link, or scan the QR code at the bottom of the page
with the same phone.

## What is in this repository

| File | What it is |
| --- | --- |
| `index.html` | the whole page: camera, preprocessing, inference, one plain JavaScript file |
| `leaf_model.onnx` | the trained network, MobileNetV3-Small exported from PyTorch (also carries the classes and the preprocessing as metadata) |
| `model_info.json` | classes, input size 96, the ImageNet mean and std used in training |
| `check_image.jpg` | one of our validation photographs |
| `check_expected.json` | the probabilities PyTorch computes for that photograph |
| `REVIEW.md` | my check of the four requirements of Task 2 |
| `REPORT.md` | Task 4: what I tried to make the classifier better, and what it brought |
| `results/` | the raw numbers of all 102 training runs of Task 4 and the script that prints the table in `REPORT.md` |

## How it works

`onnxruntime-web` (from the jsDelivr CDN, pinned to 1.21.0, WASM backend — it works in
Safari on iPhone as well as in Chrome on Android) runs `leaf_model.onnx`. Before the
network sees anything, the page builds the input exactly the way the training code did,
reading the numbers from `model_info.json`:

```
the centre square of the frame → resized to 96 × 96 → RGB pixels ÷ 255
→ each channel standardised with mean and std → float32 tensor 1 × 3 × 96 × 96
```

Classes and preprocessing are not hard-coded: change `model_info.json` and the page
follows it. Upload a photo instead of using the camera by pressing **Upload a photo**;
uploading `check_image.jpg` also prints a table of the page's probabilities next to the
notebook's, which is how requirement 2 of Task 2 is checked.

## Trained on

Photographs taken by our group in the lab, plus the previous years' photographs of the
same four species (`REPORT.md` has the measurements and the comparison). Split and
preprocessing follow the exercise notebook; the training code is `leaves_lib.py`,
`train_task2.py`, `run_task4.py` and `export_phone_model.py` in the course folder.
