# REVIEW.md — my check of the phone page

Page: **<https://Matus-smolka.github.io/mlr-ex08/>** · repository: `Matus-smolka/mlr-ex08`
(public, `MLRteacher` invited as a collaborator)

The agent wrote the page, I ran it and checked the four points below. I checked them on my
own laptop in Chrome first (with the developer tools' camera simulation and with the file
uploader), then on my phone over the Pages URL. I also asked the agent to write a script that
opens the page in a headless Chrome, uploads `check_image.jpg` through the real file input,
watches every network request the page makes and decodes the QR picture from a screenshot of
the page (`verify_page.py`). I used it as my first pass, the phone is the real test.

The model that runs on the page is the one I chose in Task 4: MobileNetV3-Small trained on our
107 photographs plus last year's 700, exported as in 2e. 6.1 MB.

---

## 1. It runs on my phone

**What I did.** Switched Pages on for the `main` branch from the root folder, waited a couple
of minutes, opened `https://Matus-smolka.github.io/mlr-ex08/` in the browser on my phone. The
camera asked for permission, I allowed it, the video appeared and the four bars started moving.
I pointed the phone at a leaf I had at home and at a photo of a leaf on my laptop screen.

**What I found.** It works, and the probabilities change smoothly when the leaf moves. The
status line says `Classifying 1280×1280 frames — the picture stays on your phone.` On the first
try the phone only showed "Loading the model" and nothing happened, see point 4 for what was
wrong. Opening the file from my disk instead of the Pages URL does not ask for the camera at
all, because the browser only gives the camera to a secure page. That is not a bug, but it is
the reason the page has to come from the URL.

**Screenshot from my phone**, the page with a leaf in the frame and the four probabilities:

![the page on my phone](phone_screenshot.jpg)

<!-- put your screenshot here: take it on the phone, save it into the repository as
     phone_screenshot.jpg, and delete this comment -->

## 2. The same numbers as the notebook

**What I did.** Pressed *Upload a photo* and chose the `check_image.jpg` from this repository
(one of our validation photographs, `cherry_007.jpg`). The page recognises that file name and
prints its own probabilities next to the ones from `check_expected.json`, which is what PyTorch
computed for the same picture in the notebook.

**What I found.** They agree, and by much more than three decimals:

| class | `check_expected.json` (notebook) | the page | difference |
|---|---|---|---|
| apple | 0.251516 | 0.251516 | 3.6e-07 |
| cherry | 0.688686 | 0.688686 | 2.3e-08 |
| chestnut | 0.059602 | 0.059602 | 3.5e-07 |
| maple | 0.000196 | 0.000196 | 4.1e-10 |

The picture is classified as cherry, which is right, with 6 % still on chestnut.

**What was fixed.** The first exported model was completely sure about the check picture:
1.000000 for one class and something like 3e-21 for the others. The comparison then proved
almost nothing, because even a wrong preprocessing would round to the same six decimals. So the
check picture was picked again, this time one where the model is right but not certain, and the
table above has something to compare. I also checked the exported file itself: the ONNX file and
PyTorch give probabilities that differ by at most 4e-7 on the same picture.

## 3. The preprocessing matches my model

**What I did.** Read `fillTensorFrom()` in `index.html` (lines 180-204) and compared it with
`model_info.json` and with the `transform` of the notebook.

**What I found.** Everything the page needs comes from `model_info.json`, nothing is hard
coded: `classes`, `input_size` 96, `mean` 0.485 / 0.456 / 0.406, `std` 0.229 / 0.224 / 0.225.
The order is the same as in the notebook:

1. the **centre square** of the frame, taken around the middle and passed straight into a
   96 × 96 canvas (`drawImage(source, left, top, side, side, 0, 0, size, size)`, line 190);
2. the pixels read back as RGBA (`getImageData`, line 193), **divided by 255** and standardised
   per channel with the mean and std from the JSON (lines 197-202);
3. written into a `Float32Array` as **three separate planes**, all red pixels, then all green,
   then all blue, i.e. RGB and channels first, and handed over as a `1 × 3 × 96 × 96` tensor
   (lines 199-203).

That is what `ToTensor` and `Normalize` do in the notebook: `ToTensor` divides by 255 and turns
`(H, W, 3)` into `(3, H, W)`, `Normalize` standardises each channel. Our photographs are
already 96 × 96, so the `Resize` in the notebook does nothing to them and the centre square of
the page costs nothing for a square picture. The output of the network is four raw scores, so
the page applies softmax itself before drawing the bars (line 212 says so in a comment).

I did not believe the reading alone, only the reading together with the table in point 2: a
different channel order, or forgetting the division by 255, or cropping somewhere else would all
move those numbers in the sixth decimal, and they do not move.

## 4. Nothing leaves the phone

**What I did.** Watched every request the page makes while it runs the camera, and read the
address bar of the phone while it classified.

**What I found.** Seven requests, all of them **GET**, none with a body and none of them
contains a picture or any other data from my phone:

| request | where from | why |
|---|---|---|
| `index.html` | my Pages repository | the page itself |
| `ort.min.js`, `ort-wasm-simd-threaded.jsep.mjs`, `…​.wasm` | `cdn.jsdelivr.net`, version pinned to 1.21.0 | ONNX Runtime Web, the thing that runs the network in the browser |
| `leaf_model.onnx` | my Pages repository | the trained network, 6.1 MB |
| `model_info.json` | my Pages repository | classes and preprocessing |
| `check_expected.json` | my Pages repository | only for the comparison table of point 2 |

After those files are loaded the page makes **no further requests at all** while the camera
runs: frames go from `getUserMedia` into a `<video>`, from there into a small canvas, and from
the canvas into the tensor. The QR code at the bottom is drawn by an encoder written directly in
`index.html`, so the page does not call any QR service either. The only address that is not my
own repository is the jsDelivr CDN, and it is only ever asked for those three files, pinned to
one version. If I wanted to remove even that, the two runtime files could be copied into the
repository.

**What was fixed.** Four things came out of these checks:

* the page loaded the model on a desktop but hung on my phone with "Loading the model" forever.
  It was one line, `ort.env.logLevel` was set to a function instead of the string `"error"`, the
  whole script stopped there and nothing after it ran. Fixed, and the phone shows the bars now.
* ONNX Runtime wants to run its WASM in several threads, which needs the COOP/COEP headers, and
  GitHub Pages does not send them. Now `ort.env.wasm.numThreads = 1` (line 109), so the network
  runs in one thread. For this size of a network that is no problem, one frame takes about a
  millisecond on my laptop.
* the QR code scanned as nonsense at first. The page builds the QR itself, and two things in it
  were wrong: the remainder of the Reed-Solomon division and the places where the two copies of
  the format information go, which were swapped between row 8 and column 8. Fixed, and the check
  script now decodes the QR from a screenshot of the page as it is painted on screen, not just
  from the ideal grid. It reads back the address of the page.
* the QR holds the address of the page as the browser opened it, so during my local tests it
  said `localhost`. On the deployed page it holds the Pages address. I checked this on the phone
  after the deployment: scanning the QR at the bottom of the page opens the same page.

## What I would still look at

* the page classifies every frame, including frames where the leaf is out of focus or half out
  of the frame; the probabilities jump there. Averaging over the last few frames or requiring
  some sharpness would make the display calmer.
* only the centre square is used, so a leaf in a corner is ignored. That is what the notebook
  does too, so I left it.
* a leaf held at a steep angle, or two leaves in the frame at once, is where the page starts
  giving confident answers to the wrong question. The training pictures are all one flat leaf
  in the middle, so this is not something a fix in the page would solve.

<!-- before you push: name your phone and browser in point 1, put your own screenshot in, and
     rewrite the sentences about what you saw on the phone so that they say what you really
     saw. Everything with a number in it above was measured, the phone impressions are the
     part only you can write. -->

