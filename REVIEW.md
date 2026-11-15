# REVIEW.md — checking the phone page

Page: **<https://Matus-smolka.github.io/mlr-ex08/>** · repository: `Matus-smolka/mlr-ex08`
(public, `MLRteacher` invited as a collaborator)

The agent wrote the page and I went through it myself. I first checked it on my laptop in Chrome, using the developer tools camera simulation and the file uploader. After that I opened the deployed page on an Oppo Reno-series phone using Chrome. I also asked the agent to make a script (`verify_page.py`) that opens the page in headless Chrome, uploads `check_image.jpg` through the actual file input, watches the network requests and decodes the QR code from a screenshot of the page. This was useful for checking things quickly, but I still treated the phone as the real test.

The model running on the page is the one I selected in Task 4: MobileNetV3-Small, trained on our 107 photographs plus last year's 700 photographs and exported as in 2e. The ONNX file is 6.1 MB.

---

## 1. Does it actually work on the phone?

I enabled Pages for the `main` branch from the root folder and waited a couple of minutes for it to deploy. Then I opened the URL on my phone. The browser asked for camera permission, I allowed it and the video appeared. The four probability bars also started moving. I tried pointing it at a leaf I had at home and also at a picture of a leaf displayed on my laptop.

From what I saw, it works as expected. The probabilities change when I move the leaf around, and the status line says `Classifying 1280×1280 frames — the picture stays on your phone.` On my first attempt it only showed "Loading the model" and did not go any further. I found the reason for that later, so I explain it in point 4.

One thing worth mentioning is that opening the HTML file directly from my disk does not give the page access to the camera. This is because the browser only allows camera access on a secure page, so it needs to be opened through the Pages URL. I do not consider this a bug.

**Screenshot from my phone:**

![the page on my phone](phone_screenshot.jpg)

## 2. Checking whether the page gives the same results as the notebook

For this test I pressed *Upload a photo* and selected `check_image.jpg` from the repository. This is one of our validation photographs, `cherry_007.jpg`. The page recognises the filename and displays its own probabilities next to the values stored in `check_expected.json`, which were calculated by PyTorch in the notebook.

The results match very closely:

| class | `check_expected.json` (notebook) | the page | difference |
|---|---|---|---|
| apple | 0.251516 | 0.251516 | 3.6e-07 |
| cherry | 0.688686 | 0.688686 | 2.3e-08 |
| chestnut | 0.059602 | 0.059602 | 3.5e-07 |
| maple | 0.000196 | 0.000196 | 4.1e-10 |

The image is classified as cherry, which is correct, although there is still around 6 % probability assigned to chestnut.

There was one problem with the original check image. The first exported model was almost completely certain about its prediction: it gave 1.000000 to one class and something around 3e-21 to the others. This was not very useful for comparing the two implementations, because even if the preprocessing was slightly wrong, the displayed values could still round to the same six decimals. I therefore picked another image where the model predicts the correct class but is not completely certain. This gives us something more useful to compare.

I also checked the exported ONNX model against PyTorch directly. The probabilities differ by at most 4e-7 for the same image, so the export itself seems to be fine as well.

## 3. Is the preprocessing the same as in my model?

I checked the `fillTensorFrom()` function in `index.html` (lines 180-204) and compared it with `model_info.json` and the `transform` used in the notebook.

From what I can see, the page gets the required settings from `model_info.json` instead of having them separately hard coded. This includes the classes, input size of 96, mean values of 0.485 / 0.456 / 0.406 and standard deviations of 0.229 / 0.224 / 0.225.

The actual order of operations is also the same:

1. It takes the **centre square** of the camera frame and draws it into a 96 × 96 canvas (`drawImage(source, left, top, side, side, 0, 0, size, size)`, line 190).
2. It reads the pixels back as RGBA using `getImageData` (line 193), divides the values by 255 and standardises each channel using the mean and standard deviation from the JSON file (lines 197-202).
3. It puts the values into a `Float32Array` as three separate planes: first red, then green and finally blue. In other words, it uses RGB with channels first and passes a `1 × 3 × 96 × 96` tensor to the network (lines 199-203).

This is what `ToTensor` and `Normalize` do in the notebook. `ToTensor` divides the pixel values by 255 and changes the image from `(H, W, 3)` to `(3, H, W)`, while `Normalize` standardises each channel. Our training photographs are already 96 × 96, so the `Resize` operation in the notebook does not actually change them. The centre crop on the page also does not change a square image.

The network returns four raw scores, so the page applies softmax before displaying the probability bars. This is also mentioned in the comment on line 212.

I did not want to rely only on reading the code and deciding that it looked correct. The comparison in point 2 is a much better check. If the channel order was different, the division by 255 was missing or the image was cropped differently, I would expect the probabilities to change. They do not, at least not beyond the very small differences shown in the table.

## 4. Does anything from the phone get sent somewhere?

I checked the network requests made by the page while the camera was running. I also watched the address bar on the phone while it was classifying.

There were seven requests in total. All of them were **GET** requests, with no request body containing a picture or other data from the phone:

| request | where from | why |
|---|---|---|
| `index.html` | my Pages repository | the page itself |
| `ort.min.js`, `ort-wasm-simd-threaded.jsep.mjs`, `…​.wasm` | `cdn.jsdelivr.net`, version pinned to 1.21.0 | ONNX Runtime Web, which runs the network in the browser |
| `leaf_model.onnx` | my Pages repository | the trained network, 6.1 MB |
| `model_info.json` | my Pages repository | classes and preprocessing settings |
| `check_expected.json` | my Pages repository | used only for the comparison in point 2 |

After these files are loaded, the page does not make any more requests while the camera is running. The frames go from `getUserMedia` into a `<video>`, then into a small canvas and finally into the tensor. The QR code at the bottom is also generated directly by code in `index.html`, so there is no external QR service involved.

The only external address is the jsDelivr CDN. It is used to load the three ONNX Runtime files, and the version is pinned to 1.21.0. If I wanted to remove this external dependency completely, I could copy those runtime files into the repository as well.

While checking this part, I found and fixed four issues:

* The page worked on my desktop but got stuck on my phone at "Loading the model". The problem was one line: `ort.env.logLevel` was assigned a function instead of the string `"error"`. This stopped the script before the rest of it could run. After fixing it, the model loaded and the probability bars appeared on the phone.
* ONNX Runtime normally wants to use multiple WASM threads, but this requires COOP/COEP headers which GitHub Pages does not provide. I set `ort.env.wasm.numThreads = 1` (line 109), so it now runs in one thread. For a network this small it does not seem to be a problem; one frame takes about a millisecond on my laptop.
* The QR code initially scanned as nonsense. The page generates the QR code itself, and there were two mistakes in the implementation: the remainder of the Reed-Solomon division and the positions of the two copies of the format information were wrong (they were swapped between row 8 and column 8). These were fixed. The verification script now decodes the QR code from a screenshot of the page as it is actually displayed, rather than only checking the ideal QR grid. It reads the page address from the code.
* During local testing the QR code contained `localhost`, because that was the address used to open the page. On the deployed version it contains the Pages address instead. I checked this after deployment by scanning the QR code with my phone, and it opened the same page.

## What I would still improve

There are a few things that work but could be better:

* The page classifies every frame, even if the leaf is out of focus or only partly visible. In those situations the probabilities can jump around quite a bit. Averaging the results over a few frames or checking image sharpness could make the display more stable.
* Only the centre square of the camera frame is used. This means that a leaf in the corner is ignored. It is the same approach as in the notebook, so I left it as it is.
* If the leaf is held at a steep angle, or if there are two leaves in the frame, the page can give a confident answer to the wrong question. The training photographs show one flat leaf in the middle, so I do not think this is something that can really be fixed just by changing the page.
