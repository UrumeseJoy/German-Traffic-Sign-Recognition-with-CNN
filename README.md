# German Traffic Sign Recognition with a CNN

A convolutional neural network, built with TensorFlow/Keras, that classifies photos of German
traffic signs into the 43 classes of the
[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html).

Everything lives in a single notebook:
[`German_Traffic_Sign_Recognition_Benchmark.ipynb`](German_Traffic_Sign_Recognition_Benchmark.ipynb).

## Results

| Split | Images | Accuracy |
|---|---|---|
| Validation (held-out physical signs from `Train.csv`) | 5,850 | 99.01% |
| **Official GTSRB test set (`Test.csv`)** | 12,630 | **98.53%** |

Trained on a Colab T4 GPU (about 10 seconds per epoch). The best epoch, chosen by validation
accuracy, was epoch 22 of 32. Results can vary by a few tenths of a percent between runs
because GPU training is not bit-for-bit deterministic.

**Most common mistakes on the test set:** *60 km/h* read as *80 km/h* (32 times), *Beware of
ice/snow* as *Bicycles crossing* (28), and *Bumpy road* as *Slippery road* (17). These are
visually similar signs that become hard to tell apart at 32×32 pixels.

> An earlier version of this notebook reported 99.93%. That figure came from a random split of
> `Train.csv`, in which near-identical frames of the same physical sign ended up in both
> training and test data, so it overstated real performance. The numbers above use the
> official test set and a leak-free split.

## Dataset

- **Source:** the GTSRB Kaggle mirror
  [`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (CC0, ~600 MB)
- **Training data:** 39,209 images from about 1,300 physical signs. Each sign is captured as a
  *track* of ~30 consecutive frames.
- **Test data:** 12,630 images (the official benchmark test set)
- **Classes:** 43, ranging from speed limits to warnings, prohibitions and mandatory directions.
  The class names are listed in the notebook (`SIGN_NAMES`).
- Each image comes with a bounding box (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`) around the sign.

## Method

1. **Preprocessing:** crop each image to its sign bounding box, then resize it to 32×32 RGB.
   Images are loaded in parallel and kept as `uint8`; scaling to `[0, 1]` happens inside the
   model.
2. **Leak-free split:** for each class, ~15% of its tracks are held out for validation. All
   frames of a sign stay on the same side, and every class is represented.
3. **Augmentation**, applied to training batches only: random rotation (±15°), shifts (10%),
   zoom (20%) and contrast changes. There are no horizontal flips, because flipping turns
   *Turn left* into *Turn right*.
4. **Model:**

   ```
   Input 32×32×3 → Rescaling(1/255)
   [Conv 32 3×3 → BN] ×2 → MaxPool → Dropout 0.25
   [Conv 64 3×3 → BN] ×2 → MaxPool → Dropout 0.25
   Flatten → Dense 256 → Dropout 0.5 → Dense 43 (softmax)
   ```
   About 487k parameters.
5. **Training:** Adam (learning rate 1e-3), sparse categorical cross-entropy, batch size 64,
   up to 50 epochs. Early stopping, checkpointing and learning-rate reduction all track
   `val_accuracy`. Seeds are fixed (`SEED = 42`) for reproducibility.
6. **Evaluation** on the official test set covers accuracy, a per-class report with sign names,
   a confusion matrix, the most common confusions, and a grid of misclassified images.

## How to run

### Google Colab (easiest)

1. Open the notebook in Colab: **File → Open notebook → GitHub**, then paste this repo's URL.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Add your Kaggle credentials as **Colab Secrets** (🔑 icon in the sidebar): `KAGGLE_USERNAME`
   and `KAGGLE_KEY`, with notebook access enabled.
4. Choose **Runtime → Run all**. The dataset downloads automatically, and the trained model is
   saved as `gtsrb_cnn.keras` and downloaded at the end.

### Locally

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn pillow kagglehub jupyter
# Optional: point to an already-downloaded copy instead of downloading again
export GTSRB_DIR=/path/to/gtsrb   # the folder that contains Train.csv, Test.csv, Meta.csv
jupyter notebook German_Traffic_Sign_Recognition_Benchmark.ipynb
```

If `GTSRB_DIR` isn't set, the notebook downloads the dataset with `kagglehub`. That uses
`~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.

> **Keep your Kaggle key out of the notebook.** Don't upload `kaggle.json` with
> `files.upload()` or print it: cell outputs are saved in the `.ipynb` file and become
> public when it's pushed.

### Using the trained model

```python
import keras, numpy as np
from PIL import Image

model = keras.models.load_model("gtsrb_cnn.keras")
img = Image.open("sign.png").convert("RGB").resize((32, 32))   # photo cropped to the sign
probs = model.predict(np.asarray(img, dtype=np.uint8)[None])[0]
print(int(probs.argmax()), float(probs.max()))                  # class id, confidence
```

The model takes raw 0–255 RGB images; no manual normalization is needed.

## Project structure

```
.
├── German_Traffic_Sign_Recognition_Benchmark.ipynb   # data, training, evaluation
└── README.md
```

## Acknowledgements

J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. *Man vs. computer: Benchmarking machine
learning algorithms for traffic sign recognition.* Neural Networks, 32, 2012.
