# Metal Tube Inspection

Count metal tubes and detect anomalous tubes in photographs using classical image processing. Task 1 locates the tubes and draws rotated bounding boxes. Task 2 checks the reflections and other evidence inside those boxes.

## Setup

Run these commands from the project root, where `TASK1.py` and `TubesDataset/` are located. The dataset is included in the repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requierments.txt
```

On Windows, create the environment with `py -m venv .venv` and activate it in PowerShell with `.\.venv\Scripts\Activate.ps1`. Then run the same install command. The `opencv-python` package provides the `cv2` import.

To run the notebook, install JupyterLab in the same environment and launch it from the project root.

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

Open `color.ipynb`, select the Python kernel for your environment, and run the cells in order.

## Run the tasks

```bash
python TASK1.py
python TASK2.py
```

By default, Task 1 evaluates and displays `0003.png`. Task 2 evaluates the entire dataset and prints the results without opening plots.

To select images, set `images` at the bottom of the task file, inside the `if __name__ == "__main__":` block. Use one of these settings.

```python
images = ["TubesDataset/0001.png"]                           # One image
images = ["TubesDataset/0001.png", "TubesDataset/0009.png"]   # Selected images
images = None                                                # All annotated images
```

Set `vis = True` to show the detections or `vis = False` to print the evaluation only. When plots are enabled, close the current plot window to continue.

Both tasks print the predicted count, expected count, and signed error for each image. A positive error means the task counted too many tubes or anomalies. The summary shows the number of images with an exact count and the standard deviation of the count errors across images.

## File guide

| File | Role |
| --- | --- |
| [`TASK1.py`](TASK1.py) | Builds the opened Otsu, reflection, and Canny masks. Starts from the Otsu regions and refines them using reflections and Canny. Each detection contains a label, center position, pixel mask, rotated box, and a flag for Canny-only detections. |
| [`TASK2.py`](TASK2.py) | Reuses preprocessing and detections from Task 1. Fits a line to the reflection inside each box and checks perpendicular spread and reflection length. It also checks the Canny-only flag and whether the box contains foreground pixels. |
| [`tube_edge_cases.py`](tube_edge_cases.py) | Contains the helpers that Task 1 uses to filter weak regions, merge fragments, split joined or unusually long clusters, and recover missed tubes from Canny edges. |
| [`color.ipynb`](color.ipynb) | Explores color channels and compares the masks visually. |
| [`TubesDataset/`](TubesDataset/) | Contains 30 numbered PNG images and the reference annotations used by the evaluation. |

## Dataset annotations

[`TubesDataset/annotations.json`](TubesDataset/annotations.json) contains one entry per image.

| Field | Meaning |
| --- | --- |
| `filename` | Name of the PNG file in `TubesDataset/`. |
| `n_tubes` | Reference tube count used by Task 1. |
| `anomalous` | Indicates whether the image contains an anomalous tube. |
| `bboxes` | Reference boxes for anomalous tubes. Task 2 uses the number of boxes as the expected anomaly count. |

The evaluation compares counts per image. The plots show where the scripts located the tubes or anomalies.

