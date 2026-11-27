# Metal Tube Inspection

Count metal tubes and detect anomalous tubes in photographs using classical image processing. Task 1 locates the tubes and colores tube-clusters. Task 2 checks the reflections and other evidence inside those boxes, and draws bounding boxes around defect tubes.

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

By default, Task 1 evaluates and displays `0014.png`. Task 2 evaluates and displays `0001.png`.

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
| [`TASK1.py`](TASK1.py) | Counts the tubes in each image and shows where they are. |
| [`TASK2.py`](TASK2.py) | Finds and marks tubes that look damaged or deformed. |
| [`tube_edge_cases.py`](tube_edge_cases.py) | Helps Task 1 handle cases such as tubes touching or appearing in separate pieces. |
| [`color.ipynb`](color.ipynb) | Shows how different image-processing steps highlight the tubes. |
| [`TubesDataset/`](TubesDataset/) | Contains the 30 images and the answers used to check the results. |

## Dataset annotations

[`TubesDataset/annotations.json`](TubesDataset/annotations.json) contains one entry per image.

