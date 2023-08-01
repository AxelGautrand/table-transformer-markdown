"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

from pathlib import Path, PurePath


def plot_logs(
    logs,
    fields=("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col=0,
    log_name="log.txt",
):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}"
            )

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument:\n{dir}"
            )
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == "mAP":
                coco_eval = (
                    pd.DataFrame(np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1])
                    .ewm(com=ewm_col)
                    .mean()
                )
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f"train_{field}", f"test_{field}"],
                    ax=axs[j],
                    color=[color] * 2,
                    style=["-", "--"],
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(
        files, sns.color_palette("Blues", n_colors=len(files)), names
    ):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs


def plot_structure(outputs, objects=["table row", "table column"]):
    """
    Performs visualization of the structure recognition on an input image. Is supposed to be run after extract() method of the TableExtractionPipeline class.

    Parameters
    ----------
    outputs : List type, output of the extract() method. Each index represents a table detected on the image and are dict objects contaning the cropted table and the bounding boxes.
    objects : List type, List of objects to plot. Default is ["table row", "table column"].
    """

    # Create a list of colors for each label
    label_colors = {
        "table": "yellow",
        "table row": "blue",
        "table column": "red",
        "table column header": "green",
        "table projected row header": "purple",
    }

    for target_object in objects:
        for output in outputs:
            image = output["image"]
            objects = output["objects"]

            # Convert PIL to numpy
            plt.imshow(image, interpolation="lanczos")
            plt.gcf().set_size_inches(20, 20)
            ax = plt.gca()

            for obj in objects:
                label = obj["label"]
                if label in [target_object]:
                    bbox = obj["bbox"]

                    # Plot bbox
                    rect = patches.Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=0.2,
                        edgecolor="none",
                        facecolor=label_colors[label],
                        alpha=0.1,
                    )
                    ax.add_patch(rect)

                    rect = patches.Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=0.4,
                        edgecolor=label_colors[label],
                        facecolor="none",
                        linestyle="-",
                    )
                    ax.add_patch(rect)

                    rect = patches.Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=0,
                        edgecolor=label_colors[label],
                        facecolor="none",
                        linestyle="-",
                        hatch="//////",
                        alpha=0.2,
                    )
                    ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        legend_elements = [
            Patch(
                facecolor=label_colors[target_object],
                edgecolor=label_colors[target_object],
                label=target_object,
                hatch="//////",
                alpha=0.3,
            )
        ]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.5, -0.02),
            loc="upper center",
            borderaxespad=0,
            fontsize=10,
            ncol=2,
        )

        plt.gcf().set_size_inches(10, 10)
        plt.axis("off")
        plt.show()
        plt.close()

    return
