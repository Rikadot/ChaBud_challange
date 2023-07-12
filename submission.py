import numpy as np
import pandas as pd
import h5py
import torch
from trimesh.voxel.runlength import dense_to_brle
from pathlib import Path
from matplotlib.colors import ListedColormap
from collections import defaultdict
from scipy import ndimage as ski
from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray
import matplotlib as plt

import chabud
from pathlib import Path
dataset = Path("A:/CodingProjekte/DataMining/src/train_eval.hdf5")
#Es liegen 15 vortrainierte Modelle auf dem Server im Verzeichnis
#/global/public/chabud-ecml-pkdd2023/checkpoints/.
#Sie können sich verfügbaren Checkpoints wie folgt anzeigen lassen:
ckpt = Path('A:/CodingProjekte/DataMining/src/lightning_logs/version_30/checkpoints/model-epoch=25-val_iou=0.00.ckpt')

#Sie können einen beliebigen Checkpoint wie folgt laden:
mdl = chabud.FireModel.load_from_checkpoint(ckpt, map_location="cpu")

# Vom Modell `mdl` benötigte Kanäle extrahieren
# channels = np.stack([c(bands) for c in mdl.channels])


# with torch.set_grad_enabled(False):
#   # Modell auf 1xlen(channels)x512x512 großen Tensor anwenden
#   # D.h. wir haben eine Batchgröße von 1 (ineffizient aber einfach).
#   print(channels)
#   # channels = channels.astype(float)
#   pred = mdl.forward(torch.Tensor(channels[np.newaxis, ...])).sigmoid() > 0.5
#   # Ersten beiden Dimensionen (batch und channel) löschen und in ein numpy Array wandeln
#   pred = pred[0, 0, ...].detach().numpy()


def process_dataset(scene, bands, true):
    rgb = ski.exposure.adjust_gamma(np.clip(bands[..., [3, 2, 1]], 0, 1), 0.4)

    channels = np.stack([c(bands) for c in mdl.channels])
    with torch.set_grad_enabled(False):
        pred = mdl.forward(torch.Tensor(channels[np.newaxis, ...])).sigmoid() > 0.5
        pred = pred[0, 0, ...].detach().numpy()

    cmap = ListedColormap(["white", "tab:brown", "tab:orange", "tab:blue"])
    mask = np.zeros_like(pred, dtype=int)
    mask = np.where(true & pred, 1, mask)
    mask = np.where(~true & pred, 2, mask)
    mask = np.where(true & ~pred, 3, mask)

    true_edge = ski.feature.canny(true.astype("float")).astype("uint8")
    pred_edge = ski.feature.canny(pred.astype("float")).astype("uint8")

    fig, (axm, axi) = plt.subplots(ncols=2, figsize=(20, 10))
    axm.imshow(mask, cmap=cmap, interpolation="nearest")

    axi.imshow(rgb, interpolation="nearest")
    axi.imshow(true_edge, cmap=ListedColormap(["#00000000", "tab:blue"]), interpolation="nearest")
    axi.imshow(pred_edge, cmap=ListedColormap(["#00000000", "tab:orange"]), interpolation="nearest")

    for ax in [axm, axi]:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

    fig.tight_layout()

    fig.savefig(f"masks/{scene}-f_{dataset.attrs['fold']}.png")
    plt.close()
# class RandomModel:
#     def __init__(self, shape):
#         self.shape = shape
#         return

#     def __call__(self, input):
#         # input is ignored, just generate some random predictions
#         return np.random.randint(0, 2, size=self.shape, dtype=bool)

class FixedModel:
    def __init__(self, shape) -> None:
        self.shape = shape
        return

    def __call__(self, input) -> Any:
        # input is ignored, just generate a mask filled with zeros, with fixed pixels set to 1
        mask = np.zeros(self.shape, dtype=bool)
        mask[100:250, 100:250] = True
        return mask


def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue

            result[uuid]['post'] = values['post_fire'][...]
            # result[uuid]['pre'] = values['pre_fire'][...]

    return dict(result)


def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.astype(bool).flatten())
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}


#der Code aus dem letzten Workshop
class PPModel:
    def __init__(self,model):
        self._model = model
        self._model.eval()

    def __call__(self,bands) -> Any:
        #preprocessing
        bands = bands /10000
        channels = np.stack([c(bands) for c in self._model.channels])
        channels = torch.Tensor(channels[np.newaxis, ...])
        #Modell auswerten
        with torch.set_grad_enabled(False):
            mask = self._model.forward(channels).sigmoid() > 0.5
        #postprocessing
        mask = mask[0,0, ...].detach().numpy()
        return mask



if __name__ == '__main__':
    validation_fold = retrieve_validation_fold('train_eval.hdf5')

    # use a list to accumulate results
    result = []
    # instantiate the model
    # model = FixedModel(shape=(512, 512))
    model = PPModel(mdl)
    for uuid in validation_fold:
        input_images = validation_fold[uuid]['post']

        # perform the prediction
        predicted = model(input_images)
        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv('predictions.csv', index=False)