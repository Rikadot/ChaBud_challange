##
## chabud.py - Hilfsfunktionen für die ChaBuD ECML Challenge 2023
##
## CHANGES:
## 2023-05-23: Erste Version veröffentlicht
##
## TODO:
## * Funktion um Vorhersage als CSV zu speichern für Leaderboard
## * Argument um Anzahl Trainingsepochen zu steuern (epoch, max_epoch, ... ?)
## * Finales Modell ausgeben und ggf. auch Vorhersage auf Validierungsdaten speichern
##
import logging
import os
from pathlib import Path
import pandas as pd
import albumentations as A
import albumentations.pytorch.transforms as Atorch
import h5py
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import xarray as xr

from pytorch_lightning.callbacks import ModelCheckpoint

fn = Path("A:/CodingProjekte/DataMining/src/train_eval.hdf5")

#Wir wollen ein Dataframe erstellen, welches nur die Namen der Datensätze enthält, die eine größere Brandfläche als 2% haben.
#Sogesehen ist es dann eine whitelist

def basic_df():
    res = []
    #Anzahl aller Datensätze ("name")
    count_ds = 0

    with h5py.File(fn, "r") as fd:

        for name, ds in fd.items():
            count_burnt_pixels = 0
            #Standardmäßig ist überall ein pre_fire verfügbar
            pre_miss = 0
            #Weil wir den Datensatz schon gecheckt haben, ist ein sicherer Zugrif auf post_fire und mask möglich (hier fehlen keine ganzen Datensätze)
            post = ds["post_fire"]
            mask = ds["mask"]

            count_burnt_pixels = np.sum(mask)
            count_pixels =512 * 512
            burnt_pixel_rel = count_burnt_pixels / count_pixels
            #Anders als bei mask und Post müssen wir vor Zugriff überprüfen ob "pre_fire" überhaupt existiert - Vermeidung einer Fehlermeldung
            if "pre_fire" not in ds:
                pre_miss = 1
            res.append({"name": name, "pre_missing": pre_miss, "burnt_pixel_abs": count_burnt_pixels, "burnt_pixel_rel": burnt_pixel_rel})

        return pd.DataFrame(res)

def miss_dp_df():
    BANDS = ["coastal_aerosol", "blue", "green", "red",
             "veg_red_1", "veg_red_2", "veg_red_3", "nir",
             "veg_red_4", "water_vapour", "swir_1", "swir_2"]

    res = basic_df().values

    # miss_dp ist eine Liste mit "name", "pre" "post" (Werte von Pre + Postt werden mit den Bandnamen selektiert)
    miss_count = 0
    miss_dp = []
    with h5py.File(fn, "r") as fd:
        for x in res:
            # skippe die Datensätze mit fehlendem Pre-Bild
            # if x["pre_missing"] == 1:
            #     continue
            pre_miss = False

            # Laden der Daten aus dem Originaldatensatz
            name = x["name"]
            ds = to_xarray(fd[name])
            pre = ds["pre"][...]
            post = ds["post"][...]
            mask = ds["mask"][...]

            if x["pre_missing"] == 1:
                # Code für den Fall, dass 'pre_missing' gleich 1 ist
                post_miss = []
                for band in range(pre.shape[2]):
                    post_miss.append((np.sum(post[band] == 0).values))
                x_post_miss = xr.DataArray(post_miss, dims=["band"], coords={"band": BANDS})
                miss_dp.append({"name": name, "pre": [], "post": x_post_miss.values})
            else:
                # Code für den Fall, dass 'pre_missing' nicht gleich 1 ist
                pre_miss = []
                post_miss = []
                for band in range(pre.shape[2]):
                    pre_miss.append((np.sum(pre[band] == 0).values))
                    post_miss.append((np.sum(post[band] == 0).values))
                x_pre_miss = xr.DataArray(pre_miss, dims=["band"], coords={"band": BANDS})
                x_post_miss = xr.DataArray(post_miss, dims=["band"], coords={"band": BANDS})
                miss_dp.append({"name": name, "pre": x_pre_miss.values, "post": x_post_miss.values})

    return miss_dp


def wl():
    whitelist = []
    df = basic_df()

    for index, row in df.iterrows():
        if row["burnt_pixel_rel"] < 0.0025:
            continue
        whitelist.append(row["name"])
    return whitelist

checkpoint_callback = ModelCheckpoint(
    #dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_iou:.2f}',
    monitor='valid_iou',
    mode='max',
    save_top_k=3
)


__version__ = "1.0.0"
logger = logging.getLogger(__name__)

ds_path = "A:/CodingProjekte/DataMining/src/train_eval.hdf5"


def to_xarray(dataset, pretty_band_names=True):
    """Konvertiert ein HDF5-Gruppenobjekt, das Vor- und Nach-Brandbilder enthält, in xarray DataArrays.

    Parameters
    ----------
    dataset : h5py.Group
        Ein HDF5-Gruppenobjekt, das die Vor- und Nach-Brandbilder, die Maske und die Metadaten enthält.
    pretty_band_names : bool, optional
        Wenn True (Standard), werden die "Pretty" Bandnamen verwendet, ansonsten die ursprünglichen MSI Bandnummern.

    Returns
    -------
    dict
        Ein Dictionary, das die xarray DataArrays für die Vor- und Nach-Brandbilder, die Maske und die Fold-Informationen enthält.
    """
    if pretty_band_names:
        BANDS = ["coastal_aerosol", "blue", "green", "red",
                 "veg_red_1", "veg_red_2", "veg_red_3", "nir",
                 "veg_red_4", "water_vapour", "swir_1", "swir_2"]
    else:
        BANDS = ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9", "11", "12"]

    post = dataset["post_fire"][...].astype("float32") / 10000.0

    try:
        pre = dataset["pre_fire"][...].astype("float32") / 10000.0
    except KeyError:
        pre = np.zeros_like(post, dtype="float32")

    mask = dataset["mask"][..., 0]

    return {"pre": xr.DataArray(pre, dims=["x", "y", "band"], coords={"x": range(512), "y": range(512), "band": BANDS}),
            "post": xr.DataArray(post, dims=["x", "y", "band"], coords={"x": range(512), "y": range(512), "band": BANDS}),
            "mask": xr.DataArray(mask, dims=["x", "y"], coords={"x": range(512), "y": range(512)}),
            "fold": dataset.attrs["fold"]}

class BandExtractor:
    def __init__(self, index, name) -> None:
        self.index = index
        self.name = name

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            return data[..., self.index]
        elif isinstance(data, xr.DataArray):
            return data.sel(band=self.name).values
        else:
            msg = "Unknown data format."
            raise Exception(msg)

    def __repr__(self) -> str:
        return f'BandExtractor({self.index}, "{self.name}")'


band_1 = BandExtractor(0, "coastal_aerosol")
band_2 = BandExtractor(1, "blue")
band_3 = BandExtractor(2, "green")
band_4 = BandExtractor(3, "red")
band_5 = BandExtractor(4, "veg_red_1")
band_6 = BandExtractor(5, "veg_red_2")
band_7 = BandExtractor(6, "veg_red_3")
band_8 = BandExtractor(7, "nir")
band_8a = BandExtractor(8, "veg_red_4")
band_9 = BandExtractor(9, "water_vapour")
band_11 = BandExtractor(10, "swir_1")
band_12 = BandExtractor(11, "swir_2")


def NBR(data):
    """Normalized Burn Ratio.

    nbr = (nir - swir_2) / (nir + swir_2)
    """
    if isinstance(data, np.ndarray):
        nir = data[..., 7]
        swir_2 = data[..., 11]
    elif isinstance(data, xr.DataArray):
        nir = data.sel(band="nir").values
        swir_2 = data.sel(band="swir_2").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = nir - swir_2
    nenner = nir + swir_2
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def NDVI(data):
    """Normalized Difference Vegetation Index."""
    if isinstance(data, np.ndarray):
        red = data[..., 3]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = nir - red
    nenner = nir + red
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def GNDVI(data):
    """Green Normalized Difference Vegetation Index."""
    if isinstance(data, np.ndarray):
        green = data[..., 2]
        red = data[..., 3]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        green = data.sel(band="green").values
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = nir - green
    nenner = nir + red
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def EVI(data):
    """Enhanced Vegetation Index."""
    if isinstance(data, np.ndarray):
        blue = data[..., 1]
        red = data[..., 3]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        blue = data.sel(band="blue").values
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = nir - red
    nenner = nir + 6 * red - 7.5 * blue + 1

    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def AVI(data):
    """Advanced Vegetation Index."""
    if isinstance(data, np.ndarray):
        red = data[..., 3]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    base = nir * (1 - red) * (nir - red)
    ## FIXME: Deal with cube roots of negative values?
    return np.power(base, 1./3., out=np.zeros_like(base), where=base>0)


def SAVI(data):
    """Soil Adjusted Vegetation Index."""
    if isinstance(data, np.ndarray):
        red = data[..., 3]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    return (nir - red) / (nir + red + 0.428) * 1.428


def NDMI(data):
    if isinstance(data, np.ndarray):
        nir = data[..., 7]
        swir_1 = data[..., 10]
    elif isinstance(data, xr.DataArray):
        nir = data.sel(band="nir").values
        swir_1 = data.sel(band="swir_1").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = nir - swir_1
    nenner = nir + swir_1
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def MSI(data):
    """Moisture Stress Index.

    Moisture Stress Index is used for canopy stress analysis, productivity
    prediction and biophysical modeling. Interpretation of the MSI is inverted
    relative to other water vegetation indices; thus, higher values of the
    index indicate greater plant water stress and in inference, less soil
    moisture content. The values of this index range from 0 to more than 3 with
    the common range for green vegetation being 0.2 to 2.
    """
    if isinstance(data, np.ndarray):
        nir = data[..., 7]
        swir_1 = data[..., 10]
    elif isinstance(data, xr.DataArray):
        nir = data.sel(band="nir").values
        swir_1 = data.sel(band="swir_1").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    return swir_1 - nir


def GCI(data):
    """Green Chlorophyll Index."""
    if isinstance(data, np.ndarray):
        green = data[..., 2]
        water_vapour = data[..., 9]
    elif isinstance(data, xr.DataArray):
        green = data.sel(band="green").values
        water_vapour = data.sel(band="water_vapour").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    return water_vapour - green


def BSI(data):
    """Bare Soil Index."""
    if isinstance(data, np.ndarray):
        blue = data[..., 1]
        red = data[..., 3]
        nir = data[..., 7]
        swir_1 = data[..., 10]
    elif isinstance(data, xr.DataArray):
        blue = data.sel(band="blue").values
        red = data.sel(band="red").values
        nir = data.sel(band="nir").values
        swir_1 = data.sel(band="swir_1").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    swir_red = swir_1 + red
    nir_blue = nir + blue
    zaehler = swir_red - nir_blue
    nenner = swir_red + nir_blue
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def NDWI(data):
    """Normalized Difference Water Index."""
    if isinstance(data, np.ndarray):
        green = data[..., 2]
        nir = data[..., 7]
    elif isinstance(data, xr.DataArray):
        green = data.sel(band="green").values
        nir = data.sel(band="nir").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = green - nir
    nenner = green + nir
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def NDSI(data):
    """Normalized Difference Snow Index."""
    if isinstance(data, np.ndarray):
        green = data[..., 2]
        swir_1 = data[..., 10]
    elif isinstance(data, xr.DataArray):
        green = data.sel(band="green").values
        swir_1 = data.sel(band="swir_1").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = green - swir_1
    nenner = green + swir_1
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)


def NDGI(data):
    if isinstance(data, np.ndarray):
        green = data[..., 2]
        red = data[..., 3]
    elif isinstance(data, xr.DataArray):
        green = data.sel(band="green").values
        red = data.sel(band="red").values
    else:
        msg = "Unknown data format."
        raise Exception(msg)

    zaehler = green - red
    nenner = green + red
    return np.divide(zaehler, nenner, out=np.zeros_like(zaehler), where=nenner != 0.0)

#Die Bänder kommen in Channels
class FiresDataset(torch.utils.data.Dataset):
    def __init__(self, filename, folds=(0, 1, 2, 3, 4),
                 channels=[],
                 include_pre=False,
                 transform=None) -> None:
        self._filename = filename
        self._fd = h5py.File(filename, "r")
        self._channels = channels
        self._transform = transform
        self._names = []

        whitelist = wl()
        for name in self._fd:
            if self._fd[name].attrs["fold"] not in folds:
                continue
            if name in whitelist:
                self._names.append((name, "post_fire"))
            if include_pre and "pre_fire" in self._fd[name]:
                pre_image = self._fd[name]["pre_fire"][...]
                # Include only "real" pre_fire images
                if np.mean(pre_image > 0) > 0.8:
                    self._names.append((name, "pre_fire"))

    def number_of_channels(self):
        return len(self._channels)

    def __getitem__(self, idx):
        name, state = self._names[idx]
        data = self._fd[name][state][...].astype("float32") / 10000.0
        if state == "pre_fire":
            mask = np.zeros((512, 512), dtype="float32")
        else:
            mask = self._fd[name]["mask"][..., 0].astype("float32")

        channels = []
        for channel in self._channels:
            channels.append(channel(data))

        # Stack indices into a new image in CHW format.
        image =  np.stack(channels)

        if self._transform:
            # Transpose image so we get HWC instead of CHW format.
            # Transform is responsible for transposing back as required by PyTorch.
            image = image.transpose((1, 2, 0))
            xfrm = self._transform(image=image, mask=mask)
            image, mask = xfrm["image"], xfrm["mask"]
        logger.debug("Final tensor shape: %s", image.shape)

        return {"image": image, "mask": mask[None, :]}

    def __len__(self) -> int:
        return len(self._names)


class FireModel(pl.LightningModule):
    def __init__(self,
                 datafile,
                 model,
                 encoder,
                 encoder_depth,
                 encoder_weights,
                 loss,
                 channels,
                 train_transform,
                 train_use_pre_fire,
                 n_cpus,
                 batch_size,
                 lr=0.00025,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.datafile = datafile
        self.lr = lr
        self.channels = channels
        if model == "unet":
            decoder_channels = [2**(8 - d) for d in range(encoder_depth, 0, -1)]
            self.model = smp.Unet(encoder_name=encoder, encoder_depth=encoder_depth, encoder_weights=encoder_weights,
                                  decoder_channels=decoder_channels,
                                  in_channels=len(channels), classes=1)
        elif model == "unetpp":
            decoder_channels = [2**(8 - d) for d in range(encoder_depth, 0, -1)]
            self.model = smp.UnetPlusPlus(encoder_name=encoder, encoder_depth=encoder_depth, encoder_weights=encoder_weights,
                                          decoder_channels=decoder_channels,
                                          in_channels=len(channels), classes=1)
        elif model == "fpn":
            if encoder_depth == 3:
                upsampling = 1
            elif encoder_depth == 4:
                upsampling = 2
            elif encoder_depth == 5:
                upsampling = 4
            else:
                raise "FPN: Unsupported encoder depth {encoder_depth}."
            self.model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                 upsampling=upsampling,
                                 in_channels=len(channels), classes=1)
        elif model == "dlv3":
            self.model = smp.DeepLabV3(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                       in_channels=len(channels), classes=1)
        elif model == "dlv3p":
            if encoder_depth != 5:
                raise f"Unsupported encoder depth {encoder_depth} for DeepLabV3+ (must be 5)."
            self.model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, encoder_depth=encoder_depth,
                                           in_channels=len(channels), classes=1)
        else:
            raise f"Unsupported model '{model}'."

        if loss == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss == "bce":
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        else:
            raise f"Unsupported loss function '{loss}'."

        self.train_transform = train_transform
        self.train_use_pre_fire = train_use_pre_fire
        self.n_cpus = n_cpus
        self.batch_size = batch_size

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch["image"], batch["mask"]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask.long(), mode="binary")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def train_dataloader(self):
        train_ds = FiresDataset(self.datafile, folds=[1, 2, 3, 4],
                                channels=self.channels,
                                transform=self.train_transform,
                                include_pre=self.train_use_pre_fire)
        train_dl = torch.utils.data.DataLoader(train_ds,
                                               batch_size=self.batch_size,
                                               num_workers=self.n_cpus,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=False)
        return train_dl

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def val_dataloader(self):
        val_ds = FiresDataset(self.datafile, folds=[0],
                              channels=self.channels,
                              transform=None,
                              include_pre=False)
        val_dl = torch.utils.data.DataLoader(val_ds,
                                             batch_size=self.batch_size,
                                             num_workers=self.n_cpus,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)
        return val_dl

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        # TODO: Can we do better? We should probably implement a learning rate schedule?
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(accelerator,
         datafile,
         batch_size,
         channels,
         n_cpus,
         model,
         encoder,
         encoder_depth,
         encoder_weights,
         loss,
         train_use_pre_fire,
         train_use_augmentation,
         learning_rate,
         ):


    if train_use_augmentation:
        train_xfrm = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            Atorch.ToTensorV2(),
            ])
    else:
        train_xfrm = None

    logger.info("Instantiating model.")
    mdl = FireModel(datafile=datafile,
                    model=model,
                    encoder=encoder,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    loss=loss,
                    channels=channels,
                    n_cpus=n_cpus,
                    train_transform=train_xfrm,
                    train_use_pre_fire=train_use_pre_fire,
                    batch_size=batch_size,
                    lr=learning_rate)

    trainer = pl.Trainer(accelerator=accelerator, devices="auto",
                         log_every_n_steps=10, max_epochs=30, callbacks=[checkpoint_callback])
#callbacks=[checkpoint_callback]
    logger.info("Start training.")
    trainer.fit(mdl)


CHANNEL_MAP = {
        "band_1": band_1,
        "band_2": band_2,
        "band_3": band_3,
        "band_4": band_4,
        "band_5": band_5,
        "band_6": band_6,
        "band_7": band_7,
        "band_8": band_8,
        "band_8a": band_8a,
        "band_9": band_9,
        "band_11": band_11,
        "band_12": band_12,
        "nbr": NBR,
        "ndvi": NDVI,
        "gndvi": GNDVI,
        "evi": EVI,
        "avi": AVI,
        "savi": SAVI,
        "ndmi": NDMI,
        "msi": MSI,
        "gci": GCI,
        "bsi": BSI,
        "ndwi": NDWI,
        "ndsi": NDSI,
        "ndgi": NDGI,
        }

if __name__ == "__main__":

    import argparse  # Only import when needed

    N_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
    parser = argparse.ArgumentParser("chabud.py")
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--datafile", type=Path, default=ds_path,
                        help="Location of data file used for training.")
    parser.add_argument("--n-cpus", type=int, default=N_CPUS, help="Number of CPU cores to use.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training and validation batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="Learning rate of optimizer.")
    parser.add_argument("--model", choices=["unet", "unetpp", "fpn", "dlv3", "dlv3p"], default="unet",
                        help="Segmentation model")
    parser.add_argument("--encoder", choices=["resnet18", "resnet34", "resnet50", "vgg13", "dpn68", "dpn92", "timm-efficientnet-b0"], default="resnet34",
                        help="Encoder of segmentation model")
    parser.add_argument("--encoder-depth", type=int, default=5,
                        help="Depth of encoder stage")
    parser.add_argument("--encoder-weights", choices=["random", "imagenet"], default="imagenet",
                        help="Weight initialization for encoder")
    parser.add_argument("--loss", choices=["dice", "bce"], default="dice",
                        help="Loss function")
    parser.add_argument("--train-use-pre_fire", action="store_true",
                        help="Use pre_fire data for training?")
    parser.add_argument("--train-use-augmentation", action="store_true",
                        help="Use data augmentation in training step?")
    parser.add_argument("--channels", nargs="+", choices=CHANNEL_MAP.keys(),
                        default=["band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7", "band_8", "band_8a", "band_9", "band_11", "band_12"],
                        help="Channels to use for prediction")
    parser.add_argument("--log-level", type=str, choices=["info", "debug"], default="info")

    args = parser.parse_args()

    LOGGING_MAP = {"info": logging.INFO, "debug": logging.DEBUG}
    logging.basicConfig(level=LOGGING_MAP[args.log_level],
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%d-%b-%y %H:%M:%S")


    if args.encoder_weights == "random":
        args.encoder_weights = None

    # Translate channel names to function that calculates the channel / index.
    logger.info(f"Selected channels: {args.channels}")
    channels = []
    for channel in args.channels:
        channels.append(CHANNEL_MAP[channel])


    torch.set_num_threads(args.n_cpus)
    torch.set_float32_matmul_precision("medium")

    main(accelerator=args.accelerator,
         datafile=args.datafile,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         channels=channels,
         n_cpus=args.n_cpus,
         model=args.model,
         encoder=args.encoder,
         encoder_depth=args.encoder_depth,
         encoder_weights=args.encoder_weights,
         loss=args.loss,
         train_use_pre_fire=args.train_use_pre_fire,
         train_use_augmentation=args.train_use_augmentation)

