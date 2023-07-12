import torch.cuda

import chabud as ch

ds_path = "A:/CodingProjekte/DataMining/src/train_eval.hdf5"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(ch.__version__)
    print(torch.cuda.is_available())
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    channels = ["band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7", "band_8", "band_8a", "band_9",
                "band_11", "band_12", "nbr", "ndvi", "gndvi", "evi", "avi", "savi", "ndmi", "msi", "gci", "bsi", "ndwi",
                "ndgi"]
    # channels = ["band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7", "band_8", "band_8a", "band_9",
    #             "band_11", "band_12", "nbr", "ndmi", "ndvi", "bsi", "ndwi"]
    channels_fun = []

    for channel in channels:
        channels_fun.append(ch.CHANNEL_MAP[channel])

    ch.main(accelerator="gpu",
            datafile=ds_path,
            batch_size=5,
            learning_rate=0.00025,
            channels=channels_fun,
            n_cpus=0,
            model="unet",
            encoder="resnet34",
            encoder_depth=5,
            encoder_weights="imagenet",
            loss="dice",
            train_use_pre_fire=False,
            train_use_augmentation=True)


