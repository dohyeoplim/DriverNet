from src.DriverNet.data.DataModule import DriverDataModule

dm = DriverDataModule(
    data_dir="./input/imgs/train",
    csv_path="./input/driver_imgs_list.csv",
    batch_size=8,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    image_size=224,
    flip_p=0.1,
    test_split=0.2,
)

dm.prepare_data()
dm.setup()

batch = next(iter(dm.train_dataloader()))
print("pixel_values:", batch["pixel_values"].shape)
print("labels:", batch["labels"])

assert dm.train_ds is not None and dm.val_ds is not None
print(f"Train samples: {len(dm.train_ds)}")
print(f"Val samples: {len(dm.val_ds)}")
print(f"Classes: {dm.class_to_idx}")
