from deeplabv3.type import (
    MODELTYPE,
    DEEPLABTYPE,
    MASKRCNNTYPE,
)

from deeplabv3.entity import (
    Deeplabv3Entity,
    MaskRcnnEntity,
    DataEntity,
    DataloaderEntity,
    TrainEntity,
)


classes = 6
DEVICE = "cpu"
MODEL_TYPE = MODELTYPE.DEEPLABV3
SENSOR_RATIO = 0.3
RESIZE_SCALE = 65

DATA_CFG = DataEntity(
    root_path="/data/share/dataset/moveawheel321321",
    image_base_path="images/",
    train_anno_path="COCO/train_without_street.json",
    valid_anno_path="COCO/valid_without_street.json",
    test_anno_path="COCO/test_without_street.json",
)


__PERSONAL_LABEL_MAP__ = {
    0: "car",
    1: "person"
}


__LOAD_LABEL_MAP__ = {
    1: "dry",
    2: "humid",
    3: "slush",
    4: "snow",
    5: "wet",
}

TRAIN_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=True
)

VALID_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=False
)

TEST_DATALOADER_CFG = DataloaderEntity(
    bathc_size=6,
    num_worker=0,
    shuffle=False
)


TRAIN_CFG = TrainEntity(
    accum_step=1,
    num_epoch=10,
    log_step=1,
)


# TODO: Augmentation 고려
TRAIN_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]


VALID_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]


TEST_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
]

MODEL_CFG = {
    MODELTYPE.DEEPLABV3: Deeplabv3Entity(
        type=DEEPLABTYPE.RESNET50,
        num_classes=classes,
        use_cbam=True,
        params=dict(
            pretrained=True,
        ),
    ),
    MODELTYPE.MASKRCNN: MaskRcnnEntity(
        type=MASKRCNNTYPE.RESNET50V2,
        params=dict(
            num_classes=classes
        ),
    )
}[MODEL_TYPE]