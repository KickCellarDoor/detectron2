import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg


def get_semantic_carla_metadata():
    meta = {
        "stuff_classes": ["Unlabeled", "Building", "Fence", "Other", "Pedestrian", "Pole", "Road line", "Road",
                          "Sidewalk", "Vegetation", "Car", "Wall", "Traffic sign"],
        "stuff_colors": [(0, 0, 0), (70, 70, 70), (190, 153, 153), (250, 170, 160), (220, 20, 60), (153, 153, 153),
                         (157, 234, 50), (128, 64, 128), (244, 35, 232), (107, 142, 35), (0, 0, 142), (102, 102, 156),
                         (220, 220, 0)],
        "stuff_dataset_id_to_contiguous_id": {
            0: 0,
            1: 1,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
        },
        "evaluator_type": "sem_seg",

    }
    return meta

# TODO: def get_semantic_bdd_metadata():

SPLITS = {
    "semantic-carla-origin-train": ("carla/origin/train", "carla/semantic/train"),
    "semantic-carla-randomized-train": ("carla/randomized/train", "carla/semantic/train"),
    "semantic-bdd-origin-train": ("bdd/origin/train", "bdd/semantic/train"),
    "semantic-bdd-randomized-train": ("bdd/randomized/train", "bdd/semantic/train"),
    "semantic-carla-origin-test": ("carla/origin/test", "carla/semantic/test"),
    "semantic-carla-randomized-test": ("carla/randomized/test", "carla/semantic/test"),
    "semantic-bdd-origin-test": ("bdd/origin/test", "bdd/semantic/test"),
    "semantic-bdd-randomized-test": ("bdd/randomized/test", "bdd/semantic/test"),
}

for key, (image_root, seg_root) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    seg_root = os.path.join("datasets", seg_root)
    image_root = os.path.join("datasets", image_root)

    DatasetCatalog.register(
        key,
        lambda key=key, image_root=image_root, seg_root=seg_root: load_sem_seg(
            gt_root=seg_root, image_root=image_root, gt_ext="png", image_ext="png"
        ),
    )

    MetadataCatalog.get(key).set(
        sem_seg_root=seg_root, image_root=image_root, **get_semantic_carla_metadata()
    )
