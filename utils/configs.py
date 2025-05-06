import hydra
import albumentations as A
from albumentations import Compose


def train2test_transforms(train_augmentations):
    """Convert train augmnetations to test augmnetations"""
    test_augmentations = []
    for augmentation in train_augmentations:
        if augmentation.__class__.__name__ == "RandomResizedCrop":
            height, width = augmentation.size
            test_augmentations.append(
                A.Resize(height=int(height / 0.8), width=int(width / 0.8))
            )
            test_augmentations.append(A.CenterCrop(height=height, width=width))
        elif augmentation.__class__.__name__ in (
            "Resize",
            "Normalize",
            "ToTensorV2",
            "ToTensor",
        ):
            test_augmentations.append(augmentation)

    return Compose(test_augmentations)


def setup_transforms(cfg_transfroms):
    """For images only"""
    size_crops = [None, None]  # height, width

    if not cfg_transfroms:
        raise RuntimeError("augms")
    train_augmentations = []
    if not cfg_transfroms.get("order"):
        raise RuntimeError(
            "Require param <order>, i.e."
            "order of augmentations as List[augmentation_name]"
        )
    for augm_name in cfg_transfroms.get("order"):
        augmentation = hydra.utils.instantiate(
            cfg_transfroms[augm_name], _convert_="all"
        )

        if aug_name := augmentation.__class__.__name__ in [
            "RandomResizedCrop",
            "Resize",
        ]:
            try:
                size_crops = augmentation.size
            except AttributeError:
                size_crops = (augmentation.height, augmentation.width)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve size for augmentation {augm_name}"
                ) from e
        elif aug_name == "GaussianBlur":
            augmentation.blur_limit = (
                int(size_crops[0] * 0.05) * 2 + 1,  # 10%, see simclr paper
                int(size_crops[1] * 0.05) * 2 + 1,
            )

        train_augmentations.append(augmentation)

    return Compose(train_augmentations)
