"""
Dataset Augmentation Module
Comprehensive augmentation support for CV datasets
"""

import os
import random
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import json
import shutil
import numpy as np


class DatasetAugmenter:
    """Apply augmentations to expand dataset size"""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    AVAILABLE_AUGMENTATIONS = {
        "flip_horizontal": {
            "name": "Horizontal Flip",
            "description": "Mirror image horizontally",
            "category": "geometric",
            "params": {}
        },
        "flip_vertical": {
            "name": "Vertical Flip",
            "description": "Mirror image vertically",
            "category": "geometric",
            "params": {}
        },
        "rotate": {
            "name": "Rotation",
            "description": "Rotate image by random angle",
            "category": "geometric",
            "params": {
                "angle_range": {"type": "range", "min": -45, "max": 45, "default": [-15, 15], "description": "Rotation angle range in degrees"}
            }
        },
        "scale": {
            "name": "Random Scale",
            "description": "Scale image up or down",
            "category": "geometric",
            "params": {
                "scale_range": {"type": "range", "min": 0.5, "max": 1.5, "default": [0.8, 1.2], "description": "Scale factor range"}
            }
        },
        "translate": {
            "name": "Translation",
            "description": "Shift image position",
            "category": "geometric",
            "params": {
                "translate_range": {"type": "range", "min": -0.2, "max": 0.2, "default": [-0.1, 0.1], "description": "Translation range as fraction of image size"}
            }
        },
        "shear": {
            "name": "Shear",
            "description": "Apply shear transformation",
            "category": "geometric",
            "params": {
                "shear_range": {"type": "range", "min": -20, "max": 20, "default": [-10, 10], "description": "Shear angle range in degrees"}
            }
        },
        "perspective": {
            "name": "Perspective",
            "description": "Apply perspective transformation",
            "category": "geometric",
            "params": {
                "strength": {"type": "range", "min": 0, "max": 0.1, "default": 0.05, "description": "Perspective distortion strength"}
            }
        },
        "crop": {
            "name": "Random Crop",
            "description": "Randomly crop and resize",
            "category": "geometric",
            "params": {
                "crop_range": {"type": "range", "min": 0.7, "max": 1.0, "default": [0.8, 0.95], "description": "Crop size range as fraction of original"}
            }
        },
        "brightness": {
            "name": "Brightness",
            "description": "Adjust image brightness",
            "category": "color",
            "params": {
                "factor_range": {"type": "range", "min": 0.5, "max": 1.5, "default": [0.8, 1.2], "description": "Brightness factor range"}
            }
        },
        "contrast": {
            "name": "Contrast",
            "description": "Adjust image contrast",
            "category": "color",
            "params": {
                "factor_range": {"type": "range", "min": 0.5, "max": 1.5, "default": [0.8, 1.2], "description": "Contrast factor range"}
            }
        },
        "saturation": {
            "name": "Saturation",
            "description": "Adjust color saturation",
            "category": "color",
            "params": {
                "factor_range": {"type": "range", "min": 0.5, "max": 1.5, "default": [0.8, 1.2], "description": "Saturation factor range"}
            }
        },
        "hue": {
            "name": "Hue Shift",
            "description": "Shift color hue",
            "category": "color",
            "params": {
                "shift_range": {"type": "range", "min": -30, "max": 30, "default": [-15, 15], "description": "Hue shift range in degrees"}
            }
        },
        "grayscale": {
            "name": "Grayscale",
            "description": "Convert to grayscale",
            "category": "color",
            "params": {
                "probability": {"type": "range", "min": 0, "max": 1, "default": 0.1, "description": "Probability of applying"}
            }
        },
        "blur": {
            "name": "Gaussian Blur",
            "description": "Apply Gaussian blur",
            "category": "noise",
            "params": {
                "radius_range": {"type": "range", "min": 0.5, "max": 3, "default": [0.5, 1.5], "description": "Blur radius range"}
            }
        },
        "noise": {
            "name": "Gaussian Noise",
            "description": "Add random noise",
            "category": "noise",
            "params": {
                "variance": {"type": "range", "min": 0.01, "max": 0.1, "default": 0.02, "description": "Noise variance"}
            }
        },
        "sharpen": {
            "name": "Sharpen",
            "description": "Sharpen image edges",
            "category": "noise",
            "params": {
                "factor": {"type": "range", "min": 1, "max": 3, "default": 1.5, "description": "Sharpening factor"}
            }
        },
        "jpeg_compression": {
            "name": "JPEG Compression",
            "description": "Simulate JPEG artifacts",
            "category": "noise",
            "params": {
                "quality_range": {"type": "range", "min": 30, "max": 95, "default": [60, 90], "description": "JPEG quality range"}
            }
        },
        "cutout": {
            "name": "Cutout",
            "description": "Randomly cut out rectangles",
            "category": "advanced",
            "params": {
                "num_holes": {"type": "range", "min": 1, "max": 5, "default": 2, "description": "Number of cutout regions"},
                "size_range": {"type": "range", "min": 0.05, "max": 0.2, "default": [0.05, 0.15], "description": "Cutout size as fraction of image"}
            }
        },
        "mosaic": {
            "name": "Mosaic",
            "description": "Combine 4 images into one (requires multiple images)",
            "category": "advanced",
            "params": {}
        },
        "mixup": {
            "name": "MixUp",
            "description": "Blend two images together",
            "category": "advanced",
            "params": {
                "alpha": {"type": "range", "min": 0.1, "max": 0.5, "default": 0.3, "description": "Blending alpha"}
            }
        },
        "elastic": {
            "name": "Elastic Deformation",
            "description": "Apply elastic transformation",
            "category": "advanced",
            "params": {
                "alpha": {"type": "range", "min": 50, "max": 200, "default": 100, "description": "Deformation strength"},
                "sigma": {"type": "range", "min": 5, "max": 15, "default": 10, "description": "Smoothness"}
            }
        },
        "grid_distortion": {
            "name": "Grid Distortion",
            "description": "Apply grid-based distortion",
            "category": "advanced",
            "params": {
                "num_steps": {"type": "range", "min": 3, "max": 10, "default": 5, "description": "Grid size"},
                "distort_limit": {"type": "range", "min": 0.1, "max": 0.5, "default": 0.3, "description": "Distortion limit"}
            }
        },
        "histogram_equalization": {
            "name": "Histogram Equalization",
            "description": "Equalize image histogram",
            "category": "color",
            "params": {}
        },
        "channel_shuffle": {
            "name": "Channel Shuffle",
            "description": "Randomly shuffle RGB channels",
            "category": "color",
            "params": {}
        },
        "invert": {
            "name": "Invert",
            "description": "Invert image colors",
            "category": "color",
            "params": {
                "probability": {"type": "range", "min": 0, "max": 1, "default": 0.1, "description": "Probability of applying"}
            }
        },
        "posterize": {
            "name": "Posterize",
            "description": "Reduce color depth",
            "category": "color",
            "params": {
                "bits": {"type": "range", "min": 2, "max": 7, "default": 4, "description": "Number of bits per channel"}
            }
        },
        "solarize": {
            "name": "Solarize",
            "description": "Invert pixels above threshold",
            "category": "color",
            "params": {
                "threshold": {"type": "range", "min": 64, "max": 192, "default": 128, "description": "Solarize threshold"}
            }
        }
    }

    def get_available_augmentations(self) -> Dict[str, Any]:
        """Return list of available augmentations with their parameters"""
        return self.AVAILABLE_AUGMENTATIONS

    def augment_dataset(
        self,
        input_path: Path,
        output_path: Path,
        format_name: str,
        target_size: int,
        augmentations: Dict[str, Dict[str, Any]],
        preserve_originals: bool = True,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Augment a dataset to reach target size, preserving train/val/test splits.
        Augmentation is applied within each split independently so no images leak
        across split boundaries.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        images = self._find_images(input_path)
        original_count = len(images)

        if original_count == 0:
            return {"success": False, "error": "No images found in dataset"}

        enabled_augs = [
            (name, config.get("params", {}))
            for name, config in augmentations.items()
            if config.get("enabled", True) and name in self.AVAILABLE_AUGMENTATIONS
        ]

        if not enabled_augs:
            return {"success": False, "error": "No valid augmentations enabled"}

        # Group images by split (None = flat dataset with no splits)
        split_groups: Dict[Optional[str], List] = {}
        for img in images:
            split = img.get("split")
            split_groups.setdefault(split, []).append(img)

        split_names = [k for k in split_groups if k is not None]
        has_splits = bool(split_names)

        self._copy_structure(input_path, output_path, format_name,
                             splits=split_names if has_splits else None)

        # Apply the same expansion factor uniformly to each split
        aug_factor = target_size / original_count if original_count > 0 else 2.0

        total_generated = 0
        img_global_idx = 0

        for split, split_images in split_groups.items():
            split_count = len(split_images)
            split_target = math.ceil(split_count * aug_factor)
            aug_needed = max(0, split_target - (split_count if preserve_originals else 0))
            aug_per_image = math.ceil(aug_needed / split_count) if split_count > 0 else 0

            split_generated = 0

            for img_info in split_images:
                img_id = img_info["id"]

                if preserve_originals:
                    self._copy_image_with_annotations(input_path, output_path, img_info, format_name)

                for aug_idx in range(aug_per_image):
                    if split_generated >= aug_needed:
                        break

                    num_augs = random.randint(1, min(3, len(enabled_augs)))
                    selected_augs = random.sample(enabled_augs, num_augs)

                    aug_result = self._apply_augmentations(
                        input_path / img_info["path"],
                        output_path,
                        img_id,
                        aug_idx,
                        selected_augs,
                        format_name,
                        input_path,
                        split=split,
                    )

                    if aug_result["success"]:
                        split_generated += 1
                        total_generated += 1

                img_global_idx += 1
                if progress_callback and original_count > 0:
                    progress_callback(int(img_global_idx / original_count * 90), total_generated)

        self._update_dataset_config(input_path, output_path, format_name,
                                    splits=split_names if has_splits else None)

        return {
            "success": True,
            "original_images": original_count,
            "augmented_images": total_generated,
            "total_images": (original_count if preserve_originals else 0) + total_generated,
            "augmentations_applied": [name for name, _ in enabled_augs],
        }

    def _find_images(self, path: Path) -> List[Dict[str, Any]]:
        """Find all images in dataset, detecting train/val/test split structure."""
        images = []
        seen: set = set()
        SPLIT_NAMES = ["train", "val", "valid", "test"]

        # Detect whether this dataset has split subdirectories
        has_splits = False
        for split in SPLIT_NAMES:
            split_path = path / split
            if not (split_path.exists() and split_path.is_dir()):
                continue
            if (split_path / "images").exists():
                has_splits = True
                break
            try:
                if any(
                    f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
                    for f in split_path.iterdir()
                ):
                    has_splits = True
                    break
            except Exception:
                pass

        if has_splits:
            for split in SPLIT_NAMES:
                split_img_dir = path / split / "images"
                if split_img_dir.exists():
                    for img_file in sorted(split_img_dir.iterdir(), key=lambda f: f.name):
                        if img_file.is_file() and img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                            rel = str(img_file.relative_to(path))
                            if rel not in seen:
                                seen.add(rel)
                                images.append({
                                    "id": img_file.stem,
                                    "path": rel,
                                    "full_path": str(img_file),
                                    "split": split,
                                })
                else:
                    split_dir = path / split
                    if split_dir.exists() and split_dir.is_dir():
                        for img_file in sorted(split_dir.iterdir(), key=lambda f: f.name):
                            if img_file.is_file() and img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                                rel = str(img_file.relative_to(path))
                                if rel not in seen:
                                    seen.add(rel)
                                    images.append({
                                        "id": img_file.stem,
                                        "path": rel,
                                        "full_path": str(img_file),
                                        "split": split,
                                    })
        else:
            for search_dir in ["images", "JPEGImages", ""]:
                search_path = path / search_dir if search_dir else path
                if not search_path.exists():
                    continue
                try:
                    for img_file in sorted(search_path.iterdir(), key=lambda f: f.name):
                        if img_file.is_file() and img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                            rel = str(img_file.relative_to(path))
                            if rel not in seen:
                                seen.add(rel)
                                images.append({
                                    "id": img_file.stem,
                                    "path": rel,
                                    "full_path": str(img_file),
                                    "split": None,
                                })
                except Exception:
                    pass

        return images

    def _copy_structure(self, src: Path, dst: Path, format_name: str, splits: List[str] = None):
        """Copy dataset directory structure, creating split subdirs when present."""
        for config_file in src.glob("*.yaml"):
            shutil.copy(config_file, dst / config_file.name)
        for config_file in src.glob("*.yml"):
            shutil.copy(config_file, dst / config_file.name)

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            if splits:
                for split in splits:
                    (dst / split / "images").mkdir(parents=True, exist_ok=True)
                    (dst / split / "labels").mkdir(parents=True, exist_ok=True)
            else:
                (dst / "images").mkdir(exist_ok=True)
                (dst / "labels").mkdir(exist_ok=True)
        elif format_name in ["pascal-voc", "voc"]:
            (dst / "JPEGImages").mkdir(exist_ok=True)
            (dst / "Annotations").mkdir(exist_ok=True)
        elif format_name == "coco":
            if splits:
                for split in splits:
                    (dst / split / "images").mkdir(parents=True, exist_ok=True)
            else:
                (dst / "images").mkdir(exist_ok=True)
            for json_file in src.glob("*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    if all(k in data for k in ["images", "annotations", "categories"]):
                        new_data = {
                            "images": [],
                            "annotations": [],
                            "categories": data["categories"]
                        }
                        with open(dst / json_file.name, "w") as out:
                            json.dump(new_data, out, indent=2)

    def _copy_image_with_annotations(
        self,
        src_path: Path,
        dst_path: Path,
        img_info: Dict[str, Any],
        format_name: str
    ):
        """Copy an image and its annotations, respecting split structure."""
        src_img = src_path / img_info["path"]
        split = img_info.get("split")

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            if split:
                img_dir = dst_path / split / "images"
                lbl_dir = dst_path / split / "labels"
            else:
                img_dir = dst_path / "images"
                lbl_dir = dst_path / "labels"

            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_img, img_dir / src_img.name)

            label_name = f"{img_info['id']}.txt"
            candidate_label_dirs = []
            if split:
                candidate_label_dirs.append(src_path / split / "labels")
            candidate_label_dirs += [
                src_path / "labels",
                src_path / "train" / "labels",
                src_path / "val" / "labels",
            ]
            for ld in candidate_label_dirs:
                src_label = ld / label_name
                if src_label.exists():
                    shutil.copy(src_label, lbl_dir / label_name)
                    break

        elif format_name in ["pascal-voc", "voc"]:
            dst_img = dst_path / "JPEGImages" / src_img.name
            shutil.copy(src_img, dst_img)

            ann_name = f"{img_info['id']}.xml"
            for ann_dir in ["Annotations", ""]:
                src_ann = src_path / ann_dir / ann_name if ann_dir else src_path / ann_name
                if src_ann.exists():
                    shutil.copy(src_ann, dst_path / "Annotations" / ann_name)
                    break

    def _apply_augmentations(
        self,
        src_img_path: Path,
        output_path: Path,
        img_id: str,
        aug_idx: int,
        augmentations: List[Tuple[str, Dict]],
        format_name: str,
        src_dataset_path: Path,
        split: str = None,
    ) -> Dict[str, Any]:
        """Apply augmentations to a single image, writing output into the correct split dir."""
        try:
            img = Image.open(src_img_path).convert("RGB")
            original_size = img.size

            transform_infos = []

            for aug_name, params in augmentations:
                img, transform_info = self._apply_single_augmentation(img, aug_name, params)
                if transform_info:
                    transform_infos.append(transform_info)

            aug_id = f"{img_id}_aug{aug_idx}"

            if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
                if split:
                    img_dir = output_path / split / "images"
                else:
                    img_dir = output_path / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                output_img_path = img_dir / f"{aug_id}.jpg"
                img.save(output_img_path, "JPEG", quality=95)

                self._transform_yolo_annotations(
                    src_dataset_path,
                    output_path,
                    img_id,
                    aug_id,
                    original_size,
                    img.size,
                    transform_infos,
                    split=split,
                )

            elif format_name in ["pascal-voc", "voc"]:
                output_img_path = output_path / "JPEGImages" / f"{aug_id}.jpg"
                img.save(output_img_path, "JPEG", quality=95)

                self._transform_voc_annotations(
                    src_dataset_path,
                    output_path,
                    img_id,
                    aug_id,
                    original_size,
                    img.size,
                    transform_infos,
                )

            else:
                if split:
                    img_dir = output_path / split / "images"
                else:
                    img_dir = output_path / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                output_img_path = img_dir / f"{aug_id}.jpg"
                img.save(output_img_path, "JPEG", quality=95)

            return {"success": True, "output_path": str(output_img_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_single_augmentation(
        self,
        img: Image.Image,
        aug_name: str,
        params: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Apply a single augmentation to an image"""
        transform_info = {}

        if aug_name == "flip_horizontal":
            img = ImageOps.mirror(img)
            transform_info["flip_h"] = True

        elif aug_name == "flip_vertical":
            img = ImageOps.flip(img)
            transform_info["flip_v"] = True

        elif aug_name == "rotate":
            angle_range = params.get("angle_range", [-15, 15])
            if isinstance(angle_range, list):
                angle = random.uniform(angle_range[0], angle_range[1])
            else:
                angle = random.uniform(-angle_range, angle_range)
            img_w, img_h = img.size
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))
            transform_info["angle"] = angle
            transform_info["img_w"] = img_w
            transform_info["img_h"] = img_h

        elif aug_name == "brightness":
            factor_range = params.get("factor_range", [0.8, 1.2])
            if isinstance(factor_range, list):
                factor = random.uniform(factor_range[0], factor_range[1])
            else:
                factor = random.uniform(1 - factor_range, 1 + factor_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        elif aug_name == "contrast":
            factor_range = params.get("factor_range", [0.8, 1.2])
            if isinstance(factor_range, list):
                factor = random.uniform(factor_range[0], factor_range[1])
            else:
                factor = random.uniform(1 - factor_range, 1 + factor_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        elif aug_name == "saturation":
            factor_range = params.get("factor_range", [0.8, 1.2])
            if isinstance(factor_range, list):
                factor = random.uniform(factor_range[0], factor_range[1])
            else:
                factor = random.uniform(1 - factor_range, 1 + factor_range)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)

        elif aug_name == "blur":
            radius_range = params.get("radius_range", [0.5, 1.5])
            if isinstance(radius_range, list):
                radius = random.uniform(radius_range[0], radius_range[1])
            else:
                radius = radius_range
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif aug_name == "sharpen":
            factor = params.get("factor", 1.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)

        elif aug_name == "grayscale":
            probability = params.get("probability", 0.1)
            if random.random() < probability:
                img = ImageOps.grayscale(img).convert("RGB")

        elif aug_name == "noise":
            variance = params.get("variance", 0.02)
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, variance, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            img = Image.fromarray((img_array * 255).astype(np.uint8))

        elif aug_name == "cutout":
            num_holes = params.get("num_holes", 2)
            size_range = params.get("size_range", [0.05, 0.15])
            img = self._apply_cutout(img, num_holes, size_range)

        elif aug_name == "histogram_equalization":
            img = ImageOps.equalize(img)

        elif aug_name == "invert":
            probability = params.get("probability", 0.1)
            if random.random() < probability:
                img = ImageOps.invert(img)

        elif aug_name == "posterize":
            bits = params.get("bits", 4)
            if isinstance(bits, float):
                bits = int(bits)
            img = ImageOps.posterize(img, bits)

        elif aug_name == "solarize":
            threshold = params.get("threshold", 128)
            if isinstance(threshold, float):
                threshold = int(threshold)
            img = ImageOps.solarize(img, threshold)

        elif aug_name == "crop":
            crop_range = params.get("crop_range", [0.8, 0.95])
            if isinstance(crop_range, list):
                scale = random.uniform(crop_range[0], crop_range[1])
            else:
                scale = crop_range

            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.Resampling.BILINEAR)
            transform_info["crop"] = {
                "left": left, "top": top,
                "crop_w": new_w, "crop_h": new_h,
                "orig_w": w, "orig_h": h,
            }

        elif aug_name == "hue":
            shift_range = params.get("shift_range", [-15, 15])
            if isinstance(shift_range, list):
                shift = random.uniform(shift_range[0], shift_range[1])
            else:
                shift = random.uniform(-shift_range, shift_range)
            img = self._shift_hue(img, shift / 360.0)

        elif aug_name == "jpeg_compression":
            quality_range = params.get("quality_range", [60, 90])
            if isinstance(quality_range, list):
                quality = random.randint(int(quality_range[0]), int(quality_range[1]))
            else:
                quality = int(quality_range)

            import io
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")

        return img, transform_info

    def _apply_cutout(self, img: Image.Image, num_holes: int, size_range: List[float]) -> Image.Image:
        """Apply cutout augmentation"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        for _ in range(num_holes):
            size = random.uniform(size_range[0], size_range[1])
            hole_h = int(h * size)
            hole_w = int(w * size)

            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)

            img_array[y:y+hole_h, x:x+hole_w] = 128  # Gray fill

        return Image.fromarray(img_array)

    def _shift_hue(self, img: Image.Image, shift: float) -> Image.Image:
        """Shift hue of an image"""
        import colorsys

        img_array = np.array(img).astype(np.float32) / 255.0
        h_shift = shift

        # Convert to HSV
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Hue calculation
        h = np.zeros_like(max_c)
        mask = diff > 0

        idx = (max_c == r) & mask
        h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360

        idx = (max_c == g) & mask
        h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360

        idx = (max_c == b) & mask
        h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

        # Apply shift
        h = (h / 360 + h_shift) % 1.0

        # Simple approximation - just return with slight color shift
        # Full HSV conversion is expensive
        shift_factor = h_shift * 2
        r_new = np.clip(r + shift_factor * 0.3, 0, 1)
        g_new = np.clip(g - shift_factor * 0.2, 0, 1)
        b_new = np.clip(b + shift_factor * 0.1, 0, 1)

        result = np.stack([r_new, g_new, b_new], axis=2)
        return Image.fromarray((result * 255).astype(np.uint8))

    def _apply_transform_to_point(self, px: float, py: float, ti: Dict) -> Tuple[float, float]:
        """Apply a single transform_info to a normalized point."""
        if ti.get("flip_h"):
            px = 1.0 - px
        if ti.get("flip_v"):
            py = 1.0 - py
        if "angle" in ti:
            theta = math.radians(ti["angle"])
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Convert to pixel space so aspect ratio is preserved correctly,
            # then normalize back — this fixes non-square (landscape/portrait) images.
            W = ti.get("img_w", 1)
            H = ti.get("img_h", 1)
            dx = (px - 0.5) * W
            dy = (py - 0.5) * H
            px = (dx * cos_t + dy * sin_t) / W + 0.5
            py = (-dx * sin_t + dy * cos_t) / H + 0.5
        if "crop" in ti:
            c = ti["crop"]
            px = (px - c["left"] / c["orig_w"]) / (c["crop_w"] / c["orig_w"])
            py = (py - c["top"] / c["orig_h"]) / (c["crop_h"] / c["orig_h"])
        return px, py

    def _apply_transform_to_bbox(
        self, x_c: float, y_c: float, w_b: float, h_b: float, ti: Dict
    ) -> Optional[Tuple[float, float, float, float]]:
        """Apply a single transform to a YOLO bbox; returns None if box exits the image."""
        half_w, half_h = w_b / 2, h_b / 2
        corners = [
            (x_c - half_w, y_c - half_h),
            (x_c + half_w, y_c - half_h),
            (x_c - half_w, y_c + half_h),
            (x_c + half_w, y_c + half_h),
        ]
        transformed = [self._apply_transform_to_point(cx, cy, ti) for cx, cy in corners]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        xmin = max(0.0, min(xs))
        xmax = min(1.0, max(xs))
        ymin = max(0.0, min(ys))
        ymax = min(1.0, max(ys))
        if xmax <= xmin or ymax <= ymin:
            return None
        return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin

    def _transform_yolo_annotations(
        self,
        src_path: Path,
        dst_path: Path,
        src_id: str,
        dst_id: str,
        original_size: Tuple[int, int],
        new_size: Tuple[int, int],
        transform_infos: List[Dict],
        split: str = None,
    ):
        """Transform YOLO annotations for an augmented image, respecting split dirs."""
        src_label = None
        candidate_dirs = []
        if split:
            candidate_dirs.append(src_path / split / "labels")
        candidate_dirs += [
            src_path / "labels",
            src_path / "train" / "labels",
            src_path / "val" / "labels",
        ]
        for d in candidate_dirs:
            potential = d / f"{src_id}.txt"
            if potential.exists():
                src_label = potential
                break

        if split:
            dst_label_dir = dst_path / split / "labels"
        else:
            dst_label_dir = dst_path / "labels"
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        if not src_label:
            (dst_label_dir / f"{dst_id}.txt").touch()
            return

        with open(src_label) as f:
            lines = f.readlines()

        transformed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = parts[0]

            if len(parts) == 5:
                # Detection format: class_id cx cy w h
                x_c = float(parts[1])
                y_c = float(parts[2])
                w_b = float(parts[3])
                h_b = float(parts[4])

                valid = True
                for ti in transform_infos:
                    result = self._apply_transform_to_bbox(x_c, y_c, w_b, h_b, ti)
                    if result is None:
                        valid = False
                        break
                    x_c, y_c, w_b, h_b = result

                if valid:
                    transformed_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_b:.6f} {h_b:.6f}")
            else:
                # Segmentation format: class_id x1 y1 x2 y2 ...
                coords = [float(p) for p in parts[1:]]
                transformed_coords = []
                for i in range(0, len(coords) - 1, 2):
                    px = coords[i]
                    py = coords[i + 1]
                    for ti in transform_infos:
                        px, py = self._apply_transform_to_point(px, py, ti)
                    transformed_coords.extend([max(0.0, min(1.0, px)), max(0.0, min(1.0, py))])

                if len(transformed_coords) >= 6:
                    transformed_lines.append(
                        f"{class_id} " + " ".join(f"{p:.6f}" for p in transformed_coords)
                    )

        dst_label = dst_label_dir / f"{dst_id}.txt"
        with open(dst_label, "w") as f:
            f.write("\n".join(transformed_lines))

    def _transform_voc_annotations(
        self,
        src_path: Path,
        dst_path: Path,
        src_id: str,
        dst_id: str,
        original_size: Tuple[int, int],
        new_size: Tuple[int, int],
        transform_infos: List[Dict],
    ):
        """Transform Pascal VOC annotations for augmented image."""
        import xml.etree.ElementTree as ET

        src_ann = None
        for ann_dir in ["Annotations", ""]:
            potential = src_path / ann_dir / f"{src_id}.xml" if ann_dir else src_path / f"{src_id}.xml"
            if potential.exists():
                src_ann = potential
                break

        if not src_ann:
            return

        tree = ET.parse(src_ann)
        root = tree.getroot()

        orig_w, orig_h = original_size

        filename_elem = root.find("filename")
        if filename_elem is not None:
            filename_elem.text = f"{dst_id}.jpg"

        to_remove = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            # Normalize to [0, 1]
            norm_x_c = (xmin + xmax) / 2 / orig_w
            norm_y_c = (ymin + ymax) / 2 / orig_h
            norm_w = (xmax - xmin) / orig_w
            norm_h = (ymax - ymin) / orig_h

            valid = True
            for ti in transform_infos:
                result = self._apply_transform_to_bbox(norm_x_c, norm_y_c, norm_w, norm_h, ti)
                if result is None:
                    valid = False
                    break
                norm_x_c, norm_y_c, norm_w, norm_h = result

            if not valid:
                to_remove.append(obj)
                continue

            bndbox.find("xmin").text = str(int((norm_x_c - norm_w / 2) * orig_w))
            bndbox.find("ymin").text = str(int((norm_y_c - norm_h / 2) * orig_h))
            bndbox.find("xmax").text = str(int((norm_x_c + norm_w / 2) * orig_w))
            bndbox.find("ymax").text = str(int((norm_y_c + norm_h / 2) * orig_h))

        for obj in to_remove:
            root.remove(obj)

        dst_ann = dst_path / "Annotations" / f"{dst_id}.xml"
        tree.write(dst_ann)

    def _update_dataset_config(
        self,
        src_path: Path,
        dst_path: Path,
        format_name: str,
        splits: List[str] = None,
    ):
        """Update dataset configuration files, using split paths when splits are present."""
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            for yaml_file in dst_path.glob("*.yaml"):
                import yaml
                with open(yaml_file) as f:
                    config = yaml.safe_load(f) or {}

                config["path"] = str(dst_path.absolute())

                if splits:
                    # Map "valid" -> "val" in the YAML key
                    key_map = {"valid": "val"}
                    written: set = set()
                    for split in splits:
                        key = key_map.get(split, split)
                        if key not in written:
                            config[key] = f"{split}/images"
                            written.add(key)
                    # Remove stale flat-image references for splits not present
                    for stale in ["train", "val", "test"]:
                        if stale not in written and config.get(stale) == "images":
                            del config[stale]
                else:
                    config["train"] = "images"
                    config["val"] = "images"

                with open(yaml_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
