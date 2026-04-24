"""
Dataset Parsers - Support for multiple CV annotation formats
"""

import os
import json
import yaml
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil
import re


class DatasetParser:
    """Universal dataset parser supporting multiple formats"""
    
    SUPPORTED_FORMATS = [
        "yolo", "yolov5", "yolov8", "yolov8-obb", "yolov9", "yolov10", "yolov11", "yolov12",
        "coco", "coco-segmentation", "coco-keypoint",
        "pascal-voc", "voc",
        "createml",
        "tensorflow-csv", "tfrecord",
        "labelme",
        "cvat",
        "supervisely",
        "classification-folder",
        "csv", "json", "txt"
    ]
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    
    def __init__(self):
        self.format_detectors = {
            "yolo": self._detect_yolo,
            "coco": self._detect_coco,
            "pascal-voc": self._detect_voc,
            "labelme": self._detect_labelme,
            "createml": self._detect_createml,
            "classification-folder": self._detect_classification_folder,
            "tensorflow-csv": self._detect_tensorflow_csv,
            "supervisely": self._detect_supervisely,
        }
    
    def _find_dataset_root(self, path: Path) -> Path:
        """
        If a zip was extracted into a single subdirectory (e.g. dataset_small/),
        return that subdirectory as the actual dataset root.
        Walk at most 2 levels deep.
        """
        path = Path(path)
        for _ in range(2):
            try:
                children = [c for c in path.iterdir()
                            if not c.name.startswith(".") and c.name != "dataset_metadata.json"]
            except Exception:
                break
            subdirs = [c for c in children if c.is_dir()]
            files = [c for c in children if c.is_file()]
            # If there are no files and exactly one subdirectory, descend into it
            if not files and len(subdirs) == 1:
                path = subdirs[0]
            else:
                break
        return path

    def parse_dataset(
        self,
        dataset_path: Path,
        format_hint: str = None,
        name: str = None
    ) -> Dict[str, Any]:
        """Parse a dataset and return its metadata"""
        dataset_path = Path(dataset_path)
        dataset_root = self._find_dataset_root(dataset_path)

        # Detect format if not provided (use resolved root for detection)
        if format_hint:
            detected_format = format_hint
        else:
            detected_format = self._detect_format(dataset_root)

        # Parse based on format (use resolved root)
        parser = self._get_parser(detected_format)
        info = parser(dataset_root)

        info["name"] = name or dataset_path.name
        info["path"] = str(dataset_path)
        info["format"] = detected_format
        info["created_at"] = datetime.now().isoformat()
        
        return info
    
    def _detect_format(self, dataset_path: Path) -> str:
        """Auto-detect dataset format"""
        for format_name, detector in self.format_detectors.items():
            if detector(dataset_path):
                return format_name

        # Second-pass: recursive search one level deeper for nested datasets
        try:
            for child in dataset_path.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    for format_name, detector in self.format_detectors.items():
                        try:
                            if detector(child):
                                return format_name
                        except Exception:
                            pass
        except Exception:
            pass

        # Fallback: if we have images but couldn't identify annotations, still
        # try to give a meaningful type rather than "unknown"
        try:
            image_count = sum(
                1 for f in dataset_path.rglob("*")
                if f.suffix.lower() in self.IMAGE_EXTENSIONS
            )
            if image_count > 0:
                return "generic-images"
        except Exception:
            pass

        return "unknown"
    
    def _detect_yolo(self, path: Path) -> bool:
        """Detect YOLO format (txt files with yaml config)"""
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        txt_files = list(path.glob("**/*.txt"))
        
        # Check for data.yaml or similar
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config or "nc" in config:
                        return True
            except Exception:
                pass
        
        # Check for YOLO txt format annotations
        for txt_file in txt_files:
            if self._is_yolo_annotation(txt_file):
                return True
        
        return False
    
    def _is_yolo_annotation(self, txt_file: Path) -> bool:
        """Check if a txt file is YOLO format"""
        try:
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # class_id x_center y_center width height
                        try:
                            int(parts[0])
                            float(parts[1])
                            float(parts[2])
                            return True
                        except ValueError:
                            return False
        except Exception:
            return False
        return False
    
    def _detect_coco(self, path: Path) -> bool:
        """Detect COCO JSON format"""
        json_files = list(path.glob("*.json")) + list(path.glob("**/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if all(key in data for key in ["images", "annotations", "categories"]):
                        return True
            except Exception:
                pass
        
        return False
    
    def _detect_voc(self, path: Path) -> bool:
        """Detect Pascal VOC XML format"""
        xml_files = list(path.glob("**/*.xml"))
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                if root.tag == "annotation" and root.find("object") is not None:
                    return True
            except Exception:
                pass
        
        return False
    
    def _detect_labelme(self, path: Path) -> bool:
        """Detect LabelMe JSON format"""
        json_files = list(path.glob("**/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "shapes" in data and "imagePath" in data:
                        return True
            except Exception:
                pass
        
        return False
    
    def _detect_createml(self, path: Path) -> bool:
        """Detect CreateML JSON format"""
        json_files = list(path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if "image" in data[0] and "annotations" in data[0]:
                            return True
            except Exception:
                pass
        
        return False
    
    def _detect_classification_folder(self, path: Path) -> bool:
        """Detect folder-based classification dataset (handles train/test/val structure)"""
        subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        
        # Check for train/test/val split structure first
        splits = ["train", "val", "valid", "validation", "test"]
        for split in splits:
            split_path = path / split
            if split_path.exists() and split_path.is_dir():
                # Check if split contains class folders with images
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith("."):
                        images = [f for f in class_dir.iterdir() 
                                 if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                        if images:
                            return True
        
        # Check flat class_name/images structure
        for subdir in subdirs:
            # Skip common non-class directories
            if subdir.name.lower() in splits + ["images", "labels", "annotations"]:
                continue
            images = [f for f in subdir.iterdir() 
                     if f.suffix.lower() in self.IMAGE_EXTENSIONS]
            if images:
                return True
        
        return False
    
    def _detect_tensorflow_csv(self, path: Path) -> bool:
        """Detect TensorFlow CSV format"""
        csv_files = list(path.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file) as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and "filename" in header and "xmin" in header:
                        return True
            except Exception:
                pass
        
        return False
    
    def _detect_supervisely(self, path: Path) -> bool:
        """Detect Supervisely format"""
        meta_file = path / "meta.json"
        return meta_file.exists()
    
    def _get_parser(self, format_name: str):
        """Get the appropriate parser for a format"""
        parsers = {
            "yolo": self._parse_yolo,
            "yolov5": self._parse_yolo,
            "yolov8": self._parse_yolo,
            "yolov8-obb": self._parse_yolo_obb,
            "yolov9": self._parse_yolo,
            "yolov10": self._parse_yolo,
            "yolov11": self._parse_yolo,
            "yolov12": self._parse_yolo,
            "coco": self._parse_coco,
            "coco-segmentation": self._parse_coco,
            "coco-keypoint": self._parse_coco,
            "pascal-voc": self._parse_voc,
            "voc": self._parse_voc,
            "labelme": self._parse_labelme,
            "createml": self._parse_createml,
            "tensorflow-csv": self._parse_tensorflow_csv,
            "classification-folder": self._parse_classification_folder,
            "supervisely": self._parse_supervisely,
            "generic-images": self._parse_generic,
        }
        return parsers.get(format_name, self._parse_generic)
    
    def _parse_yolo(self, path: Path) -> Dict[str, Any]:
        """Parse YOLO format dataset"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        # Find yaml config
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        if isinstance(config["names"], dict):
                            info["classes"] = list(config["names"].values())
                        else:
                            info["classes"] = config["names"]
                    break
            except Exception:
                pass
        
        # Count images and annotations
        image_dirs = ["images", "train/images", "val/images", "test/images"]
        label_dirs = ["labels", "train/labels", "val/labels", "test/labels"]

        images = set()
        annotations = 0

        for img_dir in image_dirs:
            img_path = path / img_dir
            if img_path.exists():
                for img in img_path.iterdir():
                    if img.suffix.lower() in self.IMAGE_EXTENSIONS:
                        images.add(img.stem)

        # Also check root for images
        for img in path.iterdir():
            if img.suffix.lower() in self.IMAGE_EXTENSIONS:
                images.add(img.stem)

        # Recursive fallback: if no images found via standard paths, search deeper
        if not images:
            for img_file in path.rglob("*"):
                if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                    # Skip files inside a "labels" directory
                    if "labels" not in img_file.parts:
                        images.add(img_file.stem)
        
        for label_dir in label_dirs:
            label_path = path / label_dir
            if label_path.exists():
                for label in label_path.glob("*.txt"):
                    with open(label) as f:
                        annotations += sum(1 for line in f if line.strip())
        
        # Also check root for labels
        for label in path.glob("*.txt"):
            if label.stem != "classes":
                try:
                    with open(label) as f:
                        annotations += sum(1 for line in f if line.strip())
                except Exception:
                    pass
        
        info["num_images"] = len(images)
        info["num_annotations"] = annotations
        
        # Detect if segmentation
        for label_dir in label_dirs + [""]:
            label_path = path / label_dir if label_dir else path
            if label_path.exists():
                for label in label_path.glob("*.txt"):
                    try:
                        with open(label) as f:
                            line = f.readline().strip()
                            parts = line.split()
                            if len(parts) > 5:
                                info["task_type"] = "segmentation"
                    except Exception:
                        pass
                    break
        
        return info
    
    def _parse_yolo_obb(self, path: Path) -> Dict[str, Any]:
        """Parse YOLO OBB (oriented bounding boxes) format"""
        info = self._parse_yolo(path)
        info["task_type"] = "obb-detection"
        return info
    
    def _parse_coco(self, path: Path) -> Dict[str, Any]:
        """Parse COCO JSON format"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        # Find COCO JSON file
        json_files = list(path.glob("*.json")) + list(path.glob("annotations/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if all(key in data for key in ["images", "annotations", "categories"]):
                    info["num_images"] = len(data["images"])
                    info["num_annotations"] = len(data["annotations"])
                    info["classes"] = [cat["name"] for cat in data["categories"]]
                    
                    # Check for segmentation
                    if data["annotations"]:
                        ann = data["annotations"][0]
                        if "segmentation" in ann and ann["segmentation"]:
                            info["task_type"] = "segmentation"
                        if "keypoints" in ann:
                            info["task_type"] = "keypoint"
                    break
            except Exception:
                pass
        
        return info
    
    def _parse_voc(self, path: Path) -> Dict[str, Any]:
        """Parse Pascal VOC XML format"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        classes = set()
        xml_files = list(path.glob("**/*.xml"))
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                if root.tag == "annotation":
                    info["num_images"] += 1
                    
                    for obj in root.findall("object"):
                        info["num_annotations"] += 1
                        name = obj.find("name")
                        if name is not None:
                            classes.add(name.text)
            except Exception:
                pass
        
        info["classes"] = list(classes)
        return info
    
    def _parse_labelme(self, path: Path) -> Dict[str, Any]:
        """Parse LabelMe JSON format"""
        info = {
            "task_type": "segmentation",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        classes = set()
        json_files = list(path.glob("**/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if "shapes" in data:
                    info["num_images"] += 1
                    info["num_annotations"] += len(data["shapes"])
                    
                    for shape in data["shapes"]:
                        classes.add(shape.get("label", "unknown"))
            except Exception:
                pass
        
        info["classes"] = list(classes)
        return info
    
    def _parse_createml(self, path: Path) -> Dict[str, Any]:
        """Parse CreateML JSON format"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        classes = set()
        json_files = list(path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if "image" in item:
                            info["num_images"] += 1
                            for ann in item.get("annotations", []):
                                info["num_annotations"] += 1
                                classes.add(ann.get("label", "unknown"))
            except Exception:
                pass
        
        info["classes"] = list(classes)
        return info
    
    def _parse_tensorflow_csv(self, path: Path) -> Dict[str, Any]:
        """Parse TensorFlow CSV format"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        classes = set()
        images = set()
        csv_files = list(path.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        images.add(row.get("filename", ""))
                        info["num_annotations"] += 1
                        classes.add(row.get("class", "unknown"))
            except Exception:
                pass
        
        info["num_images"] = len(images)
        info["classes"] = list(classes)
        return info
    
    def _parse_classification_folder(self, path: Path) -> Dict[str, Any]:
        """Parse folder-based classification dataset (handles train/test/val structure)"""
        info = {
            "task_type": "classification",
            "num_images": 0,
            "num_annotations": 0,
            "classes": [],
            "splits": {}
        }
        
        classes = set()
        splits = ["train", "val", "valid", "validation", "test"]
        
        # Check if we have train/test/val split structure
        has_splits = any((path / split).exists() for split in splits)
        
        if has_splits:
            # Handle train/test/val/class_name structure
            for split in splits:
                split_path = path / split
                if split_path.exists() and split_path.is_dir():
                    split_count = 0
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir() and not class_dir.name.startswith("."):
                            classes.add(class_dir.name)
                            images = [f for f in class_dir.iterdir() 
                                     if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                            count = len(images)
                            info["num_images"] += count
                            info["num_annotations"] += count
                            split_count += count
                    if split_count > 0:
                        info["splits"][split] = split_count
        else:
            # Handle flat class_name/images structure
            for subdir in path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    classes.add(subdir.name)
                    images = [f for f in subdir.iterdir() 
                             if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                    info["num_images"] += len(images)
                    info["num_annotations"] += len(images)
        
        info["classes"] = sorted(list(classes))
        return info
    
    def _parse_supervisely(self, path: Path) -> Dict[str, Any]:
        """Parse Supervisely format"""
        info = {
            "task_type": "detection",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }
        
        # Read meta.json for classes
        meta_file = path / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                info["classes"] = [cls["title"] for cls in meta.get("classes", [])]
        
        # Count images and annotations
        for ann_file in path.glob("**/*.json"):
            if ann_file.name != "meta.json":
                try:
                    with open(ann_file) as f:
                        data = json.load(f)
                        info["num_images"] += 1
                        info["num_annotations"] += len(data.get("objects", []))
                except Exception:
                    pass
        
        return info
    
    def _parse_generic(self, path: Path) -> Dict[str, Any]:
        """Generic parser for unknown/image-only formats — tries to infer as much as possible."""
        info = {
            "task_type": "unknown",
            "num_images": 0,
            "num_annotations": 0,
            "classes": []
        }

        classes = set()
        image_count = 0

        for img in path.rglob("*"):
            if img.suffix.lower() in self.IMAGE_EXTENSIONS:
                image_count += 1

        info["num_images"] = image_count

        # Try to infer task type and classes from subfolder names
        try:
            subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            split_names = {"train", "val", "valid", "validation", "test", "images", "labels", "annotations"}

            # Non-split subdirs may be class folders
            class_dirs = [d for d in subdirs if d.name.lower() not in split_names]
            for d in class_dirs:
                has_images = any(
                    f.suffix.lower() in self.IMAGE_EXTENSIONS for f in d.iterdir()
                    if f.is_file()
                )
                if has_images:
                    classes.add(d.name)

            if classes:
                info["task_type"] = "classification"
                info["classes"] = sorted(list(classes))
                info["num_annotations"] = image_count
        except Exception:
            pass

        # If we couldn't determine from folders, check for any annotation files
        if not classes:
            try:
                txt_files = list(path.rglob("*.txt"))
                for txt_file in txt_files:
                    if self._is_yolo_annotation(txt_file):
                        info["task_type"] = "detection"
                        break
            except Exception:
                pass

        # Ensure task_type has a user-friendly value if still unknown
        if info["task_type"] == "unknown" and image_count > 0:
            info["task_type"] = "detection"  # safe default — better than "unknown"

        return info
    
    def get_dataset_details(self, path: Path, format_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a dataset (class distribution, splits, etc.)"""
        path = Path(path)
        root = self._find_dataset_root(path)

        # Re-parse to get fresh counts
        parsed = self.parse_dataset(path, format_name)
        num_images = parsed.get("num_images", 0)
        num_annotations = parsed.get("num_annotations", 0)

        # Class distribution: {class_name: annotation_count}
        class_distribution: Dict[str, int] = {}
        try:
            dist_list = self.get_classes_with_distribution(root, format_name)
            class_distribution = {item["name"]: item["count"] for item in dist_list if item["name"]}
        except Exception:
            for cls in parsed.get("classes", []):
                class_distribution[cls] = 0

        # Splits: {split_name: image_count}
        splits: Dict[str, int] = {}
        for split in ["train", "val", "valid", "test"]:
            # Search recursively in case root is one level too high
            for split_img in [root / split / "images"] + list(root.glob(f"*/{split}/images")):
                if split_img.exists() and split_img.is_dir():
                    count = sum(1 for f in split_img.iterdir()
                                if f.suffix.lower() in self.IMAGE_EXTENSIONS)
                    if count:
                        splits[split] = splits.get(split, 0) + count
                        break
            if split not in splits:
                for split_flat in [root / split] + list(root.glob(f"*/{split}")):
                    if split_flat.exists() and split_flat.is_dir():
                        count = sum(1 for f in split_flat.iterdir()
                                    if f.suffix.lower() in self.IMAGE_EXTENSIONS)
                        if count:
                            splits[split] = count
                            break
        if not splits and num_images:
            splits["all"] = num_images

        avg = (num_annotations / num_images) if num_images else 0.0

        return {
            "class_distribution": class_distribution,
            "splits": splits,
            "avg_annotations_per_image": round(avg, 2),
            "total_images": num_images,
            "total_annotations": num_annotations,
        }
    
    def get_images_with_annotations(
        self,
        path: Path,
        format_name: str,
        page: int = 1,
        limit: int = 50,
        filter_classes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of images with their annotations"""
        path = self._find_dataset_root(Path(path))
        images = []

        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            images = self._get_yolo_images(path, format_name)
        elif format_name == "coco":
            images = self._get_coco_images(path)
        elif format_name in ["pascal-voc", "voc"]:
            images = self._get_voc_images(path)
        elif format_name == "labelme":
            images = self._get_labelme_images(path)
        elif format_name == "classification-folder":
            images = self._get_classification_images(path)
        else:
            images = self._get_generic_images(path)
        
        # Filter by classes if specified
        if filter_classes:
            filtered_images = []
            for img in images:
                # For classification, check the class_name field
                if format_name == "classification-folder":
                    if img.get("class_name") in filter_classes:
                        filtered_images.append(img)
                else:
                    # For detection/segmentation, check annotation class names
                    for ann in img.get("annotations", []):
                        if ann.get("class_name") in filter_classes:
                            filtered_images.append(img)
                            break
            images = filtered_images
        
        # Paginate
        start = (page - 1) * limit
        end = start + limit
        return images[start:end]
    
    def _get_classification_images(self, path: Path) -> List[Dict[str, Any]]:
        """Get images from classification folder dataset (handles train/test/val structure)"""
        images = []
        splits = ["train", "val", "valid", "validation", "test"]
        
        # Check if we have train/test/val split structure
        has_splits = any((path / split).exists() for split in splits)
        
        if has_splits:
            # Handle train/test/val/class_name structure
            for split in splits:
                split_path = path / split
                if split_path.exists() and split_path.is_dir():
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir() and not class_dir.name.startswith("."):
                            class_name = class_dir.name
                            for img_file in class_dir.iterdir():
                                if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                                    images.append({
                                        "id": img_file.stem,
                                        "filename": img_file.name,
                                        "path": str(img_file.relative_to(path)),
                                        "class_name": class_name,
                                        "split": split,
                                        "annotations": [{
                                            "type": "classification",
                                            "class_name": class_name
                                        }],
                                        "has_annotations": True
                                    })
        else:
            # Handle flat class_name/images structure
            for subdir in path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    class_name = subdir.name
                    for img_file in subdir.iterdir():
                        if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                            images.append({
                                "id": img_file.stem,
                                "filename": img_file.name,
                                "path": str(img_file.relative_to(path)),
                                "class_name": class_name,
                                "annotations": [{
                                    "type": "classification",
                                    "class_name": class_name
                                }],
                                "has_annotations": True
                            })
        
        return images
    
    def _get_yolo_images(self, path: Path, format_name: str) -> List[Dict[str, Any]]:
        """Get images from YOLO format dataset - with split detection and deduplication"""
        images = []
        seen_stems = set()  # track by stem to avoid duplicates across scan dirs

        # Load class names from yaml
        classes = []
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        if isinstance(config["names"], dict):
                            classes = list(config["names"].values())
                        else:
                            classes = config["names"]
                    break
            except Exception:
                pass

        def _parse_label_file(label_file):
            annotations = []
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                                if len(parts) == 5:
                                    annotations.append({
                                        "type": "bbox",
                                        "class_id": class_id,
                                        "class_name": class_name,
                                        "x_center": float(parts[1]),
                                        "y_center": float(parts[2]),
                                        "width": float(parts[3]),
                                        "height": float(parts[4])
                                    })
                                else:
                                    # Segmentation polygon - points are normalized (0-1)
                                    points = [float(p) for p in parts[1:]]
                                    annotations.append({
                                        "type": "polygon",
                                        "class_id": class_id,
                                        "class_name": class_name,
                                        "points": points,
                                        "normalized": True  # flag so frontend knows to scale
                                    })
                except Exception:
                    pass
            return annotations

        # Build list of (img_dir, label_dir, split_name) to scan
        # Prioritise explicit split dirs; fall back to flat structure
        scan_dirs = []
        for split in ["train", "val", "valid", "test"]:
            img_dir = path / split / "images"
            lbl_dir = path / split / "labels"
            if img_dir.exists():
                scan_dirs.append((img_dir, lbl_dir, split))
            elif (path / split).exists():
                # images directly in split folder
                scan_dirs.append((path / split, lbl_dir, split))

        # Also scan flat images/ directory (no split)
        flat_img = path / "images"
        flat_lbl = path / "labels"
        if flat_img.exists() and not scan_dirs:
            scan_dirs.append((flat_img, flat_lbl, None))

        # Final fallback: root directory
        if not scan_dirs:
            scan_dirs.append((path, path / "labels", None))

        for img_dir, lbl_dir, split_name in scan_dirs:
            if not img_dir.exists():
                continue
            try:
                img_files = sorted(img_dir.iterdir(), key=lambda f: f.name)
            except Exception:
                continue
            for img_file in img_files:
                if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                stem = img_file.stem
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)

                label_file = lbl_dir / f"{stem}.txt"
                annotations = _parse_label_file(label_file)

                entry = {
                    "id": stem,
                    "filename": img_file.name,
                    "path": str(img_file.relative_to(path)).replace("\\", "/"),
                    "annotations": annotations,
                    "has_annotations": len(annotations) > 0
                }
                if split_name:
                    entry["split"] = split_name
                images.append(entry)

        # Recursive fallback: if the standard scan found nothing (e.g. dataset root
        # is one level too high), search recursively for any images/ directories.
        if not images:
            for img_dir in sorted(path.rglob("images")):
                if not img_dir.is_dir():
                    continue
                parent_name = img_dir.parent.name
                split_name = parent_name if parent_name in ("train", "val", "valid", "test") else None
                lbl_dir = img_dir.parent / "labels"
                try:
                    img_files = sorted(img_dir.iterdir(), key=lambda f: f.name)
                except Exception:
                    continue
                for img_file in img_files:
                    if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                        continue
                    stem = img_file.stem
                    if stem in seen_stems:
                        continue
                    seen_stems.add(stem)
                    label_file = lbl_dir / f"{stem}.txt"
                    annotations = _parse_label_file(label_file)
                    entry = {
                        "id": stem,
                        "filename": img_file.name,
                        "path": str(img_file.relative_to(path)).replace("\\", "/"),
                        "annotations": annotations,
                        "has_annotations": len(annotations) > 0,
                    }
                    if split_name:
                        entry["split"] = split_name
                    images.append(entry)

        return images

    def _get_coco_images(self, path: Path) -> List[Dict[str, Any]]:
        """Get images from COCO format dataset"""
        images = []
        
        # Find COCO JSON
        json_files = list(path.glob("*.json")) + list(path.glob("annotations/*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if all(key in data for key in ["images", "annotations", "categories"]):
                    # Build category map
                    cat_map = {cat["id"]: cat["name"] for cat in data["categories"]}
                    
                    # Build annotation map
                    ann_map = {}
                    for ann in data["annotations"]:
                        img_id = ann["image_id"]
                        if img_id not in ann_map:
                            ann_map[img_id] = []
                        ann_map[img_id].append({
                            "type": "bbox" if not ann.get("segmentation") else "polygon",
                            "class_id": ann["category_id"],
                            "class_name": cat_map.get(ann["category_id"], "unknown"),
                            "bbox": ann.get("bbox", []),
                            "segmentation": ann.get("segmentation", [])
                        })
                    
                    for img in data["images"]:
                        images.append({
                            "id": str(img["id"]),
                            "filename": img["file_name"],
                            "path": img["file_name"],
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "annotations": ann_map.get(img["id"], []),
                            "has_annotations": img["id"] in ann_map
                        })
                    break
            except Exception:
                pass
        
        return images
    
    def _get_voc_images(self, path: Path) -> List[Dict[str, Any]]:
        """Get images from Pascal VOC format dataset"""
        images = []
        
        for xml_file in path.glob("**/*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                if root.tag == "annotation":
                    filename = root.find("filename")
                    if filename is not None:
                        annotations = []
                        for obj in root.findall("object"):
                            name = obj.find("name")
                            bndbox = obj.find("bndbox")
                            
                            if bndbox is not None:
                                annotations.append({
                                    "type": "bbox",
                                    "class_name": name.text if name is not None else "unknown",
                                    "xmin": int(bndbox.find("xmin").text),
                                    "ymin": int(bndbox.find("ymin").text),
                                    "xmax": int(bndbox.find("xmax").text),
                                    "ymax": int(bndbox.find("ymax").text)
                                })
                        
                        images.append({
                            "id": xml_file.stem,
                            "filename": filename.text,
                            "path": filename.text,
                            "annotations": annotations,
                            "has_annotations": len(annotations) > 0
                        })
            except Exception:
                pass
        
        return images
    
    def _get_labelme_images(self, path: Path) -> List[Dict[str, Any]]:
        """Get images from LabelMe format dataset"""
        images = []
        
        for json_file in path.glob("**/*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if "shapes" in data:
                    annotations = []
                    for shape in data["shapes"]:
                        annotations.append({
                            "type": shape.get("shape_type", "polygon"),
                            "class_name": shape.get("label", "unknown"),
                            "points": shape.get("points", [])
                        })
                    
                    images.append({
                        "id": json_file.stem,
                        "filename": data.get("imagePath", json_file.stem),
                        "path": data.get("imagePath", ""),
                        "width": data.get("imageWidth"),
                        "height": data.get("imageHeight"),
                        "annotations": annotations,
                        "has_annotations": len(annotations) > 0
                    })
            except Exception:
                pass
        
        return images
    
    def _get_generic_images(self, path: Path) -> List[Dict[str, Any]]:
        """Get images from unknown format"""
        images = []
        
        for img_file in path.glob("**/*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS:
                images.append({
                    "id": img_file.stem,
                    "filename": img_file.name,
                    "path": str(img_file.relative_to(path)),
                    "annotations": [],
                    "has_annotations": False
                })
        
        return images
    
    def get_image_data(self, path: Path, format_name: str, image_id: str) -> Dict[str, Any]:
        """Get a specific image with its annotations"""
        images = self.get_images_with_annotations(path, format_name, page=1, limit=10000)
        
        for img in images:
            if img["id"] == image_id:
                return img
        
        return None
    
    def get_classes(self, path: Path, format_name: str) -> List[str]:
        """Get all classes from a dataset"""
        info = self.parse_dataset(path, format_name)
        return info.get("classes", [])

    def get_classes_with_distribution(self, path: Path, format_name: str) -> List[dict]:
        """Get all classes with count distribution"""
        classes = self.get_classes(path, format_name)
        # Try to get annotation counts per class from cached images
        try:
            images = self.get_images_with_annotations(path, format_name, page=1, limit=999999)
            counts: dict = {}
            for img in images:
                for ann in img.get("annotations", []):
                    cn = ann.get("class_name", "unknown")
                    counts[cn] = counts.get(cn, 0) + 1
            result = []
            for cls in classes:
                result.append({"name": cls, "count": counts.get(cls, 0)})
            # Add any classes seen in annotations but not in class list
            for cn, cnt in counts.items():
                if cn not in classes:
                    result.append({"name": cn, "count": cnt})
            return result
        except Exception:
            return [{"name": c, "count": 0} for c in classes]
    
    def create_split_dataset(
        self,
        source_path: Path,
        output_path: Path,
        format_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Split a dataset into train/val/test sets and write to output_path."""
        import random as _random

        source_path = Path(source_path)
        output_path = Path(output_path)
        source_root = self._find_dataset_root(source_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all images
        all_images = self.get_images_with_annotations(source_path, format_name, page=1, limit=999999)

        rng = _random.Random(seed)
        if shuffle:
            rng.shuffle(all_images)

        n = len(all_images)
        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        n_test  = n - n_train - n_val

        splits = {
            "train": all_images[:n_train],
            "val":   all_images[n_train:n_train + n_val],
            "test":  all_images[n_train + n_val:],
        }

        is_yolo = format_name in (
            "yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"
        )

        # Copy YAML config if YOLO
        if is_yolo:
            for ext in ("*.yaml", "*.yml"):
                for yf in source_root.glob(ext):
                    shutil.copy(yf, output_path / yf.name)

        for split_name, imgs in splits.items():
            if not imgs:
                continue

            if is_yolo:
                out_img_dir = output_path / split_name / "images"
                out_lbl_dir = output_path / split_name / "labels"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_lbl_dir.mkdir(parents=True, exist_ok=True)

                for img in imgs:
                    # Locate source image file
                    img_path = source_root / img["path"]
                    if not img_path.exists():
                        img_path = source_path / img["path"]
                    if not img_path.exists():
                        # Recursive search by filename
                        matches = list(source_root.rglob(img["filename"]))
                        if matches:
                            img_path = matches[0]
                        else:
                            continue

                    shutil.copy(img_path, out_img_dir / img_path.name)

                    # Locate label file (look next to original image and in labels/)
                    lbl_src = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                    if not lbl_src.exists():
                        lbl_src = img_path.with_suffix(".txt")
                    if not lbl_src.exists():
                        lbl_src = source_root / "labels" / (img_path.stem + ".txt")
                    if lbl_src.exists():
                        shutil.copy(lbl_src, out_lbl_dir / lbl_src.name)
            elif format_name == "coco":
                out_img_dir = output_path / split_name / "images"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    img_path = source_root / img["path"]
                    if not img_path.exists():
                        img_path = source_path / img["path"]
                    if img_path.exists():
                        shutil.copy(img_path, out_img_dir / img_path.name)
            else:
                # Generic: just copy image files preserving relative structure
                out_dir = output_path / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    img_path = source_root / img["path"]
                    if not img_path.exists():
                        img_path = source_path / img["path"]
                    if img_path.exists():
                        shutil.copy(img_path, out_dir / img_path.name)

        return {
            "splits": {
                "train": n_train,
                "val":   n_val,
                "test":  n_test,
            },
            "total": n,
        }

    def create_filtered_dataset(
        self,
        source_path: Path,
        output_path: Path,
        kept_images: List[Dict],
        format_name: str
    ):
        """Create a new dataset with only the kept images"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        kept_ids = {img["id"] for img in kept_images}
        
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            self._filter_yolo_dataset(source_path, output_path, kept_ids)
        elif format_name == "coco":
            self._filter_coco_dataset(source_path, output_path, kept_ids)
        elif format_name in ["pascal-voc", "voc"]:
            self._filter_voc_dataset(source_path, output_path, kept_ids)
        else:
            self._filter_generic_dataset(source_path, output_path, kept_ids)
    
    def _filter_yolo_dataset(self, source: Path, output: Path, kept_ids: set):
        """Filter YOLO dataset"""
        # Copy yaml config
        for yaml_file in source.glob("*.yaml"):
            shutil.copy(yaml_file, output / yaml_file.name)
        for yaml_file in source.glob("*.yml"):
            shutil.copy(yaml_file, output / yaml_file.name)
        
        # Copy kept images and labels
        image_dirs = ["images", "train/images", "val/images", ""]
        
        for img_dir in image_dirs:
            img_path = source / img_dir if img_dir else source
            if not img_path.exists():
                continue
            
            label_dir = str(img_dir).replace("images", "labels") if img_dir else "labels"
            label_path = source / label_dir if label_dir else source
            
            out_img_dir = output / img_dir if img_dir else output / "images"
            out_label_dir = output / label_dir if label_dir else output / "labels"
            
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in img_path.iterdir():
                if img_file.suffix.lower() in self.IMAGE_EXTENSIONS and img_file.stem in kept_ids:
                    shutil.copy(img_file, out_img_dir / img_file.name)
                    
                    label_file = label_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        shutil.copy(label_file, out_label_dir / label_file.name)
    
    def _filter_coco_dataset(self, source: Path, output: Path, kept_ids: set):
        """Filter COCO dataset"""
        # Find and filter COCO JSON
        for json_file in source.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if all(key in data for key in ["images", "annotations", "categories"]):
                    # Filter images
                    kept_image_ids = set()
                    new_images = []
                    for img in data["images"]:
                        if str(img["id"]) in kept_ids:
                            new_images.append(img)
                            kept_image_ids.add(img["id"])
                    
                    # Filter annotations
                    new_annotations = [
                        ann for ann in data["annotations"]
                        if ann["image_id"] in kept_image_ids
                    ]
                    
                    # Save filtered JSON
                    output_data = {
                        "images": new_images,
                        "annotations": new_annotations,
                        "categories": data["categories"]
                    }
                    
                    with open(output / json_file.name, "w") as f:
                        json.dump(output_data, f, indent=2)
                    
                    break
            except Exception:
                pass
        
        # Copy images
        images_dir = output / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in source.glob("**/*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS and img_file.stem in kept_ids:
                shutil.copy(img_file, images_dir / img_file.name)
    
    def _filter_voc_dataset(self, source: Path, output: Path, kept_ids: set):
        """Filter Pascal VOC dataset"""
        annotations_dir = output / "Annotations"
        images_dir = output / "JPEGImages"
        
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for xml_file in source.glob("**/*.xml"):
            if xml_file.stem in kept_ids:
                shutil.copy(xml_file, annotations_dir / xml_file.name)
        
        for img_file in source.glob("**/*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS and img_file.stem in kept_ids:
                shutil.copy(img_file, images_dir / img_file.name)
    
    def _filter_generic_dataset(self, source: Path, output: Path, kept_ids: set):
        """Filter generic dataset"""
        for img_file in source.glob("**/*"):
            if img_file.suffix.lower() in self.IMAGE_EXTENSIONS and img_file.stem in kept_ids:
                dest = output / img_file.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_file, dest)
