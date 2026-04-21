"""
Annotation Tools - Manual annotation management
"""

import os
import json
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import shutil


class AnnotationManager:
    """Manage annotations for datasets"""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    
    def update_annotations(
        self,
        dataset_path: Path,
        format_name: str,
        image_id: str,
        annotations: List[Dict[str, Any]]
    ):
        """Update annotations for a specific image"""
        dataset_path = Path(dataset_path)
        
        # generic-images (raw upload) is promoted to YOLO on first annotation save
        if format_name == "generic-images":
            format_name = "yolo"

        updaters = {
            "yolo": self._update_yolo_annotations,
            "yolov5": self._update_yolo_annotations,
            "yolov8": self._update_yolo_annotations,
            "yolov9": self._update_yolo_annotations,
            "yolov10": self._update_yolo_annotations,
            "yolov11": self._update_yolo_annotations,
            "yolov12": self._update_yolo_annotations,
            "coco": self._update_coco_annotations,
            "pascal-voc": self._update_voc_annotations,
            "voc": self._update_voc_annotations,
            "labelme": self._update_labelme_annotations,
        }

        updater = updaters.get(format_name)
        if updater:
            updater(dataset_path, image_id, annotations)
    
    def _find_dataset_root(self, path: Path) -> Path:
        """Descend into single-child directories (handles ZIP-extracted nesting)."""
        for _ in range(2):
            children = [c for c in path.iterdir()
                        if not c.name.startswith(".") and c.name != "dataset_metadata.json"]
            subdirs = [c for c in children if c.is_dir()]
            files   = [c for c in children if c.is_file()]
            if not files and len(subdirs) == 1:
                path = subdirs[0]
            else:
                break
        return path

    def _update_yolo_annotations(
        self,
        dataset_path: Path,
        image_id: str,
        annotations: List[Dict[str, Any]]
    ):
        """Update YOLO format annotations"""
        dataset_path = Path(dataset_path)
        root = self._find_dataset_root(dataset_path)

        # Find image file anywhere under root
        image_file = None
        for ext in self.IMAGE_EXTENSIONS:
            for found in root.glob(f"**/{image_id}{ext}"):
                image_file = found
                break
            if image_file:
                break

        if not image_file:
            return

        with Image.open(image_file) as img:
            width, height = img.size

        # Label file lives in a sibling "labels" folder.
        # e.g. root/train/images/foo.jpg → root/train/labels/foo.txt
        # e.g. root/foo.jpg (flat)       → root/labels/foo.txt
        img_parent = image_file.parent
        if img_parent.name.lower() in ("images", "imgs"):
            label_dir = img_parent.parent / "labels"
        elif img_parent.resolve() == root.resolve():
            # Images are directly in the dataset root — put labels/ inside root
            label_dir = root / "labels"
        else:
            label_dir = img_parent.parent / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_file = label_dir / f"{image_id}.txt"

        # Load class mapping from yaml
        classes: List[str] = []
        for yaml_file in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        names = config["names"]
                        classes = list(names.values()) if isinstance(names, dict) else list(names)
                    break
            except Exception:
                pass

        class_to_id = {name: idx for idx, name in enumerate(classes)}

        with open(label_file, "w") as f:
            for ann in annotations:
                class_name = ann.get("class_name", "unknown")
                class_id = ann.get("class_id", class_to_id.get(class_name, 0))

                if ann.get("type") == "polygon" and ann.get("points"):
                    pts = ann["points"]
                    if ann.get("normalized"):
                        # Already in [0,1] — write as-is
                        norm = [float(p) for p in pts]
                    else:
                        # Pixel coordinates — normalize
                        norm = []
                        for i in range(0, len(pts) - 1, 2):
                            norm.append(float(pts[i]) / width)
                            norm.append(float(pts[i + 1]) / height)
                    if norm:
                        f.write(f"{class_id} " + " ".join(f"{p:.6f}" for p in norm) + "\n")

                elif ann.get("x_center") is not None:
                    # Already-normalized bbox (loaded from YOLO label)
                    xc = float(ann["x_center"])
                    yc = float(ann["y_center"])
                    w  = float(ann["width"])
                    h  = float(ann["height"])
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                elif ann.get("bbox"):
                    # Pixel-coordinate bbox drawn by user: [x, y, w, h]
                    bbox = ann["bbox"]
                    xc = (float(bbox[0]) + float(bbox[2]) / 2) / width
                    yc = (float(bbox[1]) + float(bbox[3]) / 2) / height
                    w  = float(bbox[2]) / width
                    h  = float(bbox[3]) / height
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    def _update_coco_annotations(
        self,
        dataset_path: Path,
        image_id: str,
        annotations: List[Dict[str, Any]]
    ):
        """Update COCO format annotations"""
        # Find COCO JSON file
        json_file = None
        for jf in list(dataset_path.glob("*.json")) + list(dataset_path.glob("annotations/*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if all(key in data for key in ["images", "annotations", "categories"]):
                        json_file = jf
                        break
            except Exception:
                pass
        
        if not json_file:
            return
        
        with open(json_file) as f:
            coco_data = json.load(f)
        
        # Find image ID
        img_id = None
        for img in coco_data["images"]:
            if str(img["id"]) == image_id or img.get("file_name", "").startswith(image_id):
                img_id = img["id"]
                break
        
        if img_id is None:
            return
        
        # Build category map
        cat_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        
        # Remove existing annotations for this image
        coco_data["annotations"] = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] != img_id
        ]
        
        # Add new annotations
        max_ann_id = max([ann["id"] for ann in coco_data["annotations"]], default=0)
        
        for ann in annotations:
            max_ann_id += 1
            class_name = ann.get("class_name", "unknown")
            category_id = cat_name_to_id.get(class_name, 0)
            
            coco_ann = {
                "id": max_ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "iscrowd": 0
            }
            
            if ann.get("bbox"):
                coco_ann["bbox"] = ann["bbox"]
                coco_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            
            if ann.get("segmentation"):
                coco_ann["segmentation"] = ann["segmentation"]
            
            coco_data["annotations"].append(coco_ann)
        
        with open(json_file, "w") as f:
            json.dump(coco_data, f, indent=2)
    
    def _update_voc_annotations(
        self,
        dataset_path: Path,
        image_id: str,
        annotations: List[Dict[str, Any]]
    ):
        """Update Pascal VOC format annotations"""
        # Find XML file
        xml_file = None
        for xf in dataset_path.glob(f"**/{image_id}.xml"):
            xml_file = xf
            break
        
        if not xml_file:
            xml_file = dataset_path / "Annotations" / f"{image_id}.xml"
            xml_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find corresponding image
        image_file = None
        for ext in self.IMAGE_EXTENSIONS:
            for pattern in [f"**/{image_id}{ext}"]:
                for found in dataset_path.glob(pattern):
                    image_file = found
                    break
                if image_file:
                    break
            if image_file:
                break
        
        # Get image dimensions
        width, height = 0, 0
        if image_file:
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception:
                pass
        
        # Create XML
        annotation_elem = ET.Element("annotation")
        
        folder = ET.SubElement(annotation_elem, "folder")
        folder.text = "JPEGImages"
        
        filename = ET.SubElement(annotation_elem, "filename")
        filename.text = image_file.name if image_file else f"{image_id}.jpg"
        
        size = ET.SubElement(annotation_elem, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth = ET.SubElement(size, "depth")
        depth.text = "3"
        
        for ann in annotations:
            obj = ET.SubElement(annotation_elem, "object")
            
            name = ET.SubElement(obj, "name")
            name.text = ann.get("class_name", "unknown")
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            if ann.get("bbox"):
                bndbox = ET.SubElement(obj, "bndbox")
                bbox = ann["bbox"]
                
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(int(bbox[0] if len(bbox) > 0 else 0))
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(int(bbox[1] if len(bbox) > 1 else 0))
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(int(bbox[2] if len(bbox) > 2 else 0))
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(int(bbox[3] if len(bbox) > 3 else 0))
        
        xml_str = minidom.parseString(ET.tostring(annotation_elem)).toprettyxml(indent="  ")
        with open(xml_file, "w") as f:
            f.write(xml_str)
    
    def _update_labelme_annotations(
        self,
        dataset_path: Path,
        image_id: str,
        annotations: List[Dict[str, Any]]
    ):
        """Update LabelMe format annotations"""
        # Find JSON file
        json_file = None
        for jf in dataset_path.glob(f"**/{image_id}.json"):
            json_file = jf
            break
        
        if not json_file:
            json_file = dataset_path / f"{image_id}.json"
        
        # Find corresponding image
        image_file = None
        image_path = ""
        for ext in self.IMAGE_EXTENSIONS:
            for found in dataset_path.glob(f"**/{image_id}{ext}"):
                image_file = found
                image_path = found.name
                break
            if image_file:
                break
        
        # Get image dimensions
        width, height = 0, 0
        if image_file:
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception:
                pass
        
        # Load existing or create new
        if json_file.exists():
            with open(json_file) as f:
                labelme_data = json.load(f)
        else:
            labelme_data = {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": image_path,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
        
        # Update shapes
        labelme_data["shapes"] = []
        
        for ann in annotations:
            shape = {
                "label": ann.get("class_name", "unknown"),
                "flags": {},
                "group_id": None
            }
            
            if ann.get("type") == "polygon" and ann.get("points"):
                points = ann["points"]
                shape["shape_type"] = "polygon"
                shape["points"] = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
            elif ann.get("bbox"):
                bbox = ann["bbox"]
                shape["shape_type"] = "rectangle"
                if len(bbox) == 4:
                    shape["points"] = [
                        [bbox[0], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    ]
                else:
                    shape["points"] = [[0, 0], [100, 100]]
            else:
                continue
            
            labelme_data["shapes"].append(shape)
        
        with open(json_file, "w") as f:
            json.dump(labelme_data, f, indent=2)
    
    def add_image(
        self,
        dataset_path: Path,
        format_name: str,
        file
    ) -> str:
        """Add a new image to the dataset"""
        dataset_path = Path(dataset_path)
        
        # Determine image directory
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            images_dir = dataset_path / "images"
        elif format_name in ["pascal-voc", "voc"]:
            images_dir = dataset_path / "JPEGImages"
        elif format_name == "coco":
            images_dir = dataset_path / "images"
        else:
            images_dir = dataset_path / "images"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_path = images_dir / file.filename
        with open(image_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        return str(image_path.relative_to(dataset_path))
    
    def add_classes(
        self,
        dataset_path: Path,
        format_name: str,
        new_classes: List[str]
    ):
        """Add new classes to a dataset"""
        dataset_path = Path(dataset_path)
        
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            self._add_yolo_classes(dataset_path, new_classes)
        elif format_name == "coco":
            self._add_coco_classes(dataset_path, new_classes)
    
    def _add_yolo_classes(self, dataset_path: Path, new_classes: List[str]):
        """Add classes to YOLO dataset"""
        # Find yaml config
        yaml_file = None
        for yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
            yaml_file = yf
            break
        
        if not yaml_file:
            yaml_file = dataset_path / "data.yaml"
        
        # Load existing config
        if yaml_file.exists():
            with open(yaml_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {
                "path": str(dataset_path.absolute()),
                "train": "images",
                "val": "images"
            }
        
        # Get existing classes
        if "names" in config:
            if isinstance(config["names"], dict):
                existing = list(config["names"].values())
            else:
                existing = config["names"]
        else:
            existing = []
        
        # Add new classes
        for cls in new_classes:
            if cls not in existing:
                existing.append(cls)
        
        config["names"] = {i: name for i, name in enumerate(existing)}
        config["nc"] = len(existing)
        
        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _add_coco_classes(self, dataset_path: Path, new_classes: List[str]):
        """Add classes to COCO dataset"""
        # Find COCO JSON file
        json_file = None
        for jf in list(dataset_path.glob("*.json")) + list(dataset_path.glob("annotations/*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if all(key in data for key in ["images", "annotations", "categories"]):
                        json_file = jf
                        break
            except Exception:
                pass
        
        if not json_file:
            return
        
        with open(json_file) as f:
            coco_data = json.load(f)
        
        # Get existing class names
        existing = {cat["name"] for cat in coco_data["categories"]}
        max_id = max([cat["id"] for cat in coco_data["categories"]], default=0)
        
        # Add new classes
        for cls in new_classes:
            if cls not in existing:
                max_id += 1
                coco_data["categories"].append({
                    "id": max_id,
                    "name": cls,
                    "supercategory": "none"
                })
        
        with open(json_file, "w") as f:
            json.dump(coco_data, f, indent=2)
    
    # ─────────────────────── CLASS MANAGEMENT ───────────────────────

    YOLO_FORMATS = {
        "yolo", "yolov5", "yolov8", "yolov9", "yolov10",
        "yolov11", "yolov12", "yolo_seg", "yolo-seg",
    }

    def _load_yolo_meta(self, root: Path):
        """Return (all_classes, yaml_file, yaml_config) for a YOLO root."""
        for yf in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            try:
                with open(yf) as f:
                    config = yaml.safe_load(f) or {}
                if "names" in config:
                    names = config["names"]
                    classes = list(names.values()) if isinstance(names, dict) else list(names)
                    return classes, yf, config
            except Exception:
                pass
        return [], None, {}

    def _save_yolo_meta(self, yaml_file: Path, config: dict, classes: List[str]):
        config = dict(config)
        config.pop("path", None)
        config["nc"] = len(classes)
        config["names"] = {i: n for i, n in enumerate(classes)}
        with open(yaml_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _find_yolo_label_files(self, root: Path) -> List[Path]:
        """Return all YOLO label .txt files under root."""
        found = []
        for lbl_dir in root.glob("**/labels"):
            if lbl_dir.is_dir():
                found.extend(lbl_dir.glob("*.txt"))
        if not found:
            # Flat layout: check every .txt
            for txt in root.glob("**/*.txt"):
                if txt.stat().st_size == 0:
                    found.append(txt)
                    continue
                try:
                    with open(txt) as f:
                        line = f.readline().strip()
                    if not line:
                        found.append(txt)
                        continue
                    parts = line.split()
                    int(parts[0])
                    float(parts[1])
                    found.append(txt)
                except Exception:
                    pass
        return found

    def _find_coco_json(self, root: Path) -> Optional[Path]:
        for jf in list(root.glob("*.json")) + list(root.glob("annotations/*.json")):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if all(k in data for k in ("images", "annotations", "categories")):
                    return jf
            except Exception:
                pass
        return None

    # ── extract ──────────────────────────────────────────────────────

    def extract_classes(
        self,
        dataset_path: Path,
        output_path: Path,
        format_name: str,
        classes_to_extract: List[str],
    ) -> Dict[str, Any]:
        """Extract specific classes to a new dataset (only images that have those classes)."""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        fmt = format_name.lower()
        if fmt in self.YOLO_FORMATS:
            return self._extract_yolo_classes(dataset_path, output_path, classes_to_extract)
        elif fmt == "coco":
            return self._extract_coco_classes(dataset_path, output_path, classes_to_extract)
        elif fmt in ("pascal-voc", "voc"):
            return self._extract_voc_classes(dataset_path, output_path, classes_to_extract)
        elif fmt == "labelme":
            return self._extract_labelme_classes(dataset_path, output_path, classes_to_extract)
        return {"extracted_images": 0, "extracted_annotations": 0}

    def _extract_yolo_classes(self, dataset_path: Path, output_path: Path, classes_to_extract: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        all_classes, yaml_file, yaml_config = self._load_yolo_meta(root)

        class_to_id = {n: i for i, n in enumerate(all_classes)}
        valid = [c for c in classes_to_extract if c in class_to_id]
        target_ids = {class_to_id[c] for c in valid}
        new_class_to_id = {n: i for i, n in enumerate(valid)}
        old_to_new = {class_to_id[n]: new_class_to_id[n] for n in valid}

        extracted_images = extracted_annotations = 0

        for img_file in root.glob("**/*"):
            if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            stem = img_file.stem
            parent = img_file.parent
            if parent.name == "images":
                lbl_file = parent.parent / "labels" / f"{stem}.txt"
            else:
                lbl_file = parent.parent / "labels" / f"{stem}.txt"
            if not lbl_file.exists():
                lbl_file = parent / f"{stem}.txt"
            if not lbl_file.exists():
                continue

            filtered = []
            try:
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        if cid in target_ids:
                            filtered.append(f"{old_to_new[cid]} " + " ".join(parts[1:]))
            except Exception:
                continue

            if not filtered:
                continue

            try:
                img_rel = img_file.relative_to(root)
            except ValueError:
                img_rel = Path("images") / img_file.name
            try:
                lbl_rel = lbl_file.relative_to(root)
            except ValueError:
                lbl_rel = Path("labels") / lbl_file.name

            out_img = output_path / img_rel
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, out_img)

            out_lbl = output_path / lbl_rel
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            with open(out_lbl, "w") as f:
                f.write("\n".join(filtered) + "\n")

            extracted_images += 1
            extracted_annotations += len(filtered)

        # Write YAML
        new_config = dict(yaml_config)
        new_config.pop("path", None)
        new_config["nc"] = len(valid)
        new_config["names"] = {i: n for i, n in enumerate(valid)}
        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        return {"extracted_images": extracted_images, "extracted_annotations": extracted_annotations}

    def _extract_coco_classes(self, dataset_path: Path, output_path: Path, classes_to_extract: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        jf = self._find_coco_json(root)
        if not jf:
            return {"extracted_images": 0, "extracted_annotations": 0}
        with open(jf) as f:
            data = json.load(f)

        extract_set = set(classes_to_extract)
        sel_cat_ids = {c["id"] for c in data["categories"] if c["name"] in extract_set}
        filtered_anns = [a for a in data["annotations"] if a["category_id"] in sel_cat_ids]
        img_ids = {a["image_id"] for a in filtered_anns}
        filtered_imgs = [i for i in data["images"] if i["id"] in img_ids]
        filtered_cats = [c for c in data["categories"] if c["name"] in extract_set]

        new_data = {k: data[k] for k in ("info", "licenses") if k in data}
        new_data.update({"images": filtered_imgs, "annotations": filtered_anns, "categories": filtered_cats})

        rel = jf.relative_to(root)
        out_jf = output_path / rel
        out_jf.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jf, "w") as f:
            json.dump(new_data, f, indent=2)

        extracted = 0
        for img_info in filtered_imgs:
            fname = Path(img_info.get("file_name", "")).name
            for candidate in root.glob(f"**/{fname}"):
                try:
                    out = output_path / candidate.relative_to(root)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, out)
                    extracted += 1
                    break
                except Exception:
                    pass

        return {"extracted_images": extracted, "extracted_annotations": len(filtered_anns)}

    def _extract_voc_classes(self, dataset_path: Path, output_path: Path, classes_to_extract: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        extract_set = set(classes_to_extract)
        ann_dir = root / "Annotations"
        if not ann_dir.exists():
            ann_dir = root

        img_dirs = [root / "JPEGImages", root / "images", root]

        extracted_images = extracted_annotations = 0

        for xf in list(ann_dir.glob("*.xml")) + ([] if ann_dir == root else list(root.glob("**/*.xml"))):
            try:
                tree = ET.parse(xf)
                re_ = tree.getroot()
            except Exception:
                continue

            matching = [o for o in re_.findall("object")
                        if o.find("name") is not None and o.find("name").text in extract_set]
            if not matching:
                continue

            for obj in list(re_.findall("object")):
                name_el = obj.find("name")
                if name_el is None or name_el.text not in extract_set:
                    re_.remove(obj)

            fn_el = re_.find("filename")
            fname = fn_el.text if fn_el is not None else xf.stem
            img_file = None
            for d in img_dirs:
                for ext in self.IMAGE_EXTENSIONS:
                    c = d / f"{Path(fname).stem}{ext}"
                    if c.exists():
                        img_file = c
                        break
                if img_file:
                    break

            out_ann = output_path / "Annotations"
            out_ann.mkdir(parents=True, exist_ok=True)
            xml_str = minidom.parseString(ET.tostring(re_)).toprettyxml(indent="  ")
            with open(out_ann / xf.name, "w") as f:
                f.write(xml_str)

            if img_file:
                out_img = output_path / "JPEGImages"
                out_img.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, out_img / img_file.name)

            extracted_images += 1
            extracted_annotations += len(matching)

        return {"extracted_images": extracted_images, "extracted_annotations": extracted_annotations}

    def _extract_labelme_classes(self, dataset_path: Path, output_path: Path, classes_to_extract: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        extract_set = set(classes_to_extract)
        extracted_images = extracted_annotations = 0

        for jf in root.glob("**/*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "shapes" not in data:
                    continue
            except Exception:
                continue

            matching = [s for s in data["shapes"] if s.get("label") in extract_set]
            if not matching:
                continue

            data["shapes"] = matching
            try:
                rel = jf.relative_to(root)
            except ValueError:
                rel = Path(jf.name)
            out_jf = output_path / rel
            out_jf.parent.mkdir(parents=True, exist_ok=True)
            with open(out_jf, "w") as f:
                json.dump(data, f, indent=2)

            img_path = data.get("imagePath", "")
            img_file = None
            if img_path:
                c = jf.parent / img_path
                if c.exists():
                    img_file = c
            if not img_file:
                for ext in self.IMAGE_EXTENSIONS:
                    c = jf.parent / f"{jf.stem}{ext}"
                    if c.exists():
                        img_file = c
                        break
            if img_file:
                try:
                    out_img = output_path / img_file.relative_to(root)
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, out_img)
                except Exception:
                    pass

            extracted_images += 1
            extracted_annotations += len(matching)

        return {"extracted_images": extracted_images, "extracted_annotations": extracted_annotations}

    # ── delete ───────────────────────────────────────────────────────

    def delete_classes(
        self,
        dataset_path: Path,
        format_name: str,
        classes_to_delete: List[str],
    ) -> Dict[str, Any]:
        """Remove all annotations for the given classes from the dataset."""
        dataset_path = Path(dataset_path)
        fmt = format_name.lower()
        if fmt in self.YOLO_FORMATS:
            return self._delete_yolo_classes(dataset_path, classes_to_delete)
        elif fmt == "coco":
            return self._delete_coco_classes(dataset_path, classes_to_delete)
        elif fmt in ("pascal-voc", "voc"):
            return self._delete_voc_classes(dataset_path, classes_to_delete)
        elif fmt == "labelme":
            return self._delete_labelme_classes(dataset_path, classes_to_delete)
        return {"deleted_annotations": 0, "affected_images": 0}

    def _delete_yolo_classes(self, dataset_path: Path, classes_to_delete: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        all_classes, yaml_file, yaml_config = self._load_yolo_meta(root)

        del_set = set(classes_to_delete)
        class_to_id = {n: i for i, n in enumerate(all_classes)}
        del_ids = {class_to_id[c] for c in del_set if c in class_to_id}

        remaining = [c for c in all_classes if c not in del_set]
        old_to_new: Dict[int, int] = {}
        new_idx = 0
        for i, name in enumerate(all_classes):
            if name not in del_set:
                old_to_new[i] = new_idx
                new_idx += 1

        deleted = affected = 0
        for lbl in self._find_yolo_label_files(root):
            try:
                with open(lbl) as f:
                    lines = f.readlines()
            except Exception:
                continue
            new_lines, changed = [], False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    new_lines.append(line)
                    continue
                if cid in del_ids:
                    deleted += 1
                    changed = True
                else:
                    new_id = old_to_new.get(cid, cid)
                    new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
            if changed:
                affected += 1
                with open(lbl, "w") as f:
                    f.writelines(new_lines)

        if yaml_file:
            self._save_yolo_meta(yaml_file, yaml_config, remaining)

        return {"deleted_annotations": deleted, "affected_images": affected}

    def _delete_coco_classes(self, dataset_path: Path, classes_to_delete: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        jf = self._find_coco_json(root)
        if not jf:
            return {"deleted_annotations": 0, "affected_images": 0}
        with open(jf) as f:
            data = json.load(f)

        del_set = set(classes_to_delete)
        del_ids = {c["id"] for c in data["categories"] if c["name"] in del_set}
        orig = len(data["annotations"])
        data["annotations"] = [a for a in data["annotations"] if a["category_id"] not in del_ids]
        data["categories"] = [c for c in data["categories"] if c["name"] not in del_set]
        affected = len({a["image_id"] for a in data["annotations"]})

        with open(jf, "w") as f:
            json.dump(data, f, indent=2)
        return {"deleted_annotations": orig - len(data["annotations"]), "affected_images": affected}

    def _delete_voc_classes(self, dataset_path: Path, classes_to_delete: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        del_set = set(classes_to_delete)
        deleted = affected = 0
        for xf in root.glob("**/*.xml"):
            try:
                tree = ET.parse(xf)
                re_ = tree.getroot()
            except Exception:
                continue
            to_rm = [o for o in re_.findall("object")
                     if o.find("name") is not None and o.find("name").text in del_set]
            if to_rm:
                for o in to_rm:
                    re_.remove(o)
                    deleted += 1
                affected += 1
                with open(xf, "w") as f:
                    f.write(minidom.parseString(ET.tostring(re_)).toprettyxml(indent="  "))
        return {"deleted_annotations": deleted, "affected_images": affected}

    def _delete_labelme_classes(self, dataset_path: Path, classes_to_delete: List[str]) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        del_set = set(classes_to_delete)
        deleted = affected = 0
        for jf in root.glob("**/*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "shapes" not in data:
                    continue
            except Exception:
                continue
            orig = len(data["shapes"])
            data["shapes"] = [s for s in data["shapes"] if s.get("label") not in del_set]
            rm = orig - len(data["shapes"])
            if rm:
                deleted += rm
                affected += 1
                with open(jf, "w") as f:
                    json.dump(data, f, indent=2)
        return {"deleted_annotations": deleted, "affected_images": affected}

    # ── merge ────────────────────────────────────────────────────────

    def merge_classes(
        self,
        dataset_path: Path,
        format_name: str,
        source_classes: List[str],
        target_class: str,
    ) -> Dict[str, Any]:
        """Merge source_classes into target_class."""
        dataset_path = Path(dataset_path)
        fmt = format_name.lower()
        if fmt in self.YOLO_FORMATS:
            return self._merge_yolo_classes(dataset_path, source_classes, target_class)
        elif fmt == "coco":
            return self._merge_coco_classes(dataset_path, source_classes, target_class)
        elif fmt in ("pascal-voc", "voc"):
            return self._merge_voc_classes(dataset_path, source_classes, target_class)
        elif fmt == "labelme":
            return self._merge_labelme_classes(dataset_path, source_classes, target_class)
        return {"merged_annotations": 0}

    def _merge_yolo_classes(self, dataset_path: Path, source_classes: List[str], target_class: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        all_classes, yaml_file, yaml_config = self._load_yolo_meta(root)

        src_set = set(source_classes)

        # Ensure target exists
        if target_class not in all_classes:
            all_classes.append(target_class)

        class_to_id = {n: i for i, n in enumerate(all_classes)}
        target_old_id = class_to_id[target_class]
        src_ids = {class_to_id[c] for c in src_set if c in class_to_id and c != target_class}

        # New class list: remove source classes that aren't the target
        new_classes = [c for c in all_classes if c not in src_set or c == target_class]
        new_class_to_id = {n: i for i, n in enumerate(new_classes)}
        old_to_new: Dict[int, int] = {}
        for i, name in enumerate(all_classes):
            if name in src_set and name != target_class:
                old_to_new[i] = new_class_to_id[target_class]
            else:
                old_to_new[i] = new_class_to_id.get(name, i)

        merged = 0
        for lbl in self._find_yolo_label_files(root):
            try:
                with open(lbl) as f:
                    lines = f.readlines()
            except Exception:
                continue
            new_lines, changed = [], False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    new_lines.append(line)
                    continue
                new_id = old_to_new.get(cid, cid)
                if new_id != cid:
                    merged += 1
                    changed = True
                new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
            if changed:
                with open(lbl, "w") as f:
                    f.writelines(new_lines)

        if yaml_file:
            self._save_yolo_meta(yaml_file, yaml_config, new_classes)

        return {"merged_annotations": merged}

    def _merge_coco_classes(self, dataset_path: Path, source_classes: List[str], target_class: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        jf = self._find_coco_json(root)
        if not jf:
            return {"merged_annotations": 0}
        with open(jf) as f:
            data = json.load(f)

        src_set = set(source_classes)
        target_cat = next((c for c in data["categories"] if c["name"] == target_class), None)
        if not target_cat:
            max_id = max((c["id"] for c in data["categories"]), default=0)
            target_cat = {"id": max_id + 1, "name": target_class, "supercategory": "none"}
            data["categories"].append(target_cat)
        target_id = target_cat["id"]

        src_ids = {c["id"] for c in data["categories"] if c["name"] in src_set and c["name"] != target_class}
        merged = 0
        for ann in data["annotations"]:
            if ann["category_id"] in src_ids:
                ann["category_id"] = target_id
                merged += 1
        data["categories"] = [c for c in data["categories"] if c["name"] not in src_set or c["name"] == target_class]

        with open(jf, "w") as f:
            json.dump(data, f, indent=2)
        return {"merged_annotations": merged}

    def _merge_voc_classes(self, dataset_path: Path, source_classes: List[str], target_class: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        src_set = set(source_classes)
        merged = 0
        for xf in root.glob("**/*.xml"):
            try:
                tree = ET.parse(xf)
                re_ = tree.getroot()
            except Exception:
                continue
            changed = False
            for obj in re_.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text in src_set and name_el.text != target_class:
                    name_el.text = target_class
                    merged += 1
                    changed = True
            if changed:
                with open(xf, "w") as f:
                    f.write(minidom.parseString(ET.tostring(re_)).toprettyxml(indent="  "))
        return {"merged_annotations": merged}

    def _merge_labelme_classes(self, dataset_path: Path, source_classes: List[str], target_class: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        src_set = set(source_classes)
        merged = 0
        for jf in root.glob("**/*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "shapes" not in data:
                    continue
            except Exception:
                continue
            changed = False
            for s in data["shapes"]:
                if s.get("label") in src_set and s.get("label") != target_class:
                    s["label"] = target_class
                    merged += 1
                    changed = True
            if changed:
                with open(jf, "w") as f:
                    json.dump(data, f, indent=2)
        return {"merged_annotations": merged}

    # ── rename ───────────────────────────────────────────────────────

    def rename_class(
        self,
        dataset_path: Path,
        format_name: str,
        old_name: str,
        new_name: str,
    ) -> Dict[str, Any]:
        """Rename a single class across all annotations."""
        dataset_path = Path(dataset_path)
        fmt = format_name.lower()
        if fmt in self.YOLO_FORMATS:
            return self._rename_yolo_class(dataset_path, old_name, new_name)
        elif fmt == "coco":
            return self._rename_coco_class(dataset_path, old_name, new_name)
        elif fmt in ("pascal-voc", "voc"):
            return self._rename_voc_class(dataset_path, old_name, new_name)
        elif fmt == "labelme":
            return self._rename_labelme_class(dataset_path, old_name, new_name)
        return {"renamed_annotations": 0}

    def _rename_yolo_class(self, dataset_path: Path, old_name: str, new_name: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        all_classes, yaml_file, yaml_config = self._load_yolo_meta(root)
        if old_name not in all_classes:
            return {"renamed_annotations": 0}
        new_classes = [new_name if c == old_name else c for c in all_classes]
        old_id = all_classes.index(old_name)
        renamed = 0
        for lbl in self._find_yolo_label_files(root):
            try:
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and int(parts[0]) == old_id:
                            renamed += 1
            except Exception:
                pass
        if yaml_file:
            self._save_yolo_meta(yaml_file, yaml_config, new_classes)
        return {"renamed_annotations": renamed}

    def _rename_coco_class(self, dataset_path: Path, old_name: str, new_name: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        jf = self._find_coco_json(root)
        if not jf:
            return {"renamed_annotations": 0}
        with open(jf) as f:
            data = json.load(f)
        cat_id = None
        for cat in data["categories"]:
            if cat["name"] == old_name:
                cat["name"] = new_name
                cat_id = cat["id"]
                break
        if cat_id is None:
            return {"renamed_annotations": 0}
        renamed = sum(1 for a in data["annotations"] if a["category_id"] == cat_id)
        with open(jf, "w") as f:
            json.dump(data, f, indent=2)
        return {"renamed_annotations": renamed}

    def _rename_voc_class(self, dataset_path: Path, old_name: str, new_name: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        renamed = 0
        for xf in root.glob("**/*.xml"):
            try:
                tree = ET.parse(xf)
                re_ = tree.getroot()
            except Exception:
                continue
            changed = False
            for obj in re_.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text == old_name:
                    name_el.text = new_name
                    renamed += 1
                    changed = True
            if changed:
                with open(xf, "w") as f:
                    f.write(minidom.parseString(ET.tostring(re_)).toprettyxml(indent="  "))
        return {"renamed_annotations": renamed}

    def _rename_labelme_class(self, dataset_path: Path, old_name: str, new_name: str) -> Dict[str, Any]:
        root = self._find_dataset_root(dataset_path)
        renamed = 0
        for jf in root.glob("**/*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "shapes" not in data:
                    continue
            except Exception:
                continue
            changed = False
            for s in data["shapes"]:
                if s.get("label") == old_name:
                    s["label"] = new_name
                    renamed += 1
                    changed = True
            if changed:
                with open(jf, "w") as f:
                    json.dump(data, f, indent=2)
        return {"renamed_annotations": renamed}

    def create_empty_annotation(
        self,
        dataset_path: Path,
        format_name: str,
        image_id: str,
        image_filename: str,
        width: int,
        height: int
    ):
        """Create an empty annotation file for an image"""
        dataset_path = Path(dataset_path)
        
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            labels_dir = dataset_path / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            (labels_dir / f"{image_id}.txt").touch()
        
        elif format_name == "coco":
            # Add to COCO JSON
            json_file = None
            for jf in list(dataset_path.glob("*.json")) + list(dataset_path.glob("annotations/*.json")):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                        if all(key in data for key in ["images", "annotations", "categories"]):
                            json_file = jf
                            break
                except Exception:
                    pass
            
            if json_file:
                with open(json_file) as f:
                    coco_data = json.load(f)
                
                max_img_id = max([img["id"] for img in coco_data["images"]], default=0)
                coco_data["images"].append({
                    "id": max_img_id + 1,
                    "file_name": image_filename,
                    "width": width,
                    "height": height
                })
                
                with open(json_file, "w") as f:
                    json.dump(coco_data, f, indent=2)
        
        elif format_name in ["pascal-voc", "voc"]:
            ann_dir = dataset_path / "Annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            annotation = ET.Element("annotation")
            
            folder = ET.SubElement(annotation, "folder")
            folder.text = "JPEGImages"
            
            filename = ET.SubElement(annotation, "filename")
            filename.text = image_filename
            
            size = ET.SubElement(annotation, "size")
            width_elem = ET.SubElement(size, "width")
            width_elem.text = str(width)
            height_elem = ET.SubElement(size, "height")
            height_elem.text = str(height)
            depth = ET.SubElement(size, "depth")
            depth.text = "3"
            
            xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
            with open(ann_dir / f"{image_id}.xml", "w") as f:
                f.write(xml_str)
        
        elif format_name == "labelme":
            labelme_data = {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": image_filename,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
            
            with open(dataset_path / f"{image_id}.json", "w") as f:
                json.dump(labelme_data, f, indent=2)
