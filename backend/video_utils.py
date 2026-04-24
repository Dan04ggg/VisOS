"""
Video Frame Extraction and Duplicate Detection Utilities
"""

import os
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import json


class VideoFrameExtractor:
    """Extract frames from video files for dataset creation"""
    
    SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        nth_frame: int = 1,
        max_frames: Optional[int] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        image_format: str = "jpg",
        quality: int = 95,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Extract every nth frame from a video file
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            nth_frame: Extract every nth frame (1 = all frames, 30 = 1 per second at 30fps)
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds (None = until end)
            image_format: Output image format (jpg, png)
            quality: JPEG quality (1-100)
        
        Returns:
            Dictionary with extraction results
        """
        try:
            import cv2
        except ImportError:
            return {
                "success": False,
                "error": "OpenCV not installed. Run: pip install opencv-python",
                "extracted_frames": 0
            }
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            return {"success": False, "error": "Video file not found", "extracted_frames": 0}
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file", "extracted_frames": 0}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)

        # Build list of frame positions to extract (direct seek = much faster)
        frames_to_extract = list(range(start_frame, end_frame, max(1, nth_frame)))
        if max_frames:
            frames_to_extract = frames_to_extract[:max_frames]

        total_to_extract = len(frames_to_extract)
        extracted_frames = []

        for i, frame_pos in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"frame_{frame_pos:08d}.{image_format}"
            frame_path = output_dir / frame_filename

            if image_format == "jpg":
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(frame_path), frame)

            extracted_frames.append({
                "filename": frame_filename,
                "frame_number": frame_pos,
                "timestamp": frame_pos / fps if fps > 0 else 0
            })

            if progress_callback and total_to_extract > 0:
                progress_callback(int((i + 1) / total_to_extract * 100), i + 1)

        cap.release()
        
        return {
            "success": True,
            "extracted_frames": len(extracted_frames),
            "frames": extracted_frames,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 0,
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 0
            }
        }
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get information about a video file"""
        try:
            import cv2
        except ImportError:
            return {"success": False, "error": "OpenCV not installed"}
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "success": True,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
            "filename": video_path.name
        }


class DuplicateDetector:
    """Find and manage duplicate/similar images in datasets"""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
    
    def __init__(self):
        self.hash_cache: Dict[str, str] = {}
    
    def compute_perceptual_hash(self, image_path: Path, hash_size: int = 16) -> str:
        """
        Compute perceptual hash (pHash) for an image
        Similar images will have similar hashes
        """
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            
            # Compute difference between adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Convert to hash string
            return ''.join(str(int(b)) for b in diff.flatten())
        except Exception as e:
            return ""
    
    def compute_average_hash(self, image_path: Path, hash_size: int = 8) -> str:
        """
        Compute average hash (aHash) for an image
        Faster but less accurate than pHash
        """
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            avg = pixels.mean()
            diff = pixels > avg
            return ''.join(str(int(b)) for b in diff.flatten())
        except Exception as e:
            return ""
    
    def compute_md5_hash(self, image_path: Path) -> str:
        """Compute MD5 hash for exact duplicate detection"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes"""
        if len(hash1) != len(hash2):
            return -1
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def find_duplicates(
        self,
        dataset_path: Path,
        method: str = "perceptual",
        threshold: int = 10,
        include_near_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Find duplicate and near-duplicate images in a dataset
        
        Args:
            dataset_path: Path to the dataset
            method: Detection method ('md5', 'perceptual', 'average')
            threshold: Hamming distance threshold for near-duplicates (perceptual/average only)
            include_near_duplicates: Include near-duplicates or only exact matches
        
        Returns:
            Dictionary with duplicate groups and statistics
        """
        dataset_path = Path(dataset_path)
        
        # Find all images
        images = []
        for ext in self.IMAGE_EXTENSIONS:
            images.extend(dataset_path.glob(f"**/*{ext}"))
            images.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        # Compute hashes
        image_hashes: Dict[str, Tuple[str, Path]] = {}
        
        for img_path in images:
            if method == "md5":
                img_hash = self.compute_md5_hash(img_path)
            elif method == "perceptual":
                img_hash = self.compute_perceptual_hash(img_path)
            else:  # average
                img_hash = self.compute_average_hash(img_path)
            
            if img_hash:
                image_hashes[str(img_path)] = (img_hash, img_path)
        
        # Find duplicates
        duplicate_groups: List[List[Dict[str, Any]]] = []
        processed: set = set()
        
        hash_list = list(image_hashes.items())
        
        for i, (path1, (hash1, img_path1)) in enumerate(hash_list):
            if path1 in processed:
                continue
            
            group = [{
                "path": str(img_path1.relative_to(dataset_path)),
                "full_path": str(img_path1),
                "hash": hash1,
                "is_original": True
            }]
            
            for path2, (hash2, img_path2) in hash_list[i+1:]:
                if path2 in processed:
                    continue
                
                if method == "md5":
                    is_duplicate = hash1 == hash2
                    distance = 0 if is_duplicate else -1
                else:
                    distance = self.hamming_distance(hash1, hash2)
                    is_duplicate = distance <= threshold if include_near_duplicates else distance == 0
                
                if is_duplicate:
                    group.append({
                        "path": str(img_path2.relative_to(dataset_path)),
                        "full_path": str(img_path2),
                        "hash": hash2,
                        "distance": distance,
                        "is_original": False
                    })
                    processed.add(path2)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(path1)
        
        # Calculate statistics
        total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
        
        return {
            "success": True,
            "method": method,
            "threshold": threshold,
            "total_images": len(images),
            "duplicate_groups": len(duplicate_groups),
            "total_duplicates": total_duplicates,
            "unique_images": len(images) - total_duplicates,
            "groups": duplicate_groups
        }
    
    def remove_duplicates(
        self,
        dataset_path: Path,
        format_name: str,
        duplicate_groups: List[List[Dict[str, Any]]],
        keep_strategy: str = "first"
    ) -> Dict[str, Any]:
        """
        Remove duplicate images from dataset
        
        Args:
            dataset_path: Path to the dataset
            format_name: Dataset format
            duplicate_groups: Groups of duplicate images
            keep_strategy: How to decide which to keep ('first', 'largest', 'smallest')
        
        Returns:
            Dictionary with removal results
        """
        dataset_path = Path(dataset_path)
        removed_files = []
        
        for group in duplicate_groups:
            # Determine which to keep
            if keep_strategy == "largest":
                group.sort(key=lambda x: Path(x["full_path"]).stat().st_size, reverse=True)
            elif keep_strategy == "smallest":
                group.sort(key=lambda x: Path(x["full_path"]).stat().st_size)
            # else: keep first (default order)
            
            # Keep first, remove rest
            for item in group[1:]:
                img_path = Path(item["full_path"])
                
                if img_path.exists():
                    # Remove image
                    img_path.unlink()
                    removed_files.append(str(img_path))
                    
                    # Remove corresponding annotation
                    self._remove_annotation(dataset_path, format_name, img_path.stem)
        
        return {
            "success": True,
            "removed_count": len(removed_files),
            "removed_files": removed_files
        }
    
    def _remove_annotation(self, dataset_path: Path, format_name: str, image_id: str):
        """Remove annotation file for an image"""
        if format_name in ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]:
            # Remove YOLO label file
            for label_dir in ["labels", "train/labels", "val/labels", "test/labels"]:
                label_file = dataset_path / label_dir / f"{image_id}.txt"
                if label_file.exists():
                    label_file.unlink()
        
        elif format_name in ["pascal-voc", "voc"]:
            # Remove VOC XML file
            for ann_dir in ["Annotations", ""]:
                xml_file = dataset_path / ann_dir / f"{image_id}.xml"
                if xml_file.exists():
                    xml_file.unlink()
        
        elif format_name == "labelme":
            # Remove LabelMe JSON file
            json_file = dataset_path / f"{image_id}.json"
            if json_file.exists():
                json_file.unlink()


class CLIPEmbeddingManager:
    """
    Use CLIP embeddings for semantic similarity detection
    Requires: pip install transformers torch
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        # Per-dataset cache: dataset_path -> (valid_images_relative, similarities_matrix, total_images)
        self.dataset_cache: Dict[str, Any] = {}
        # Per-dataset cancellation events for cooperative cancellation
        self._cancel_events: Dict[str, threading.Event] = {}

    def cancel_scan(self, dataset_id: str):
        """Signal an in-progress CLIP scan to stop."""
        event = self._cancel_events.get(dataset_id)
        if event:
            event.set()
    
    def load_model(self):
        """Load CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            return {"success": True}
        except ImportError:
            return {
                "success": False,
                "error": "transformers and torch not installed. Run: pip install transformers torch"
            }
    
    def compute_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Compute CLIP embedding for an image"""
        if self.model is None:
            return None
        
        try:
            import torch
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            return features.cpu().numpy().flatten()
        except Exception:
            return None
    
    def _build_groups(
        self,
        valid_images: List[Path],
        similarities: np.ndarray,
        similarity_threshold: float,
        dataset_path: Path,
        total_images: int,
    ) -> Dict[str, Any]:
        """Group images by similarity threshold using a precomputed similarity matrix."""
        similar_groups: List[List[Dict[str, Any]]] = []
        processed: set = set()

        for i in range(len(valid_images)):
            if i in processed:
                continue

            group: List[Dict[str, Any]] = [{
                "path": str(valid_images[i].relative_to(dataset_path)),
                "full_path": str(valid_images[i]),
                "is_original": True,
            }]

            for j in range(i + 1, len(valid_images)):
                if j in processed:
                    continue
                if similarities[i, j] >= similarity_threshold:
                    group.append({
                        "path": str(valid_images[j].relative_to(dataset_path)),
                        "full_path": str(valid_images[j]),
                        "similarity": float(similarities[i, j]),
                        "is_original": False,
                    })
                    processed.add(j)

            if len(group) > 1:
                similar_groups.append(group)
                processed.add(i)

        return {
            "success": True,
            "total_images": total_images,
            "similar_groups": len(similar_groups),
            "total_similar": sum(len(g) - 1 for g in similar_groups),
            "groups": similar_groups,
        }

    def find_similar_images(
        self,
        dataset_path: Path,
        similarity_threshold: float = 0.9,
        batch_size: int = 32,
        dataset_id: str = "",
    ) -> Dict[str, Any]:
        """
        Find semantically similar images using CLIP embeddings.
        Embeddings are cached so regroup_by_threshold can be called
        without recomputing them.
        Pass dataset_id to enable cooperative cancellation via cancel_scan().
        """
        if self.model is None:
            result = self.load_model()
            if not result["success"]:
                return result

        dataset_path = Path(dataset_path)
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

        # Set up a fresh cancel event for this scan
        cancel_event = threading.Event()
        if dataset_id:
            self._cancel_events[dataset_id] = cancel_event

        # Find all images
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(dataset_path.glob(f"**/*{ext}"))
            images.extend(dataset_path.glob(f"**/*{ext.upper()}"))

        # Compute embeddings
        embeddings = []
        valid_images = []

        for img_path in images:
            if cancel_event.is_set():
                if dataset_id:
                    self._cancel_events.pop(dataset_id, None)
                return {"success": False, "error": "Scan cancelled"}
            emb = self.compute_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
                valid_images.append(img_path)

        if dataset_id:
            self._cancel_events.pop(dataset_id, None)

        if not embeddings:
            return {"success": False, "error": "No valid images found"}

        # Convert to numpy array and normalize
        emb_array = np.array(embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        emb_array = emb_array / norms

        # Compute pairwise similarities
        similarities = np.dot(emb_array, emb_array.T)

        # Cache for regroup_by_threshold
        self.dataset_cache[str(dataset_path)] = (valid_images, similarities, len(images))

        return self._build_groups(valid_images, similarities, similarity_threshold, dataset_path, len(images))

    def regroup_by_threshold(
        self,
        dataset_path: Path,
        similarity_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Re-group images using cached embeddings — no recomputation needed.
        Returns an error if find_similar_images has not been run for this dataset yet.
        """
        cache_key = str(Path(dataset_path))
        if cache_key not in self.dataset_cache:
            return {
                "success": False,
                "error": "No cached embeddings for this dataset. Run a full scan first.",
            }

        valid_images, similarities, total_images = self.dataset_cache[cache_key]
        return self._build_groups(valid_images, similarities, similarity_threshold, Path(dataset_path), total_images)
