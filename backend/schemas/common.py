from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    format: str
    task_type: str
    num_images: int
    num_annotations: int
    classes: List[str]
    created_at: str
    splits: Optional[Dict[str, int]] = None


class AnnotationUpdate(BaseModel):
    image_id: str
    annotations: List[Dict[str, Any]]


class ConversionRequest(BaseModel):
    dataset_id: str
    target_format: str
    output_name: Optional[str] = None


class MergeRequest(BaseModel):
    dataset_ids: List[str]
    output_name: str
    output_format: str
    class_mapping: Optional[Dict[str, str]] = None


class TrainingConfig(BaseModel):
    dataset_id: str
    name: str = ""
    model_type: str
    model_arch: str = "yolov8n"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    pretrained: bool = True
    device: str = "auto"
    save_period: int = 0  # save checkpoint every N epochs (0 = disabled)
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer: str = "SGD"
    patience: int = 50
    cos_lr: bool = False
    warmup_epochs: float = 3.0
    weight_decay: float = 0.0005
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.5
    amp: bool = True
    dropout: float = 0.0


class SettingsConfig(BaseModel):
    models_path: Optional[str] = None
    datasets_path: Optional[str] = None
    output_path: Optional[str] = None
    use_gpu: bool = True
    gpu_device: str = "0"


class ClassAddRequest(BaseModel):
    dataset_id: str
    new_classes: List[str]
    use_model: bool = False
    model_id: Optional[str] = None


class ClassExtractRequest(BaseModel):
    dataset_id: str
    classes_to_extract: List[str]
    output_name: str
    output_format: Optional[str] = None


class ClassDeleteRequest(BaseModel):
    dataset_id: str
    classes_to_delete: List[str]


class ClassMergeRequest(BaseModel):
    dataset_id: str
    source_classes: List[str]
    target_class: str


class ClassRenameRequest(BaseModel):
    dataset_id: str
    old_name: str
    new_name: str


class SplitRequest(BaseModel):
    dataset_id: Optional[str] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    output_name: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None
    stratify: bool = False


class AugmentationConfig(BaseModel):
    dataset_id: str
    output_name: str
    target_size: int
    augmentations: Dict[str, Dict[str, Any]]


class SortingAction(BaseModel):
    image_id: str
    action: str


class AnnotationHistoryEntry(BaseModel):
    timestamp: str
    action: str
    annotation_data: Dict[str, Any]


class LocalFolderRequest(BaseModel):
    folder_path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


class LocalPathRequest(BaseModel):
    path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


class VideoExtractRequest(BaseModel):
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    output_name: str
    mode: str = "interval"
    nth_frame: int = 30
    frame_interval: Optional[int] = None
    uniform_count: Optional[int] = 100
    manual_frames: Optional[List[int]] = None
    max_frames: Optional[int] = None
    start_time: float = 0
    end_time: Optional[float] = None
    existing_dataset_id: Optional[str] = None


class DuplicateDetectionRequest(BaseModel):
    method: str = "perceptual"
    threshold: int = 10
    include_near_duplicates: bool = True


class ClipRegroupRequest(BaseModel):
    threshold: int = 90


class RemoveDuplicatesRequest(BaseModel):
    groups: List[List[Dict[str, Any]]]
    keep_strategy: str = "first"


class SimplePreviewRequest(BaseModel):
    dataset_id: str
    config: Dict[str, Any]
    num_previews: int = 6


class SimpleAugmentRequest(BaseModel):
    dataset_id: str
    config: Dict[str, Any]
    augment_factor: Optional[float] = None
    target_size: Optional[int] = None
    output_name: Optional[str] = None
    class_targets: Optional[Dict[str, int]] = None


class EnhancedAugmentationRequest(BaseModel):
    output_name: str
    target_size: int
    target_multiplier: Optional[float] = None
    augmentations: Dict[str, Dict[str, Any]]
    preserve_originals: bool = True


class EnhancedSplitRequest(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    output_name: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None
    stratified: bool = False
    by_folder: bool = False


class YamlWizardConfig(BaseModel):
    class_names: List[str]
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    image_ids: List[str]


class BatchSplitRequest(BaseModel):
    image_ids: List[str]
    split: str


class SnapshotRequest(BaseModel):
    name: str
    description: str = ""


class DatasetRenameRequest(BaseModel):
    new_name: str
