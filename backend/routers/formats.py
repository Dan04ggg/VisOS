import shutil
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from core.workspace import _make_dataset_folder, _save_dataset_metadata
from schemas.common import ConversionRequest

router = APIRouter()


@router.get("/api/formats")
async def list_formats():
    return {
        "formats": [
            {"id": "yolo",          "name": "YOLO",               "description": "YOLOv5/v8/v9/v10/v11 txt annotations",    "task": ["detection", "segmentation"]},
            {"id": "coco",          "name": "COCO JSON",          "description": "COCO JSON format",                        "task": ["detection", "segmentation", "keypoints"]},
            {"id": "pascal-voc",    "name": "Pascal VOC",         "description": "XML annotations per image",               "task": ["detection"]},
            {"id": "labelme",       "name": "LabelMe",            "description": "LabelMe JSON format",                     "task": ["detection", "segmentation"]},
            {"id": "createml",      "name": "CreateML",           "description": "Apple CreateML format",                   "task": ["detection"]},
            {"id": "tfrecord",      "name": "TFRecord",           "description": "TensorFlow Record format",                "task": ["detection", "classification"]},
            {"id": "csv",           "name": "CSV",                "description": "Simple CSV format",                       "task": ["detection"]},
            {"id": "yolo_seg",      "name": "YOLO Segmentation",  "description": "YOLO polygon segmentation",               "task": ["segmentation"]},
            {"id": "coco_panoptic", "name": "COCO Panoptic",      "description": "COCO panoptic segmentation",              "task": ["panoptic-segmentation"]},
            {"id": "cityscapes",    "name": "Cityscapes",         "description": "Cityscapes polygon + mask format",        "task": ["segmentation"]},
            {"id": "ade20k",        "name": "ADE20K",             "description": "ADE20K PNG segmentation masks",           "task": ["segmentation"]},
            {"id": "classification","name": "Classification Folder","description": "Folder-per-class structure",            "task": ["classification"]},
            {"id": "yolo_obb",      "name": "YOLO OBB",           "description": "YOLO oriented bounding boxes",            "task": ["obb-detection"]},
            {"id": "dota",          "name": "DOTA",               "description": "DOTA quad-polygon format",                "task": ["obb-detection"]},
        ]
    }


@router.post("/api/convert")
async def convert_dataset(request: ConversionRequest):
    if request.dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[request.dataset_id]
    source_path = settings.DATASETS_DIR / request.dataset_id

    format_map = {
        "csv": "tensorflow-csv",
        "tensorflow-csv": "tensorflow-csv",
        "voc": "pascal-voc",
        "pascal-voc": "pascal-voc",
        "yolo_seg": "yolo",
        "yolo-seg": "yolo",
        "yolo_obb": "yolo-obb",
        "yolov8-obb": "yolo-obb",
        "coco_panoptic": "coco-panoptic",
        "classification": "classification-folder",
        "cityscapes": "cityscapes",
        "ade20k": "ade20k",
        "dota": "dota",
        "tfrecord": "tfrecord",
    }
    target_format = format_map.get(request.target_format, request.target_format)

    output_name = request.output_name or f"{dataset['name']}_{request.target_format}"
    new_dataset_id, output_path = _make_dataset_folder(output_name)

    try:
        settings.format_converter.convert(source_path, output_path, dataset["format"], target_format)
    except Exception as e:
        shutil.rmtree(output_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Conversion failed: {str(e)}")

    new_info = settings.dataset_parser.parse_dataset(output_path, target_format, new_dataset_id)
    new_info["id"] = new_dataset_id
    new_info["format"] = request.target_format
    settings.active_datasets[new_dataset_id] = new_info
    _save_dataset_metadata(new_dataset_id, new_info)

    return {"success": True, "new_dataset": new_info}


@router.get("/api/export/{dataset_id}")
async def export_dataset(dataset_id: str, target_format: str = None):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    if target_format and target_format != dataset["format"]:
        export_path = settings.TEMP_DIR / str(uuid.uuid4())
        settings.format_converter.convert(dataset_path, export_path, dataset["format"], target_format)
    else:
        export_path = dataset_path

    zip_path = settings.EXPORTS_DIR / f"{dataset['name']}_{target_format or dataset['format']}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", export_path)

    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")
