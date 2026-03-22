"""API routes for managing model weight files (list, upload, delete)."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

ALLOWED_EXTENSIONS = {".pt", ".pth", ".safetensors", ".ckpt", ".bin"}


@router.get("")
async def list_model_files():
    """List all model files in the models directory."""
    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(models_dir.iterdir()):
        if f.is_file() and f.suffix in ALLOWED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "size": f.stat().st_size,
            })
    return files


@router.post("/upload")
async def upload_model_file(file: UploadFile):
    """Upload a model weight file (.pt, .pth, .safetensors, .ckpt, .bin)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    logger.info("Uploaded model file: %s (%d bytes)", file.filename, len(content))
    return {"filename": file.filename, "size": len(content)}


@router.delete("/{filename}")
async def delete_model_file(filename: str):
    """Delete a model weight file."""
    models_dir = settings.MODELS_DIR
    filepath = models_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    if not filepath.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    if filepath.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a model file")

    filepath.unlink()
    logger.info("Deleted model file: %s", filename)
    return {"message": f"Deleted {filename}"}
