from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "CodefyUI"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB

    NODES_DIR: Path = Path(__file__).parent / "nodes"
    CUSTOM_NODES_DIR: Path = Path(__file__).parent / "custom_nodes"
    GRAPHS_DIR: Path = Path(__file__).parent.parent / "data" / "graphs"
    PRESETS_DIR: Path = Path(__file__).parent / "presets"
    MODELS_DIR: Path = Path(__file__).parent.parent / "data" / "models"
    EXAMPLES_DIR: Path = Path(__file__).parent.parent.parent / "examples"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path | None = None
    LOG_JSON: bool = False

    MAX_PARALLEL_NODES: int = 4

    model_config = {"env_prefix": "CODEFYUI_"}


settings = Settings()
