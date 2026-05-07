from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP Configuration
    gcp_project: str = Field(..., env="GCP_PROJECT")
    gcp_location: str = Field("us-central1", env="GCP_LOCATION")

    # Model Configuration
    gemini_model: str = Field("gemini-2.0-flash-001", env="GEMINI_MODEL")
    embedding_model: str = Field("text-embedding-004", env="EMBEDDING_MODEL")

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    indices_dir: Path = data_dir / "indices"

    # Vector Store Config
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    text_collection_name: str = Field("ifc_text", env="TEXT_COLLECTION_NAME")

    # Qdrant mode: "server" = Docker/remote server, "local" = on-disk file
    qdrant_mode: Literal["server", "local"] = Field("server", env="QDRANT_MODE")

settings = Settings()
