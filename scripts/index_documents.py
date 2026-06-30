from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.indexer import index_documents


def main() -> None:
    settings = Settings.from_env()
    count = index_documents(settings)
    print(
        f"Indexed {count} chunks into {settings.effective_collection_name} "
        f"using {settings.embedding_model}."
    )


if __name__ == "__main__":
    main()
