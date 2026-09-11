import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# .env 를 먼저 읽어야 실제 키로 live 테스트를 돌릴 수 있다.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("DATA_GO_KR_SERVICE_KEY", "test-key")
