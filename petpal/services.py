"""프로세스 수명 동안 유지되는 앱 레벨 자원(설계서 3.1 '앱 레벨 캐시')."""

from __future__ import annotations

import functools

from .api import PublicDataClient
from .codes import CodeTables
from .config import Settings


class Services:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.client = PublicDataClient(self.settings)
        self.codes = CodeTables.load(self.client)


@functools.lru_cache(maxsize=1)
def get_services() -> Services:
    """기동 시 1회만 코드표를 적재한다(일일 트래픽 1,000건 제한 대응)."""
    return Services()
