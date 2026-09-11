"""설계서 3.1 Context — Runtime Context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PetPalContext:
    """앱이 호출 시 주입한다. 모델과 Tool 이 변경할 수 없다."""

    user_id: str
    app_name: str = "petpal"
