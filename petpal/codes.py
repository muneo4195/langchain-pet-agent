"""지역·품종 코드 매핑 — 설계서 3.1의 '앱 레벨 캐시'.

유저와 무관한 정적 참조표라 Store(유저별 장기기억)가 아니라 프로세스 캐시에 둔다.
두 API 의 코드 체계가 완전히 다르므로 이 계층이 반드시 필요하다.

    구조동물   upr_cd 7자리(6110000) / org_cd 7자리
    동반여행   lDongRegnCd 2자리(11) / lDongSignguCd 3자리(150)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .api import PublicDataClient
from .config import UPKIND

log = logging.getLogger("petpal.codes")

CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "codes.json"
CACHE_TTL = 60 * 60 * 24 * 30  # 코드 체계는 거의 바뀌지 않는다

_SIDO_SUFFIX = ("특별자치도", "특별자치시", "특별시", "광역시", "도")
# 구어 축약형 — API 표기는 '충청남도' 인데 사용자는 '충남' 이라고 쓴다.
_SIDO_ALIAS = {
    "충남": "충청남", "충북": "충청북", "전남": "전라남", "전북": "전라북",
    "경남": "경상남", "경북": "경상북", "서울시": "서울", "제주도": "제주",
}
_SGG_SUFFIX = ("시", "군", "구")


def norm_sido(name: str) -> str:
    """'서울특별시' · '서울시' · '서울' 을 모두 '서울' 로 맞춘다."""
    n = (name or "").strip()
    if n in _SIDO_ALIAS:
        return _SIDO_ALIAS[n]
    for suf in _SIDO_SUFFIX:
        if n.endswith(suf) and len(n) > len(suf):
            return _SIDO_ALIAS.get(n[: -len(suf)], n[: -len(suf)])
    return _SIDO_ALIAS.get(n, n)


def norm_sgg(name: str) -> str:
    """'강릉시' → '강릉'. 단, '중구' 처럼 짧은 이름은 그대로 둔다."""
    n = (name or "").strip()
    for suf in _SGG_SUFFIX:
        if n.endswith(suf) and len(n) > 2:
            return n[: -len(suf)]
    return n


@dataclass
class RegionMatch:
    """지역명 한 건에 대한 두 API 의 코드 묶음."""

    sido: str = ""
    sigungu: str = ""
    upr_cd: str | None = None            # 구조동물 시도
    org_cd: str | None = None            # 구조동물 시군구
    l_dong_regn_cd: str | None = None    # 동반여행 시도
    l_dong_signgu_cd: str | None = None  # 동반여행 시군구

    @property
    def label(self) -> str:
        return " ".join(x for x in (self.sido, self.sigungu) if x)

    def is_empty(self) -> bool:
        return not (self.upr_cd or self.l_dong_regn_cd)


@dataclass
class CodeTables:
    animal_sido: dict[str, str] = field(default_factory=dict)          # 정규화명 → upr_cd
    animal_sigungu: dict[str, dict[str, str]] = field(default_factory=dict)  # upr_cd → {정규화명: org_cd}
    travel_regn: dict[str, str] = field(default_factory=dict)          # 정규화명 → lDongRegnCd
    travel_signgu: dict[str, dict[str, str]] = field(default_factory=dict)   # regn → {정규화명: signgu}
    kinds: dict[str, dict[str, str]] = field(default_factory=dict)     # upkind → {품종명: kindCd}
    fetched_at: float = 0.0

    # ------------------------------------------------------------- 적재/캐시
    @classmethod
    def build(cls, client: PublicDataClient) -> "CodeTables":
        t = cls(fetched_at=time.time())

        def named(rows, name_key, code_key):
            """일부 지자체 행은 이름 필드가 통째로 빠져 온다(1.5 데이터 품질 편차)."""
            for r in rows:
                name, code = r.get(name_key), r.get(code_key)
                if name and code:
                    yield str(name), str(code)

        for name, code in named(client.animal_sido(), "orgdownNm", "orgCd"):
            t.animal_sido[norm_sido(name)] = code
        for upr in set(t.animal_sido.values()):
            t.animal_sigungu[upr] = {
                norm_sgg(n): c for n, c in named(client.animal_sigungu(upr), "orgdownNm", "orgCd")
            }
        for name, code in named(client.travel_ldong_code(), "name", "code"):
            t.travel_regn[norm_sido(name)] = code
        for regn in set(t.travel_regn.values()):
            t.travel_signgu[regn] = {
                norm_sgg(n): c for n, c in named(client.travel_ldong_code(regn), "name", "code")
            }
        for code in UPKIND.values():
            t.kinds[code] = {n: c for n, c in named(client.animal_kinds(code), "kindNm", "kindCd")}
        return t

    @classmethod
    def load(cls, client: PublicDataClient, *, refresh: bool = False) -> "CodeTables":
        """기동 시 1회 로드. 일일 트래픽 1,000건 제한이 있어 디스크에도 캐싱한다."""
        if not refresh and CACHE_PATH.exists():
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                if time.time() - data.get("fetched_at", 0) < CACHE_TTL:
                    return cls(**data)
            except (OSError, ValueError, TypeError) as exc:
                log.warning("코드 캐시를 읽지 못해 새로 받습니다: %s", exc)
        tables = cls.build(client)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(asdict(tables), ensure_ascii=False), encoding="utf-8")
        return tables

    # ---------------------------------------------------------------- 조회
    @staticmethod
    def _lookup(table: dict[str, str], token: str) -> tuple[str, str] | None:
        """정확 일치 → 접두 일치 순으로 찾는다. (원래 키, 코드) 반환."""
        if token in table:
            return token, table[token]
        for key, code in table.items():
            if key and (token.startswith(key) or key.startswith(token)):
                return key, code
        return None

    def resolve(self, text: str) -> RegionMatch:
        """'경기도 가평군', '강릉', '제주특별자치도' 같은 자유 텍스트를 코드로 바꾼다."""
        m = RegionMatch()
        if not text:
            return m
        tokens = [norm_sido(tok) for tok in str(text).replace(",", " ").split() if tok]
        tokens = [t for t in tokens if t]

        for token in tokens:  # 시도 먼저
            hit = self._lookup(self.animal_sido, token) or None
            travel = self._lookup(self.travel_regn, token)
            if hit or travel:
                m.sido = (hit or travel)[0]
                m.upr_cd = hit[1] if hit else None
                m.l_dong_regn_cd = travel[1] if travel else None
                break

        for token in tokens:  # 시군구
            norm = norm_sgg(token)
            if norm == m.sido:
                continue
            a = self._lookup(self.animal_sigungu.get(m.upr_cd or "", {}), norm)
            t = self._lookup(self.travel_signgu.get(m.l_dong_regn_cd or "", {}), norm)
            if a or t:
                m.sigungu = (a or t)[0]
                m.org_cd = a[1] if a else None
                m.l_dong_signgu_cd = t[1] if t else None
                return m

        if m.is_empty():  # '강릉' 처럼 시군구만 준 경우 전국에서 역으로 찾는다
            for token in tokens:
                found = self._find_sigungu_anywhere(norm_sgg(token))
                if found:
                    return found
        return m

    def _find_sigungu_anywhere(self, token: str) -> RegionMatch | None:
        """시도를 생략한 시군구명으로 소속 시도까지 되짚는다."""
        upr_to_sido = {code: name for name, code in self.animal_sido.items()}
        for upr, table in self.animal_sigungu.items():
            hit = self._lookup(table, token)
            if not hit:
                continue
            sido = upr_to_sido.get(upr, "")
            regn = self.travel_regn.get(sido)
            t = self._lookup(self.travel_signgu.get(regn or "", {}), token)
            return RegionMatch(
                sido=sido, sigungu=hit[0], upr_cd=upr, org_cd=hit[1],
                l_dong_regn_cd=regn, l_dong_signgu_cd=t[1] if t else None,
            )
        return None

    def kind_code(self, name: str, upkind: str | None = None) -> str | None:
        """'푸들' → kindCd. 축종을 알면 그 안에서만 찾는다."""
        pools = [self.kinds[upkind]] if upkind and upkind in self.kinds else list(self.kinds.values())
        for pool in pools:
            for kind_name, code in pool.items():
                if name and (name in kind_name or kind_name in name):
                    return code
        return None
