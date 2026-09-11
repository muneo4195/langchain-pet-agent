# 함께갈개 (PetPal)

유기동물 입양 매칭과 반려동물 동반여행 추천을 함께 처리하는 LangChain 에이전트.

```
사용자 ─▶ before_agent(가드레일·의도) ─▶ wrap_model_call(Tool 가시성·선호 주입)
        ─▶ GPT-5-mini ⇄ Tool 5종 ─▶ wrap_tool_call(지역코드·캐시·재시도·결과필터)
        ─▶ Structured Output ─▶ after_agent(PII·grounded 검증) ─▶ 응답
```

## 설치

모든 명령은 이 폴더(`petpal-agent/`) 안에서 실행한다.

```bash
cd petpal-agent
python -m venv venv && source venv/bin/activate   # 상위 폴더의 venv 를 그대로 써도 된다
pip install -r requirements.txt
cp .env.example .env      # DATA_GO_KR_SERVICE_KEY, OPENAI_API_KEY 채우기
```

상위 폴더에 이미 만들어 둔 venv 를 쓴다면 `source ../venv/bin/activate` 로 대신한다.

공공데이터포털에서 아래 두 서비스의 활용신청이 승인돼 있어야 한다. 인증키는 계정당 하나이며
인코딩 키·디코딩 키 어느 쪽을 넣어도 동작한다.

| 서비스 | Base URL |
|---|---|
| 국가동물보호정보시스템 구조동물 조회 | `http://apis.data.go.kr/1543061/abandonmentPublicService_v2` |
| 한국관광공사 반려동물 동반여행 | `https://apis.data.go.kr/B551011/KorPetTourService2` |

## 실행

```bash
python -m petpal.cli                                   # 대화형
python -m petpal.cli "강릉에 반려견 동반 숙소 알려줘"       # 한 번만 질의
python scripts/demo_pipeline.py                        # OpenAI 키 없이 Tool·미들웨어만 확인
python scripts/probe_temperature.py                    # temperature 지원 여부 1회 확인
pytest                                                 # 단위 테스트
pytest -m live                                         # 공공 API 실호출 (트래픽 소모)
```

## 폴더 구조

```
petpal-agent/
  petpal/      에이전트 패키지
  tests/       단위 테스트 58개 + 실호출 테스트 3개
  scripts/     demo_pipeline.py · probe_temperature.py
  .cache/      지역·품종 코드표 캐시 (자동 생성)
  .env         인증키 (git 추적 제외)
```

## 모듈 ↔ 설계서 대응

| 파일 | 설계서 |
|---|---|
| `petpal/schemas.py` | 2.4 Structured Output — `GuardrailClassification` / `IntentClassification` / `AnimalCard` / `PetTravelCard` / `AgentResponse` |
| `petpal/tools.py` | 2.5 Tool 5종 |
| `petpal/middleware.py` | 3.2 Middleware 7종(Custom) |
| `petpal/state.py` · `context.py` · `services.py` | 3.1 Context — State / Runtime Context / 앱 레벨 캐시 |
| `petpal/guardrails.py` | 3.3 Guardrails — 규칙 기반 1차 필터, PII 마스킹 |
| `petpal/codes.py` | 지역·품종 코드 매핑 (두 API 의 코드 체계가 다름) |
| `petpal/api.py` | 1.5 안정성 — 타임아웃·지수 백오프 3회 재시도, 응답 봉투 검증 |
| `petpal/agent.py` | 2.3 System Prompt + 내장 미들웨어(ToolRetry / ToolCallLimit / Summarization) |

## 구현하면서 확인한 API 사실

이 세 가지가 설계의 뼈대를 결정했다. 모두 실호출로 확인했다.

1. **동반 조건은 목록 API 에 없다.** `areaBasedList2` 응답에는 `acmpyTypeCd` 가 없고
   `detailPetTour2` 를 `contentId` 별로 따로 불러야 한다. 그래서 Tool 이 5개다.
   장소 수만큼 호출이 늘어나므로 상위 3~5건만 병렬 호출한다.
2. **나이·성별·체중은 검색 파라미터가 아니다.** 응답 필드로만 오고 `"0.7(Kg)"`, `"2026(년생)"`
   처럼 단위가 붙은 문자열이다. 크기·나이 조건은 `ResultFilterMiddleware` 가 후처리로 거른다.
3. **두 API 의 지역코드 체계가 다르다.**

   | | 시도 | 시군구 |
   |---|---|---|
   | 구조동물 | `upr_cd` = `6530000` (7자리) | `org_cd` = `4201000` |
   | 동반여행 | `lDongRegnCd` = `51` (2자리) | `lDongSignguCd` = `150` (3자리) |

   변환표는 `CodeTables`(앱 레벨 캐시) 하나가 갖고 있고, Tool 이 자연어 지역명을 넘겨 조회한다.
   `RegionCodeResolverMiddleware` 는 지역이 생략된 지시어성 요청("그 아이 사는 곳 근처")을
   `State.selected_animal` 의 지역으로 채우는 일을 맡는다.

## 운영상 제약

- **일일 트래픽 1,000건 / 서비스.** 개발·테스트 호출도 같은 쿼터를 쓴다. 코드표는
  `.cache/codes.json` 에, 조회 결과는 TTL 캐시(공고 5분 / 여행지 24시간)에 담아 호출을 줄인다.
- **동반여행 데이터 커버리지 편차가 크다.** 서울 3,170건인 반면 가평군 음식점은 1건이다.
  `acmpyPsblCpam`·`acmpyNeedMtr` 는 대부분 빈 값이라 `미확인` 으로 표기한다.
  0건·소량일 때 결과를 지어내지 않고 `no_result_hint` 로 대안을 제시한다.
- **temperature.** GPT-5 계열이 커스텀 값을 거부할 수 있어 기본값은 인자를 생략한다.
  `scripts/probe_temperature.py` 로 한 번 확인한 뒤 `.env` 의 `PETPAL_TEMPERATURE` 를 고정하면 된다.

## LangGraph 제약 두 가지

구현하면서 부딪힌 프레임워크 제약이라 설계서와 다르게 간 부분이다.

1. **미들웨어가 Tool 에 값을 주입할 수 없다.** `ToolNode._inject_tool_args` 는 모델의 위조를
   막으려고 `InjectedToolArg` 키를 `tool_call.args` 에서 **전부 제거**한다. `ToolRuntime` 도
   미들웨어 체인보다 **먼저** 그래프 상태로 만들어져 `request.override(state=...)` 가 닿지 않는다.
   그래서 지역코드 조회는 Tool 안으로 옮기고, 미들웨어는 지시어 해석만 맡는다.
2. **`dict` 인자는 strict 스키마를 통과하지 못한다.** `preferences: dict[str, Any]` 는
   `properties` 없는 빈 오브젝트로 직렬화돼 OpenAI 가 400 을 낸다
   (`Invalid schema for function ...`). `save_user_preference` 는 원시 타입 인자로 받는다.

## grounded 판정

모델이 `grounded=true` 라고 신고해도 믿지 않는다. `ResultFilterMiddleware` 가 Tool 응답의
`desertionNo` / `contentid` 를 `State.last_tool_results` 에 기록하고, `OutputGuardrailMiddleware`
가 최종 카드의 ID 를 그 목록과 대조해 없는 항목을 제거한 뒤 `grounded` 를 다시 계산한다.
