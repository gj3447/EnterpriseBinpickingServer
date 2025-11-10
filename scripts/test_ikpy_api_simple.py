"""ikpy 백엔드 downward IK 엔드포인트를 간단히 호출해 보는 스크립트.

사용 예:

    python scripts/test_ikpy_api_simple.py --base-url http://192.168.0.196:53000
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import requests


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple IKPy downward API probe")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:53000",
        help="FastAPI 서버 기본 URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="HTTP 요청 타임아웃(초)",
    )
    return parser.parse_args(argv)


def build_payload() -> dict:
    translations = [
        [-0.4106, 0.3083, 0.2155],
        [-0.5067, 0.0502, 0.3147],
        [-0.4076, 0.0763, 0.2002],
    ]

    return {
        "target_frame": "tool",
        "translations": translations,
        "mode": "auto",
        "coordinate_mode": "base",
        "grip_offsets": [0.0],
        "max_iterations": 200,
        "tolerance": 1e-4,
        "damping": 0.6,
    }


def call_endpoint(base_url: str, timeout: float) -> requests.Response:
    url = base_url.rstrip("/") + "/api/robot/ik/ikpy/downward"
    payload = build_payload()
    print("요청 URL:", url)
    print("요청 페이로드:")
    print(json.dumps(payload, indent=2))
    response = requests.post(url, json=payload, timeout=timeout)
    return response


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    try:
        response = call_endpoint(args.base_url, args.timeout)
    except requests.RequestException as exc:
        print("[요청 실패]", exc)
        return 1

    print("응답 코드:", response.status_code)
    try:
        print("응답 본문:")
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(response.text)

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))


