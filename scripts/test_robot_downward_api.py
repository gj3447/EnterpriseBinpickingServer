"""REST API를 통해 Downward IK 기능을 점검하는 스크립트.

두 백엔드(Pinocchio, ikpy)에 대해 `/api/robot/ik/{backend}/downward` 엔드포인트에
일괄 요청을 보내고, 20개(기본값)의 후보 좌표가 허용 오차 내에서 풀리는지 검증합니다.

실행 예:

```bash
python scripts/test_robot_downward_api.py --backend both --target-frame tool
```
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Dict, Iterable, List, Sequence

import requests


DEFAULT_BOUNDS = {
    "x": (-0.55, -0.30),
    "y": (0.05, 0.45),
    "z": (0.12, 0.32),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot IK downward API smoke test")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI 서버 기본 URL (프로토콜 포함)",
    )
    parser.add_argument(
        "--backend",
        default="pinocchio",
        choices=["pinocchio", "ikpy", "both"],
        help="테스트할 백엔드 선택",
    )
    parser.add_argument(
        "--target-frame",
        default="tool",
        help="IK를 계산할 타깃 프레임 이름",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="생성할 후보 좌표 개수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="좌표 생성 랜덤 시드",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "fixed", "prismatic"],
        help="IK 계산 모드",
    )
    parser.add_argument(
        "--grip-offset",
        type=float,
        action="append",
        dest="grip_offsets",
        help="추가 그리퍼 오프셋 (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="IK 반복 최대 횟수",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="수렴 허용 오차",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.6,
        help="댐핑 파라미터",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=5e-3,
        help="오차 허용 임계값 (이 값을 넘으면 실패로 간주)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="HTTP 요청 타임아웃(초)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="포즈별 상세 결과 출력",
    )
    return parser.parse_args(argv)


def generate_translations(
    count: int,
    bounds: Dict[str, tuple[float, float]],
    seed: int,
) -> List[List[float]]:
    rng = random.Random(seed)
    translations: List[List[float]] = []
    for _ in range(count):
        x = rng.uniform(*bounds["x"])
        y = rng.uniform(*bounds["y"])
        z = rng.uniform(*bounds["z"])
        translations.append([round(x, 4), round(y, 4), round(z, 4)])
    return translations


def compose_payload(args: argparse.Namespace, translations: Sequence[Sequence[float]]) -> dict:
    return {
        "target_frame": args.target_frame,
        "translations": translations,
        "mode": args.mode,
        "coordinate_mode": "base",
        "grip_offsets": args.grip_offsets or [0.0],
        "max_iterations": args.max_iter,
        "tolerance": args.tol,
        "damping": args.damping,
    }


def summarize_response(
    response_json: dict,
    translations: Sequence[Sequence[float]],
    error_threshold: float,
) -> dict:
    per_pose: dict[int, dict] = {}
    for candidate in response_json.get("candidates", []):
        pose_index = candidate["pose_index"]
        current_error = candidate["error"]
        entry = per_pose.get(pose_index)
        if entry is None or current_error < entry["error"]:
            per_pose[pose_index] = {
                "error": current_error,
                "iterations": candidate["iterations"],
                "grip_offset": candidate["grip_offset"],
                "mode_used": candidate["mode_used"],
            }

    solved = sum(1 for idx in range(len(translations)) if idx in per_pose)
    errors = [info["error"] for info in per_pose.values() if info]
    max_error = max(errors) if errors else None
    median_error = statistics.median(errors) if errors else None
    threshold_failures = [
        (idx, info["error"])
        for idx, info in per_pose.items()
        if info["error"] > error_threshold
    ]

    return {
        "solved": solved,
        "max_error": max_error,
        "median_error": median_error,
        "threshold_failures": threshold_failures,
        "per_pose": per_pose,
    }


def print_summary(
    backend: str,
    summary: dict,
    total: int,
    error_threshold: float,
    translations: Sequence[Sequence[float]],
    verbose: bool,
) -> None:
    print("\n========================================")
    print(f"Backend: {backend}")
    print(f"Solved poses: {summary['solved']}/{total}")
    if summary["max_error"] is not None:
        print(f"Max best-error: {summary['max_error']:.6f}")
    if summary["median_error"] is not None:
        print(f"Median best-error: {summary['median_error']:.6f}")
    fails = summary["threshold_failures"]
    if fails:
        print(f"Failed (error > {error_threshold}): {len(fails)} pose(s)")
        for idx, err in fails:
            print(f"  - pose#{idx:02d} error={err:.6f}")
    else:
        print(f"All poses within error threshold ({error_threshold}).")

    if verbose:
        print("\n[세부 결과]")
        per_pose = summary["per_pose"]
        for idx, vec in enumerate(translations):
            info = per_pose.get(idx)
            if not info:
                print(f" - pose#{idx:02d} @ {vec} -> 실패 (결과 없음)")
                continue
            status = "FAIL" if info["error"] > error_threshold else "OK"
            print(
                " - pose#{:02d} @ {} -> error={:.6f} ({}) offset={:.3f} iterations={}".format(
                    idx,
                    vec,
                    info["error"],
                    status,
                    info["grip_offset"],
                    info["iterations"],
                )
            )
    print("========================================")


def call_downward_endpoint(
    base_url: str,
    backend: str,
    payload: dict,
    timeout: float,
) -> dict:
    endpoint = f"/api/robot/ik/{backend}/downward"
    url = base_url.rstrip("/") + endpoint
    print(f"\n요청: {url}")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"[{backend}] 요청 실패: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"[{backend}] HTTP {response.status_code} -> {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"[{backend}] JSON 파싱 실패: {response.text}") from exc


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    translations = generate_translations(args.count, DEFAULT_BOUNDS, args.seed)
    print(f"생성된 후보 좌표 {len(translations)}개 (seed={args.seed})")
    for idx, vec in enumerate(translations):
        print(f"  #{idx:02d}: {vec}")

    payload = compose_payload(args, translations)

    if args.backend == "both":
        backends: Iterable[str] = ("pinocchio", "ikpy")
    else:
        backends = (args.backend,)

    for backend in backends:
        try:
            response_json = call_downward_endpoint(args.base_url, backend, payload, args.timeout)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            continue

        summary = summarize_response(response_json, translations, args.error_threshold)
        print_summary(backend, summary, len(translations), args.error_threshold, translations, args.verbose)

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

