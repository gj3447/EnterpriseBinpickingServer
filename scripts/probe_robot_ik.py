"""로컬 IK 서버에 다수의 후보 위치를 전송하고 응답 품질을 점검하는 유틸리티.

기본 구성은 `/api/robot/ik/downward` 엔드포인트로 50개의 후보 위치를 전송합니다.
랜덤 시드를 고정해 재현 가능한 샘플을 생성하며, 응답에서 포즈별 최적 후보와
전체 최적 후보를 요약해 출력합니다.

사용 예시는 다음과 같습니다.

```
python scripts/probe_robot_ik.py --base-url http://127.0.0.1:8000 --target-frame link_6
```
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import requests


DEFAULT_TRANSLATION_BOUNDS = {
    "x": (-0.55, -0.25),
    "y": (0.10, 0.45),
    "z": (0.12, 0.32),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IK 서버 품질 점검용 배치 요청 도구")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI 서버 베이스 URL (프로토콜 포함)",
    )
    parser.add_argument(
        "--endpoint",
        default="/api/robot/ik/downward",
        help="호출할 엔드포인트 경로",
    )
    parser.add_argument(
        "--target-frame",
        default="tool",
        help="IK를 계산할 말단 프레임 이름",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="생성할 후보 위치 개수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="랜덤 샘플 reproducibility를 위한 시드값",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "fixed", "prismatic"],
        help="IK 모드",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["fixed", "prismatic", None],
        help="사용할 URDF variant (미지정 시 기본값)",
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
        default=1e-3,
        help="DLS 감쇠 계수 초기값",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        type=float,
        help="후보 위치를 생성할 박스 범위 (미터 단위)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="HTTP 요청 타임아웃 (초)",
    )
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="모든 후보 결과를 상세 출력",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="요약 결과를 CSV 파일로 저장할 경로",
    )
    return parser.parse_args(argv)


def build_translation_box(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    if args.bounds:
        x_min, x_max, y_min, y_max, z_min, z_max = args.bounds
        return {
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "z": (z_min, z_max),
        }
    return DEFAULT_TRANSLATION_BOUNDS


def generate_translations(
    count: int,
    bounds: dict[str, tuple[float, float]],
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


def compose_payload(
    args: argparse.Namespace,
    translations: Sequence[Sequence[float]],
) -> dict:
    payload = {
        "target_frame": args.target_frame,
        "translations": translations,
        "mode": args.mode,
        "coordinate_mode": "base",
        "max_iterations": args.max_iter,
        "tolerance": args.tol,
        "damping": args.damping,
    }
    if args.variant:
        payload["urdf_variant"] = args.variant
    grip_offsets = args.grip_offsets or [0.0]
    payload["grip_offsets"] = grip_offsets
    return payload


def summarize_candidates(
    translations: Sequence[Sequence[float]],
    response_json: dict,
) -> tuple[list[dict], dict | None]:
    summary_by_pose: list[dict] = [{} for _ in translations]
    candidates = response_json.get("candidates", [])
    for candidate in candidates:
        pose_index = candidate["pose_index"]
        store = summary_by_pose[pose_index]
        current_error = candidate["error"]
        if not store or current_error < store["error"]:
            summary_by_pose[pose_index] = {
                "error": current_error,
                "iterations": candidate["iterations"],
                "grip_offset": candidate["grip_offset"],
                "joint_positions": candidate["joint_positions"],
                "mode_used": candidate["mode_used"],
            }

    return summary_by_pose, response_json.get("best")


def pretty_print_summary(
    translations: Sequence[Sequence[float]],
    response_json: dict,
    summary_by_pose: Sequence[dict],
    best_candidate: dict | None,
    print_candidates: bool,
) -> None:
    valid_errors = [item["error"] for item in summary_by_pose if item]

    print("\n=== 전체 요약 ===")
    print(f"응답 모드: {response_json.get('mode')} | 사용 variant: {response_json.get('urdf_variant_used')}")
    print(f"그리퍼 관절 존재: {response_json.get('has_gripper_joint')} ({response_json.get('gripper_joint_name')})")
    if best_candidate:
        idx = best_candidate["pose_index"]
        print(
            f"전체 최적: pose#{idx} @ translation={translations[idx]} | "
            f"offset={best_candidate['grip_offset']:.3f} | error={best_candidate['error']:.6f} | iterations={best_candidate['iterations']}"
        )
    if valid_errors:
        print(
            "오차 통계: min={:.6f} | median={:.6f} | max={:.6f}".format(
                min(valid_errors), statistics.median(valid_errors), max(valid_errors)
            )
        )

    print("\n=== 포즈별 최적 후보 ===")
    for idx, info in enumerate(summary_by_pose):
        translation = translations[idx]
        if not info:
            print(f"pose#{idx:02d} @ {translation} -> 결과 없음")
            continue
        print(
            f"pose#{idx:02d} @ {translation} | error={info['error']:.6f} | iterations={info['iterations']} | "
            f"offset={info['grip_offset']:.3f} | mode={info['mode_used']}"
        )

    if print_candidates:
        print("\n=== 전체 후보 상세 ===")
        print(json.dumps(response_json.get("candidates", []), indent=2, ensure_ascii=False))


def write_summary_csv(
    output_path: Path,
    translations: Sequence[Sequence[float]],
    summary_by_pose: Sequence[dict],
    best_candidate: dict | None,
    response_json: dict,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "pose_index",
                "x",
                "y",
                "z",
                "best_error",
                "iterations",
                "mode_used",
                "grip_offset",
                "joint_positions",
                "is_global_best",
                "solver_mode",
                "urdf_variant",
            ]
        )

        best_index = best_candidate["pose_index"] if best_candidate else None

        for idx, translation in enumerate(translations):
            info = summary_by_pose[idx]
            if not info:
                writer.writerow([
                    idx,
                    translation[0],
                    translation[1],
                    translation[2],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    response_json.get("mode"),
                    response_json.get("urdf_variant_used"),
                ])
                continue

            writer.writerow([
                idx,
                translation[0],
                translation[1],
                translation[2],
                f"{info['error']:.9f}",
                info["iterations"],
                info["mode_used"],
                f"{info['grip_offset']:.6f}",
                json.dumps(info["joint_positions"]),
                "Y" if best_index == idx else "",
                response_json.get("mode"),
                response_json.get("urdf_variant_used"),
            ])

    return output_path


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    bounds = build_translation_box(args)
    translations = generate_translations(args.count, bounds, args.seed)
    payload = compose_payload(args, translations)

    url = args.base_url.rstrip("/") + args.endpoint

    print("요청 URL:", url)
    print("후보 개수:", len(translations))
    print("번들 범위:", bounds)

    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        print("[ERROR] 요청 실패:", exc)
        return 1

    if not response.ok:
        print(f"[ERROR] HTTP {response.status_code} -> {response.text}")
        return 1

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        print("[ERROR] JSON 파싱 실패:", response.text)
        return 1

    summary_by_pose, best_candidate = summarize_candidates(translations, response_json)
    pretty_print_summary(translations, response_json, summary_by_pose, best_candidate, args.print_candidates)

    if args.output:
        csv_path = write_summary_csv(args.output, translations, summary_by_pose, best_candidate, response_json)
        print(f"\nCSV 저장 완료: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


