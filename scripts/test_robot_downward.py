"""로봇 IK `downward` 기능을 간단히 검증하는 스크립트.

두 백엔드(Pinocchio, ikpy)를 각각 테스트할 수 있으며, 기본으로 20개의
집기 후보 좌표를 생성해 내려찍는 자세로 IK를 계산합니다.

실행 예:

```bash
python scripts/test_robot_downward.py --backend both --target-frame tool
```
"""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from app.core.config import settings
from app.schemas.robot import PoseTarget, RobotIkRequest
from app.services.robot_service import RobotBackend
from app.services.robot_service_ikpy import RobotServiceIkpy
from app.services.robot_service_pinocchio import RobotService as RobotServicePinocchio
from app.stores.application_store import ApplicationStore


DEFAULT_BOUNDS = {
    "x": (-0.55, -0.30),
    "y": (0.05, 0.45),
    "z": (0.12, 0.32),
}


@dataclass
class TestResult:
    backend: RobotBackend
    total: int
    solved: int
    max_error: float | None
    median_error: float | None
    threshold_failures: list[tuple[int, float]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot downward IK smoke test")
    parser.add_argument(
        "--backend",
        default="pinocchio",
        choices=["pinocchio", "ikpy", "both"],
        help="테스트할 백엔드 선택 (both는 두 백엔드 모두 순차 실행)",
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
        default=120,
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
        help="오차 허용 임계값 (이 값을 넘는 포즈는 실패로 간주)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="각 포즈별 세부 결과 출력",
    )
    return parser.parse_args(argv)


def generate_translations(
    count: int,
    bounds: dict[str, tuple[float, float]],
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    translations: list[list[float]] = []
    for _ in range(count):
        x = rng.uniform(*bounds["x"])
        y = rng.uniform(*bounds["y"])
        z = rng.uniform(*bounds["z"])
        translations.append([round(x, 4), round(y, 4), round(z, 4)])
    return translations


def build_request(args: argparse.Namespace, translations: Sequence[Sequence[float]]) -> RobotIkRequest:
    downward_quaternion = [1.0, 0.0, 0.0, 0.0]
    pose_targets = [
        PoseTarget(translation=list(vec), rotation_quaternion=downward_quaternion)
        for vec in translations
    ]
    return RobotIkRequest(
        target_frame=args.target_frame,
        pose_targets=pose_targets,
        grip_offsets=args.grip_offsets or [0.0],
        mode=args.mode,
        coordinate_mode="base",
        max_iterations=args.max_iter,
        tolerance=args.tol,
        damping=args.damping,
    )


async def run_backend_test(
    backend: RobotBackend,
    args: argparse.Namespace,
    translations: Sequence[Sequence[float]],
) -> TestResult:
    store = ApplicationStore()
    if backend == "pinocchio":
        service = RobotServicePinocchio(
            store=store,
            fixed_urdf_path=settings.ROBOT_URDF_PATH_FIXED,
            prismatic_urdf_path=settings.ROBOT_URDF_PATH_PRISMATIC,
        )
    else:
        service = RobotServiceIkpy(
            store=store,
            fixed_urdf_path=settings.ROBOT_URDF_PATH_FIXED,
            prismatic_urdf_path=settings.ROBOT_URDF_PATH_PRISMATIC,
        )

    await service.start()
    try:
        request = build_request(args, translations)
        response = await service.solve_ik(request)
    finally:
        await service.stop()

    per_pose: dict[int, list[float]] = {}
    for candidate in response.candidates:
        per_pose.setdefault(candidate.pose_index, []).append(candidate.error)

    solved = sum(1 for idx in range(len(translations)) if per_pose.get(idx))
    best_errors: list[float] = [min(errors) for errors in per_pose.values() if errors]
    max_error = max(best_errors) if best_errors else None
    median_error = None
    if best_errors:
        sorted_errors = sorted(best_errors)
        mid = len(sorted_errors) // 2
        if len(sorted_errors) % 2 == 0:
            median_error = (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.0
        else:
            median_error = sorted_errors[mid]

    threshold_failures: list[tuple[int, float]] = []
    for idx, errors in per_pose.items():
        best = min(errors)
        if best > args.error_threshold:
            threshold_failures.append((idx, best))

    if args.verbose:
        print(f"\n[세부 결과] backend={backend}")
        for idx, vec in enumerate(translations):
            errors = per_pose.get(idx)
            if not errors:
                print(f" - pose#{idx:02d} @ {vec} -> 실패 (결과 없음)")
                continue
            best = min(errors)
            status = "FAIL" if best > args.error_threshold else "OK"
            print(f" - pose#{idx:02d} @ {vec} -> best_error={best:.6f} ({status})")

    return TestResult(
        backend=backend,
        total=len(translations),
        solved=solved,
        max_error=max_error,
        median_error=median_error,
        threshold_failures=threshold_failures,
    )


def print_summary(result: TestResult, threshold: float) -> None:
    print("\n========================================")
    print(f"Backend: {result.backend}")
    print(f"Solved poses: {result.solved}/{result.total}")
    if result.max_error is not None:
        print(f"Max best-error: {result.max_error:.6f}")
    if result.median_error is not None:
        print(f"Median best-error: {result.median_error:.6f}")
    if result.threshold_failures:
        print(f"Failed (error > {threshold}): {len(result.threshold_failures)} pose(s)")
        for idx, err in result.threshold_failures:
            print(f"  - pose#{idx:02d} error={err:.6f}")
    else:
        print(f"All poses within error threshold ({threshold}).")
    print("========================================")


async def async_main(args: argparse.Namespace) -> int:
    translations = generate_translations(args.count, DEFAULT_BOUNDS, args.seed)
    print(f"생성된 후보 좌표 {len(translations)}개 (seed={args.seed})")
    for idx, vec in enumerate(translations):
        print(f"  #{idx:02d}: {vec}")

    backends: Iterable[RobotBackend]
    if args.backend == "both":
        backends = ("pinocchio", "ikpy")
    elif args.backend == "pinocchio":
        backends = ("pinocchio",)
    else:
        backends = ("ikpy",)

    for backend in backends:
        result = await run_backend_test(backend, args, translations)
        print_summary(result, args.error_threshold)

    return 0


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

