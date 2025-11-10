"""ikpy URDF 체인 정보를 점검하는 디버그 스크립트."""

from __future__ import annotations

import argparse
from typing import Sequence

from ikpy.chain import Chain


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print ikpy chain details")
    parser.add_argument(
        "urdf_path",
        nargs="?",
        default="app/static/urdf/dsr_description2/urdf/a0509_prismatic.urdf",
        help="체크할 URDF 파일 경로",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    chain = Chain.from_urdf_file(args.urdf_path)

    print(f"URDF: {args.urdf_path}")
    print(f"Total links: {len(chain.links)}")
    print(f"Active mask: {chain.active_links_mask}")

    for idx, link in enumerate(chain.links):
        joint_type = getattr(link, "joint_type", None)
        bounds = getattr(link, "bounds", None)
        print(
            f"#{idx:02d} name={link.name!r} active={chain.active_links_mask[idx]} "
            f"joint_type={joint_type} bounds={bounds}"
        )

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))


