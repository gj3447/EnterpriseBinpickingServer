import asyncio
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.dependencies import get_robot_service


async def main():
    service = get_robot_service()
    await service.start()
    robot_obj = service.get_robot_object()
    if not robot_obj:
        print("Robot model not loaded")
        return
    model = robot_obj["model"]
    for frame in model.frames:
        print(frame.index, frame.name)


if __name__ == "__main__":
    asyncio.run(main())













