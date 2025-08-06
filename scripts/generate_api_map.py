"""
FastAPI 애플리케이션에 등록된 모든 API 및 WebSocket 엔드포인트 목록을 생성하여
'api_map.md' 파일로 저장하는 스크립트입니다.

이 스크립트를 실행하려면, 먼저 가상환경을 활성화해야 합니다.
`conda activate binpicking_env`

그런 다음, 프로젝트 루트 디렉토리에서 아래 명령어를 실행하세요.
`python -m scripts.generate_api_map`
"""
import os
import inspect
from pathlib import Path

# --- 스크립트 실행 경로 문제 해결 ---
# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트 디렉토리를 찾습니다.
# (scripts/generate_api_map.py -> scripts -> 프로젝트 루트)
ROOT_DIR = Path(__file__).parent.parent
# 현재 작업 디렉토리를 프로젝트 루트로 변경합니다.
# 이렇게 하면 'app/config/...' 같은 상대 경로를 항상 올바르게 찾을 수 있습니다.
os.chdir(ROOT_DIR)

# 작업 디렉토리 변경 후, FastAPI 앱을 임포트합니다.
from app.main import app
from fastapi.routing import APIRoute, APIWebSocketRoute


def generate_api_map():
    """FastAPI 앱의 모든 라우트를 분석하여 텍스트 파일로 저장합니다."""
    # ... (나머지 로직은 이전과 동일)
    output_lines = ["# Enterprise Binpicking Server API & WebSocket Map\n\n"]
    
    http_routes = []
    ws_routes = []

    for route in app.routes:
        if isinstance(route, APIRoute):
            http_routes.append(route)
        elif isinstance(route, APIWebSocketRoute):
            ws_routes.append(route)

    if http_routes:
        output_lines.append("## HTTP API Endpoints\n\n")
        grouped_routes = {}
        for route in http_routes:
            tag = (route.tags[0] if route.tags else "Default")
            if tag not in grouped_routes:
                grouped_routes[tag] = []
            grouped_routes[tag].append(route)
            
        for tag, routes_in_group in sorted(grouped_routes.items()):
            output_lines.append(f"### {tag}\n\n")
            for route in sorted(routes_in_group, key=lambda r: r.path):
                methods = ", ".join(route.methods)
                summary = inspect.getdoc(route.endpoint)
                summary_line = f"    - {summary.strip().splitlines()[0]}" if summary else ""
                output_lines.append(f"- **`{methods}`** `{route.path}`\n{summary_line}\n")
            output_lines.append("\n")

    if ws_routes:
        output_lines.append("## WebSocket Endpoints\n\n")
        for route in sorted(ws_routes, key=lambda r: r.path):
            summary = inspect.getdoc(route.endpoint)
            summary_line = f"    - {summary.strip().splitlines()[0]}" if summary else ""
            output_lines.append(f"- **`WEBSOCKET`** `{route.path}`\n{summary_line}\n")

    output_path = ROOT_DIR / "api_map.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"API map successfully generated at: {output_path.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    generate_api_map()
