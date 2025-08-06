import uvicorn
import os

from app.main import app

if __name__ == "__main__":
    # 포트 번호는 환경 변수 또는 기본값으로 설정
    port = int(os.environ.get("PORT", 52000))
    
    # Uvicorn 서버 실행
    # --reload 옵션은 개발 중에 코드가 변경될 때 서버를 자동으로 재시작해줍니다.
    # 프로덕션 환경에서는 이 옵션을 False로 설정하거나 uvicorn을 직접 실행하는 것이 좋습니다.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
