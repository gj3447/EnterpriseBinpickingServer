# ArUco 보드 포즈 추정 정확도 개선 방안

## 문제점 분석
현재 구현에서 보드 포즈가 불안정한 주요 원인:
1. ArUco 코너가 객체 가장자리에 위치해 깊이 불연속 지점에서 노이즈 발생
2. SVD 기반 3D-3D 정합이 outlier에 민감
3. 깊이 기반 ↔ PnP 폴백 시 포즈 점프
4. 최소 3개 포인트만으로 6DOF 추정 시도
5. 깊이 품질 검증 없이 raw 값 사용

## 해결 방안

### 1. 깊이 샘플링 개선
```python
def _sample_depth_robust(self, depth_m, u, v, marker_size_px):
    """마커 내부 영역에서 안정적인 깊이 샘플링"""
    # 코너에서 내부로 10% 오프셋
    offset = marker_size_px * 0.1
    
    # 마커 중심 방향으로 이동한 점에서 샘플링
    # 또는 마커 내부 전체 영역의 평면 피팅
    
    # 깊이 유효성 검사: 표준편차가 임계값 이하인지 확인
    if depth_std > threshold:
        return None
```

### 2. RANSAC 기반 Robust 3D-3D 정합
```python
def _rigid_transform_3d_ransac(self, A, B, inlier_threshold=0.01):
    """RANSAC으로 outlier 제거 후 정합"""
    best_inliers = []
    best_R, best_t = None, None
    
    for _ in range(iterations):
        # 랜덤 3점 선택
        sample_idx = np.random.choice(len(A), 3, replace=False)
        
        # 3점으로 변환 계산
        R, t = self._rigid_transform_3d(A[sample_idx], B[sample_idx])
        
        # 모든 점에 대해 오차 계산
        transformed = (R @ A.T).T + t
        errors = np.linalg.norm(transformed - B, axis=1)
        
        # Inlier 카운트
        inliers = errors < inlier_threshold
        
        if np.sum(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R, t
    
    # 최종 정합: 모든 inlier 사용
    if len(best_inliers) >= min_inliers:
        return self._rigid_transform_3d(A[best_inliers], B[best_inliers])
```

### 3. 하이브리드 접근: PnP + 깊이 검증
```python
def _estimate_board_pose_hybrid(self, ...):
    """PnP 우선, 깊이로 검증/정제"""
    # 1단계: 안정적인 PnP(IPPE)로 초기 포즈
    rvec_pnp, tvec_pnp = self._estimate_pose_board_ippe(...)
    
    # 2단계: 깊이 정보로 스케일/Z축 정제
    if depth_available:
        # PnP 결과를 초기값으로 깊이 일관성 체크
        predicted_depths = self._project_to_depth(rvec_pnp, tvec_pnp, ...)
        observed_depths = self._sample_marker_depths(...)
        
        # 깊이 일관성이 높으면 스케일 보정
        if depth_consistency > threshold:
            scale_factor = np.median(observed_depths / predicted_depths)
            tvec_refined = tvec_pnp * scale_factor
            
    # 3단계: 시간적 필터링 (칼만 필터)
```

### 4. 마커 내부 평면 피팅
```python
def _fit_marker_plane(self, depth_m, marker_corners_px, intrinsics):
    """마커 내부 영역의 깊이로 평면 피팅"""
    # 마커 내부를 그리드로 샘플링
    mask = self._create_marker_mask(marker_corners_px)
    
    # 유효한 깊이 포인트 추출
    points_3d = []
    for y, x in mask_points:
        z = depth_m[y, x]
        if z > 0 and z < max_depth:
            point_3d = self._backproject(x, y, z, ...)
            points_3d.append(point_3d)
    
    # RANSAC 평면 피팅
    plane_model = self._fit_plane_ransac(points_3d)
    
    # 평면 위에 마커 코너 투영
    refined_corners_3d = self._project_corners_to_plane(...)
```

### 5. 시간적 일관성 (Temporal Filtering)
```python
class PoseKalmanFilter:
    """보드 포즈의 시간적 안정화"""
    def __init__(self):
        # 상태: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.kf = cv2.KalmanFilter(13, 7)
        # 프로세스/측정 노이즈 설정
        
    def predict(self, dt):
        # 모션 모델로 예측
        
    def update(self, pose, confidence):
        # 신뢰도에 따라 측정 노이즈 조정
        if confidence < threshold:
            # 높은 측정 노이즈 = 기존 추정치 더 신뢰
```

### 6. 멀티 프레임 번들 조정
```python
def _bundle_adjustment_multiframe(self, frames_buffer):
    """최근 N 프레임의 관측을 동시 최적화"""
    # 모든 프레임의 reprojection error + depth consistency 최소화
    # Levenberg-Marquardt 또는 g2o 사용
```

### 7. 적응적 파라미터 조정
```python
def _adaptive_parameters(self, detection_history):
    """검출 이력에 따라 파라미터 동적 조정"""
    if recent_detections_stable:
        # 안정적일 때: 작은 window, 엄격한 threshold
        self.depth_window = 3
        self.ransac_threshold = 0.005
    else:
        # 불안정할 때: 큰 window, 느슨한 threshold
        self.depth_window = 7
        self.ransac_threshold = 0.015
```

## 구현 우선순위
1. **즉시 적용 가능**: 
   - RANSAC 3D-3D 정합
   - 깊이 품질 검증 (표준편차 체크)
   - 최소 포인트 수 증가 (3→6)

2. **중기 개선**:
   - 마커 내부 평면 피팅
   - 칼만 필터 시간적 안정화
   - 하이브리드 PnP+깊이 접근

3. **장기 최적화**:
   - 멀티 프레임 번들 조정
   - 적응적 파라미터 조정
   - GPU 가속 (대량 포인트 처리)

## 5cm 마커 특화 팁
- 작은 마커일수록 깊이 노이즈 영향이 크므로:
  - 마커 내부 전체 영역 활용 (코너만 X)
  - 여러 마커의 공동 평면 가정 활용
  - 더 엄격한 outlier rejection
  - 시간적 일관성 가중치 증가
