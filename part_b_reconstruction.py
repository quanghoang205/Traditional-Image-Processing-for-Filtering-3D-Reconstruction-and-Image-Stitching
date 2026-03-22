import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def process_stereo_uncalibrated(imgL_path, imgR_path):
    # 1. Đọc ảnh và giảm kích thước để chạy nhanh hơn trên Colab
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    if imgL is None or imgR is None:
        print("Lỗi: Không đọc được ảnh!")
        return

    imgL = cv2.resize(imgL, (600, 450))
    imgR = cv2.resize(imgR, (600, 450))

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 2. Tìm điểm đặc trưng và Matching bằng SIFT
    print("1. Đang trích xuất đặc trưng và nối điểm...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayL, None)
    kp2, des2 = sift.detectAndCompute(grayR, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # Lowe's ratio
            good_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # 3. Tính Fundamental Matrix (F)
    print("2. Đang tính Fundamental Matrix...")
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # 4. Stereo Rectification (Nắn chỉnh ảnh)
    print("3. Đang nắn chỉnh ảnh (Rectification) để căn bằng các đường cực...")
    h, w = grayL.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))

    # Warp (bóp méo) ảnh theo ma trận H1, H2 để chúng thẳng hàng ngang
    rectifiedL = cv2.warpPerspective(imgL, H1, (w, h))
    rectifiedR = cv2.warpPerspective(imgR, H2, (w, h))
    rectifiedL_gray = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    rectifiedR_gray = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

    # 5. Tính Disparity Map trên ảnh đã nắn chỉnh
    print("4. Đang tính Disparity Map bằng SGBM...")
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64, # Có thể tăng lên 80, 96 nếu vật thể gần
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(rectifiedL_gray, rectifiedR_gray).astype(np.float32) / 16.0

    # Hiển thị Disparity
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map sau khi Rectify')
    plt.axis('off')
    plt.show()

    # 6. Tái tạo 3D và hiển thị bằng Plotly trên Colab
    print("5. Đang tái tạo 3D Point Cloud...")
    focal_length = 0.8 * w
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0,  h / 2.0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2RGB)

    mask_3d = disparity > disparity.min()
    out_points = points_3D[mask_3d]
    out_colors = colors[mask_3d]

    # Vẽ 3D trên Colab bằng Plotly (lấy mẫu giảm số điểm để tránh đơ trình duyệt)
    print("6. Đang vẽ biểu đồ 3D tương tác...")
    sample_rate = 1 # Chỉ lấy 1/15 số điểm ảnh để render cho nhẹ
    fig = go.Figure(data=[go.Scatter3d(
        x=out_points[::sample_rate, 0],
        y=out_points[::sample_rate, 1],
        z=out_points[::sample_rate, 2],
        mode='markers',
        marker=dict(size=2, color=out_colors[::sample_rate]/255.0)
    )])
    # Lật ngược trục Z để nhìn thuận mắt hơn
    fig.update_layout(scene=dict(zaxis=dict(autorange="reversed")), margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

# --- Chạy hàm ---

process_stereo_uncalibrated('/workspaces/Traditional-Image-Processing-for-Filtering-3D-Reconstruction-and-Image-Stitching/a.jpg', '/workspaces/Traditional-Image-Processing-for-Filtering-3D-Reconstruction-and-Image-Stitching/b.jpg')