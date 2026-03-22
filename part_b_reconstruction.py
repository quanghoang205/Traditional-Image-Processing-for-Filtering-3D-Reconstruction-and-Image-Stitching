import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- HÀM PHỤ: Vẽ đường cực (Đã sửa lỗi float -> int) ---
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        
        # Ép kiểu tọa độ tâm (x, y) về số nguyên
        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0]), int(pt2[1]))
        
        img1 = cv2.circle(img1, pt1_int, 5, color, -1)
        img2 = cv2.circle(img2, pt2_int, 5, color, -1)
    return img1, img2
# ------------------------------------------------------

def process_stereo_uncalibrated(imgL_path, imgR_path):
    # 1. Đọc ảnh và giảm kích thước
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

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # 3. Tính Fundamental Matrix (F)
    print("2. Đang tính Fundamental Matrix...")
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]

    # --- 3.5 Vẽ Epipolar Lines ---
    print("3. Đang vẽ Epipolar Lines...")
    # Lấy ngẫu nhiên 15 điểm để vẽ cho đỡ rối mắt
    idx = np.random.choice(len(pts1_in), min(15, len(pts1_in)), replace=False)
    pts1_sample = pts1_in[idx]
    pts2_sample = pts2_in[idx]

    lines1 = cv2.computeCorrespondEpilines(pts2_sample.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img5, _ = drawlines(grayL, grayR, lines1, pts1_sample, pts2_sample)
    
    lines2 = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img3, _ = drawlines(grayR, grayL, lines2, pts2_sample, pts1_sample)

    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines (Left Image)')
    plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines (Right Image)')
    plt.axis('off')
    plt.show()

    # 4. Stereo Rectification (Nắn chỉnh ảnh)
    print("4. Đang nắn chỉnh ảnh (Rectification) để căn bằng các đường cực...")
    h, w = grayL.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_in, pts2_in, F, (w, h))

    rectifiedL = cv2.warpPerspective(imgL, H1, (w, h))
    rectifiedR = cv2.warpPerspective(imgR, H2, (w, h))
    rectifiedL_gray = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
    rectifiedR_gray = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

    # 5. Tính Disparity Map trên ảnh đã nắn chỉnh
    print("5. Đang tính Disparity Map bằng SGBM...")
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
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
    print("6. Đang tái tạo 3D Point Cloud...")
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

    # Vẽ 3D trên Colab bằng Plotly
    print("7. Đang vẽ biểu đồ 3D tương tác...")
    sample_rate = 5 # Lấy 1/5 số điểm để trình duyệt không bị đơ
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
# Cập nhật ảnh có chứa tờ báo (1a.jpg, 2a.jpg) vào đây
process_stereo_uncalibrated('/content/im2.png', '/content/im6.png')
