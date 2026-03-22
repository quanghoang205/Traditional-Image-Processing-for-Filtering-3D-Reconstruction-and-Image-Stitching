import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_matches_and_homography(img1, img2):
    """
    Tìm điểm đặc trưng, nối điểm và tính toán ma trận Homography.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1. Phát hiện đặc trưng bằng SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 2. Matching bằng FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lọc điểm tốt (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Trực quan hóa các điểm nối (Deliverable 1)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 3. Tính Homography bằng RANSAC
    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Lấy ma trận Homography (H) chuyển từ ảnh 2 sang ảnh 1
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H, img_matches
    else:
        print(f"Không đủ điểm khớp - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None, None

def warp_and_blend(img1, img2, H):
    """
    Warp ảnh 2 theo ma trận H và ghép vào ảnh 1.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Tính toán kích thước của khung canvas mới chứa cả 2 ảnh
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    # Ma trận dịch chuyển để ảnh không bị cắt mất phần âm
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Warp ảnh 2 vào canvas
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))

    # Blend: Chèn ảnh 1 vào đúng vị trí
    # (Ở đây ta dùng cách ghi đè đơn giản. Em có thể nghiên cứu thêm cv2.addWeighted để blend mượt mép hơn nếu muốn điểm cộng)
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1

    return result

def process_stitching_pipeline(image_paths):
    # Đọc và giảm kích thước 4 ảnh để tránh quá tải RAM trên Colab
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Lỗi: Không đọc được {path}")
            return
        img = cv2.resize(img, (600, 450)) # Resize để tính toán nhanh hơn
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển sang RGB cho Matplotlib
        images.append(img)

    panorama = images[0]

    for i in range(1, len(images)):
        print(f"Đang ghép ảnh {i} và ảnh {i+1}...")
        img_next = images[i]

        # Tìm ma trận biến đổi H từ ảnh tiếp theo sang panorama hiện tại
        H, img_matches = find_matches_and_homography(panorama, img_next)

        if H is not None:
            # Hiển thị Matched keypoints
            plt.figure(figsize=(15, 5))
            plt.imshow(img_matches)
            plt.title(f'Matched Keypoints: Panorama tạm & Ảnh {i+1}')
            plt.axis('off')
            plt.show()

            # Thực hiện Warp và Blend
            panorama = warp_and_blend(panorama, img_next, H)
        else:
            print("Quá trình ghép bị dừng do không tìm thấy đủ điểm chung.")
            break

    # Hiển thị kết quả Panorama cuối cùng
    print("Hoàn thành! Đang hiển thị Final Panorama...")
    plt.figure(figsize=(20, 10))
    plt.imshow(panorama)
    plt.title('Final Panorama Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Cập nhật danh sách 4 bức ảnh chụp liên tiếp của em vào đây (từ trái sang phải)
    my_images = [
        "c.jpg",
        "d.jpg",
        "e.jpg",
        "f.jpg"
    ]
    process_stitching_pipeline(my_images)