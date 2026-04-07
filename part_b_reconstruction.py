"""
===================================================================
HỆ THỐNG TÁI TẠO 3D TỪ ẢNH STEREO CHƯA HIỆU CHỈNH
===================================================================
Thư viện : OpenCV, Open3D, NumPy
Mô tả    : Tái tạo đám mây điểm 3D có màu từ cặp ảnh stereo
           sử dụng SIFT + FLANN + RANSAC + SGBM + Open3D
===================================================================
"""

import sys
import os
import numpy as np
import cv2

# ---------------------------------------------------------------
# Kiểm tra Open3D trước khi import để báo lỗi rõ ràng
# ---------------------------------------------------------------
try:
    import open3d as o3d
except ImportError:
    print("[LỖI] Chưa cài thư viện Open3D.")
    print("      Hãy chạy: pip install open3d")
    sys.exit(1)


# ===================================================================
# PHẦN 1: XỬ LÝ ẢNH VÀ TÍNH TOÁN HÌNH HỌC
# ===================================================================

class StereoProcessor:
    """
    Lớp xử lý ảnh stereo: tìm điểm đặc trưng, tính ma trận cơ bản,
    chỉnh sửa ảnh và tạo bản đồ độ chênh lệch (disparity map).
    """

    def __init__(self, img_left_path: str, img_right_path: str):
        """
        Khởi tạo bộ xử lý stereo.

        Args:
            img_left_path : Đường dẫn ảnh trái (im2.png)
            img_right_path: Đường dẫn ảnh phải (im6.png)
        """
        print("=" * 60)
        print("  KHỞI ĐỘNG HỆ THỐNG TÁI TẠO 3D TỪ ẢNH STEREO")
        print("=" * 60)

        # --- Đọc và kiểm tra file ảnh ---
        self.img_left_color  = self._load_image(img_left_path,  "ảnh trái")
        self.img_right_color = self._load_image(img_right_path, "ảnh phải")

        # Chuyển sang grayscale để xử lý đặc trưng
        self.img_left_gray  = cv2.cvtColor(self.img_left_color,  cv2.COLOR_BGR2GRAY)
        self.img_right_gray = cv2.cvtColor(self.img_right_color, cv2.COLOR_BGR2GRAY)

        self.h, self.w = self.img_left_gray.shape
        print(f"[OK] Kích thước ảnh: {self.w} x {self.h} pixel")

    # ------------------------------------------------------------------
    # Bước 1: Đọc ảnh từ file
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(path: str, label: str) -> np.ndarray:
        """Đọc ảnh và ném ngoại lệ nếu không tìm thấy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[LỖI] Không tìm thấy {label}: '{path}'\n"
                f"      Vui lòng đặt file ảnh cùng thư mục với script."
            )
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"[LỖI] Không thể đọc {label}: '{path}'")
        print(f"[OK] Đọc {label}: {path}")
        return img

    # ------------------------------------------------------------------
    # Bước 2: Tìm điểm đặc trưng bằng SIFT + khớp bằng FLANN
    # ------------------------------------------------------------------
    def detect_and_match_features(self) -> tuple:
        """
        Sử dụng SIFT (Scale-Invariant Feature Transform) để phát hiện
        keypoints và mô tả đặc trưng, sau đó dùng FLANN (Fast Library
        for Approximate Nearest Neighbors) để khớp các điểm tương ứng.

        Returns:
            pts_left, pts_right: Mảng tọa độ 2D các điểm khớp (Nx2)
        """
        print("\n[BƯỚC 1] Phát hiện và khớp điểm đặc trưng (SIFT + FLANN)...")

        # Tạo detector SIFT
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)

        # Tìm keypoints và mô tả đặc trưng trên cả 2 ảnh
        kp1, des1 = sift.detectAndCompute(self.img_left_gray,  None)
        kp2, des2 = sift.detectAndCompute(self.img_right_gray, None)
        print(f"   - Số điểm đặc trưng ảnh trái : {len(kp1)}")
        print(f"   - Số điểm đặc trưng ảnh phải: {len(kp2)}")

        if len(kp1) < 8 or len(kp2) < 8:
            raise RuntimeError("[LỖI] Quá ít điểm đặc trưng để tính toán.")

        # Cấu hình FLANN matcher cho SIFT (dùng KD-Tree)
        index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Khớp điểm với kNN (k=2) để áp dụng Lowe's ratio test
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's Ratio Test: giữ chỉ những cặp khớp "chắc chắn"
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        print(f"   - Số cặp khớp sau Ratio Test: {len(good_matches)}")

        if len(good_matches) < 8:
            raise RuntimeError(
                "[LỖI] Không đủ cặp điểm khớp tốt (cần ít nhất 8)."
            )

        # Trích xuất tọa độ 2D của các điểm khớp
        pts_left  = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        self.pts_left  = pts_left
        self.pts_right = pts_right
        return pts_left, pts_right

    # ------------------------------------------------------------------
    # Bước 3: Tính ma trận cơ bản (Fundamental Matrix) bằng RANSAC
    # ------------------------------------------------------------------
    def compute_fundamental_matrix(self) -> np.ndarray:
        """
        Tính Fundamental Matrix F liên kết các điểm tương ứng giữa
        2 ảnh theo quan hệ: x'^T * F * x = 0
        RANSAC được dùng để loại bỏ các điểm nhiễu (outliers).

        Returns:
            F: Ma trận cơ bản (3x3)
        """
        print("\n[BƯỚC 2] Tính Fundamental Matrix bằng RANSAC...")

        F, mask = cv2.findFundamentalMat(
            self.pts_left,
            self.pts_right,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,   # Ngưỡng reprojection error (pixel)
            confidence=0.999             # Độ tin cậy yêu cầu
        )

        if F is None:
            raise RuntimeError("[LỖI] Không tính được Fundamental Matrix.")

        # Lọc chỉ giữ inliers (điểm được RANSAC chấp nhận)
        mask = mask.ravel().astype(bool)
        self.pts_left  = self.pts_left[mask]
        self.pts_right = self.pts_right[mask]

        print(f"   - Số inliers sau RANSAC: {mask.sum()} / {len(mask)}")
        print(f"   - Fundamental Matrix:\n{np.round(F, 6)}")

        self.F = F
        return F

    # ------------------------------------------------------------------
    # Bước 4: Chỉnh sửa ảnh stereo chưa hiệu chỉnh (Uncalibrated Rectification)
    # ------------------------------------------------------------------
    def rectify_images(self) -> tuple:
        """
        Stereo Rectification biến đổi 2 ảnh sao cho các dòng quét
        (epipolar lines) nằm ngang hàng — điều kiện cần để tính disparity.

        Vì camera chưa được hiệu chỉnh (uncalibrated), dùng hàm
        cv2.stereoRectifyUncalibrated để tính homography H1, H2.

        Returns:
            img_rect_left, img_rect_right: Ảnh sau khi rectify (BGR)
        """
        print("\n[BƯỚC 3] Thực hiện Stereo Rectification (Uncalibrated)...")

        # Tính homography H1, H2 từ Fundamental Matrix và các điểm khớp
        ret, H1, H2 = cv2.stereoRectifyUncalibrated(
            self.pts_left,
            self.pts_right,
            self.F,
            imgSize=(self.w, self.h),
            threshold=5
        )

        if not ret:
            raise RuntimeError("[LỖI] stereoRectifyUncalibrated thất bại.")

        # Áp dụng phép biến đổi perspective (warpPerspective) lên cả 2 ảnh
        img_rect_left  = cv2.warpPerspective(self.img_left_color,  H1, (self.w, self.h))
        img_rect_right = cv2.warpPerspective(self.img_right_color, H2, (self.w, self.h))

        self.img_rect_left  = img_rect_left
        self.img_rect_right = img_rect_right
        self.H1 = H1
        self.H2 = H2

        print("   - Rectification hoàn tất.")
        return img_rect_left, img_rect_right

    # ------------------------------------------------------------------
    # Bước 5: Tính Disparity Map bằng StereoSGBM
    # ------------------------------------------------------------------
    def compute_disparity(self) -> np.ndarray:
        """
        Semi-Global Block Matching (SGBM) tính độ chênh lệch (disparity)
        giữa các pixel tương ứng trên 2 ảnh đã rectify.
        Disparity tỉ lệ nghịch với khoảng cách đến camera: Z = f*B/d

        Returns:
            disparity: Bản đồ disparity dạng float32 (đã chuẩn hóa)
        """
        print("\n[BƯỚC 4] Tính Disparity Map bằng StereoSGBM...")

        # Số kênh màu (dùng để tính P1, P2)
        num_channels = 3

        # Cấu hình StereoSGBM — các tham số ảnh hưởng trực tiếp đến chất lượng
        min_disparity   = 0
        num_disparities = 16 * 8   # Phải là bội số của 16; tăng nếu cảnh sâu
        block_size      = 5        # Kích thước block so khớp (lẻ, 3–11)

        sgbm = cv2.StereoSGBM_create(
            minDisparity      = min_disparity,
            numDisparities    = num_disparities,
            blockSize         = block_size,
            # P1, P2: tham số điều hòa độ mượt — P2 > P1
            P1                = 8  * num_channels * block_size ** 2,
            P2                = 32 * num_channels * block_size ** 2,
            disp12MaxDiff     = 1,     # Kiểm tra tính nhất quán trái-phải
            uniquenessRatio   = 10,    # Lọc điểm không rõ ràng (%)
            speckleWindowSize = 100,   # Loại bỏ nhiễu nhỏ
            speckleRange      = 32,    # Biên độ nhiễu
            preFilterCap      = 63,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Chất lượng cao nhất
        )

        # Chuyển ảnh rectify sang grayscale để SGBM xử lý
        left_gray  = cv2.cvtColor(self.img_rect_left,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.img_rect_right, cv2.COLOR_BGR2GRAY)

        # Tính disparity (kết quả nhân 16 — cần chia lại)
        disparity_raw = sgbm.compute(left_gray, right_gray)
        disparity = disparity_raw.astype(np.float32) / 16.0

        # Thay thế giá trị âm (vùng không tính được) bằng NaN
        disparity[disparity < 0] = np.nan

        self.disparity = disparity
        print(f"   - Disparity range: [{np.nanmin(disparity):.1f}, {np.nanmax(disparity):.1f}]")
        return disparity


# ===================================================================
# PHẦN 2: TÁI TẠO 3D VÀ HIỂN THỊ
# ===================================================================

class Reconstructor3D:
    """
    Lớp tái tạo đám mây điểm 3D từ disparity map và hiển thị
    tương tác bằng Open3D.
    """

    def __init__(self, processor: StereoProcessor):
        self.proc = processor

    # ------------------------------------------------------------------
    # Bước 6: Xây dựng ma trận Q (reprojection matrix)
    # ------------------------------------------------------------------
    def build_Q_matrix(self) -> np.ndarray:
        """
        Ma trận Q dùng để tái chiếu (reproject) disparity map thành
        tọa độ 3D theo công thức: [X, Y, Z, W]^T = Q * [x, y, d, 1]^T

        Vì camera chưa hiệu chỉnh, ước tính thông số nội (intrinsics):
          - Tiêu cự f  ≈ 0.8 × width  (giả định góc nhìn ~67°)
          - Điểm chính cx = w/2, cy = h/2
          - Baseline B ≈ w / 10  (ước tính tương đối)

        Returns:
            Q: Ma trận tái chiếu 4x4
        """
        w, h = self.proc.w, self.proc.h
        f  = 0.8 * w          # Tiêu cự ước tính
        cx = w / 2.0          # Điểm chính x
        cy = h / 2.0          # Điểm chính y
        B  = w / 10.0         # Baseline ước tính

        # Ma trận Q chuẩn cho stereo rectified
        Q = np.float32([
            [1,  0,   0,  -cx    ],
            [0,  1,   0,  -cy    ],
            [0,  0,   0,   f     ],
            [0,  0,  -1/B, 0     ]
        ])

        self.Q = Q
        print(f"\n[BƯỚC 5] Xây dựng ma trận Q:")
        print(f"   - Tiêu cự f  = {f:.1f} px")
        print(f"   - Điểm chính = ({cx:.1f}, {cy:.1f})")
        print(f"   - Baseline B ≈ {B:.1f} px (ước tính)")
        return Q

    # ------------------------------------------------------------------
    # Bước 7: Chuyển disparity → tọa độ 3D (Point Cloud)
    # ------------------------------------------------------------------
    def disparity_to_pointcloud(self) -> tuple:
        """
        Dùng cv2.reprojectImageTo3D để chuyển mỗi pixel (x, y, d)
        thành điểm 3D (X, Y, Z) theo ma trận Q.

        Lọc bỏ:
          - Điểm có Z quá lớn (nhiễu ở vô cực)
          - Điểm có disparity = NaN hoặc = 0

        Returns:
            points: Tọa độ 3D (Nx3, float32)
            colors: Màu RGB tương ứng (Nx3, float32 trong [0,1])
        """
        print("\n[BƯỚC 6] Tái chiếu Disparity Map → Point Cloud 3D...")

        # cv2.reprojectImageTo3D trả về array (H, W, 3) chứa (X, Y, Z)
        points_3d = cv2.reprojectImageTo3D(self.proc.disparity, self.Q)

        # Lấy màu từ ảnh rectify trái (BGR → RGB)
        colors_bgr = self.proc.img_rect_left

        # --- Tạo mặt nạ để lọc điểm hợp lệ ---
        # 1) Loại điểm có disparity không hợp lệ
        valid_disp = np.isfinite(self.proc.disparity) & (self.proc.disparity > 0)
        # 2) Loại điểm Z quá lớn (thường là background nhiễu)
        z_vals = points_3d[:, :, 2]
        z_max  = np.percentile(z_vals[valid_disp], 97)   # Ngưỡng 97th percentile
        valid_z = z_vals < z_max
        # 3) Loại điểm Z âm
        valid_pos = z_vals > 0

        mask = valid_disp & valid_z & valid_pos

        # Trích xuất điểm và màu theo mặt nạ
        pts    = points_3d[mask]                         # (N, 3)
        cols   = colors_bgr[mask].astype(np.float32) / 255.0  # (N, 3) trong [0,1]
        # Đảo kênh BGR → RGB
        cols   = cols[:, ::-1]

        print(f"   - Số điểm 3D hợp lệ: {len(pts):,}")

        self.points = pts
        self.colors = cols
        return pts, cols

    # ------------------------------------------------------------------
    # Bước 8: Lưu và hiển thị Point Cloud bằng Open3D
    # ------------------------------------------------------------------
    def visualize(self, save_path: str = "pointcloud.ply"):
        """
        Tạo Open3D PointCloud object và mở cửa sổ visualizer tương tác.
        Người dùng có thể:
          - Xoay  : giữ chuột trái + kéo
          - Thu phóng: cuộn chuột
          - Di chuyển: giữ chuột phải + kéo
          - Reset: phím 'R'

        Args:
            save_path: Đường dẫn lưu file PLY (tùy chọn)
        """
        print("\n[BƯỚC 7] Khởi tạo Open3D Visualizer...")

        # Tạo đối tượng PointCloud của Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)

        # --- Làm sạch thêm: loại bỏ điểm thống kê ngoại lai ---
        print("   - Lọc statistical outlier...")
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,   # Số láng giềng để ước tính
            std_ratio=2.0      # Ngưỡng độ lệch chuẩn
        )
        print(f"   - Điểm sau lọc: {len(pcd_clean.points):,}")

        # Lưu file PLY để dùng lại
        o3d.io.write_point_cloud(save_path, pcd_clean)
        print(f"   - Đã lưu point cloud: {save_path}")

        # --- Mở cửa sổ tương tác ---
        print("\n[HIỂN THỊ] Mở cửa sổ 3D Visualizer...")
        print("   Điều khiển:")
        print("   • Chuột trái  : Xoay mô hình")
        print("   • Chuột phải  : Di chuyển")
        print("   • Cuộn chuột  : Thu phóng")
        print("   • Phím R      : Reset góc nhìn")
        print("   • Phím Q / Esc: Thoát")

        # Tùy chỉnh cửa sổ visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="3D Stereo Reconstruction — Point Cloud",
            width=1280,
            height=720
        )
        vis.add_geometry(pcd_clean)

        # Thiết lập render options
        opt = vis.get_render_option()
        opt.point_size          = 1.5    # Kích thước điểm
        opt.background_color    = np.array([0.05, 0.05, 0.1])  # Nền tối
        opt.show_coordinate_frame = True

        # Đặt góc nhìn ban đầu
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)

        vis.run()       # Vòng lặp chính — blocking cho đến khi đóng cửa sổ
        vis.destroy_window()
        print("\n[OK] Visualizer đã đóng.")


# ===================================================================
# PHẦN 3: PIPELINE CHÍNH
# ===================================================================

def run_pipeline(left_path: str = "im2.png", right_path: str = "im6.png"):
    """
    Hàm điều phối toàn bộ pipeline từ đầu đến cuối.

    Args:
        left_path : Đường dẫn ảnh trái
        right_path: Đường dẫn ảnh phải
    """
    try:
        # --- Khởi tạo bộ xử lý stereo ---
        proc = StereoProcessor(left_path, right_path)

        # --- Xử lý ảnh theo trình tự ---
        proc.detect_and_match_features()
        proc.compute_fundamental_matrix()
        proc.rectify_images()
        proc.compute_disparity()

        # --- Tái tạo và hiển thị 3D ---
        recon = Reconstructor3D(proc)
        recon.build_Q_matrix()
        recon.disparity_to_pointcloud()

        # Hiển thị bản đồ disparity (tùy chọn — nhấn bất kỳ phím để tiếp tục)
        disp_vis = proc.disparity.copy()
        disp_vis = np.nan_to_num(disp_vis, nan=0.0)
        disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Map (nhấn phím bất kỳ để mở 3D Viewer)", disp_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Hiển thị cửa sổ 3D tương tác
        recon.visualize(save_path="pointcloud_output.ply")

        print("\n" + "=" * 60)
        print("  HOÀN THÀNH! File PLY đã lưu: pointcloud_output.ply")
        print("=" * 60)

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[LỖI KHÔNG XÁC ĐỊNH] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ===================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ===================================================================

if __name__ == "__main__":
    # Có thể truyền đường dẫn ảnh qua argument dòng lệnh:
    #   python stereo_3d_reconstruction.py im2.png im6.png
    if len(sys.argv) == 3:
        left_img  = sys.argv[1]
        right_img = sys.argv[2]
    else:
        # Mặc định tìm ảnh trong cùng thư mục với script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        left_img   = os.path.join(script_dir, "im2.png")
        right_img  = os.path.join(script_dir, "im6.png")

    run_pipeline(left_img, right_img)
