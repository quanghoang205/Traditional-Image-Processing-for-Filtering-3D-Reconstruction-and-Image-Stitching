import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image):
    """them nhieu gaussian vao anh de test cac bo loc"""
    row, col, ch = image.shape
    mean = 0
    sigma = 25 # do lech chuan nhieu nhat
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row,col,ch)
    noisy =  image + gauss
    return np.clip(noisy, 0, 255).astype (np.uint8)

def process_part_a(image_path):
    # doc anh goc bang opne cv
    # chuyen doi mau BGR sang RGB cho matplotlib
    original = cv2.imread(image_path)
    if original is None:
        print(f" khon the doc duoc anh taij {image_path}. hay kiem tra lai duong dan")
        return
    original_image = cv2.cvtColor( original, cv2.COLOR_BGR2RGB)
    #tao anh nhieu
    noisy_img = add_noise(original)

    #2 ap dung Mean Filter ( kernel size 5x5)
    mean_filtered = cv2.blur(noisy_img, (5,5))

    #3 ap dung Gaussian Filter ( kernel size 5x5, sigma tu dong )
    gaussian_filtered = cv2.GaussianBlur(noisy_img, (5,5), 0)

    #4 ap dung Median Filtered (kernel size 5x5, sigma tu dong)
    median_filtered = cv2.medianBlur(noisy_img,5)

    #5 ap dung Laplacian sharpening len anh goc ( vi sharpening anh nhieu se lamf nhieu nang hon)
    # lay dao ham bac 2 bang Laplacian
    laplacian = cv2.Laplacian(original, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Cong vien canh vua tim duoc anh goc ( cong thuc Unsharp Masking co ban)
    sharpened = cv2.addWeighted(original, 1.5, laplacian, -0.5, 0)

    # hien thi side-by-side bang Matplotlib
    titles = ['Original', 'Noisy Image', 'Mean Filtered', 'Gaussian Filtered', 'Median Filtered', 'Laplacian Sharpened']
    images = [original, noisy_img, mean_filtered, gaussian_filtered, median_filtered, sharpened]

    plt.figure(figsize=(18,10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return original, noisy_img, mean_filtered, gaussian_filtered, median_filtered

if __name__ == "__main__":
    original, noisy_img, mean_filtered, gaussian_filtered, median_filtered = process_part_a("//content/1.jpg")
    # Tính PSNR để so sánh khả năng khử nhiễu
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (PSNR) ---")

    # Hàm cv2.PSNR có sẵn của OpenCV giúp tính toán rất nhanh
    psnr_noisy = cv2.PSNR(original, noisy_img)
    print(f"1. Ảnh nhiễu (chưa lọc) so với ảnh gốc: {psnr_noisy:.2f} dB")

    psnr_mean = cv2.PSNR(original, mean_filtered)
    print(f"2. Mean Filter: {psnr_mean:.2f} dB")

    psnr_gaussian = cv2.PSNR(original, gaussian_filtered)
    print(f"3. Gaussian Filter: {psnr_gaussian:.2f} dB")

    psnr_median = cv2.PSNR(original, median_filtered)
    print(f"4. Median Filter: {psnr_median:.2f} dB")


