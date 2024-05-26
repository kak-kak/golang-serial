import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
# 画像をぼかしてノイズを低減
blurred = cv2.medianBlur(image, 5)

# ハフ円変換を適用して円を検出
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=50, param2=30, minRadius=20, maxRadius=100)

# 検出された円を確認
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (x, y, r) in circles:
        # 円の周囲に円を描画
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        # 円の中心に点を描画
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    # 結果を表示
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.show()
else:
    print("No circles were found")

# 円弧を抽出する関数
def extract_arc(image, center, radius, start_angle, end_angle):
    mask = np.zeros_like(image)
    axes = (radius, radius)
    angle = 0  # Angle of rotation of the arc
    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, 255, -1)
    arc = cv2.bitwise_and(image, image, mask=mask)
    return arc

# 例として、検出された最初の円から円弧を抽出
if circles is not None:
    x, y, r = circles[0]
    start_angle = 0
    end_angle = 180  # 180度の円弧を抽出
    arc = extract_arc(image, (x, y), r, start_angle, end_angle)
    
    # 円弧の結果を表示
    plt.figure(figsize=(10, 8))
    plt.imshow(arc, cmap='gray')
    plt.title('Extracted Arc')
    plt.show()
else:
    print("No circles to extract arcs from")
