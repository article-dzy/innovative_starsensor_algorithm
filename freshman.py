import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class StarTracker:
    def __init__(self):
        self.frame_rate = 4
        self.horizontal_fov = 29.56
        self.vertical_fov = 24.9
        self.image_width = 1224
        self.image_height = 1024

    def calculate_centroid(self, x1, y1, x2, y2, x3, y3):
        """计算三角形的几何中心（质心）坐标"""
        centroid_x = (x1 + x2 + x3) / 3
        centroid_y = (y1 + y2 + y3) / 3
        return centroid_x, centroid_y

    def is_within_range(self, point, selected_points, radius=10):
        """检查目标点是否与已选点距离过近"""
        for p in selected_points:
            if abs(point[0] - p[0]) <= radius and abs(point[1] - p[1]) <= radius:
                return True
        return False

    def is_near_edge_1(self, x, y, image_shape, margin_ratio=0.2):
        """
        检查点是否位于图像边缘区域（版本1）
        特点：左侧边缘区域比右侧宽一倍
        """
        height, width = image_shape
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)

        return (x < 2 * margin_x or x > width - margin_x or
                y < margin_y or y > height - margin_y)

    def is_near_edge_2(self, x, y, image_shape, margin_ratio=0.2):
        """
        检查点是否位于图像边缘区域（版本2）
        特点：右侧边缘区域比左侧宽一倍
        """
        height, width = image_shape
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)

        return (x < margin_x or x > width - 2 * margin_x or
                y < margin_y or y > height - margin_y)

    def weighted_center_original(self, region, region_coordinates):
        """根据区域内像素亮度计算加权质心（亚像素级精确定位）"""
        y_coords, x_coords = np.meshgrid(
            np.arange(region.shape[0]),
            np.arange(region.shape[1]),
            indexing='ij'
        )

        weights = region.flatten()
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        # 归一化权重
        max_brightness = np.max(region)
        if max_brightness != 0:
            weights = weights / max_brightness

        # 计算加权平均坐标
        weighted_x = np.sum(weights * x_coords) / np.sum(weights)
        weighted_y = np.sum(weights * y_coords) / np.sum(weights)

        # 转换为全局坐标
        weighted_x += region_coordinates[1]
        weighted_y += region_coordinates[0]

        return weighted_x, weighted_y

    def detect_stars(self, image_path, num_stars=6, edge_check_version=1):
        """
        检测图像中的星点
        edge_check_version: 1-使用is_near_edge_1, 2-使用is_near_edge_2
        """
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("无法读取图像: " + image_path)

        # 高斯模糊处理
        blur1 = cv2.GaussianBlur(image, (5, 5), 0)
        blur2 = cv2.GaussianBlur(image, (15, 15), 0)

        # 计算高斯差分
        dog = cv2.subtract(blur1, blur2)

        # 筛选最亮的特征点
        selected_points = []
        dog_flat = dog.flatten().astype(np.float64)

        while len(selected_points) < num_stars:
            max_idx = np.argmax(dog_flat)
            y, x = np.unravel_index(max_idx, dog.shape)

            # 根据版本选择边缘检查函数
            if edge_check_version == 1:
                is_near_edge = self.is_near_edge_1
            else:
                is_near_edge = self.is_near_edge_2

            # 跳过边缘区域的点
            if is_near_edge(x, y, image.shape):
                dog_flat[max_idx] = -1e10
                continue

            # 检查点是否与已选点距离足够远
            if not self.is_within_range((x, y), selected_points):
                selected_points.append((x, y))

            dog_flat[max_idx] = -1e10

        # 精化特征点坐标
        refined_points = []
        for point in selected_points:
            x, y = point
            region_x_min = max(int(x) - 2, 0)
            region_x_max = min(int(x) + 2, image.shape[1])
            region_y_min = max(int(y) - 2, 0)
            region_y_max = min(int(y) + 2, image.shape[0])

            region = image[region_y_min:region_y_max, region_x_min:region_x_max]
            weighted_x, weighted_y = self.weighted_center_original(region, (region_y_min, region_x_min))
            refined_points.append((weighted_x, weighted_y))

        return refined_points, image

    def calculate_distance(self, star1, star2):
        """计算两点之间的距离"""
        return math.sqrt((star1["x"] - star2["x"]) ** 2 + (star1["y"] - star2["y"]) ** 2)

    def calculate_angles(self, a, b, c):
        """计算三角形的三个角"""
        angle_a = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        angle_b = math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
        angle_c = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        return math.degrees(angle_a), math.degrees(angle_b), math.degrees(angle_c)

    def star_identification(self, stars):
        """星点识别和三角形匹配"""
        N = 0

        while N <= len(stars) - 3:
            n = N + 2

            while n < len(stars):
                triangle = [stars[N], stars[N + 1], stars[n]]

                # 计算三角形边长
                sides = [
                    self.calculate_distance(triangle[0], triangle[1]),
                    self.calculate_distance(triangle[1], triangle[2]),
                    self.calculate_distance(triangle[0], triangle[2]),
                ]
                min_side = min(sides)

                # 约束检查
                if min_side <= 15:
                    if n >= len(stars) - 1:
                        break
                    n += 1
                    continue

                angles = self.calculate_angles(*sides)
                min_angle = min(angles)

                if min_angle <= 20:
                    if n >= len(stars) - 1:
                        break
                    n += 1
                    continue

                # 通过约束检查，返回三角形信息
                x1, y1 = triangle[0]['x'], triangle[0]['y']
                x2, y2 = triangle[1]['x'], triangle[1]['y']
                x3, y3 = triangle[2]['x'], triangle[2]['y']

                centroid = self.calculate_centroid(x1, y1, x2, y2, x3, y3)
                triangle_info = {
                    "sides": sides,
                    "coordinates": [[x1, y1], [x2, y2], [x3, y3]]
                }

                print("识别成功")
                return triangle_info, centroid

            N += 1
            if N > len(stars) - 3:
                break

        print("识别失败：无法找到合适的三角形")
        return None

    def compare_elements(self, a, b, rel_tol=0.05, abs_tol=0.0):
        """比较两个数值是否相近"""
        if isinstance(a, (np.floating, float)):
            a = float(a)
        if isinstance(b, (np.floating, float)):
            b = float(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def compare_sides(self, triangle1, triangle2, rel_tol=0.05, abs_tol=0.0):
        """比较两个三角形的边长是否匹配"""
        try:
            sides1 = triangle1.get("sides", [])
            sides2 = triangle2.get("sides", [])
            if not isinstance(sides1, list) or not isinstance(sides2, list):
                return False
            if len(sides1) != 3 or len(sides2) != 3:
                return False
        except Exception as e:
            print("比较边长时出错: " + str(e))
            return False

        sides1_sorted = sorted(sides1)
        sides2_sorted = sorted(sides2)

        for i in range(3):
            if not self.compare_elements(sides1_sorted[i], sides2_sorted[i], rel_tol, abs_tol):
                return False
        return True

    def calculate_angular_velocity(self, delta_pixels, delta_time, fov, image_dimension):
        """计算角速度"""
        angular_velocity = (delta_pixels / delta_time) * (fov / image_dimension)
        return angular_velocity

    def visualize_matched_triangles(self, image1, triangle1_coords, image2, triangle2_coords,
                                    centroid1, centroid2, delta_x, delta_y):
        """专门可视化匹配的两个三角形"""
        # 创建彩色图像用于绘制
        img1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # 定义颜色
        triangle_color = (0, 255, 0)  # 绿色三角形
        centroid_color = (255, 0, 0)  # 蓝色质心
        point_color = (0, 0, 255)  # 红色顶点

        # 在第一张图像上绘制三角形
        for i in range(len(triangle1_coords)):
            coord = triangle1_coords[i]
            x, y = int(coord[0]), int(coord[1])
            # 绘制顶点
            cv2.circle(img1_color, (x, y), 6, point_color, -1)
            # 添加顶点编号
            cv2.putText(img1_color, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)

        # 绘制三角形边
        for i in range(3):
            start_point = (int(triangle1_coords[i][0]), int(triangle1_coords[i][1]))
            end_point = (int(triangle1_coords[(i + 1) % 3][0]), int(triangle1_coords[(i + 1) % 3][1]))
            cv2.line(img1_color, start_point, end_point, triangle_color, 2)

        # 绘制质心
        cx1, cy1 = int(centroid1[0]), int(centroid1[1])
        cv2.circle(img1_color, (cx1, cy1), 8, centroid_color, -1)
        cv2.putText(img1_color, "Centroid", (cx1 + 10, cy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, centroid_color, 2)

        # 在第二张图像上绘制三角形
        for i in range(len(triangle2_coords)):
            coord = triangle2_coords[i]
            x, y = int(coord[0]), int(coord[1])
            # 绘制顶点
            cv2.circle(img2_color, (x, y), 6, point_color, -1)
            # 添加顶点编号
            cv2.putText(img2_color, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)

        # 绘制三角形边
        for i in range(3):
            start_point = (int(triangle2_coords[i][0]), int(triangle2_coords[i][1]))
            end_point = (int(triangle2_coords[(i + 1) % 3][0]), int(triangle2_coords[(i + 1) % 3][1]))
            cv2.line(img2_color, start_point, end_point, triangle_color, 2)

        # 绘制质心
        cx2, cy2 = int(centroid2[0]), int(centroid2[1])
        cv2.circle(img2_color, (cx2, cy2), 8, centroid_color, -1)
        cv2.putText(img2_color, "Centroid", (cx2 + 10, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, centroid_color, 2)

        # 创建可视化图像
        plt.figure(figsize=(15, 6))

        # 第一张图像
        plt.subplot(1, 2, 1)
        plt.title("First Image - Matched Triangle", fontsize=14, fontweight='bold')
        plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 第二张图像
        plt.subplot(1, 2, 2)
        plt.title("Second Image - Matched Triangle", fontsize=14, fontweight='bold')
        plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 添加整体标题和信息
        plt.suptitle("Matched Triangles for Angular Velocity Calculation\n"
                     "Centroid Shift: Δx=" + str(round(delta_x, 2)) + "px, Δy=" + str(round(delta_y, 2)) + "px",
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.show()

        # 保存图像
        cv2.imwrite('matched_triangle_image1.jpg', img1_color)
        cv2.imwrite('matched_triangle_image2.jpg', img2_color)
        print("匹配三角形图像已保存为 'matched_triangle_image1.jpg' 和 'matched_triangle_image2.jpg'")

    def process_images(self, image1_path, image2_path):
        """处理两张图像并计算角速度"""
        # 处理第一张图像（使用版本1的边缘检查）
        print("处理第一张图像...")
        refined_points1, image1 = self.detect_stars(image1_path, 6, edge_check_version=1)
        stars1 = []
        for point in refined_points1:
            x, y = point
            stars1.append({"x": x, "y": y})

        # 星点识别
        result1 = self.star_identification(stars1)
        if result1 is None:
            print("第一张图像识别失败")
            return

        triangle_info1, centroid1 = result1
        print("第一张图像三角形信息:", triangle_info1)

        # 处理第二张图像（使用版本2的边缘检查）
        print("\n处理第二张图像...")
        refined_points2, image2 = self.detect_stars(image2_path, 10, edge_check_version=2)
        stars2 = []
        for point in refined_points2:
            x, y = point
            stars2.append({"x": x, "y": y})

        # 星点识别
        result2 = self.star_identification(stars2)
        if result2 is None:
            print("第二张图像识别失败")
            return

        triangle_info2, centroid2 = result2
        print("第二张图像三角形信息:", triangle_info2)

        # 比较三角形
        if self.compare_sides(triangle_info1, triangle_info2, rel_tol=1.0):
            print("\n三角形匹配成功!")

            # 计算质心偏移
            delta_x = centroid1[0] - centroid2[0]
            delta_y = centroid1[1] - centroid2[1]

            print("质心偏移: Δx=" + str(round(delta_x, 2)) + " pixels, Δy=" + str(round(delta_y, 2)) + " pixels")

            # 计算时间间隔
            delta_time = 1 / self.frame_rate

            # 计算角速度
            omega_x = self.calculate_angular_velocity(delta_x, delta_time, self.horizontal_fov, self.image_width)
            omega_y = self.calculate_angular_velocity(delta_y, delta_time, self.vertical_fov, self.image_height)

            print("角速度 - X方向: " + str(round(omega_x, 4)) + " 度/秒, Y方向: " + str(round(omega_y, 4)) + " 度/秒")
            print("合成角速度: " + str(round(math.sqrt(omega_x ** 2 + omega_y ** 2), 4)) + " 度/秒")

            # 可视化匹配的三角形
            self.visualize_matched_triangles(
                image1, triangle_info1["coordinates"],
                image2, triangle_info2["coordinates"],
                centroid1, centroid2,
                delta_x, delta_y
            )

        else:
            print("\n三角形不匹配")



if __name__ == "__main__":
    tracker = StarTracker()

    # 替换为您的图像路径
    image1_path = "C:/Users/14553\Desktop/1111.png"
    image2_path = "C:/Users/14553\Desktop/2222.png"
    try:
        tracker.process_images(image1_path, image2_path)
    except Exception as e:
        print("处理过程中出错: " + str(e))