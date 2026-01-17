import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


class StarTracker:
    def __init__(self):
        self.frame_rate = 4
        self.horizontal_fov = 29.56
        self.vertical_fov = 24.9
        self.image_width = None
        self.image_height = None

    def calculate_centroid(self, x1, y1, x2, y2, x3, y3):
        """计算三角形的几何中心（质心）坐标"""
        centroid_x = (x1 + x2 + x3) / 3
        centroid_y = (y1 + y2 + y3) / 3
        return centroid_x, centroid_y

    def is_within_range(self, point, selected_points, radius=1):
        """检查目标点是否与已选点距离过近"""
        for p in selected_points:
            if abs(point[0] - p[0]) <= radius and abs(point[1] - p[1]) <= radius:
                return True
        return False

    def is_near_edge_1(self, x, y, image_shape, margin_ratio=0.2):
        """边缘检查版本1"""
        height, width = image_shape
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        return (x < 2 * margin_x or x > width - margin_x or
                y < margin_y or y > height - margin_y)

    def is_near_edge_2(self, x, y, image_shape, margin_ratio=0.2):
        """边缘检查版本2"""
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
        max_brightness = np.max(region)
        if max_brightness != 0:
            weights = weights / max_brightness
        weighted_x = np.sum(weights * x_coords) / np.sum(weights)
        weighted_y = np.sum(weights * y_coords) / np.sum(weights)
        weighted_x += region_coordinates[1]
        weighted_y += region_coordinates[0]
        return weighted_x, weighted_y

    def detect_stars(self, image_path, num_stars=6, edge_check_version=1):
        """检测图像中的星点"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("无法读取图像: " + image_path)
        if self.image_width is None or self.image_height is None:
            self.image_height, self.image_width = image.shape
            print(f"图像尺寸: {self.image_width}×{self.image_height}")
        blur1 = cv2.GaussianBlur(image, (5, 5), 0)
        blur2 = cv2.GaussianBlur(image, (15, 15), 0)
        dog = cv2.subtract(blur1, blur2)
        selected_points = []
        dog_flat = dog.flatten().astype(np.float64)
        while len(selected_points) < num_stars:
            max_idx = np.argmax(dog_flat)
            y, x = np.unravel_index(max_idx, dog.shape)
            is_near_edge = self.is_near_edge_1 if edge_check_version == 1 else self.is_near_edge_2
            if is_near_edge(x, y, image.shape):
                dog_flat[max_idx] = -1e10
                continue
            if not self.is_within_range((x, y), selected_points):
                selected_points.append((x, y))
            dog_flat[max_idx] = -1e10
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

    def is_valid_triangle(self, triangle):
        """检查三角形是否满足约束条件"""
        sides = [
            self.calculate_distance(triangle[0], triangle[1]),
            self.calculate_distance(triangle[1], triangle[2]),
            self.calculate_distance(triangle[0], triangle[2]),
        ]
        min_side = min(sides)
        if min_side <= 40:
            return False
        angles = self.calculate_angles(*sides)
        min_angle = min(angles)
        if min_angle <= 20:
            return False
        return True

    def find_reference_triangle(self, stars):
        """从星点中找到一个有效的参考三角形"""
        n_stars = len(stars)
        for i in range(n_stars):
            for j in range(i + 1, n_stars):
                for k in range(j + 1, n_stars):
                    triangle = [stars[i], stars[j], stars[k]]
                    if self.is_valid_triangle(triangle):
                        sides = [
                            self.calculate_distance(triangle[0], triangle[1]),
                            self.calculate_distance(triangle[1], triangle[2]),
                            self.calculate_distance(triangle[0], triangle[2]),
                        ]
                        x1, y1 = triangle[0]['x'], triangle[0]['y']
                        x2, y2 = triangle[1]['x'], triangle[1]['y']
                        x3, y3 = triangle[2]['x'], triangle[2]['y']
                        centroid = self.calculate_centroid(x1, y1, x2, y2, x3, y3)
                        triangle_info = {
                            "sides": sides,
                            "coordinates": [[x1, y1], [x2, y2], [x3, y3]],
                            "stars": triangle,
                            "centroid": centroid
                        }
                        return triangle_info, centroid
        return None

    def find_all_valid_triangles(self, stars, max_triangles=50):
        """从星点中找出所有有效的三角形"""
        valid_triangles = []
        n_stars = len(stars)
        for i in range(n_stars):
            for j in range(i + 1, n_stars):
                for k in range(j + 1, n_stars):
                    triangle = [stars[i], stars[j], stars[k]]
                    if self.is_valid_triangle(triangle):
                        sides = [
                            self.calculate_distance(triangle[0], triangle[1]),
                            self.calculate_distance(triangle[1], triangle[2]),
                            self.calculate_distance(triangle[0], triangle[2]),
                        ]
                        x1, y1 = triangle[0]['x'], triangle[0]['y']
                        x2, y2 = triangle[1]['x'], triangle[1]['y']
                        x3, y3 = triangle[2]['x'], triangle[2]['y']
                        centroid = self.calculate_centroid(x1, y1, x2, y2, x3, y3)
                        triangle_info = {
                            "sides": sides,
                            "coordinates": [[x1, y1], [x2, y2], [x3, y3]],
                            "stars": triangle,
                            "centroid": centroid
                        }
                        valid_triangles.append(triangle_info)
                        if len(valid_triangles) >= max_triangles:
                            return valid_triangles
        return valid_triangles

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

    def calculate_sides_difference(self, triangle1, triangle2):
        """计算两个三角形边长之间的差异度（越小越相似）"""
        sides1 = sorted(triangle1.get("sides", []))
        sides2 = sorted(triangle2.get("sides", []))
        if len(sides1) != 3 or len(sides2) != 3:
            return float('inf')
        diff = 0
        for i in range(3):
            if sides2[i] == 0:
                return float('inf')
            diff += abs(sides1[i] - sides2[i]) / sides2[i]
        return diff / 3

    def find_best_match(self, triangle1, triangles2):
        """在第二张图像的所有三角形中寻找与triangle1最匹配的三角形"""
        best_match = None
        best_difference = float('inf')
        for triangle2 in triangles2:
            if self.compare_sides(triangle1, triangle2, rel_tol=0.1):
                difference = self.calculate_sides_difference(triangle1, triangle2)
                if difference < best_difference:
                    best_difference = difference
                    best_match = triangle2
        return best_match, best_difference

    def calculate_angular_velocity(self, delta_pixels, delta_time, fov, image_dimension):
        """计算角速度"""
        angular_velocity = (delta_pixels / delta_time) * (fov / image_dimension)
        return angular_velocity

    def visualize_matched_triangles(self, image1, triangle1_coords, image2, triangle2_coords,
                                    centroid1, centroid2, delta_x, delta_y, match_info=""):
        """可视化匹配的两个三角形"""
        img1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        triangle_color = (0, 255, 0)
        centroid_color = (255, 0, 0)
        point_color = (0, 0, 255)

        for i in range(len(triangle1_coords)):
            coord = triangle1_coords[i]
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(img1_color, (x, y), 6, point_color, -1)
            cv2.putText(img1_color, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)

        for i in range(3):
            start_point = (int(triangle1_coords[i][0]), int(triangle1_coords[i][1]))
            end_point = (int(triangle1_coords[(i + 1) % 3][0]), int(triangle1_coords[(i + 1) % 3][1]))
            cv2.line(img1_color, start_point, end_point, triangle_color, 2)

        cx1, cy1 = int(centroid1[0]), int(centroid1[1])
        cv2.circle(img1_color, (cx1, cy1), 8, centroid_color, -1)
        cv2.putText(img1_color, "Centroid", (cx1 + 10, cy1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, centroid_color, 2)

        for i in range(len(triangle2_coords)):
            coord = triangle2_coords[i]
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(img2_color, (x, y), 6, point_color, -1)
            cv2.putText(img2_color, str(i + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)

        for i in range(3):
            start_point = (int(triangle2_coords[i][0]), int(triangle2_coords[i][1]))
            end_point = (int(triangle2_coords[(i + 1) % 3][0]), int(triangle2_coords[(i + 1) % 3][1]))
            cv2.line(img2_color, start_point, end_point, triangle_color, 2)

        cx2, cy2 = int(centroid2[0]), int(centroid2[1])
        cv2.circle(img2_color, (cx2, cy2), 8, centroid_color, -1)
        cv2.putText(img2_color, "Centroid", (cx2 + 10, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, centroid_color, 2)

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.title("First Image - Reference Triangle", fontsize=14, fontweight='bold')
        plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Second Image - Matched Triangle", fontsize=14, fontweight='bold')
        plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.suptitle(f"Matched Triangles for Angular Velocity Calculation\n"
                     f"Image Size: {self.image_width}×{self.image_height}\n"
                     f"Centroid Shift: Δx={round(delta_x, 2)}px, Δy={round(delta_y, 2)}px\n"
                     f"{match_info}",
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.show()

        cv2.imwrite('matched_triangle_image1.jpg', img1_color)
        cv2.imwrite('matched_triangle_image2.jpg', img2_color)
        print("匹配三角形图像已保存为 'matched_triangle_image1.jpg' 和 'matched_triangle_image2.jpg'")

    def process_images(self, image1_path, image2_path):
        """处理两张图像并计算角速度"""
        print("=== 星点追踪系统 - 多三角形匹配版 ===")
        
        # 处理第一张图像（获取参考三角形）
        print("\n1. 处理第一张图像...")
        refined_points1, image1 = self.detect_stars(image1_path, 6, edge_check_version=1)
        stars1 = [{"x": x, "y": y} for x, y in refined_points1]

        result1 = self.find_reference_triangle(stars1)
        if result1 is None:
            print("第一张图像中未找到有效三角形")
            return
        triangle_info1, centroid1 = result1
        print(f"参考三角形边长: {[round(s, 2) for s in triangle_info1['sides']]}")

        # 处理第二张图像（检测更多星点）
        print("\n2. 处理第二张图像...")
        refined_points2, image2 = self.detect_stars(image2_path, 15, edge_check_version=2)
        stars2 = [{"x": x, "y": y} for x, y in refined_points2]
        print(f"第二张图像检测到 {len(stars2)} 个星点")

        # 生成所有可能的三角形
        print("\n3. 生成所有可能的三角形...")
        all_triangles2 = self.find_all_valid_triangles(stars2, max_triangles=100)
        print(f"第二张图像中找到 {len(all_triangles2)} 个有效三角形")

        if len(all_triangles2) == 0:
            print("第二张图像中没有找到有效三角形")
            return

        # 寻找最佳匹配
        print("\n4. 尝试匹配三角形...")
        best_match, best_difference = self.find_best_match(triangle_info1, all_triangles2)
        
        # 容错机制：如果严格匹配失败，尝试宽松匹配
        if best_match is None:
            print("未找到匹配的三角形，尝试放宽容差匹配...")
            for triangle2 in all_triangles2:
                if self.compare_sides(triangle_info1, triangle2, rel_tol=0.15):
                    best_match = triangle2
                    break
            
            if best_match is None:
                print("匹配失败")
                return

        # 计算质心偏移
        centroid2 = best_match.get("centroid")
        delta_x = centroid1[0] - centroid2[0]
        delta_y = centroid1[1] - centroid2[1]

        print(f"\n5. 匹配成功!")
        print(f"   参考三角形边长: {[round(s, 2) for s in triangle_info1['sides']]}")
        print(f"   匹配三角形边长: {[round(s, 2) for s in best_match['sides']]}")
        print(f"   边长差异度: {best_difference:.4f}")
        print(f"   质心偏移: Δx={round(delta_x, 2)} pixels, Δy={round(delta_y, 2)} pixels")

        # 计算角速度
        delta_time = 1 / self.frame_rate
        omega_x = self.calculate_angular_velocity(delta_x, delta_time, self.horizontal_fov, self.image_width)
        omega_y = self.calculate_angular_velocity(delta_y, delta_time, self.vertical_fov, self.image_height)

        print(f"\n6. 角速度计算结果:")
        print(f"   X方向角速度: {round(omega_x, 4)} 度/秒")
        print(f"   Y方向角速度: {round(omega_y, 4)} 度/秒")
        print(f"   合成角速度: {round(math.sqrt(omega_x ** 2 + omega_y ** 2), 4)} 度/秒")

        # 可视化
        match_info = f"从 {len(all_triangles2)} 个三角形中找到最佳匹配"
        self.visualize_matched_triangles(
            image1, triangle_info1["coordinates"],
            image2, best_match["coordinates"],
            centroid1, centroid2,
            delta_x, delta_y,
            match_info
        )


if __name__ == "__main__":
    tracker = StarTracker()
    
    # 修改为你的图像路径
    image1_path = "C:/Users/Lenovo/Desktop/star/77.png"
    image2_path = "C:/Users/Lenovo/Desktop/star/77.png"
    
    try:
        tracker.process_images(image1_path, image2_path)
    except Exception as e:
        print("处理过程中出错: " + str(e))
