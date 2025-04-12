import os
import numpy as np
import cv2
import open3d as o3d
import logging
import subprocess
import tempfile
import shutil
import json
import trimesh
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class SFMMVSProcessor:
    def __init__(self, work_dir=None):
        """
        Khởi tạo bộ xử lý SfM và Nerfstudio (Instant-NGP)
        Args:
            work_dir: Thư mục làm việc (nếu None sẽ tạo thư mục tạm)
        """
        if work_dir is None:
            self.work_dir = tempfile.mkdtemp()
        else:
            self.work_dir = work_dir
            os.makedirs(self.work_dir, exist_ok=True)

        # Tạo các thư mục con
        self.images_dir = os.path.join(self.work_dir, "images")
        self.sparse_dir = os.path.join(self.work_dir, "sparse")
        self.nerfstudio_data_dir = os.path.join(
            self.work_dir, "nerfstudio_data")
        self.nerfstudio_output_dir = os.path.join(
            self.work_dir, "nerfstudio_output")
        self.mesh_output_dir = os.path.join(self.work_dir, "mesh_output")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.nerfstudio_data_dir, exist_ok=True)
        os.makedirs(self.nerfstudio_output_dir, exist_ok=True)
        os.makedirs(self.mesh_output_dir, exist_ok=True)

        logger.info(f"Initialized SFM-Nerfstudio processor in {self.work_dir}")

    def process_views(self, views, camera_poses=None, max_iterations=10000, resolution=512,
                      export_method='poisson', simplify_mesh_in_processing=False,
                      simplify_target_faces=100000, remove_small_components_threshold=0.05):
        """
        Xử lý các view để tạo mesh 3D chất lượng cao sử dụng SfM và Nerfstudio (Instant-NGP)
        Args:
            views: List of novel view numpy arrays
            camera_poses: List of camera poses (nếu có từ Zero123++)
            max_iterations: Số lần lặp tối đa cho quá trình huấn luyện Instant-NGP
            resolution: Độ phân giải của mesh khi trích xuất
            export_method: Phương thức export mesh từ Nerfstudio ('poisson' hoặc 'textured_mesh')
            simplify_mesh_in_processing: Có đơn giản hóa mesh ngay trong quá trình xử lý hay không
            simplify_target_faces: Số mặt mục tiêu nếu simplify_mesh_in_processing=True
            remove_small_components_threshold: Tỷ lệ phần trăm diện tích/số mặt tối thiểu so với thành phần lớn nhất
        Returns:
            Trimesh mesh object
        """
        try:
            # 1. Lưu các view vào thư mục images
            image_paths = self._save_views(views)

            # 2. Chạy COLMAP SfM nếu không có camera poses
            if camera_poses is None:
                logger.info("Running COLMAP SfM to estimate camera poses...")
                self._run_colmap_sfm(image_paths)
                camera_poses = self._load_camera_poses()

            # 3. Chuyển đổi dữ liệu COLMAP sang định dạng Nerfstudio
            logger.info("Converting COLMAP data to Nerfstudio format...")
            self._convert_colmap_to_nerfstudio()

            # 4. Huấn luyện Instant-NGP thông qua Nerfstudio
            logger.info("Training Instant-NGP model with Nerfstudio...")
            model_path = self._train_instant_ngp(max_iterations=max_iterations)

            # 5. Trích xuất mesh từ mô hình NeRF đã huấn luyện
            logger.info("Extracting mesh from trained NeRF model...")
            mesh_path = self._extract_mesh_from_nerf(
                model_path, resolution=resolution, export_method=export_method)

            # 6. Đọc và xử lý mesh
            logger.info("Processing extracted mesh...")
            mesh = self._process_mesh(
                mesh_path,
                simplify_mesh=simplify_mesh_in_processing,
                simplify_target_faces=simplify_target_faces,
                remove_small_components_threshold=remove_small_components_threshold
            )

            return mesh

        except Exception as e:
            logger.error(f"Error in SFM-Nerfstudio processing: {e}")
            raise

    def _save_views(self, views):
        """
        Lưu các view vào thư mục images
        """
        image_paths = []
        for i, view in enumerate(views):
            # Chuyển đổi sang định dạng BGR cho OpenCV
            if len(view.shape) == 3 and view.shape[2] == 3:
                view_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            else:
                view_bgr = view

            # Lưu ảnh
            image_path = os.path.join(self.images_dir, f"view_{i:04d}.jpg")
            cv2.imwrite(image_path, view_bgr)
            image_paths.append(image_path)

        logger.info(f"Saved {len(image_paths)} views to {self.images_dir}")
        return image_paths

    def _run_colmap_sfm(self, image_paths):
        """
        Chạy COLMAP SfM để ước tính camera poses
        """
        # Tạo file database
        db_path = os.path.join(self.work_dir, "database.db")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", self.images_dir,
            "--ImageReader.single_camera", "1"
        ], check=True)

        # Chạy feature matching
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1"
        ], check=True)

        # Chạy SfM
        subprocess.run([
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", self.images_dir,
            "--output_path", self.sparse_dir,
            "--Mapper.init_min_num_inliers", "15",
            "--Mapper.multiple_models", "1",
            "--Mapper.extract_colors", "1"
        ], check=True)

        logger.info("COLMAP SfM completed successfully")

    def _load_camera_poses(self):
        """
        Đọc camera poses từ kết quả SfM
        """
        # Đọc file cameras.txt từ thư mục sparse/0
        cameras_file = os.path.join(self.sparse_dir, "0", "cameras.txt")
        images_file = os.path.join(self.sparse_dir, "0", "images.txt")

        # Parse camera poses
        camera_poses = []
        with open(images_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue

                # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                parts = line.strip().split()
                if len(parts) >= 10:
                    # Chuyển đổi từ quaternion sang ma trận quay
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])

                    # Tạo ma trận quay từ quaternion
                    R = self._quaternion_to_rotation(qw, qx, qy, qz)

                    # Tạo ma trận pose
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = [tx, ty, tz]

                    camera_poses.append(pose)

        return camera_poses

    def _quaternion_to_rotation(self, qw, qx, qy, qz):
        """
        Chuyển đổi từ quaternion sang ma trận quay
        """
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        R[0, 1] = 2*qx*qy - 2*qz*qw
        R[0, 2] = 2*qx*qz + 2*qy*qw
        R[1, 0] = 2*qx*qy + 2*qz*qw
        R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        R[1, 2] = 2*qy*qz - 2*qx*qw
        R[2, 0] = 2*qx*qz - 2*qy*qw
        R[2, 1] = 2*qy*qz + 2*qx*qw
        R[2, 2] = 1 - 2*qx*qx - 2*qy*qy

        return R

    def _convert_colmap_to_nerfstudio(self):
        """
        Chuyển đổi dữ liệu COLMAP sang định dạng Nerfstudio sử dụng ns-process-data
        """
        # Xác định đường dẫn tới thư mục COLMAP
        colmap_dir = self.work_dir
        output_dir = self.nerfstudio_data_dir

        # Gọi ns-process-data để chuyển đổi dữ liệu COLMAP sang định dạng Nerfstudio
        command = [
            "ns-process-data", "colmap",
            "--data", colmap_dir,
            "--output-dir", output_dir,
            "--skip-colmap",  # Bỏ qua bước chạy COLMAP vì đã có kết quả SfM
            "--verbose"
        ]

        subprocess.run(command, check=True)

        logger.info(
            f"Converted COLMAP data to Nerfstudio format in {output_dir}")

        # Kiểm tra xem transforms.json đã được tạo ra hay chưa
        transforms_path = os.path.join(output_dir, "transforms.json")
        if not os.path.exists(transforms_path):
            logger.error("transforms.json not found after conversion")
            raise FileNotFoundError(
                f"transforms.json not found at {transforms_path}")

        logger.info("Successfully created Nerfstudio dataset from COLMAP data")

    def _train_instant_ngp(self, max_iterations=10000):
        """
        Huấn luyện mô hình Instant-NGP sử dụng Nerfstudio

        Args:
            max_iterations: Số lần lặp tối đa cho quá trình huấn luyện
                            (Giảm xuống cho phù hợp với môi trường giới hạn như Kaggle)

        Returns:
            Path to the trained model config
        """
        # Xác định đường dẫn dữ liệu và output
        data_path = self.nerfstudio_data_dir
        output_path = self.nerfstudio_output_dir
        method = "instant-ngp"  # Sử dụng instant-ngp thay vì nerfacto cho hiệu quả tốt hơn
        experiment_name = "model"

        # Tạo lệnh huấn luyện mô hình
        command = [
            "ns-train", method,
            "--data", data_path,
            "--output-dir", output_path,
            "--experiment-name", experiment_name,
            "--max-num-iterations", str(max_iterations),
            "--pipeline.model.background-color", "white",
            "--viewer.quit-on-train-completion",  # Tự động thoát sau khi huấn luyện hoàn tất
            # Lưu lại checkpoint sau mỗi 10% tổng số bước
            "--steps-per-save", str(max_iterations // 10),
            "--pipeline.model.cone-angle", "0.0",  # Tối ưu góc nhìn cho mô hình InstantNGP
            "--pipeline.datamanager.train-num-rays-per-batch", "4096",  # Tối ưu hóa memory usage
            "--pipeline.model.render-step-size", "0.001"  # Tăng chất lượng rendering
        ]

        # Chạy huấn luyện
        logger.info(
            f"Starting Instant-NGP training for {max_iterations} iterations...")
        subprocess.run(command, check=True)

        # Xác định đường dẫn đến thư mục mô hình đã huấn luyện
        # Nerfstudio tạo ra một thư mục với tên dựa trên timestamp
        # Tìm thư mục mới nhất trong output_path/method/experiment_name
        model_dir = os.path.join(output_path, method, experiment_name)
        subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, d))]

        if not subdirs:
            logger.error("No model directory found after training")
            raise FileNotFoundError(f"No model directory found in {model_dir}")

        # Sắp xếp theo thời gian sửa đổi để lấy thư mục mới nhất
        latest_model_dir = max(subdirs, key=os.path.getmtime)
        model_config_path = os.path.join(latest_model_dir, "config.yml")

        if not os.path.exists(model_config_path):
            logger.error(f"Model config file not found at {model_config_path}")
            raise FileNotFoundError(
                f"Model config file not found at {model_config_path}")

        logger.info(
            f"Instant-NGP training completed. Model saved to {latest_model_dir}")

        return model_config_path

    def _extract_mesh_from_nerf(self, model_config_path, resolution=512, export_method='poisson'):
        """
        Trích xuất mesh từ mô hình NeRF đã huấn luyện

        Args:
            model_config_path: Đường dẫn đến file config.yml của mô hình
            resolution: Độ phân giải của mesh (cao hơn = chi tiết hơn nhưng tốn tài nguyên hơn)
            export_method: Phương thức export mesh từ Nerfstudio ('poisson' hoặc 'textured_mesh')

        Returns:
            Path to the extracted mesh file
        """
        # Xác định đường dẫn output cho mesh
        mesh_output_path = os.path.join(self.mesh_output_dir, "mesh.ply")

        # Thiết lập lệnh cơ bản
        base_cmd = [
            "ns-export",
            "--load-config", model_config_path,
            "--output-path", mesh_output_path,
            "--resolution", str(resolution),
        ]

        # Tùy chỉnh tham số dựa trên phương thức export
        if export_method == 'poisson':
            # Phương pháp Poisson Surface Reconstruction - tốt cho bề mặt mượt mà
            command = base_cmd + [
                "poisson",
                "--normal-method", "open3d",  # Sử dụng Open3D để ước tính normals chính xác hơn
                "--color-interpolation", "barycentric",  # Tối ưu hóa chất lượng màu sắc
                "--num-points", "500000"  # Số lượng điểm để ước tính, cao hơn = tốt hơn nhưng chậm hơn
            ]
            logger.info(
                f"Using Poisson mesh extraction method with resolution {resolution}")

        elif export_method == 'textured_mesh':
            # Phương pháp Textured Mesh - tốt cho bề mặt với texture chất lượng cao
            command = base_cmd + [
                "textured-mesh",
                "--use-vertex-colors",  # Sử dụng vertex colors
                "--unwrap-method", "smart",  # Lựa chọn phương pháp unwrap thông minh
                "--create-texture-atlas",  # Tạo texture atlas cho mesh
                # Số lượng mặt ước lượng
                "--target-num-faces", str(resolution * 5)
            ]
            logger.info(
                f"Using Textured Mesh extraction method with resolution {resolution}")

        elif export_method == 'marching_cubes':
            # Phương pháp Marching Cubes - tốt cho chi tiết geometry
            command = base_cmd + [
                "marching-cubes",
                "--isosurface-threshold", "50.0",  # Ngưỡng isosurface
                "--feature-grid-resolution", str(resolution),
                "--normal-weighting-mode", "geometric"  # Sử dụng normal weighting mode
            ]
            logger.info(
                f"Using Marching Cubes extraction method with resolution {resolution}")

        else:
            logger.warning(
                f"Unknown export method: {export_method}, falling back to poisson")
            command = base_cmd + ["poisson"]

        # Thực hiện lệnh export mesh
        try:
            logger.info(f"Extracting mesh with command: {' '.join(command)}")
            subprocess.run(command, check=True)

            if not os.path.exists(mesh_output_path):
                logger.error(
                    f"Mesh extraction failed: {mesh_output_path} not found")
                raise FileNotFoundError(
                    f"Extracted mesh file not found at {mesh_output_path}")

            # Kiểm tra kích thước file
            file_size = os.path.getsize(mesh_output_path)
            # Nếu file quá nhỏ (dưới 1KB), có thể có vấn đề
            if file_size < 1000:
                logger.warning(
                    f"Extracted mesh file is suspiciously small ({file_size} bytes)")
            else:
                logger.info(
                    f"Mesh extracted successfully to {mesh_output_path} ({file_size/1024:.2f} KB)")

            return mesh_output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Mesh extraction failed with error: {e}")

            # Nếu export_method ban đầu thất bại, thử phương pháp khác
            if export_method != 'poisson':
                logger.info("Trying fallback export method: poisson")
                return self._extract_mesh_from_nerf(model_config_path, resolution, 'poisson')
            raise

    def _process_mesh(self, mesh_path, simplify_mesh=False, simplify_target_faces=100000, remove_small_components_threshold=0.05):
        """
        Đọc và xử lý mesh: fill holes, remove duplicate vertices

        Args:
            mesh_path: Đường dẫn đến file mesh (.ply hoặc .obj)
            simplify_mesh: Có đơn giản hóa mesh ngay trong quá trình xử lý hay không
            simplify_target_faces: Số mặt mục tiêu nếu simplify_mesh=True
            remove_small_components_threshold: Tỷ lệ phần trăm diện tích/số mặt tối thiểu so với thành phần lớn nhất

        Returns:
            Đối tượng trimesh.Trimesh đã được xử lý
        """
        # Đọc mesh bằng trimesh
        logger.info(f"Loading mesh from {mesh_path}")
        try:
            mesh = trimesh.load_mesh(mesh_path)
        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            raise

        # Kiểm tra xem mesh có phải là instance của Trimesh không
        if not isinstance(mesh, trimesh.Trimesh):
            logger.warning(
                "Loaded mesh is not a triangle mesh, attempting to convert...")
            try:
                scene = mesh
                # Nếu mesh là Scene, kết hợp tất cả các geometry
                meshes = []
                for g in scene.geometry.values():
                    if hasattr(g, 'vertices') and hasattr(g, 'faces'):
                        # Nếu geometry có vertices và faces, chuyển đổi thành Trimesh
                        try:
                            tmesh = trimesh.Trimesh(
                                vertices=g.vertices, faces=g.faces)
                            # Giữ lại thuộc tính visual nếu có
                            if hasattr(g, 'visual') and g.visual is not None:
                                tmesh.visual = g.visual
                            meshes.append(tmesh)
                        except Exception as e:
                            logger.warning(
                                f"Could not convert geometry to Trimesh: {e}")

                if meshes:
                    # Nối các mesh lại với nhau
                    mesh = trimesh.util.concatenate(meshes)
                    logger.info(
                        f"Converted scene to Trimesh with {len(mesh.faces)} faces")
                else:
                    raise ValueError("No valid geometries found in mesh scene")
            except Exception as e:
                logger.error(f"Error converting mesh: {e}")
                raise

        # Loại bỏ các điểm trùng lặp
        initial_vertices = len(mesh.vertices)
        mesh.remove_duplicate_vertices()
        logger.info(
            f"Removed duplicate vertices: {initial_vertices} -> {len(mesh.vertices)}")

        # Loại bỏ các mặt suy biến
        initial_faces = len(mesh.faces)
        mesh.remove_degenerate_faces()
        if initial_faces > len(mesh.faces):
            logger.info(
                f"Removed degenerate faces: {initial_faces} -> {len(mesh.faces)}")

        # Fill các lỗ nhỏ nếu mesh không watertight
        if mesh.is_watertight:
            logger.info("Mesh is already watertight, no need to fill holes")
        else:
            logger.info("Filling holes in mesh...")
            try:
                # Thử sử dụng pymeshfix để lấp các lỗ
                try:
                    import pymeshfix
                    logger.info("Using pymeshfix to repair mesh")
                    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                    meshfix.repair()
                    vertices, faces = meshfix.get_mesh()
                    fixed_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=faces)

                    # Chỉ cập nhật mesh nếu số mặt tương đương hoặc nhiều hơn
                    if len(fixed_mesh.faces) >= len(mesh.faces) * 0.9:
                        # Lưu các thuộc tính quan trọng từ mesh gốc
                        original_vertex_colors = None
                        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                            original_vertex_colors = mesh.visual.vertex_colors

                        # Cập nhật mesh với phiên bản đã sửa
                        mesh = fixed_mesh
                        logger.info(
                            f"Filled holes using pymeshfix: {len(mesh.faces)} faces")

                        # Khôi phục vertex colors nếu có thể
                        # Chú ý: Việc mapping vertex colors sau khi sửa mesh là phức tạp và có thể không chính xác hoàn toàn
                        # vì cấu trúc đỉnh có thể thay đổi. Đây chỉ là phương pháp đơn giản.
                        if original_vertex_colors is not None:
                            logger.info("Attempting to restore vertex colors")
                            # Gán màu cơ bản cho tất cả đỉnh (ví dụ: màu trắng)
                            default_color = np.array(
                                [255, 255, 255, 255], dtype=np.uint8)
                            mesh.visual.vertex_colors = np.tile(
                                default_color, (len(mesh.vertices), 1))

                            # Thử khôi phục màu cho đỉnh trùng lặp (chỉ là ước lượng đơn giản)
                            min_len = min(
                                len(original_vertex_colors), len(mesh.vertices))
                            mesh.visual.vertex_colors[:min_len] = original_vertex_colors[:min_len]
                    else:
                        logger.warning(
                            "pymeshfix resulted in too few faces, keeping original mesh")
                except ImportError:
                    logger.warning(
                        "pymeshfix not available - install with 'pip install pymeshfix' for better hole filling")
                    # Sử dụng phương pháp tích hợp của trimesh nếu không có pymeshfix
                    mesh.fill_holes()
            except Exception as e:
                logger.warning(f"Error during hole filling: {e}")
                logger.info("Attempting basic hole filling with trimesh")
                try:
                    mesh.fill_holes()
                except Exception as e2:
                    logger.warning(f"Basic hole filling also failed: {e2}")

        # Bảo đảm mesh có normals
        logger.info("Fixing mesh normals...")
        mesh.fix_normals()

        # Loại bỏ các thành phần không kết nối
        logger.info("Processing disconnected components...")
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            logger.info(f"Mesh has {len(components)} disconnected components")

            # Tính toán diện tích/số faces của từng thành phần
            metrics = [(i, comp.area, len(comp.faces))
                       for i, comp in enumerate(components)]
            # Sắp xếp theo diện tích giảm dần
            metrics.sort(key=lambda x: x[1], reverse=True)

            # Thành phần lớn nhất
            largest_area = metrics[0][1]

            # Lọc các thành phần đủ lớn để giữ lại (dựa vào ngưỡng)
            kept_indices = [i for i, area, _ in metrics if area >=
                            largest_area * remove_small_components_threshold]
            kept_components = [components[i] for i in kept_indices]

            logger.info(
                f"Keeping {len(kept_components)}/{len(components)} components with threshold {remove_small_components_threshold}")
            logger.info(
                f"Largest component area: {largest_area:.2f}, smallest kept component area: {metrics[kept_indices[-1]][1]:.2f}")

            if len(kept_components) > 0:
                # Nối các thành phần được giữ lại
                mesh = trimesh.util.concatenate(kept_components)
                logger.info(f"Combined mesh has {len(mesh.faces)} faces")
            else:
                # Nếu không có thành phần nào đạt ngưỡng, giữ lại thành phần lớn nhất
                logger.warning(
                    "No components met the threshold, keeping only the largest component")
                mesh = components[metrics[0][0]]
                logger.info(f"Largest component has {len(mesh.faces)} faces")

        # Đơn giản hóa mesh nếu được yêu cầu và cần thiết
        if simplify_mesh and len(mesh.faces) > simplify_target_faces:
            logger.info(
                f"Simplifying mesh from {len(mesh.faces)} to {simplify_target_faces} faces...")
            try:
                original_vertex_colors = None
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    original_vertex_colors = mesh.visual.vertex_colors.copy()

                # Thực hiện đơn giản hóa
                simplified_mesh = mesh.simplify_quadratic_decimation(
                    simplify_target_faces)

                # Kiểm tra kết quả
                if len(simplified_mesh.faces) < len(mesh.faces) * 0.1:
                    logger.warning(
                        f"Simplification resulted in too few faces ({len(simplified_mesh.faces)}), keeping original mesh")
                else:
                    # Đơn giản hóa thành công
                    mesh = simplified_mesh
                    logger.info(f"Mesh simplified to {len(mesh.faces)} faces")

                    # Cập nhật visual properties nếu bị mất
                    if original_vertex_colors is not None and (
                        not hasattr(mesh.visual, 'vertex_colors') or
                        mesh.visual.vertex_colors is None or
                        np.all(mesh.visual.vertex_colors == np.array(
                            [255, 255, 255, 255], dtype=np.uint8))
                    ):
                        logger.info(
                            "Restoring vertex colors after simplification")
                        # Tạo một visual mới với màu mặc định
                        default_color = np.array(
                            [255, 255, 255, 255], dtype=np.uint8)
                        mesh.visual.vertex_colors = np.tile(
                            default_color, (len(mesh.vertices), 1))
                        # Chú ý: Đây chỉ là cách tiếp cận đơn giản. Việc giữ nguyên vertex colors
                        # sau đơn giản hóa là phức tạp và cần interpolation/resampling phức tạp hơn.
            except Exception as e:
                logger.error(f"Error during mesh simplification: {e}")
                logger.warning("Keeping original unsimplified mesh")

        # Lưu mesh đã xử lý
        processed_mesh_path = os.path.join(
            self.mesh_output_dir, "processed_mesh.ply")
        logger.info(f"Saving processed mesh to {processed_mesh_path}")
        mesh.export(processed_mesh_path)
        logger.info(
            f"Processed mesh saved with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

        return mesh

    def cleanup(self):
        """
        Dọn dẹp thư mục tạm
        """
        if os.path.exists(self.work_dir) and self.work_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.work_dir)
            logger.info(f"Cleaned up temporary directory: {self.work_dir}")
