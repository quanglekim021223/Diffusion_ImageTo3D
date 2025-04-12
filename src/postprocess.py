import numpy as np
import trimesh
import pyrender
from pathlib import Path
import open3d as o3d
import cv2
import os
import logging
import shutil
import subprocess
from PIL import Image
from .sfm_mvs import SFMMVSProcessor

# Configure logging
logger = logging.getLogger(__name__)


class ModelPostProcessor:
    def __init__(self):
        """
        Khởi tạo ModelPostProcessor để tạo mesh 3D chất lượng cao từ novel views
        """
        self.device = "cuda" if os.environ.get(
            "CUDA_VISIBLE_DEVICES") is not None else "cpu"
        logger.info(
            f"Initializing ModelPostProcessor with device: {self.device}")
        self.sfm_mvs_processor = SFMMVSProcessor()
        self.mesh = None
        self.mesh_path = None

    def create_mesh_from_views(self, novel_views, output_path=None, max_iterations=5000, mesh_resolution=512,
                               export_method='poisson', simplify_mesh_in_processing=False,
                               simplify_target_faces=100000, remove_small_components_threshold=0.05):
        """
        Tạo mesh 3D chất lượng cao từ các novel views sử dụng pipeline COLMAP SfM + Nerfstudio (Instant-NGP)

        Args:
            novel_views: Danh sách các novel view tensors hoặc PIL Images
            output_path: Đường dẫn để lưu mesh đầu ra (tùy chọn)
            max_iterations: Số lần lặp tối đa cho quá trình huấn luyện Instant-NGP
            mesh_resolution: Độ phân giải của mesh (cao hơn = chi tiết hơn nhưng tốn tài nguyên hơn)
            export_method: Phương thức export mesh từ Nerfstudio ('poisson', 'textured_mesh', hoặc 'marching_cubes')
            simplify_mesh_in_processing: Có đơn giản hóa mesh ngay trong quá trình xử lý hay không
            simplify_target_faces: Số mặt mục tiêu nếu simplify_mesh_in_processing=True
            remove_small_components_threshold: Tỷ lệ phần trăm diện tích/số mặt tối thiểu so với thành phần lớn nhất

        Returns:
            Đối tượng trimesh.Trimesh
        """
        try:
            # Chuyển đổi novel views sang numpy arrays
            views_np = self._convert_views_to_numpy(novel_views)

            # Sử dụng SFMMVSProcessor đã cải tiến để tạo mesh chất lượng cao
            logger.info("Processing views with SFM + Nerfstudio pipeline...")
            self.mesh = self.sfm_mvs_processor.process_views(
                views_np,
                max_iterations=max_iterations,
                resolution=mesh_resolution,
                export_method=export_method,
                simplify_mesh_in_processing=simplify_mesh_in_processing,
                simplify_target_faces=simplify_target_faces,
                remove_small_components_threshold=remove_small_components_threshold
            )

            # Lưu mesh nếu output_path được cung cấp
            if output_path:
                self._save_mesh(self.mesh, output_path)

            logger.info("Mesh creation completed successfully")
            return self.mesh

        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            raise

    def _convert_views_to_numpy(self, views):
        """
        Chuyển đổi các novel views từ nhiều định dạng khác nhau sang numpy arrays

        Args:
            views: Danh sách các view (tensor, PIL Image, hoặc numpy array)

        Returns:
            Danh sách các numpy arrays
        """
        views_np = []
        for view in views:
            if isinstance(view, Image.Image):
                # PIL Image → numpy array
                views_np.append(np.array(view))
            elif isinstance(view, np.ndarray):
                # Đã là numpy array
                views_np.append(view)
            elif hasattr(view, 'numpy'):
                # PyTorch/TensorFlow tensor → numpy array
                if view.ndim == 4 and view.shape[0] == 1:  # Batch dimension
                    view = view.squeeze(0)
                if view.ndim == 3 and view.shape[0] == 3:  # CHW format
                    view = view.permute(1, 2, 0).numpy()  # → HWC
                else:
                    view = view.numpy()

                # Đảm bảo giá trị trong khoảng [0, 255]
                if view.max() <= 1.0:
                    view = (view * 255).astype(np.uint8)
                views_np.append(view)
            else:
                raise TypeError(f"Unsupported view type: {type(view)}")

        logger.info(f"Converted {len(views_np)} views to numpy arrays")
        return views_np

    def optimize_mesh(self, simplify=False, target_faces=100000, fill_holes=True, clean_small_components=True, remove_small_components_threshold=0.05):
        """
        Tối ưu hóa mesh sau khi tạo bằng phương thức create_mesh_from_views

        Chú ý: Nếu simplify_mesh_in_processing=True đã được sử dụng trong create_mesh_from_views, 
        việc đơn giản hóa mesh có thể đã được thực hiện. Phương thức này vẫn cung cấp các bước xử lý
        bổ sung hoặc có thể được sử dụng cho các mesh không được tạo bằng pipeline tích hợp.

        Args:
            simplify: Có đơn giản hóa mesh không
            target_faces: Số mặt mục tiêu nếu simplify=True
            fill_holes: Có lấp các lỗ không
            clean_small_components: Có loại bỏ các thành phần nhỏ không
            remove_small_components_threshold: Tỷ lệ phần trăm diện tích/số mặt tối thiểu so với thành phần lớn nhất

        Returns:
            Mesh đã tối ưu hóa
        """
        if self.mesh is None:
            logger.error(
                "No mesh to optimize. Call create_mesh_from_views first.")
            return None

        logger.info("Optimizing mesh...")

        # Lấp các lỗ
        if fill_holes and not self.mesh.is_watertight:
            logger.info("Filling holes...")
            try:
                # Thử sử dụng pymeshfix trước nếu có
                try:
                    import pymeshfix
                    logger.info("Using pymeshfix for hole filling")

                    # Lưu trữ vertex colors nếu có
                    original_vertex_colors = None
                    if hasattr(self.mesh.visual, 'vertex_colors') and self.mesh.visual.vertex_colors is not None:
                        original_vertex_colors = self.mesh.visual.vertex_colors.copy()

                    # Sửa mesh
                    meshfix = pymeshfix.MeshFix(
                        self.mesh.vertices, self.mesh.faces)
                    meshfix.repair()
                    vertices, faces = meshfix.get_mesh()
                    fixed_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=faces)

                    # Kiểm tra kết quả sửa chữa
                    if len(fixed_mesh.faces) >= len(self.mesh.faces) * 0.9:
                        self.mesh = fixed_mesh
                        logger.info(
                            f"Filled holes using pymeshfix: {len(self.mesh.faces)} faces")

                        # Cố gắng khôi phục vertex colors
                        if original_vertex_colors is not None:
                            logger.info(
                                "Restoring vertex colors after hole filling")
                            min_vertices = min(
                                len(original_vertex_colors), len(self.mesh.vertices))
                            default_color = np.array(
                                [255, 255, 255, 255], dtype=np.uint8)
                            self.mesh.visual.vertex_colors = np.tile(
                                default_color, (len(self.mesh.vertices), 1))
                            self.mesh.visual.vertex_colors[:min_vertices] = original_vertex_colors[:min_vertices]
                    else:
                        logger.warning(
                            "pymeshfix resulted in too few faces, using trimesh's built-in method")
                        self.mesh.fill_holes()
                except ImportError:
                    logger.warning(
                        "pymeshfix not available, using trimesh's built-in hole filling")
                    self.mesh.fill_holes()
            except Exception as e:
                logger.warning(f"Error filling holes: {e}")

        # Loại bỏ các điểm trùng lặp
        logger.info("Removing duplicate vertices...")
        initial_vertices = len(self.mesh.vertices)
        self.mesh.remove_duplicate_vertices()
        logger.info(
            f"Removed duplicates: {initial_vertices} → {len(self.mesh.vertices)} vertices")

        # Loại bỏ các mặt suy biến
        initial_faces = len(self.mesh.faces)
        self.mesh.remove_degenerate_faces()
        if initial_faces > len(self.mesh.faces):
            logger.info(
                f"Removed degenerate faces: {initial_faces} → {len(self.mesh.faces)}")

        # Loại bỏ các thành phần nhỏ
        if clean_small_components:
            try:
                logger.info("Removing small disconnected components...")
                components = self.mesh.split(only_watertight=False)

                if len(components) > 1:
                    # Tính toán diện tích và số mặt
                    metrics = [(i, comp.area, len(comp.faces))
                               for i, comp in enumerate(components)]
                    # Sắp xếp theo diện tích
                    metrics.sort(key=lambda x: x[1], reverse=True)

                    # Lấy diện tích lớn nhất để tính ngưỡng
                    largest_area = metrics[0][1]
                    threshold = largest_area * remove_small_components_threshold

                    # Lọc các thành phần dựa trên ngưỡng
                    kept_indices = [i for i, area,
                                    _ in metrics if area >= threshold]
                    kept_components = [components[i] for i in kept_indices]

                    if len(kept_components) < len(components):
                        logger.info(
                            f"Keeping {len(kept_components)}/{len(components)} components "
                            f"(threshold: {remove_small_components_threshold*100:.1f}% of largest)")
                        # Nối các thành phần được giữ lại
                        if kept_components:
                            self.mesh = trimesh.util.concatenate(
                                kept_components)
                        else:
                            logger.warning(
                                "No components met the threshold, keeping largest component")
                            self.mesh = components[metrics[0][0]]
            except Exception as e:
                logger.warning(f"Error cleaning small components: {e}")

        # Đơn giản hóa mesh nếu được yêu cầu và cần thiết
        if simplify and len(self.mesh.faces) > target_faces:
            try:
                # Lưu vertex colors trước khi đơn giản hóa
                original_vertex_colors = None
                if hasattr(self.mesh.visual, 'vertex_colors') and self.mesh.visual.vertex_colors is not None:
                    original_vertex_colors = self.mesh.visual.vertex_colors.copy()

                logger.info(
                    f"Simplifying mesh from {len(self.mesh.faces)} to {target_faces} faces...")

                simplified_mesh = self.mesh.simplify_quadratic_decimation(
                    target_faces)

                # Kiểm tra kết quả đơn giản hóa
                if simplified_mesh is not None and len(simplified_mesh.faces) > 0:
                    self.mesh = simplified_mesh
                    logger.info(
                        f"Mesh simplified to {len(self.mesh.faces)} faces")

                    # Khôi phục vertex colors nếu có thể
                    if original_vertex_colors is not None:
                        # Kiểm tra xem vertex colors có bị mất trong quá trình đơn giản hóa không
                        if not hasattr(self.mesh.visual, 'vertex_colors') or self.mesh.visual.vertex_colors is None:
                            logger.info(
                                "Attempting to restore vertex colors after simplification")
                            # Tạo màu mặc định cho tất cả đỉnh
                            default_color = np.array(
                                [255, 255, 255, 255], dtype=np.uint8)
                            vertex_colors = np.tile(
                                default_color, (len(self.mesh.vertices), 1))

                            # Gán màu sắc cho visual
                            if not hasattr(self.mesh, 'visual'):
                                self.mesh.visual = trimesh.visual.ColorVisuals(
                                    self.mesh)
                            self.mesh.visual.vertex_colors = vertex_colors

                            logger.info("Basic vertex colors restored")
                else:
                    logger.warning(
                        "Simplification failed, keeping original mesh")
            except Exception as e:
                logger.warning(f"Error simplifying mesh: {e}")

        # Đảm bảo mesh có normals
        self.mesh.fix_normals()

        logger.info("Mesh optimization completed")
        return self.mesh

    def _save_mesh(self, mesh, output_path, format='glb'):
        """
        Lưu mesh vào file

        Args:
            mesh: Đối tượng trimesh.Trimesh
            output_path: Đường dẫn để lưu mesh
            format: Định dạng đầu ra (glb, obj, ply, stl)
        """
        # Tạo thư mục nếu không tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lấy phần mở rộng file
        ext = os.path.splitext(output_path)[1].lower()[1:]

        # Nếu không có phần mở rộng hoặc không khớp với format, cập nhật đường dẫn
        if not ext or ext != format:
            output_path = os.path.splitext(output_path)[0] + '.' + format

        # Xuất mesh
        try:
            mesh.export(output_path, file_type=format)
            logger.info(f"Mesh saved to {output_path}")
            self.mesh_path = output_path
            return output_path
        except Exception as e:
            logger.error(f"Error saving mesh to {output_path}: {e}")
            raise

    def export_multiple_formats(self, base_path):
        """
        Xuất mesh ở nhiều định dạng khác nhau

        Args:
            base_path: Đường dẫn cơ sở cho các file đầu ra
        """
        if self.mesh is None:
            logger.error(
                "No mesh to export. Call create_mesh_from_views first.")
            return

        # Lấy đường dẫn cơ sở không có phần mở rộng
        base_path = os.path.splitext(base_path)[0]

        # Xuất ở các định dạng khác nhau
        formats = ['glb', 'obj', 'ply', 'stl']
        paths = []

        for format in formats:
            try:
                output_path = f"{base_path}.{format}"
                self._save_mesh(self.mesh, output_path, format)
                paths.append(output_path)
            except Exception as e:
                logger.warning(f"Failed to export as {format}: {e}")

        return paths

    def clean_up(self):
        """
        Dọn dẹp tài nguyên tạm thời
        """
        logger.info("Cleaning up temporary resources...")
        if hasattr(self, 'sfm_mvs_processor'):
            self.sfm_mvs_processor.cleanup()

    def _apply_texture(self, mesh=None, texture_image=None):
        """
        Áp dụng đơn giản vertex colors cho mesh

        Chú ý: Bước này chỉ áp dụng vertex colors đơn giản làm placeholder. Để có texture 
        chất lượng game, cần thực hiện UV unwrapping và texture baking chuyên dụng sau khi
        có mesh low-poly đã retopology.

        Args:
            mesh: Mesh để áp dụng texture (nếu None, sử dụng self.mesh)
            texture_image: Ảnh texture để áp dụng (nếu None, tạo texture mặc định)

        Returns:
            Mesh với texture đã áp dụng
        """
        if mesh is None:
            if self.mesh is None:
                logger.error(
                    "No mesh to apply texture. Create or provide a mesh first.")
                return None
            mesh = self.mesh

        logger.info("Applying basic vertex colors to mesh...")

        # Đây chỉ là một placeholder đơn giản
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            logger.info("No vertex colors found, creating default white color")
            default_color = np.array([255, 255, 255, 255], dtype=np.uint8)
            if not hasattr(mesh, 'visual'):
                mesh.visual = trimesh.visual.ColorVisuals(mesh)
            mesh.visual.vertex_colors = np.tile(
                default_color, (len(mesh.vertices), 1))

        logger.info("Basic vertex colors applied")
        logger.warning(
            "Note: For production-quality textures, you should use dedicated 3D "
            "modeling software to perform proper UV unwrapping and texture baking "
            "after obtaining a retopologized low-poly mesh."
        )

        return mesh
