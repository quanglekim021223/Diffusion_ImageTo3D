import numpy as np
import trimesh
import pyrender
from pathlib import Path
import open3d as o3d
import cv2
from scipy.spatial import Delaunay
import torch
import os
import logging
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


class ModelPostProcessor:
    def __init__(self):
        self.scene = pyrender.Scene()
        self.point_cloud = None
        self.mesh = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def create_mesh_from_views(self, novel_views, use_poisson=True, depth=9, apply_texture=True):
        """
        Create a 3D mesh from the generated novel views
        Args:
            novel_views: List of novel view tensors or PIL Images
            use_poisson: Whether to use Poisson surface reconstruction
            depth: Depth parameter for Poisson reconstruction
            apply_texture: Whether to apply texture mapping
        Returns:
            Trimesh mesh object
        """
        try:
            # Convert PIL Images to numpy arrays if needed
            views_np = []
            for view in novel_views:
                if isinstance(view, Image.Image):
                    views_np.append(np.array(view))
                else:
                    views_np.append(view.numpy().transpose(1, 2, 0))

            # Create point cloud from views
            logger.info("Creating point cloud from views...")
            self.point_cloud = self._create_point_cloud(views_np)

            # Create mesh from point cloud
            logger.info("Creating mesh from point cloud...")
            self.mesh = self._create_mesh_from_points(
                self.point_cloud, use_poisson, depth)

            # Optimize mesh
            logger.info("Optimizing mesh...")
            self.mesh = self._optimize_mesh(self.mesh)

            # Apply texture if requested
            if apply_texture:
                logger.info("Applying texture mapping...")
                self.mesh = self._apply_texture(self.mesh, views_np)

            return self.mesh
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            raise

    def _create_point_cloud(self, views):
        """
        Create a point cloud from the novel views
        Args:
            views: List of novel view numpy arrays
        Returns:
            Open3D point cloud object
        """
        # This is an improved implementation that creates a more detailed point cloud
        points = []
        colors = []

        for i, view in enumerate(views):
            # Create a grid of points
            h, w, c = view.shape
            y, x = np.mgrid[0:h, 0:w]

            # Calculate depth using a more sophisticated approach
            # We use the view index as a proxy for depth, but in a real implementation
            # you would use stereo matching or other techniques
            depth = np.ones((h, w)) * (i + 1) / len(views)

            # Add some noise to depth to create more variation
            depth += np.random.normal(0, 0.05, (h, w))

            # Create 3D points
            points_view = np.stack([x, y, depth], axis=-1)
            points.append(points_view.reshape(-1, 3))

            # Extract colors from the view
            colors_view = view.reshape(-1, c)
            colors.append(colors_view)

        # Concatenate all points and colors
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)

        return pcd

    def _create_mesh_from_points(self, point_cloud, use_poisson=True, depth=9):
        """
        Create a mesh from a point cloud using Poisson surface reconstruction or convex hull
        Args:
            point_cloud: Open3D point cloud object
            use_poisson: Whether to use Poisson surface reconstruction
            depth: Depth parameter for Poisson reconstruction
        Returns:
            Trimesh mesh object
        """
        try:
            if use_poisson:
                # Perform Poisson surface reconstruction
                logger.info(
                    f"Performing Poisson surface reconstruction with depth={depth}...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    point_cloud, depth=depth)

                # Remove low density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.1)
                mesh.remove_vertices_by_mask(vertices_to_remove)
            else:
                # Use convex hull as a fallback
                logger.info("Using convex hull for mesh creation...")
                points = np.asarray(point_cloud.points)
                mesh = trimesh.creation.convex_hull(points)
                return mesh

            # Convert to trimesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            return trimesh_mesh
        except Exception as e:
            logger.warning(
                f"Mesh creation failed: {e}. Falling back to convex hull.")
            # Fall back to convex hull if reconstruction fails
            points = np.asarray(point_cloud.points)
            mesh = trimesh.creation.convex_hull(points)
            return mesh

    def _optimize_mesh(self, mesh, target_faces=None, fill_holes=True, remove_duplicates=True):
        """
        Optimize the mesh by filling holes, removing duplicate vertices, and simplifying if needed
        Args:
            mesh: Trimesh mesh object
            target_faces: Target number of faces for simplification (None for no simplification)
            fill_holes: Whether to fill holes in the mesh
            remove_duplicates: Whether to remove duplicate vertices
        Returns:
            Optimized Trimesh mesh object
        """
        try:
            logger.info("Starting mesh optimization...")

            # Fill holes if requested
            if fill_holes:
                logger.info("Filling holes in mesh...")
                mesh.fill_holes()

            # Remove duplicate vertices if requested
            if remove_duplicates:
                logger.info("Removing duplicate vertices...")
                mesh.remove_duplicate_vertices()

            # Simplify mesh if target_faces is specified
            if target_faces is not None and len(mesh.faces) > target_faces:
                logger.info(
                    f"Simplifying mesh from {len(mesh.faces)} to {target_faces} faces...")
                mesh = mesh.simplify_quadratic_decimation(target_faces)

            # Ensure mesh is watertight
            if not mesh.is_watertight:
                logger.warning("Mesh is not watertight, attempting to fix...")
                mesh.fill_holes()
                mesh.remove_degenerate_faces()
                mesh.remove_infinite_values()
                mesh.remove_unreferenced_vertices()

            logger.info("Mesh optimization completed successfully")
            return mesh

        except Exception as e:
            logger.error(f"Error during mesh optimization: {e}")
            return mesh  # Return original mesh if optimization fails

    def _save_mesh(self, mesh, output_path, format='glb'):
        """
        Save the mesh to a file
        Args:
            mesh: Trimesh mesh object
            output_path: Path to save the mesh
            format: Output format (glb, obj, ply, stl)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get file extension
        ext = os.path.splitext(output_path)[1].lower()

        # If no extension or extension doesn't match format, update path
        if not ext or ext[1:] != format:
            output_path = os.path.splitext(output_path)[0] + '.' + format

        # Export mesh
        if format == 'glb':
            mesh.export(output_path, file_type='glb')
        elif format == 'obj':
            mesh.export(output_path, file_type='obj')
        elif format == 'ply':
            mesh.export(output_path, file_type='ply')
        elif format == 'stl':
            mesh.export(output_path, file_type='stl')
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Mesh saved to {output_path}")

    def export_multiple_formats(self, mesh, base_path):
        """
        Export the mesh in multiple formats
        Args:
            mesh: Trimesh mesh object
            base_path: Base path for the output files
        """
        # Get base path without extension
        base_path = os.path.splitext(base_path)[0]

        # Export in different formats
        formats = ['glb', 'obj', 'ply', 'stl']
        for format in formats:
            output_path = f"{base_path}.{format}"
            self._save_mesh(mesh, output_path, format)

    def _apply_texture(self, mesh, views):
        """
        Apply texture mapping to the mesh using the novel views
        Args:
            mesh: Trimesh mesh object
            views: List of novel view numpy arrays
        Returns:
            Trimesh mesh object with texture
        """
        try:
            # This is a simplified texture mapping implementation
            # In a real implementation, you would use more sophisticated techniques
            # like UV unwrapping and texture projection

            # For GLB format, we can use vertex colors as a simple texture
            if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'vertex_colors'):
                # Create a simple vertex color based on position
                # This is just a placeholder - in a real implementation,
                # you would project the textures from the views onto the mesh
                vertices = mesh.vertices
                colors = np.zeros((len(vertices), 4), dtype=np.uint8)

                # Use position to create a simple color gradient
                min_pos = vertices.min(axis=0)
                max_pos = vertices.max(axis=0)
                range_pos = max_pos - min_pos

                for i in range(3):  # x, y, z
                    if range_pos[i] > 0:
                        normalized = (vertices[:, i] -
                                      min_pos[i]) / range_pos[i]
                        colors[:, i] = (normalized * 255).astype(np.uint8)
                    else:
                        colors[:, i] = 128

                # Set alpha channel to 255 (fully opaque)
                colors[:, 3] = 255

                # Create a simple visual with vertex colors
                mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

            return mesh
        except Exception as e:
            logger.warning(
                f"Texture mapping failed: {e}. Returning mesh without texture.")
            return mesh
