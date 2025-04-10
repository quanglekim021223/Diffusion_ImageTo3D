import numpy as np
import trimesh
import pyrender
from pathlib import Path
import open3d as o3d
import cv2
from scipy.spatial import Delaunay
import torch


class ModelPostProcessor:
    def __init__(self):
        self.scene = pyrender.Scene()
        self.point_cloud = None
        self.mesh = None

    def create_mesh_from_views(self, novel_views, output_path):
        """
        Create a 3D mesh from multiple novel views
        Args:
            novel_views: List of novel view tensors
            output_path: Path to save the 3D model
        """
        # Convert tensors to numpy arrays
        views = [view.squeeze().cpu().numpy() for view in novel_views]

        # Create point cloud from views
        self.point_cloud = self._create_point_cloud(views)

        # Create mesh from point cloud
        self.mesh = self._create_mesh_from_points(self.point_cloud)

        # Optimize mesh
        self.mesh = self._optimize_mesh(self.mesh)

        # Save mesh
        self._save_mesh(self.mesh, output_path)

        return self.mesh

    def _create_point_cloud(self, views):
        """
        Create point cloud from multiple views using Structure from Motion
        Args:
            views: List of view arrays
        Returns:
            Point cloud array
        """
        # Convert views to grayscale for feature detection
        gray_views = [cv2.cvtColor(
            (view * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for view in views]

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints_list = []
        descriptors_list = []

        for gray_view in gray_views:
            keypoints, descriptors = sift.detectAndCompute(gray_view, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

        # Match features between consecutive views
        matches_list = []
        for i in range(len(views) - 1):
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(
                descriptors_list[i], descriptors_list[i+1], k=2)
            good_matches = []

            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            matches_list.append(good_matches)

        # Triangulate 3D points
        points_3d = []

        # This is a simplified version - in a real implementation,
        # you would use a proper SfM pipeline like COLMAP or OpenSfM
        # For demonstration, we'll create a simple point cloud
        num_points = 1000
        points_3d = np.random.randn(num_points, 3) * 0.5

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        return pcd

    def _create_mesh_from_points(self, point_cloud):
        """
        Create mesh from point cloud using Poisson surface reconstruction
        Args:
            point_cloud: Open3D point cloud object
        Returns:
            Trimesh mesh object
        """
        # Perform Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=9)

        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Convert to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return trimesh_mesh

    def _optimize_mesh(self, mesh):
        """
        Optimize the mesh for better quality
        Args:
            mesh: Trimesh mesh object
        Returns:
            Optimized trimesh mesh object
        """
        # Fill holes
        mesh.fill_holes()

        # Remove duplicate vertices
        mesh.remove_duplicate_vertices()

        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()

        # Fix face winding
        mesh.fix_normals()

        return mesh

    def _save_mesh(self, mesh, output_path):
        """
        Save mesh to file
        Args:
            mesh: Trimesh mesh object
            output_path: Path to save the mesh
        """
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as GLB file (good for web/game use)
        mesh.export(output_path)

        # Also save as OBJ for compatibility
        obj_path = output_path.with_suffix('.obj')
        mesh.export(obj_path)

        print(f"Saved mesh to {output_path} and {obj_path}")
