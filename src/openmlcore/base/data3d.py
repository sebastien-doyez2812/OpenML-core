"""
Author: Seb Doyez

3D dataset should be a folder, where inside we can find subfolder corresponding to a scene
1 scene = 1 subfolder
Inside a subfolder, there qre .pcd and ..._label.txt

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os

class Segmentation3dDataset(Dataset):
    def __init__(self, dataset_path, num_points = 2048, split="train"):
        super().__init__()
        self.num_points = num_points     
        self.dataset_path = dataset_path

        self.scene_dirs = self._get_object_paths(self.dataset_path) # Only treat the airplane!
        total_data = len(self.scene_dirs)
        split_idx = int(total_data * 0.8)
        if split.lower() == "train":
            self.scene_dirs = self.scene_dirs[:split_idx]
        else: 
            self.scene_dirs = self.scene_dirs[split_idx:]

    def _get_object_paths(self, dataset_path):
        # Dataset should be like:
        # subfolders, and inside, you will find a .pcd and the _label.txt 
        # .pcd is the point cloud and _label.txt is the label
        # Label goes from 1 to n ( number of classes)
                
        object_paths = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for file in filenames:
                if file.endswith(".pcd"):
                    base_path = os.path.join(dirpath, file.replace(".pcd", ""))
                    if os.path.exists(base_path +"_label.txt"):
                        object_paths.append(base_path)
        return object_paths
                    

    def _load_scene_data(self, scene_base_path):
        label_path = scene_base_path + "_label.txt"
        pcd_path   = scene_base_path + ".pcd"

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"No labels found for {label_path}")    

        
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"No PCD found for {pcd_path}")

        with open(pcd_path, 'rb') as f:
            header = []
            data_type = 'ascii'
            num_pts = 0
            fields = []
            sizes = []
            types = []
                
            # Skip the header
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('latin-1')
                header.append(line)
                if line.startswith("DATA"):
                    parts = line.strip().split()
                    if len(parts) > 1:
                        data_type = parts[1]
                    break
            
            for line in header:
                if line.startswith("POINTS"):
                    num_pts = int(line.strip().split()[1])
                elif line.startswith("WIDTH") and num_pts == 0:
                    num_pts = int(line.strip().split()[1])
                elif line.startswith("FIELDS"):
                    fields = line.strip().split()[1:]
                elif line.startswith("SIZE"):
                    sizes = [int(s) for s in line.strip().split()[1:]]
                elif line.startswith("TYPE"):
                    types = line.strip().split()[1:]

            # Loading the data:
            if data_type.lower() == 'ascii':
                points_data = np.loadtxt(f, encoding="latin-1")
                points = points_data[:, :3]
            else:
                binary_data = f.read()
                try:
                    dt_list = []
                    for field, size, t in zip(fields, sizes, types):
                        if t == 'F':
                            np_t = np.float32 if size == 4 else np.float64
                        elif t in ['I', 'U']:
                            np_t = np.int32 if size == 4 else (np.int16 if size == 2 else np.int8)
                        else:
                            np_t = np.float32
                        dt_list.append((field, np_t))
                        
                    dt = np.dtype(dt_list)
                    data = np.frombuffer(binary_data, dtype=dt, count=num_pts)
                    points = np.stack([data['x'], data['y'], data['z']], axis=-1)
                except Exception:
                    raw_data = np.frombuffer(binary_data, dtype=np.float32)
                    stride = len(raw_data) // num_pts
                    points = raw_data.reshape(num_pts, stride)[:, :3]

        # Loading the labels
        labels = np.loadtxt(label_path, dtype=np.int64, encoding="latin-1")
        
        assert len(points) == len(labels), f"[-] Error, number of points mismatch: {len(points)} pts vs {len(labels)} labels."

        return points, labels

    def _normalize_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))

        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered

        return points_normalized

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_path = self.scene_dirs[idx]
        points, labels = self._load_scene_data(scene_path)

        totals_pts = points.shape[0] 
        if totals_pts >= self.num_points:
            # Block_sampling:
            center_idx = np.random.choice(points.shape[0])
            center = points[center_idx]

            radius = 1.5
            distances = np.sqrt(np.sum((points - center) ** 2, axis = 1))

            mask = distances < radius
            block_points = points[mask]
            block_labels = labels[mask]

            total_pts = block_points.shape[0]
            if total_pts >= self.num_points:
                choice = np.random.choice(total_pts, self.num_points, replace=False)
            else:
                choice = np.random.choice(total_pts, self.num_points, replace=True)

            points_sampled = block_points[choice]
            labels_sampled = block_labels[choice]

        else:
            choice = np.random.choice(totals_pts, self.num_points, replace=True)
            points_sampled = points[choice]
            labels_sampled = labels[choice]
                    
        points_normalized =  self._normalize_point_cloud(points_sampled)
        points_tensor = torch.tensor(points_normalized, dtype=torch.float32).transpose(1, 0)
        labels_tensor = torch.tensor(labels_sampled, dtype=torch.long)

        return points_tensor, labels_tensor
           