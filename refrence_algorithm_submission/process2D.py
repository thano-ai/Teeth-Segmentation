import glob
import json
import os
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import traceback
import cv2


# Custom Encoder to handle NumPy types in JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# 2D U-Net Model Definition
class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=48):
        super(UNet2D, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        # test

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder with padding to handle size mismatches
        self.up2 = up_block(512, 256)
        self.dec2 = conv_block(512, 256)
        self.up1 = up_block(256, 128)
        self.dec1 = conv_block(256, 128)

        # Final output
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)          # [batch, 64, h, w]
        enc2 = self.enc2(self.pool(enc1))  # [batch, 128, h/2, w/2]
        enc3 = self.enc3(self.pool(enc2))  # [batch, 256, h/4, w/4]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # [batch, 512, h/8, w/8]

        # Decoder with padding to handle odd dimensions
        dec2 = self.up2(bottleneck)
        # Pad if dimensions don't match
        diffY = enc3.size()[2] - dec2.size()[2]
        diffX = enc3.size()[3] - dec2.size()[3]
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        dec2 = self.dec2(torch.cat([dec2, enc3], dim=1))

        dec1 = self.up1(dec2)
        # Pad if dimensions don't match
        diffY = enc2.size()[2] - dec1.size()[2]
        diffX = enc2.size()[3] - dec1.size()[3]
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        dec1 = self.dec1(torch.cat([dec1, enc2], dim=1))

        return self.final(dec1)


# Preprocessing function
def preprocess_function(mesh):
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    vertices -= vertices.mean(dim=0)  # Center the mesh
    vertices /= vertices.max()  # Normalize
    return vertices


# Downsampling function
def downsample_vertices(vertices, target_num_vertices):
    if vertices.shape[0] > target_num_vertices:
        step = vertices.shape[0] // target_num_vertices
        indices = np.arange(0, vertices.shape[0], step)[:target_num_vertices]
        vertices = vertices[indices]
    return vertices


def downsample_labels(labels, output_size):
    """Downsample labels to match model output size"""
    labels = labels.unsqueeze(1).float()  # Add channel dimension
    labels = F.interpolate(labels, size=output_size, mode='nearest')
    return labels.squeeze(1).long()


# Padding function to handle variable-sized meshes
def pad_vertices(vertices, max_vertices):
    if vertices.shape[0] < max_vertices:
        padding = np.zeros((max_vertices - vertices.shape[0], 3))
        vertices = np.vstack((vertices, padding))
    return vertices[:max_vertices]


# Custom Dataset for .obj files
class MeshDataset(Dataset):
    def __init__(self, root_dir, max_vertices=8192):
        self.root_dir = root_dir
        self.max_vertices = max_vertices
        self.file_pairs = self._find_file_pairs()[:100]  # Use only 2 samples for testing

        if not self.file_pairs:
            raise ValueError(f"No valid OBJ+JSON pairs found in {root_dir}")

    def _find_file_pairs(self):
        file_pairs = []
        # Search for lower and upper jaw folders
        for jaw_type in ['lower', 'upper']:
            jaw_dir = os.path.join(self.root_dir, jaw_type)
            if not os.path.exists(jaw_dir):
                continue

            # Find all case folders
            for case_dir in glob.glob(os.path.join(jaw_dir, '*')):
                if not os.path.isdir(case_dir):
                    continue

                # Find OBJ and JSON files
                obj_files = glob.glob(os.path.join(case_dir, '*.obj'))
                json_files = glob.glob(os.path.join(case_dir, '*.json'))

                # Pair them up (assuming one OBJ and one JSON per folder)
                for obj_file in obj_files:
                    # Find matching JSON (same stem)
                    base_name = os.path.splitext(os.path.basename(obj_file))[0]
                    matching_json = [j for j in json_files if os.path.splitext(os.path.basename(j))[0] == base_name]

                    if matching_json:
                        file_pairs.append((obj_file, matching_json[0], jaw_type))

        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        obj_path, json_path, jaw_type = self.file_pairs[idx]

        # Load mesh
        mesh = trimesh.load(obj_path, process=False)
        vertices = preprocess_function(mesh)

        # Downsample and pad vertices
        vertices = downsample_vertices(vertices.numpy(), self.max_vertices)
        vertices = pad_vertices(vertices, self.max_vertices)

        # Load JSON data (assuming it contains segmentation info)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Create labels - you'll need to adapt this based on your JSON structure
        labels = np.zeros(vertices.shape[0], dtype=np.int64)

        # Example: If JSON contains tooth labels
        if 'labels' in json_data:
            raw_labels = np.array(json_data['labels'], dtype=np.int64)

            # Downsample and pad labels same way as vertices
            if raw_labels.shape[0] > self.max_vertices:
                step = raw_labels.shape[0] // self.max_vertices
                indices = np.arange(0, raw_labels.shape[0], step)[:self.max_vertices]
                labels = raw_labels[indices]
            else:
                # Pad if fewer labels
                pad_size = self.max_vertices - raw_labels.shape[0]
                labels = np.pad(raw_labels, (0, pad_size), 'constant', constant_values=0)
        else:
            # Fallback: create artificial labels
            y_coords = vertices[:, 1]  # Get y-coordinates
            median_y = np.median(y_coords)
            labels[y_coords > median_y] = 1

        return (
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            jaw_type
        )


# 2D ScanSegmentation class
class ScanSegmentation2D:
    def __init__(self, max_vertices=8192):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UNet2D(in_channels=3, out_channels=49)  # 48 classes for teeth segmentation
        self.model.to(self.device)
        self.max_vertices = max_vertices
        self.grid_size = int(np.ceil(np.sqrt(max_vertices)))  # Size of 2D grid
        if self.grid_size % 2 != 0:
            self.grid_size += 1

    def vertices_to_grid(self, vertices):
        """Convert 1D vertex array to 2D grid"""
        batch_size = vertices.shape[0]
        num_vertices = vertices.shape[1]

        # Calculate how many vertices we can fit in the grid
        max_fit = self.grid_size * self.grid_size
        vertices_to_use = min(num_vertices, max_fit)

        # Create 2D grid (batch_size, channels, grid_size, grid_size)
        grid = torch.zeros((batch_size, 3, self.grid_size, self.grid_size),
                           dtype=torch.float32, device=self.device)

        # Fill the grid
        for i in range(batch_size):
            # Get the vertices we'll use (trim if necessary)
            flat_vertices = vertices[i, :vertices_to_use, :]

            # Calculate how many vertices we actually have
            actual_vertices = flat_vertices.shape[0]

            # Pad if we don't have enough to fill the grid
            if actual_vertices < max_fit:
                padding = torch.zeros((max_fit - actual_vertices, 3),
                                      device=self.device)
                flat_vertices = torch.cat([flat_vertices, padding], dim=0)

            # Reshape to 2D grid (3 channels, grid_size x grid_size)
            grid[i] = flat_vertices.transpose(0, 1).reshape(3, self.grid_size, self.grid_size)

        return grid

    def grid_to_vertices(self, grid, original_num_vertices):
        """Convert 2D grid predictions back to 1D vertex predictions"""
        batch_size = grid.shape[0]

        # Flatten the grid
        flat_predictions = grid.reshape(batch_size, -1)

        # Truncate to original number of vertices
        return flat_predictions[:, :original_num_vertices]

    def load_dataset(self, input_dir, max_vertices=8192):
        """Load the dataset from the input directory"""
        return MeshDataset(input_dir, max_vertices=max_vertices)

    def train(self, input_dir, epochs=20, batch_size=2):
        """Complete training pipeline"""
        dataset = self.load_dataset(input_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print(f"Starting training with {len(dataset)} samples...")

        self.model.train()
        for epoch in range(epochs):
            print(f"\nStarting epoch {epoch + 1}/{epochs}", flush=True)
            epoch_loss = 0.0
            correct = 0
            total = 0
            batch_counter = 0

            for batch in dataloader:
                batch_counter += 1
                print(f"Processing batch {batch_counter}/{len(dataloader)}", end='\r', flush=True)

                if len(batch) == 2:
                    vertices, labels = batch
                else:
                    vertices, labels, _ = batch  # ignore jaw_type if present

                # Move data to device and convert to 2D grid
                vertices = vertices.to(self.device)
                labels = labels.to(self.device)

                # Convert vertices to grid
                vertices_grid = self.vertices_to_grid(vertices)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(vertices_grid)  # [batch, num_classes, h, w]

                # Reshape outputs and labels
                outputs = outputs.permute(0, 2, 3, 1)  # [batch, h, w, num_classes]
                outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch*h*w, num_classes]

                # Reshape labels to match outputs
                labels = labels.reshape(-1)  # [batch*num_vertices]

                # Ensure we only use valid labels (trim to match output spatial dimensions)
                output_pixels = outputs.shape[0] // batch_size
                labels = labels[:output_pixels * batch_size]

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Calculate metrics
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()

            # Print epoch statistics
            accuracy = 100 * correct / total
            print(
                f"\nEpoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

    @staticmethod
    def load_input(input_dir):
        """Find all .obj files in the directory structure"""
        inputs = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.obj'):
                    inputs.append(os.path.join(root, file))
        if not inputs:
            raise FileNotFoundError(f"No .obj files found in {input_dir}")
        return inputs

    def predict(self, obj_path, json_path):
        self.model.eval()
        with torch.no_grad():
            try:
                # 1. Load and preprocess mesh
                mesh = trimesh.load(obj_path, process=False)
                if mesh is None:
                    raise ValueError(f"Failed to load mesh from {obj_path}")

                # Get vertices and ensure proper shape [n_vertices, 3]
                vertices = np.asarray(mesh.vertices)
                if vertices.ndim == 1:
                    vertices = vertices.reshape(-1, 3)
                elif vertices.shape[1] != 3:
                    vertices = vertices.reshape(-1, 3)

                # Preprocessing
                vertices = vertices - vertices.mean(axis=0)  # Center
                vertices /= np.abs(vertices).max()  # Normalize
                original_vertex_count = len(vertices)

                # 2. Determine jaw type from path
                jaw = "lower" if "lower" in obj_path.lower() else "upper"

                # 3. Prepare input tensor
                if original_vertex_count > self.max_vertices:
                    indices = np.random.choice(original_vertex_count, self.max_vertices, replace=False)
                    vertices = vertices[indices]
                else:
                    # Pad with zeros if needed
                    padding = np.zeros((self.max_vertices - original_vertex_count, 3))
                    vertices = np.vstack([vertices, padding])

                # Convert to tensor [1, max_vertices, 3]
                input_tensor = torch.from_numpy(vertices).float().unsqueeze(0).to(self.device)

                # Convert to 2D grid
                input_grid = self.vertices_to_grid(input_tensor)

                # 4. Get predictions
                logits = self.model(input_grid)
                predictions = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                # Flatten predictions and trim to original size
                predictions = predictions.reshape(-1)[:original_vertex_count]

                # 5. Create instances from predictions
                # First create a binary mask for each tooth class
                instances = np.zeros_like(predictions)
                unique_labels = np.unique(predictions)

                current_instance_id = 1
                for label in unique_labels:
                    if label == 0:  # Skip background
                        continue

                    # Create mask for current label
                    mask = (predictions == label).astype(np.uint8)

                    # Find connected components
                    # Since we're working with 1D data (after flattening), we'll use a simple approach
                    diff = np.diff(mask.astype(int))
                    starts = np.where(diff == 1)[0] + 1
                    ends = np.where(diff == -1)[0] + 1

                    # Handle edge cases
                    if mask[0]:
                        starts = np.insert(starts, 0, 0)
                    if mask[-1]:
                        ends = np.append(ends, len(mask))

                    # Assign instance IDs
                    for start, end in zip(starts, ends):
                        instances[start:end] = current_instance_id
                        current_instance_id += 1

                return predictions, instances, jaw

            except Exception as e:
                print(f"Error processing {obj_path}: {str(e)}")
                traceback.print_exc()
                # Return zero arrays with correct length
                mesh = trimesh.load(obj_path, process=False)
                vertex_count = len(mesh.vertices) if mesh else self.max_vertices
                jaw = "lower" if "lower" in obj_path.lower() else "upper"
                return np.zeros(vertex_count, dtype=np.int64), np.zeros(vertex_count, dtype=np.int64), jaw

    @staticmethod
    def write_output(labels, instances, jaw, output_dir='./output'):
        """Write the output JSON with proper labels, instances, and jaw type"""
        pred_output = {
            'id_patient': "",
            'jaw': jaw,
            'labels': labels,
            'instances': instances
        }
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'expected_output2D.json')
        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)
        print(f"Output written to {output_path}")

    def visualize_results(self, obj_path, labels):
        """Visualize predictions on the original mesh"""
        import matplotlib.pyplot as plt
        import trimesh

        # Load the original mesh
        mesh = trimesh.load(obj_path, process=False)

        # Create colormap (one color per unique label)
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        # Assign colors to vertices
        vertex_colors = np.zeros((len(labels), 4))  # RGBA colors
        for i, label in enumerate(unique_labels):
            vertex_colors[labels == label] = colors[i]

        # Apply colors to mesh
        mesh.visual.vertex_colors = vertex_colors

        # Display interactive 3D view
        mesh.show()

    def process(self, input_dir):
        inputs = self.load_input(input_dir)
        obj_path = inputs[0]
        json_path = os.path.splitext(obj_path)[0] + ".json"

        labels, instances, jaw = self.predict(obj_path, json_path)

        # Write output
        self.write_output(labels=labels.tolist(), instances=instances.tolist(), jaw=jaw)

        # Visualize predictions
        self.visualize_results(obj_path, labels)


if __name__ == "__main__":
    input_dir = "D:\Teeth3DS\data_part_1"
    scan_segmentation = ScanSegmentation2D(max_vertices=8192)

    try:
        scan_segmentation.train(input_dir, epochs=100)
        scan_segmentation.process(input_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
