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


# U-Net Model Definition (1D Version)
class UNet1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet1D, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up2 = up_block(512, 256)
        self.dec2 = conv_block(512, 256)  # 256*2 due to skip connection
        self.up1 = up_block(256, 128)
        self.dec1 = conv_block(256, 128)  # 128*2 due to skip connection

        # Final output
        self.final = nn.Conv1d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat((dec2, enc3), dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc2), dim=1))

        # Final output
        return self.final(dec1)

# Preprocessing function
def preprocess_function(mesh):
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    vertices -= vertices.mean(dim=0)  # Center the mesh
    vertices /= vertices.max()        # Normalize
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
        # self.file_pairs = self._find_file_pairs()
        self.file_pairs = self._find_file_pairs()[:2]  # Use only 2 samples for testing

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
        # This is just an example - modify according to your actual JSON format
        labels = np.zeros(vertices.shape[0], dtype=np.int64)

        # Example: If JSON contains tooth labels
        # Load labels from JSON
        if 'labels' in json_data:
            raw_labels = np.array(json_data['labels'], dtype=np.int64)
            print(np.unique(raw_labels))

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
            labels = np.zeros(vertices.shape[0], dtype=np.int64)
            labels[y_coords > median_y] = 1

        # Fallback: Create artificial labels if no labels in JSON
        # y_coords = vertices[:, 1]  # Get y-coordinates
        # median_y = np.median(y_coords)
        # labels[y_coords > median_y] = 1  # Class 1 for upper half

        # Debug print (optional)
        # print(f"Vertices shape: {vertices.shape}, Labels shape: {labels.shape}")
        print(f"Vertices dtype: {vertices.dtype}, shape: {vertices.shape}")
        print(f"Labels dtype: {labels.dtype}, shape: {labels.shape}")
        print(f"Loaded vertices shape: {vertices.shape}")
        print(f"Vertex sample: {vertices[0]}")
        return (
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            jaw_type
        )


# Training Loop
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
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

            # Move data to device
            vertices = vertices.to(device).permute(0, 2, 1)  # [batch, channels, vertices]
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(vertices)  # [batch, num_classes, num_vertices]

            # Downsample labels to match output size
            labels = downsample_labels(labels, outputs.shape[2])

            # Reshape for loss
            outputs = outputs.permute(0, 2, 1)  # [batch, vertices, num_classes]
            outputs = outputs.reshape(-1, outputs.shape[-1])
            labels = labels.reshape(-1)

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
        print(f"\nEpoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}", flush=True)


# Testing/Inference Loop
def test_model(model, dataloader, device):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for vertices, _ in dataloader:
            vertices = vertices.to(device)

            # Reshape vertices to [batch_size, channels, num_vertices]
            vertices = vertices.permute(0, 2, 1)  # Shape: [batch_size, 3, num_vertices]

            outputs = model(vertices)
            all_outputs.append(outputs.cpu().numpy())

    return all_outputs


# ScanSegmentation Class
class ScanSegmentation:
    def __init__(self, max_vertices=8192):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UNet1D(in_channels=3, out_channels=48)  # 2 classes for example
        self.model.to(self.device)
        self.max_vertices = max_vertices  # Initialize the attribute

    def load_dataset(self, input_dir, max_vertices=8192):
        """Load the dataset from the input directory"""
        return MeshDataset(input_dir, max_vertices=max_vertices)

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

    def train(self, input_dir, epochs=20, batch_size=2):
        """Complete training pipeline"""
        dataset = self.load_dataset(input_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print(f"Starting training with {len(dataset)} samples...")


        # Call train_model instead of duplicating training logic
        train_model(
            model=self.model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=epochs
        )

    @staticmethod
    def load_input(input_dir):
        inputs = glob.glob(f'{input_dir}/**/*.obj', recursive=True)
        if len(inputs) == 0:
            raise FileNotFoundError(f"No .obj files found in the directory: {input_dir}")
        return inputs

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
        output_path = os.path.join(output_dir, 'expected_output.json')
        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)
        print(f"Output written to {output_path}")

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

                # Convert to tensor [1, 3, max_vertices]
                input_tensor = torch.from_numpy(vertices.T).float().unsqueeze(0).to(self.device)

                # 4. Get predictions
                logits = self.model(input_tensor)
                predictions = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                predictions = predictions[:original_vertex_count]  # Trim to original size
                print(f"Predictions shape: {predictions.shape}")
                print(f"Predictions sample: {predictions[:10]}")
                print(f"Unique predictions: {np.unique(predictions)}")

                # 5. NEW - Simplified Instance Segmentation (1D only)
                instances = np.zeros_like(predictions)
                unique_labels = np.unique(predictions)

                current_instance_id = 1
                for label in unique_labels:
                    if label == 0:  # Skip background
                        continue

                    # Create mask for current label
                    mask = (predictions == label)

                    # Find connected components using 1D method
                    # This replaces cv2.connectedComponents for 1D data
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

    def process(self, input_dir):
        inputs = self.load_input(input_dir)
        # Find matching JSON file (assuming same name but .json extension)
        obj_path = inputs[0]
        json_path = os.path.splitext(obj_path)[0] + ".json"

        # Call predict and get all three return values
        labels, instances, jaw = self.predict(obj_path, json_path)

        # Convert numpy arrays to lists for JSON serialization
        self.write_output(labels=labels.tolist(), instances=instances.tolist(), jaw=jaw)


if __name__ == "__main__":
    input_dir = 'D:/UST/AI/3DTeethSeg22_challenge/data/3dteethseg/raw'
    scan_segmentation = ScanSegmentation(max_vertices=8192)

    try:
        scan_segmentation.train(input_dir, epochs=3)
        scan_segmentation.process(input_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()