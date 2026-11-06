"""
Core Computer Vision Engine: 1:1 Identity Verification Gate
- MTCNN: Face detection, boundary validation, and landmark alignment
- InceptionResnetV1 (pretrained on VGGFace2): 512-dimensional feature embedding extraction
- Metric Engine: Cosine similarity and Euclidean distance calculation
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


class FaceVerificationEngine:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # MTCNN: Aligns faces to 160x160 with facial landmark rotation
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=40,
            select_largest=True,
            post_process=True,
            device=self.device,
        )

        # Inception-ResNet-v1: 512-D embedding extractor
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def extract_face(
        self, image: Image.Image
    ) -> Tuple[Optional[torch.Tensor], Optional[Image.Image], Optional[str]]:
        """
        Validates KYC compliance (checks for exactly 1 face), aligns the face
        tensor for the embedding model, and extracts a display crop.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Step 1: Detect face count and bounding boxes
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None or len(boxes) == 0:
            return None, None, "No face detected. Please provide a clear, well-lit photo."

        # Filter out low-confidence ghost detections
        valid_boxes = [box for box, prob in zip(boxes, probs) if prob is not None and prob >= 0.85]
        if len(valid_boxes) == 0:
            return None, None, "No high-confidence face found in the image."

        if len(valid_boxes) > 1:
            return (
                None,
                None,
                f"Multiple faces detected ({len(valid_boxes)}). 1:1 verification requires exactly one subject.",
            )

        # Step 2: Extract aligned PyTorch tensor (3, 160, 160)
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            return None, None, "Failed to align facial landmarks."

        # Step 3: Extract clean visual crop for UI inspection
        x1, y1, x2, y2 = valid_boxes[0]
        # Add 10% padding around bounding box for display
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - 0.1 * w))
        y1 = max(0, int(y1 - 0.1 * h))
        x2 = min(image.width, int(x2 + 0.1 * w))
        y2 = min(image.height, int(y2 + 0.1 * h))
        preview_crop = image.crop((x1, y1, x2, y2))

        return face_tensor, preview_crop, None

    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Passes the aligned face tensor through InceptionResnetV1 to produce a 512-D vector."""
        with torch.no_grad():
            tensor_batch = face_tensor.unsqueeze(0).to(self.device)
            embedding = self.resnet(tensor_batch)
            # Normalize embedding to unit length (L2 norm = 1)
            return F.normalize(embedding, p=2, dim=1)

    def verify(
        self,
        ref_image: Image.Image,
        probe_image: Image.Image,
        threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end 1:1 pairwise identity verification.
        Returns similarity metrics, decision verdict, and cropped visual previews.
        """
        # Process Reference image
        ref_tensor, ref_crop, ref_err = self.extract_face(ref_image)
        if ref_err:
            return {"success": False, "error": f"Reference Image: {ref_err}"}

        # Process Probe image
        probe_tensor, probe_crop, probe_err = self.extract_face(probe_image)
        if probe_err:
            return {"success": False, "error": f"Probe Image: {probe_err}"}

        # Extract normalized 512-D embeddings
        emb_ref = self.get_embedding(ref_tensor)
        emb_probe = self.get_embedding(probe_tensor)

        # Calculate metrics:
        # 1. Cosine Similarity: Dot product of L2-normalized vectors ([-1.0, 1.0])
        cosine_sim = torch.mm(emb_ref, emb_probe.t()).item()
        
        # 2. Euclidean Distance: Geometric distance in 512-D space
        euclidean_dist = torch.dist(emb_ref, emb_probe).item()

        # Decision rule
        is_match = cosine_sim >= threshold

        return {
            "success": True,
            "match": is_match,
            "cosine_similarity": float(cosine_sim),
            "euclidean_distance": float(euclidean_dist),
            "threshold_used": float(threshold),
            "ref_crop": ref_crop,
            "probe_crop": probe_crop,
            "error": None,
        }