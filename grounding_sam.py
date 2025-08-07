from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

class GroundingSAM:
    def __init__(
            self,
            detector_id: Optional[str] = None,
            segmenter_id: Optional[str] = None):

        if detector_id is None:
            self.detector_id = "IDEA-Research/grounding-dino-tiny"
        else:
            self.detector_id = detector_id

        if segmenter_id is None:
            self.segmenter_id = "facebook/sam-vit-base"
        else:
            self.segmenter_id = segmenter_id

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # DINO
        self.detector = pipeline(model=self.detector_id, task="zero-shot-object-detection", device=self._device)

        # SAM
        self.segmenter = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self._device)
        self.segmenter_processor = AutoProcessor.from_pretrained(self.segmenter_id)

    def _get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]
    
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def _polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask
    
    def _refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self._mask_to_polygon(mask)
                mask = self._polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def _detect(
            self,
            image: Image.Image,
            labels: List[str],
            threshold: float = 0.3
            ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label+"." for label in labels]

        results = self.detector(image,  candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def _segment(
            self,
            image: Image.Image,
            detection_results: List[Dict[str, Any]],
            polygon_refinement: bool = False,
            ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """

        boxes = self._get_boxes(detection_results)
        inputs = self.segmenter_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self._device)

        outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self._refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
            self,
            image: Image.Image,
            labels: List[str],
            threshold: float = 0.3,
            polygon_refinement: bool = False):
        
        detections = self._detect(image, labels, threshold)
        detections = self._segment(image, detections, polygon_refinement)

        return detections
    
    def __call__(
            self,
            image: Image.Image,
            labels: List[str],
            threshold: float = 0.3,
            polygon_refinement: bool = False):
        return self.grounded_segmentation(image, labels, threshold, polygon_refinement)