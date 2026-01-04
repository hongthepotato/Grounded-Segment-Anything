"""
Base protocol for exporters.

Defines the interface that all exporters must implement.
"""

from typing import Protocol, List, Dict, Any


class ExporterProtocol(Protocol):
    """
    Protocol for annotation exporters.
    
    Implementations convert detection results to specific formats
    (COCO, YOLO, Pascal VOC, etc.).
    """

    @staticmethod
    def export(
        results: List[Dict[str, Any]],
        class_prompts: List[str],
        output_mode: str = "both"
    ) -> Dict[str, Any]:
        """
        Export detection results to the target format.
        
        Args:
            results: List of detection results from AutoLabeler
            class_prompts: List of class names
            output_mode: "boxes", "masks", or "both"
            
        Returns:
            Formatted output (structure depends on format)
        """
        ...
