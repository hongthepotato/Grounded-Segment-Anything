"""
Model Report Generator for creating user-friendly evaluation reports.

This module provides:
- ModelReportGenerator: Generates JSON reports with metrics and recommendations
- Identifies performance issues and provides improvement suggestions
- Supports both detection (Grounding DINO) and segmentation (SAM) models
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ModelReportGenerator:
    """
    Generate user-friendly evaluation reports.
    
    Creates comprehensive reports that include:
    - Overall model score and grade
    - Technical metrics (mAP, IoU, etc.)
    - Simple metrics (detection rate, accuracy rate)
    - Per-class performance breakdown
    - Improvement recommendations
    
    Example:
        >>> generator = ModelReportGenerator()
        >>> report = generator.generate_report(
        ...     evaluation_results=results,
        ...     model_name='grounding_dino',
        ...     test_set_size=150
        ... )
        >>> generator.save_report(report, 'evaluation_report.json')
    """
    
    # Thresholds for generating recommendations
    LOW_SAMPLE_THRESHOLD = 50
    LOW_SCORE_THRESHOLD = 50
    MEDIUM_SCORE_THRESHOLD = 70
    
    def __init__(self):
        """Initialize report generator."""
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        model_name: str,
        test_set_size: int,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from ModelEvaluator
            model_name: Name of the model ('grounding_dino' or 'sam')
            test_set_size: Number of images in test set
            extra_info: Optional additional information to include
        
        Returns:
            Complete report dictionary
        """
        model_type = evaluation_results.get('model_type', 'detection')
        technical_metrics = evaluation_results.get('technical_metrics', {})
        simple_metrics = evaluation_results.get('simple_metrics', {})
        samples = evaluation_results.get('samples', {})
        
        # Generate per-class performance from simple metrics
        per_class_performance = simple_metrics.get('per_class', [])
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(
            technical_metrics, simple_metrics, per_class_performance, model_type
        )
        
        # Build the report
        report = {
            'model_name': model_name,
            'model_type': model_type,
            'evaluation_date': datetime.now().isoformat(),
            'test_set_size': test_set_size,
            
            'simple_metrics': {
                'overall_score': simple_metrics.get('overall_score', 0),
                'grade': simple_metrics.get('grade', 'Unknown'),
                'summary': simple_metrics.get('summary', ''),
            },
            
            'technical_metrics': technical_metrics,
            
            'per_class_performance': per_class_performance,
            
            'recommendations': recommendations,
            
            'samples': {
                'success_count': len(samples.get('success', [])),
                'failure_count': len(samples.get('failure', [])),
                'success_files': [s['file_name'] for s in samples.get('success', [])],
                'failure_files': [s['file_name'] for s in samples.get('failure', [])]
            }
        }
        
        # Add model-specific simple metrics
        if model_type == 'segmentation':
            report['simple_metrics']['coverage_rate'] = simple_metrics.get('coverage_rate', 0)
            report['simple_metrics']['quality_rate'] = simple_metrics.get('quality_rate', 0)
        
        # Add extra info if provided
        if extra_info:
            report['extra_info'] = extra_info
        
        return report
    
    def _generate_recommendations(
        self,
        technical_metrics: Dict[str, Any],
        simple_metrics: Dict[str, Any],
        per_class_performance: List[Dict],
        model_type: str
    ) -> List[str]:
        """
        Generate improvement recommendations based on metrics.
        
        Analyzes:
        - Overall performance
        - Per-class weaknesses
        - Sample count issues
        - Precision/recall balance
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Overall score analysis
        overall_score = simple_metrics.get('overall_score', 0)
        
        if overall_score < self.LOW_SCORE_THRESHOLD:
            recommendations.append(
                f"Overall performance is low ({overall_score:.0f}/100). "
                "Consider: (1) Adding more training data, (2) Training for more epochs, "
                "(3) Using data augmentation."
            )
        elif overall_score < self.MEDIUM_SCORE_THRESHOLD:
            recommendations.append(
                f"Overall performance is average ({overall_score:.0f}/100). "
                "Focus on improving weak classes to boost overall score."
            )
        
        # mAP analysis for detection
        if model_type == 'detection':
            mAP50 = technical_metrics.get('mAP50', 0)
            mAP50_95 = technical_metrics.get('mAP50_95', 0)
            
            if mAP50 > 0.8 and mAP50_95 < 0.5:
                recommendations.append(
                    "Good detection at IoU=0.5 but lower at stricter thresholds. "
                    "Consider improving localization accuracy with more training."
                )
        
        # Per-class analysis
        weak_classes = []
        low_sample_classes = []
        
        for class_info in per_class_performance:
            class_name = class_info.get('class', 'unknown')
            class_score = class_info.get('score', 0)
            sample_count = class_info.get('sample_count', 0)
            
            if sample_count < self.LOW_SAMPLE_THRESHOLD:
                low_sample_classes.append((class_name, sample_count))
            
            if class_score < self.LOW_SCORE_THRESHOLD:
                weak_classes.append((class_name, class_score))
        
        # Low sample warnings
        if low_sample_classes:
            class_list = ", ".join([f"'{name}' ({count} samples)" for name, count in low_sample_classes])
            recommendations.append(
                f"Classes with few training samples: {class_list}. "
                "Results may be unreliable. Add more examples for these classes."
            )
        
        # Weak class warnings
        if weak_classes:
            # Sort by score (worst first)
            weak_classes.sort(key=lambda x: x[1])
            worst_classes = weak_classes[:3]  # Top 3 worst
            
            class_list = ", ".join([f"'{name}' ({score:.0f}%)" for name, score in worst_classes])
            recommendations.append(
                f"Weak performing classes: {class_list}. "
                "Consider adding more training examples or improving annotation quality for these classes."
            )
        
        # Class imbalance check
        if per_class_performance:
            scores = [c.get('score', 0) for c in per_class_performance]
            score_range = max(scores) - min(scores) if scores else 0
            
            if score_range > 40:
                recommendations.append(
                    f"Large performance gap between classes ({score_range:.0f}% difference). "
                    "Consider balancing your dataset or using class-weighted loss."
                )
        
        # If everything is good
        if not recommendations:
            if overall_score >= 90:
                recommendations.append(
                    "Excellent performance! Model is ready for deployment. "
                    "Consider testing on edge cases and monitoring performance in production."
                )
            elif overall_score >= 80:
                recommendations.append(
                    "Very good performance! For further improvement, "
                    "focus on edge cases and difficult examples."
                )
        
        return recommendations
    
    def generate_summary_text(self, report: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the report.
        
        Args:
            report: The generated report dictionary
        
        Returns:
            Formatted string summary
        """
        lines = []
        
        lines.append("=" * 60)
        lines.append("MODEL EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Model info
        lines.append(f"Model: {report['model_name']}")
        lines.append(f"Type: {report['model_type']}")
        lines.append(f"Evaluation Date: {report['evaluation_date']}")
        lines.append(f"Test Set Size: {report['test_set_size']} images")
        lines.append("")
        
        # Simple metrics
        simple = report['simple_metrics']
        lines.append("-" * 60)
        lines.append("OVERALL PERFORMANCE")
        lines.append("-" * 60)
        lines.append(f"Score: {simple['overall_score']:.1f}/100 ({simple['grade']})")
        lines.append("")
        lines.append(simple.get('summary', ''))
        lines.append("")
        
        if report['model_type'] == 'segmentation':
            lines.append(f"Coverage Rate: {simple.get('coverage_rate', 0):.1f}%")
            lines.append(f"Quality Rate: {simple.get('quality_rate', 0):.1f}%")
            lines.append("")
        
        # Technical metrics
        tech = report['technical_metrics']
        lines.append("-" * 60)
        lines.append("TECHNICAL METRICS")
        lines.append("-" * 60)
        
        if report['model_type'] == 'detection':
            lines.append(f"mAP@50: {tech.get('mAP50', 0):.3f}")
            lines.append(f"mAP@50-95: {tech.get('mAP50_95', 0):.3f}")
        else:
            lines.append(f"mIoU: {tech.get('mIoU', 0):.3f}")
            lines.append(f"Mean Dice: {tech.get('mean_dice', 0):.3f}")
            lines.append(f"Precision: {tech.get('precision', 0):.3f}")
            lines.append(f"Recall: {tech.get('recall', 0):.3f}")
        lines.append("")
        
        # Per-class performance
        per_class = report.get('per_class_performance', [])
        if per_class:
            lines.append("-" * 60)
            lines.append("PER-CLASS PERFORMANCE")
            lines.append("-" * 60)
            
            for class_info in per_class:
                warning = f" [!{class_info.get('warning', '')}]" if class_info.get('warning') else ""
                lines.append(
                    f"  {class_info['class']:20s}: {class_info['score']:5.1f}% "
                    f"({class_info['grade']}) - {class_info['sample_count']} samples{warning}"
                )
            lines.append("")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            lines.append("-" * 60)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 60)
            
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        also_save_text: bool = True
    ) -> None:
        """
        Save report to JSON file.
        
        Args:
            report: The generated report dictionary
            output_path: Path to save JSON file
            also_save_text: Whether to also save a text summary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved evaluation report to: %s", output_path)
        
        # Save text summary
        if also_save_text:
            text_path = output_path.with_suffix('.txt')
            summary_text = self.generate_summary_text(report)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            logger.info("Saved text summary to: %s", text_path)
    
    def combine_reports(
        self,
        reports: List[Dict[str, Any]],
        combined_name: str = 'combined'
    ) -> Dict[str, Any]:
        """
        Combine multiple model reports into a single report.
        
        Useful when evaluating both detection and segmentation models.
        
        Args:
            reports: List of individual reports
            combined_name: Name for the combined report
        
        Returns:
            Combined report dictionary
        """
        combined = {
            'report_name': combined_name,
            'evaluation_date': datetime.now().isoformat(),
            'model_reports': reports,
            'summary': {}
        }
        
        # Calculate combined summary
        total_score = 0
        for report in reports:
            score = report.get('simple_metrics', {}).get('overall_score', 0)
            total_score += score
        
        avg_score = total_score / len(reports) if reports else 0
        combined['summary'] = {
            'average_score': round(avg_score, 1),
            'num_models': len(reports),
            'model_names': [r.get('model_name', 'unknown') for r in reports]
        }
        
        # Combine all recommendations
        all_recommendations = []
        for report in reports:
            model_name = report.get('model_name', 'unknown')
            for rec in report.get('recommendations', []):
                all_recommendations.append(f"[{model_name}] {rec}")
        
        combined['combined_recommendations'] = all_recommendations
        
        return combined

