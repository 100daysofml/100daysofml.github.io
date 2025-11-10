#!/usr/bin/env python3
"""
Suggest ML topics for lessons 36-100 based on a comprehensive curriculum map.
Usage: python suggest_topics.py <lesson_number>
"""

import sys
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Week:
    week_number: int
    theme: str
    topics: List[str]

# Comprehensive curriculum map for weeks 8-20 (lessons 36-100)
CURRICULUM_MAP: Dict[int, Week] = {
    8: Week(
        week_number=8,
        theme="Transfer Learning & Domain Adaptation",
        topics=[
            "Transfer Learning fundamentals and use cases",
            "Fine-tuning pre-trained models",
            "Feature extraction vs end-to-end learning",
            "Domain adaptation techniques",
            "Multi-task learning"
        ]
    ),
    9: Week(
        week_number=9,
        theme="Generative Adversarial Networks (GANs)",
        topics=[
            "GAN architecture and training dynamics",
            "Generator and discriminator networks",
            "Loss functions and convergence issues",
            "Conditional GANs (cGANs)",
            "Applications: image synthesis and style transfer"
        ]
    ),
    10: Week(
        week_number=10,
        theme="Transformer Architecture Fundamentals",
        topics=[
            "Attention mechanisms and self-attention",
            "Transformer encoder-decoder architecture",
            "Positional encoding and embeddings",
            "Multi-head attention",
            "Vision Transformers (ViT)"
        ]
    ),
    11: Week(
        week_number=11,
        theme="BERT and Advanced NLP",
        topics=[
            "BERT pre-training and fine-tuning",
            "Masked Language Modeling (MLM)",
            "Next Sentence Prediction (NSP)",
            "Named Entity Recognition and text classification",
            "Semantic similarity and sentence embeddings"
        ]
    ),
    12: Week(
        week_number=12,
        theme="Reinforcement Learning",
        topics=[
            "Markov Decision Processes (MDPs)",
            "Q-Learning and Deep Q-Networks (DQN)",
            "Policy gradient methods and PPO",
            "Actor-Critic algorithms",
            "Multi-agent reinforcement learning"
        ]
    ),
    13: Week(
        week_number=13,
        theme="Advanced Computer Vision",
        topics=[
            "Object detection: YOLO, Faster R-CNN, SSD",
            "Instance segmentation and Mask R-CNN",
            "Semantic segmentation techniques",
            "Pose estimation and keypoint detection",
            "3D computer vision fundamentals"
        ]
    ),
    14: Week(
        week_number=14,
        theme="Time Series & Sequence Modeling",
        topics=[
            "Time series decomposition and analysis",
            "ARIMA and statistical forecasting",
            "LSTM and GRU for sequence learning",
            "Temporal Convolutional Networks (TCN)",
            "Attention-based sequence models and Temporal Fusion"
        ]
    ),
    15: Week(
        week_number=15,
        theme="AutoML & Hyperparameter Tuning",
        topics=[
            "Grid search, random search, and Bayesian optimization",
            "Hyperband and successive halving",
            "Neural Architecture Search (NAS)",
            "AutoML frameworks: Auto-sklearn, TPOT",
            "Meta-learning and few-shot learning"
        ]
    ),
    16: Week(
        week_number=16,
        theme="Model Deployment & Serving",
        topics=[
            "Model serialization: pickle, joblib, ONNX",
            "REST API development with Flask/FastAPI",
            "Docker containerization for ML models",
            "Model versioning and management",
            "Batch prediction and real-time inference"
        ]
    ),
    17: Week(
        week_number=17,
        theme="MLOps & Production ML",
        topics=[
            "ML pipeline orchestration: Airflow, Kubeflow",
            "Data versioning and DVC",
            "Model monitoring and drift detection",
            "A/B testing and canary deployments",
            "Feature stores and data governance"
        ]
    ),
    18: Week(
        week_number=18,
        theme="Graph Neural Networks",
        topics=[
            "Graph representations and graph convolutions",
            "Message passing and graph embedding",
            "Graph Convolutional Networks (GCN)",
            "Attention-based graph networks (GAT)",
            "Knowledge graphs and link prediction"
        ]
    ),
    19: Week(
        week_number=19,
        theme="Federated & Distributed Learning",
        topics=[
            "Federated learning fundamentals",
            "Communication-efficient learning",
            "Privacy-preserving machine learning",
            "Differential privacy techniques",
            "Distributed training and gradient synchronization"
        ]
    ),
    20: Week(
        week_number=20,
        theme="Advanced Optimization & Research",
        topics=[
            "Gradient descent variants and adaptive methods",
            "Loss landscape visualization",
            "Lottery ticket hypothesis",
            "Model pruning, quantization, and distillation",
            "Neural ODE and continuous learning"
        ]
    )
}


def get_topic_for_lesson(lesson_number: int) -> Dict:
    """
    Get suggested topic for a given lesson number.

    Args:
        lesson_number: Lesson number (36-100)

    Returns:
        Dictionary with week, theme, suggested topics, and context
    """
    if lesson_number < 36 or lesson_number > 100:
        return {
            "error": "Lesson number must be between 36 and 100",
            "lesson": lesson_number
        }

    # Calculate week number (lesson 36 = week 8, lesson 41 = week 9, etc.)
    week_number = 8 + (lesson_number - 36) // 5
    day_in_week = (lesson_number - 36) % 5 + 1

    if week_number not in CURRICULUM_MAP:
        return {
            "error": f"Week {week_number} not found in curriculum map",
            "lesson": lesson_number
        }

    week = CURRICULUM_MAP[week_number]
    topic = week.topics[day_in_week - 1]

    return {
        "lesson": lesson_number,
        "week": week_number,
        "day_in_week": day_in_week,
        "theme": week.theme,
        "topic": topic,
        "all_week_topics": week.topics,
        "difficulty": get_difficulty_level(week_number),
        "prerequisite_weeks": get_prerequisites(week_number)
    }


def get_difficulty_level(week_number: int) -> str:
    """Determine difficulty level based on week."""
    if week_number <= 10:
        return "Intermediate"
    elif week_number <= 15:
        return "Advanced"
    else:
        return "Expert"


def get_prerequisites(week_number: int) -> List[str]:
    """Return prerequisite topics for a given week."""
    prerequisites = {
        8: ["Deep Learning fundamentals", "CNN and RNN basics"],
        9: ["Neural network training", "Loss functions"],
        10: ["Attention mechanisms basics", "Sequence models"],
        11: ["NLP fundamentals", "Word embeddings"],
        12: ["Probability and statistics", "Deep learning"],
        13: ["CNN architecture", "Image processing"],
        14: ["RNN/LSTM basics", "Sequence modeling"],
        15: ["Model evaluation", "Hyperparameter basics"],
        16: ["Python development", "Model training"],
        17: ["Model deployment", "Data pipelines"],
        18: ["Graph theory basics", "Neural networks"],
        19: ["Distributed systems", "Privacy concepts"],
        20: ["Optimization theory", "Neural networks"]
    }
    return prerequisites.get(week_number, [])


def print_topic_suggestion(suggestion: Dict) -> None:
    """Pretty print topic suggestion."""
    if "error" in suggestion:
        print(f"Error: {suggestion['error']}")
        return

    print(f"\n{'='*60}")
    print(f"LESSON {suggestion['lesson']} - Week {suggestion['week']}, Day {suggestion['day_in_week']}")
    print(f"{'='*60}")
    print(f"Theme: {suggestion['theme']}")
    print(f"Difficulty: {suggestion['difficulty']}")
    print(f"\nToday's Topic: {suggestion['topic']}")
    print(f"\nThis Week's Topics:")
    for i, t in enumerate(suggestion['all_week_topics'], 1):
        marker = "→" if t == suggestion['topic'] else " "
        print(f"  {marker} Day {i}: {t}")
    print(f"\nPrerequisites: {', '.join(suggestion['prerequisite_weeks'])}")
    print(f"{'='*60}\n")


def list_all_topics() -> None:
    """List all topics in the curriculum."""
    print(f"\n{'='*60}")
    print("ML CURRICULUM: Weeks 8-20 (Lessons 36-100)")
    print(f"{'='*60}\n")

    for week_num in sorted(CURRICULUM_MAP.keys()):
        week = CURRICULUM_MAP[week_num]
        lesson_start = 36 + (week_num - 8) * 5
        lesson_end = lesson_start + 4
        print(f"Week {week_num} (Lessons {lesson_start}-{lesson_end}): {week.theme}")
        for i, topic in enumerate(week.topics, 1):
            print(f"  Day {i}: {topic}")
        print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python suggest_topics.py <lesson_number>")
        print("       python suggest_topics.py --list")
        print("\nExample: python suggest_topics.py 42")
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_all_topics()
        return

    try:
        lesson_number = int(sys.argv[1])
        suggestion = get_topic_for_lesson(lesson_number)
        print_topic_suggestion(suggestion)
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid lesson number")
        sys.exit(1)


if __name__ == "__main__":
    main()
