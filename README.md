# PEFT-LORA
Paramater Efficient Fine Tuning using LORA to predict review sentiments

PEFT Movie Review Sentiment Analysis
Overview
This project demonstrates Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) to improve sentiment analysis performance on movie reviews. We fine-tune a pre-trained DistilBERT model on the IMDB dataset, comparing its performance before and after PEFT.

Key Features

Efficient Fine-tuning: Uses LoRA to train only ~0.3% of the model's parameters
Performance Comparison: Clear before/after metrics showing improvement
Interactive Demo: Test the model with custom movie reviews
GPU Support: Automatically detects and uses CUDA if available

Hardware Requirements


Minimum: 4GB RAM, CPU only

Recommended: 8GB RAM, GPU with 4GB+ VRAM

Software Dependencies
bashtorch>=1.13.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.12.0
evaluate>=0.4.0
accelerate>=0.20.0
numpy>=1.21.0

Model Architecture
Base Model: DistilBERT

66M parameters (40% smaller than BERT)
6 transformer layers
Optimized for sequence classification

PEFT Method: LoRA

Adds low-rank decomposition matrices to attention layers
Trains only ~298K parameters (0.45% of total)
Maintains original model weights frozen

Performance Metrics
MetricBase ModelPEFT ModelImprovementAccuracy~50%~87%+37%F1 Score~50%~87%+37%Training TimeN/A~10 min (GPU)-Trainable Params66.6M298K99.5% reduction
