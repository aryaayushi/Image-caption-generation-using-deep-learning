Image Caption Generator using Deep Learning on Flickr8K Dataset
📋 Project Overview
This project implements an end-to-end Image Caption Generator using deep learning techniques. The system automatically generates descriptive captions for images by combining Computer Vision (CV) and Natural Language Processing (NLP) methodologies. The model is trained on the Flickr8K dataset, which contains 8,000 images with 5 captions each.

🚀 Key Features
Multi-modal Architecture: Combines CNN for image feature extraction with LSTM for sequence generation

Dual Decoding Strategies: Implements both Greedy Search and Beam Search for caption generation

Comprehensive Evaluation: Uses BLEU scores (BLEU-1, BLEU-2) for quantitative performance assessment

Visualization Tools: Interactive visualization of images with corresponding captions

End-to-End Pipeline: From data preprocessing to model training and inference

🏗️ Architecture & Algorithm
1. Feature Extraction Module
Model: InceptionV3 (pretrained on ImageNet)

Input Size: 299×299×3 pixels

Feature Dimension: 2048-dimensional feature vectors

Modification: Removed final classification layer, used penultimate layer outputs

2. Sequence Processing Module
Tokenizer: Keras Tokenizer with vocabulary size of 8,586 words

Text Cleaning: Regex-based preprocessing for caption standardization

Sequence Padding: Fixed length of 34 tokens (max caption length + start/end tokens)

3. Core Model Architecture
text
Input Layers:
├── Image Features: (2048-dim CNN output)
└── Caption Sequence: (34-token padded sequence)

Processing Layers:
├── Image Pathway: BatchNormalization → Dense(256, ReLU) → BatchNormalization
├── Text Pathway: Embedding(256-dim) → LSTM(256)
└── Fusion: Add() operation combines both pathways

Output Layer:
└── Dense(8586, softmax) - Vocabulary probability distribution
4. Training Configuration
Optimizer: Adam with learning rate scheduling (exponential decay)

Loss Function: Categorical Crossentropy

Regularization: Early Stopping (patience=3), Batch Normalization

Batch Sizes: Train=270, Validation=150

Epochs: 8 (with early stopping)

📊 Dataset Details
Flickr8K Dataset Structure
Total Images: 8,000

Captions per Image: 5

Total Captions: 40,455

Average Caption Length: ~10-15 words

Vocabulary Size: 8,585 unique words

Data Split
Training Set: 6,877 images (85.96%)

Validation Set: 1,092 images (13.65%)

Test Set: 122 images (1.53%)

🔧 Implementation Details
Preprocessing Pipeline
Image Processing:

Resize to 299×299

Apply InceptionV3 preprocessing

Extract 2048-dim feature vectors

Text Processing:

Lowercase conversion

Special character removal

Tokenization and sequencing

Add 'start' and 'end' tokens

Data Generator
Custom generator that:

Shuffles data each epoch

Creates input-output pairs for sequence prediction

Handles variable-length sequences with padding

Batches data efficiently for GPU training

🎯 Caption Generation Strategies
1. Greedy Search
Selects the most probable word at each time step

Simple and computationally efficient

May produce suboptimal sequences

2. Beam Search (K=3)
Maintains multiple candidate sequences

Explores K-best alternatives at each step

Uses log probabilities for sequence scoring

Produces more diverse and accurate captions

📈 Performance Metrics
Evaluation Metrics
BLEU-1: Unigram precision

BLEU-2: Bigram precision

BLEU Scores: Combined evaluation of caption quality

Model Performance
Training Loss: Reduced from 4.52 to 2.37

Validation Loss: Reduced from 3.76 to 3.22

BLEU Scores: ~0.70-0.82 across test images

🛠️ Technical Stack
Core Libraries
TensorFlow/Keras: Model building and training

NumPy/Pandas: Numerical operations and data handling

Matplotlib/Seaborn: Visualization and plotting

NLTK: BLEU score calculation

PIL: Image processing

Hardware Requirements
GPU recommended for training (Google Colab/Kaggle compatible)

Minimum 8GB RAM for feature extraction

Storage: ~2GB for dataset and models

📁 Project Structure
text
├── data/
│   ├── images/               # Flickr8K images
│   └── captions.txt          # Image-caption mappings
├── notebooks/
│   └── Image_Caption_Generator.ipynb  # Main implementation
├── models/
│   └── caption_model.h5      # Trained model weights
├── utils/
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── visualization.py      # Visualization utilities
│   └── evaluation.py         # BLEU score calculation
└── README.md                 # This file
🔍 Key Techniques & Algorithms
Deep Learning Techniques
Transfer Learning: Leveraging pretrained InceptionV3 for feature extraction

Attention Mechanism: Implicit attention through LSTM fusion

Sequence-to-Sequence: Image-to-text generation framework

Teacher Forcing: Training with ground truth previous words

NLP Techniques
Word Embeddings: 256-dimensional learned embeddings

Sequence Padding: Handling variable-length captions

Tokenization: Word-level tokenization with OOV handling

Optimization Techniques
Learning Rate Scheduling: Exponential decay

Early Stopping: Preventing overfitting

Batch Normalization: Stabilizing training

Gradient Clipping: Preventing exploding gradients
