[explainability_report.txt](https://github.com/user-attachments/files/19619949/explainability_report.txt)# hackrush25

Pranjal Goyal
24110274

Model link:
Google Colab: [https://colab.research.google.com/drive/1eCQIYKwD6cRcA4bki3G3f3y5uRV11LVl#scrollTo=SkbQy09YltrS]

## Explainability Report
AI vs. Real Image Classification - Explainability Report
============================================================

Model: ResNet50
Explainability Method: Gradient-weighted Class Activation Mapping (Grad-CAM)

Example Images:

Example 1:
  Filename: 1.jpg
  Prediction: AI
  Confidence: 0.9883
  Visualization saved to: explainability_examples/example_1.png
  Explanation: The model detected patterns typical of AI-generated content,
    such as perfect symmetry, unnatural textures, or artifacts in details like eyes and hair.

Example 2:
  Filename: 1000.jpg
  Prediction: AI
  Confidence: 0.9914
  Visualization saved to: explainability_examples/example_2.png
  Explanation: The model detected patterns typical of AI-generated content,
    such as perfect symmetry, unnatural textures, or artifacts in details like eyes and hair.

Example 3:
  Filename: 1001.jpg
  Prediction: Real
  Confidence: 0.9911
  Visualization saved to: explainability_examples/example_3.png
  Explanation: The model identified natural features consistent with real photography,
    including natural imperfections, realistic lighting, and authentic texture details.

Example 4:
  Filename: 1004.jpg
  Prediction: AI
  Confidence: 0.9998
  Visualization saved to: explainability_examples/example_4.png
  Explanation: The model detected patterns typical of AI-generated content,
    such as perfect symmetry, unnatural textures, or artifacts in details like eyes and hair.

Example 5:
  Filename: 1008.jpg
  Prediction: AI
  Confidence: 0.9996
  Visualization saved to: explainability_examples/example_5.png
  Explanation: The model detected patterns typical of AI-generated content,
    such as perfect symmetry, unnatural textures, or artifacts in details like eyes and hair.

Decision Boundary Analysis:
  The model's decision boundary separates AI from real images based on learned
  features that distinguish synthetic from natural content. The heatmaps highlight
  regions that contribute most strongly to classification decisions.

Heuristic Analysis:
  Of 5000 total images, 2201 (44.0%) were classified
  using the width-height ratio heuristic, which significantly improved inference speed.
