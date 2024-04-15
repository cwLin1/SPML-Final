# SPML-Final

train.py: Training with normal data for comparing with experimental results.

noisy_training.py: Training with noisy labeled data  <br /><br />

Algorithm: Multi-Model Multi-Round Outlier Removal Label Refurbishment Learning Algorithm

  

For each round of training, predict the feature space and remove part of the outlier samples.

Next, we train other model with remaining data, and refurbish the label for the next round of training.    <br /><br />


Result comparison: (r := label noise ratio)

* r = 0:
  * natural training: 0.95
  * noisy training: 0.95
  * Label Accuracy: 0.956

* r = 0.1:
  * natural training: 0.88
  * noisy training: 0.925
  * Label Accuracy: 0.901

* r = 0.2:
  * natural training: 0.785
  * noisy training: 0.915
  * Label Accuracy: 0.883

* r = 0.3:
  * natural training: 0.635
  * noisy training: 0.91
  * Label Accuracy: 0.863

* r = 0.4:
  * natural training: 0.525
  * noisy training: 0.85
  * Label Accuracy: 0.825

* r = 0.45:
  * natural training: 0.515
  * noisy training: 0.78
  * Label Accuracy: 0.72

* r = 0.5:
  * natural training: 0.55
  * noisy training: 0.42
  * Label Accuracy: 0.433
