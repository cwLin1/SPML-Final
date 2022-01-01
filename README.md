# SPML-Final

train.py: 自然訓練演算法，用來跟實驗結果比較

noisy_training.py: 對於 noisy label 的訓練演算法  <br /><br />

演算法: Multi-Model Multi-Round Outlier Removal Label Refurbishment Learning Algorithm

(其實就是整合論文提供的一些建議訓練方法XD)  <br /><br />
  
  

每輪訓練使用一個 model，計算 Feature Space，將部分 Outlier Sample 移除

接著用另一個 model，使用剩下的資料訓練，然後用此 model 翻新 Label，然後跑下一輪訓練    <br /><br />

因為 Dropout Layer 的隨機性，每次訓練結果不盡相同，Label Accuracy 大約能達到 85%-88% 左右，Test Accuracy 能達到 88% 以上，
有時候運氣好可以達到 90% 以上的 Test Accuracy<br /><br />

結果比較: (r := label noise ratio)

* r = 0.1:
  * natural training: 0.88
  * noisy training: 0.925

* r = 0.2:
  * natural training: 0.785
  * noisy training: 0.915
