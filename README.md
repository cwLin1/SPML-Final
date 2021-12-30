# SPML-Final

train.py: 自然訓練演算法，用來跟實驗結果比較

noisy_training.py: 對於 noisy label 的訓練演算法

演算法: Multi-Model Multi-Round Outlier Removal Label Refurbishment Learning Algorithm

(其實就是整合論文提供的訓練方法XD)

\ 

每輪訓練使用一個 model，計算 Feature Space，將部分 Outlier Sample 移除

接著用另一個 model，使用剩下的資料訓練，然後用此 model 翻新 Label

然後跑下一輪訓練

\ 

因為 Dropout Layer 的隨機性，每次訓練結果不盡相同，運氣好可以達到 90% 以上的 Test Accuracy
