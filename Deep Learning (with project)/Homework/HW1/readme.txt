1. read_data.py：讀取 MNIST 資料，並將資料整理後分割成 train, validation, test 三份，匯出成 data_set.npy.npz
2. train.py：匯入 data_set.npy.npz 資料用以訓練模型，訓練完成後的參數匯出成 weight.npy (dictionary 型態，key : "W1","b1","W2","b2","W3","b3")
3. test.py：匯入 data_set.npy.npz 中的 test data 和 weight.py，並計算模型對 test data 預測的 accuracy 和 loss