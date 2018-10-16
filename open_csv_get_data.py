#读取文件
data_file = open("mnist_test_10.csv", "r")
data_list = data_file.readlines()
print(data_list[5])
data_file.close()

#这个代码我们来将上面的数字来用像素表示，使用imshow()函数绘制数字矩形数组
import numpy as np
import matplotlib.pyplot as plot


all_value = data_list[0].split(',')
image_array = np.asfarray(all_value[1:]).reshape((28,28))
plot.imshow(image_array, cmap='Greys', interpolation='None')
plot.show()

print(type(data_list[0]))
print(len(data_list[0]))
print(image_array.shape)