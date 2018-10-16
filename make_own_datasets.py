
#使用自己的图片创建数据集
import imageio#image io 该类包含一些用来查找imageReader(读取图片)和写入图片以及执行的简单编码和解码的静态快捷键
import glob
#glob是python自己带的一个文件操作相关模块，用它可以查找符合自己的母的文件，就类似于windows下的文件搜索，支持通配操作符，*，？，[]这三个通配符
#  ‘*’：代表0个或多个字符；‘？’：代表一个字符；‘[]’：匹配指定范围内的字符，如[0-9]匹配数字
import numpy
import matplotlib.pyplot as plot


our_own_dataset = []

for image_file_name in glob.glob('2828_my_own_?.png'):
    print("Loading ...", image_file_name)
    label = int(image_file_name[-5:-4])
    img_array = imageio.imread(image_file_name,as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data/255.0*0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    record = numpy.append(label, img_data)
    print(record)
    our_own_dataset.append(record)
    print(record.shape)
    print(type(record))
    print(our_own_dataset)
    pass
plot.imshow(our_own_dataset[1][1:].reshape(28, 28), cmap='Greys', interpolation='None')
plot.show()