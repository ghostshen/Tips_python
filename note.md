1. ##### split()自带空格分割，而且自动删除空白内容；

2. ##### list元素计数：
	
	```python
	1------------------------
	from collections import Counter
	a = [1, 2, 3, 1, 1, 2]
	result = Counter(a)
	print result
	
	2------------------------
	import pandas as pd
	a = [1, 2, 3, 1, 1, 2]
	result = pd.value_counts(a)
	print result
	```
	
3. ##### 字典排序：
	
	```python
	1-按key：sorted(dict1.items(), key=lambda d: d[0]，reverse = True)
	2-按value：sorted(dict1.items(), key=lambda d: d[1])
	```
	
4. ##### 字典key和value反转——仅在一对一情况下操作：
	
	```python
	m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
	mi = dict(zip(m.values(), m.keys())) # mi:{1: 'a', 2: 'b', 3: 'c', 4: 'd'}
	```

5. ##### 进度条
	
	```python
	from tqdm import tqdm
	for i in tqdm(range(num_walks)):
	    pass
	```

6. ##### 随机抽取元素
	
	```python
	from random import sample
	l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	print(sample(l, 5)) # 随机抽取5个元素
	```
	
7. ##### list求交并补等
	
	```python
	#求交集
	retB = list(set(listA).intersection(set(listB)))
	#求并集
	retC = list(set(listA).union(set(listB)))
	#求差集
	retD = list(set(listB).difference(set(listA)))
	```

7. ##### 确认文件或者路径是否存在
	
	```python
	import os
	os.path.isfile('test.txt') #如果不存在就返回False
	os.path.exists(directory) #如果目录不存在就返回False
	```
	
8. ##### terminal跑代码设置参数（基于tensorflow）
	
	```python
	import tensorflow as tf
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_string('dataset', 'test_2_multi', 'Dataset string.') 
	#[(名称)，(参数值)，(备注)]
	```
	
9. ##### split()计数分割
	
	```python
	str.split(str=" ",num=string.count(str))[n]
	# 参数说明：
	# str： 表示为分隔符，默认为空格，但是不能为空。若字符串中没有分隔符，则把整个字符串作为列表的一个元素
	# num：表示分割次数。如果存在参数num，则仅分隔成 num+1 个子字符串，并且每一个子字符串可以赋给新的变量
	# [n]： 表示选取第n个分片，也可以是[0:n]表示选取第0片到第n片，函数返回一个list.
	```
	
10. ##### numpy输出设置
	
	```python
	np.set_printoptions(linewidth=90)
	```
	
11. ##### 服务器查看GPU使用情况：
	
	```python
	nvidia-smi
	fuser -v /dev/nvidia* 发现僵尸进程（连号的）
	pmap -d PID  查看具体这个进程调用GPU的情况
	kill -9 PID  强行关掉所有当前并未执行的僵尸进程
	
	gpustat -c -p -u
	查看用户gpu使用情况
	
	free -m
	查看内存
	
	top/htop
	cpu使用情况
	```
	
13. ##### Python字典的clear()方法（删除字典内所有元素）

    ```python
    dict = {'name': '我的博客地址', 'alexa': 10000, 'url': 'http://blog.csdn.net/uuihoo/'}
    dict.clear();  # 清空词典所有条目
    ```

14. ##### Python字典的pop()方法（删除字典给定键 key 所对应的值，返回值为被删除的值）

    ```python
    site= {'name': '我的博客地址', 'alexa': 10000, 'url':'http://blog.csdn.net/uuihoo/'}
    pop_obj=site.pop('name') # 删除要删除的键值对，如{'name':'我的博客地址'}这个键值对
    print pop_obj   # 输出 ：我的博客地址
    ```

15. ##### Python字典的popitem()方法（随机返回并删除字典中的一对键和值）

    ```python
    site= {'name': '我的博客地址', 'alexa': 10000, 'url':'http://blog.csdn.net/uuihoo/'}
    pop_obj=site.popitem() # 随机返回并删除一个键值对
    print pop_obj   # 输出结果可能是{'url','http://blog.csdn.net/uuihoo/'}
    ```

16. ##### del 全局方法（能删单一的元素也能清空字典，清空只需一项操作）

    ```python
    site= {'name': '我的博客地址', 'alexa': 10000, 'url':'http://blog.csdn.net/uuihoo/'}
    del site['name'] # 删除键是'name'的条目 
    del site  # 清空字典所有条目
    ```

17. ##### 分词库、词云图

    ```python
    import jieba  # 分词库
    from wordcloud import WordCloud  # 词云库
    
    cut_text = jieba.cut(text)
    
    wc = WordCloud(font_path=r"simsun.ttf", max_font_size=100,background_color="white",height=500,width=500,max_words=500)
    wc.generate(result)
    wc.to_file("background.jpg") 
    ```

17. ##### tf.squeeze()

    ```python
    # 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果。
    # tf.squeeze和tf.expand_dims互为逆操作。
    def squeeze(input, axis=None, name=None, squeeze_dims=None)
    
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    tf.shape(tf.squeeze(t))  # [2, 3]
    
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
    
    ```

18. ##### @运算符

    ```python
    a = np.array([[1,2,1],[0,0,1],[1,0,0]])
    b = a @ a
    # b = [[2 2 3][1 0 0][1 2 1]]
    # 矩阵乘法
    ```

19. ##### np.zeros_like（）

    ```python
    scores = np.zeros_like(ys, dtype=np.float) 
    # 返回和ys的shape相同的全为0的np数组
    ```

20. ##### np.greater()

    ```python
    np.greater(a, b) # 返回a每个元素是否大于b中对应位置元素的bool值
    ```

21. ##### tf.gather()

    ```python
    temp = tf.range(0,10)*10 + tf.constant(1,shape=[10])
    temp2 = tf.gather(temp,[1,5,9])
    
    with tf.Session() as sess:
        print sess.run(temp)
        print sess.run(temp2)
    #输出
    #[ 1 11 21 31 41 51 61 71 81 91]
    #[11 51 91]
    
    ```
    

23. ##### numpy.delete()

    ```python
    import numpy as np
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    index = [2, 3, 6]
    new_a = np.delete(a, index)
    # new_a:[1, 2, 5, 6, 8, 9]
    ```

24. ##### numpy.append()

    ```python
    a = np.array([1,2,3,4])
    b = np.append(a,[1,1])
    # b:[1,2,3,4,1,1]
    
    a = array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
    b = np.append(a,[[1,1,1,1]],axis=0)
    # b:[[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11],[ 1,  1,  1,  1]]
    ```

25. ##### numpy.insert()

    ```python
    # numpy.insert(arr,obj,value,axis=None)
    # arr:为目标向量
    # obj:为目标位置
    # value:为想要插入的数值
    # axis:为插入的维度
    a = array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
    b = np.insert(a,1,[1,1,1,1],0)
    """
    b:array([[ 0,  1,  2,  3],
           [ 1,  1,  1,  1],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    """
    ```

    

