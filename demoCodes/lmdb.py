"""
LMDB文件可以同时由多个进程打开，具有极高的数据存取速度，访问简单，不需要运行单独的数据库管理进程，只要在访问数据的代码里引用LMDB库，访问时给文件路径即可。

让系统访问大量小文件的开销很大，而LMDB使用内存映射的方式访问文件，使得文件内寻址的开销非常小，使用指针运算就能实现。数据库单文件还能减少数据集复制 / 传输过程的开销。

在python中使用lmdb： linux中，可以使用指令‘pip
install
lmdb’ 安装lmdb包。
"""



# 1.
# 生成一个空的lmdb数据库文件

# -*- coding: utf-8 -*-
import lmdb

# 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
# map_size定义最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./train"，map_size = 1099511627776)
env.close()

# 2.
# LMDB数据的添加、修改、删除

# -*- coding: utf-8 -*-
import lmdb

# map_size定义最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./train", map_size=1099511627776)

txn = env.begin(write=True)

# 添加数据和键值
txn.put(key='1', value='aaa')
txn.put(key='2', value='bbb')
txn.put(key='3', value='ccc')

# 通过键值删除数据
txn.delete(key='1')

# 修改数据
txn.put(key='3', value='ddd')

# 通过commit()函数提交更改
txn.commit()
env.close()

# 3.
# 查询lmdb数据库内容

# -*- coding: utf-8 -*-
import lmdb

env = lmdb.open("./train")

# 参数write设置为True才可以写入
txn = env.begin(write=True)
############################################添加、修改、删除数据

# 添加数据和键值
txn.put(key='1', value='aaa')
txn.put(key='2', value='bbb')
txn.put(key='3', value='ccc')

# 通过键值删除数据
txn.delete(key='1')

# 修改数据
txn.put(key='3', value='ddd')

# 通过commit()函数提交更改
txn.commit()
############################################查询lmdb数据
txn = env.begin()

# get函数通过键值查询数据
print
txn.get(str(2))

# 通过cursor()遍历所有数据和键值
for key, value in txn.cursor():
    print(key, value)

############################################


env.close()

# 4.
# 读取已有.mdb文件内容

# -*- coding: utf-8 -*-
import lmdb

env_db = lmdb.Environment('trainC')
# env_db = lmdb.open("./trainC")

txn = env_db.begin()

# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
print
txn.get(str(200))

for key, value in txn.cursor():  # 遍历
    print(key, value)

env_db.close()
# ---------------------
# 作者：-牧野 -
# 来源：CSDN
# 原文：https: // blog.csdn.net / dcrmg / article / details / 79144507
# 版权声明：本文为博主原创文章，转载请附上博文链接！