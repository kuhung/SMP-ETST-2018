from lib.sentence2vec import Sentence2Vec

model = Sentence2Vec('../feature/content_200.model')

# turn job title to vector
print(model.get_vector(' FT IR 产物 结构 进行 分析 扫描电镜 查看 喷洒 抑尘剂 煤粉 固化 表面 形貌'))

# not similar job
print(model.similarity(' FT IR 产物 结构 进行 分析 扫描电镜 查看 喷洒 抑尘剂 煤粉 固化 表面 形貌',
                       '方法 利用 硅胶 ODS 及大孔 树脂 HP 多种 色谱 方法 对密 花石 豆兰 乙醇 提取物 进行 分离 反相 HPLC 进行 纯化'))

# a bit similar job
print(model.similarity(' FT IR 产物 结构 进行 分析 扫描电镜 查看 喷洒 抑尘剂 煤粉 固化 表面 形貌',
                       ' FT IR 产物 结构 进行 分析 扫描电镜 查看 喷洒 兴奋剂 煤粉 表面 形貌'))
