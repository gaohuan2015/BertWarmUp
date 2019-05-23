#import re
import math
from io import open

##const  常量
UNK = None  #未登陆的词
START_TOKEN = '<s>'  #开始
END_TOKEN = '</s>'  #结束

##processing  字典的生成等处理办法
#加入起始标记
def build_sentences(sentences):
        out = []
        for sentence in sentences:
				#lower() 方法转换字符串中所有大写字符为小写。
                words = [x.lower() for x in sentence]
				# insert() 函数用于将指定对象插入列表的指定位置。
				#list.insert(index, obj) //index -- 对象 obj 需要插入的索引位置。obj -- 要插入列表中的对象。
                words.insert(0, START_TOKEN)
				#append() 方法用于在列表末尾添加新的对象。list.append(obj)//obj -- 添加到列表末尾的对象。
                words.append(END_TOKEN)
                out.append(words)
        return out

# 构建ungram词频词典
def build_undict(sentences):
        undict = {} #字典
        total = 0
        for words in sentences:
                for word in words:
                        if word not in undict:
                                undict[word] = 1
                        else:
                                undict[word] += 1
                        if word != START_TOKEN and word != END_TOKEN:
                                total += 1
        return undict, total
#？？问题：为什么undict返回有total，而bidict和tridict没有
#构建bigram词频词典，其中以二元组(u, v)作为词典的键
def build_bidict(sentences):
    bidict = {}
    for words in sentences:
            for i in range(len(words)-1):
                    tup = (words[i], words[i+1])  #二元组
                    if tup not in bidict:
                            bidict[tup] = 1
                    else:
                            bidict[tup] += 1
    return bidict

# 构建trigram词频词典，其中以三元组(u, v, w)作为词典的键
def build_tridict(sentences):
        tridict = {}
        for words in sentences:
                for i in range(len(words) -2):
                        tup = (words[i], words[i+1], words[i+2])  #三元组
                        if tup not in tridict:
                                tridict[tup] = 1
                        else:
                                tridict[tup] += 1
        return tridict

##ngram model   UnGram、BiGram和TriGram模型以及一些求解模型方法
'''
@function calc_prob 			计算单词word条件概率，这里使用最大似然估计(max-likelihood estimate)去计算概率
@function calc_sentence_prob	计算句子sentence的条件概率
'''
class UnGram(object):
	def __init__(self, sentences, smooth = None):
		self.undict, self.total = build_undict(sentences)
		self.smooth = smooth
		#？？问题smooth的具体形式
#计算单词word条件概率，这里使用最大似然估计(max-likelihood estimate)去计算概率
	def calc_prob(self, word):
		prob = 0
		if self.smooth != None:
			prob = self.smooth(word, undict=self.undict, total=self.total)
		else:
			if word in self.undict:
				prob = float(self.undict[word]) / self.total
		return prob
#计算句子sentence的条件概率
	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		for word in sentence:
			if word != START_TOKEN and word != END_TOKEN:
				word_prob = self.calc_prob(word)
				if word_prob != 0:
					'''
					log() 返回 x 的自然对数。
					math.log(x[, base])//x -- 数值表达式。 base -- 可选，底数，默认为 e。
					注意：log()是不能直接访问的，需要导入 math 模块，通过静态对象调用该方法。
					'''
					prob_log_sum += math.log(word_prob, 2)
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum


def sort_vocab(self):
	# ？？问题：下面的undict.key指的是什么？前面没有提到
	vocabs = list(self.undict.keys())

	'''
	remove() 函数用于移除列表中某个值的第一个匹配项。
    list.remove(obj)//obj -- 列表中要移除的对象。
    '''

	# vocabs.remove(const.START_TOKEN)
	# vocabs.remove(const.END_TOKEN)

	'''
	sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
	list.sort(cmp=None, key=None, reverse=False)
	cmp -- 可选参数, 如果指定了该参数会使用该参数的方法进行排序。
	key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
	reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。
	'''
	vocabs.sort()
	vocabs.append(UNK)
	vocabs.append(START_TOKEN)
	vocabs.append(END_TOKEN)
	return vocabs

class BiGram(UnGram):
	def __init__(self, sentences, smooth = None):
		UnGram.__init__(self, sentences, smooth)
		self.bidict = build_bidict(sentences)
#计算条件概率，这里使用最大似然估计(max-likelihood estimate)去计算概率
	'''
	python函数——形参中的：*args和**kwargs 
	多个实参，放到一个元组里面,以*开头，可以传多个参数；**是形参中按照关键字传值把多余的传值以字典的方式呈现
	*args：（表示的就是将实参中按照位置传值，多出来的值都给args，且以元祖的方式呈现）
	**kwargs：（表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现）
	'''
	def calc_prob(self, *args):
		if len(args) != 2:
			raise ValueError('two words is required')

		prob = 0
		if self.smooth != None:
			prob = self.smooth(args[0], args[1], bidict=self.bidict, undict=self.undict)
		else:
			if args in self.bidict and args[0] in self.undict:
				return float(self.bidict[args]) / self.undict[args[0]]
		return prob
#计算句子的条件概率
	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		prev_word = None
		for word in sentence:
			if prev_word != None:
				word_prob = self.calc_prob(prev_word, word)
				prob_log_sum += word_prob
			prev_word = word
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum
		#pow() 方法返回 x^y（x的y次方） 的值。

class TriGram(BiGram):
	def __init__(self, sentences, smooth = None):
		BiGram.__init__(self, sentences, smooth)
		self.tridict = build_tridict(sentences)
#计算条件概率，这里使用最大似然估计(MLE:max-likelihood estimate)去计算概率
	def calc_prob(self, *args):
		if len(args) != 3:
			raise ValueError('three words is required')
		'''
		Python异常处理raise
		raise 引发一个异常,如引发一个<传入无效的参数>ValueError()
		https://www.cnblogs.com/Lival/p/6203111.html
		'''
		prob = 0
		if self.smooth != None:
			prob = self.smooth(args[0], args[1], args[2], tridict=self.tridict, bidict=self.bidict, undict=self.undict)
		else:
			bitup = (args[0], args[1])
			if args in self.tridict and bitup in self.bidict:
				return float(self.tridict[args]) / self.bidict[bitup]
		return prob
#计算句子的条件概率
	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		prev_stack = []
		for word in sentence:
			if len(prev_stack) < 2:
				prev_stack.append(word)
			elif len(prev_stack) == 2:
				word_prob = self.calc_prob(prev_stack[0], prev_stack[1], word)
				prob_log_sum += word_prob
				prev_stack[0] = prev_stack[1]
				prev_stack[1] = word
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum


'''
@function: calc_**gram_count   主要用来统计语料库中词的总数
@function: print_**gram_probas 格式化输出概率 
'''

class GramUtil(object):

	@staticmethod
	#calc_**gram_count   主要用来统计语料库中词的总数
	def calc_ungram_count(sentences):
		count = 0
		for sentence in sentences:
			# except START_TOKEN and END_TOKEN
			count += len(sentence) - 2
		return count

	@staticmethod
	# calc_**gram_count   主要用来统计语料库中词的总数
	def calc_bigram_count(sentences):
		count = 0
		for sentence in sentences:
			count += len(sentence) - 1
		return count

	@staticmethod
	# calc_**gram_count   主要用来统计语料库中词的总数
	def calc_trigram_count(sentences):
		count = 0
		for sentence in sentences:
			count += len(sentence)
		return count

	@staticmethod
	#print_**gram_probas 格式化输出概率
	def print_ungram_probs(model, vocabs):
		for vocab in vocabs:
			if vocab != START_TOKEN and vocab != END_TOKEN:
				print("{} \t {}".format(vocab if vocab != UNK else 'UNK', model.calc_prob(vocab)))
				#UNK也就是unknown单词，即词频率少于一定数量的稀有词的代号
				#Python format 格式化函数  https://www.runoob.com/python/att-string-format.html
				#\t 横向制表符
	@staticmethod
	# print_**gram_probas 格式化输出概率
	def print_bigram_probs(model, vocabs):
		print("\t\t", end="")
		for vocab in vocabs:
			if vocab != START_TOKEN:
				print(vocab if vocab != UNK else "UNK", end="\t\t")
		print("")
		for vocab in vocabs:
			if vocab != END_TOKEN:
				print(vocab if vocab != UNK else "UNK", end="\t\t")
				for vocab2 in vocabs:
					if vocab2 != START_TOKEN:
						print("{0:.3f}".format(model.calc_prob(vocab, vocab2)), end="\t\t")
				print("")

	@staticmethod
	# print_**gram_probas 格式化输出概率
	def print_trigram_probs(model, vocabs):
		print("\t\t", end="")
		for vocab in vocabs:
			if vocab != START_TOKEN:
				print(vocab if vocab != UNK else "UNK", end="\t")
		print("")
		for vocab in vocabs:
			if vocab != END_TOKEN:
				for vocab2 in vocabs:
					if vocab2 != START_TOKEN and vocab != UNK and vocab2 != UNK and vocab2 != END_TOKEN:
						print(vocab, vocab2 if vocab2 != UNK else "UNK", end="\t\t")
						for vocab3 in vocabs:
							if vocab3 != END_TOKEN:
								print("{0:.3f}".format(model.calc_prob(vocab, vocab2, vocab3)), end="\t")
						print("")

##evaluate  模型评估方法
# 计算困惑度
def perplexity(model, sentences, cal_gram_func):
    # gram_count 词的总数，对应教程中的 M
	gram_count = cal_gram_func(sentences)
	prob_log_sum = 0
	for sentence in sentences:
		try:
			prob_log_sum -= math.log(model.calc_sentence_prob(sentence), 2)
		except:
			prob_log_sum -= float('-inf')
		return math.pow(2, prob_log_sum/gram_count)
#smooth
#@description: 平滑估计计算
class Smooth(object):
	@staticmethod
	#形参**kwargs：（表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现）
	def discounting(*args, **kwargs):
		discount_value = 0.5
		if 'discount_value' in kwargs:
			discount_value = kwargs['discount_value']
		if len(args) == 1:
			if 'undict' not in kwargs:
				raise ValueError('undict is required')
			if 'total' not in kwargs:
				raise ValueError('total (words count in sentences) is required')
			undict = kwargs['undict']
			total = kwargs['total']
			word = args[0]
			if word in undict:
				return float(undict[word] - discount_value) / total
		if len(args) == 2:
			if 'bidict' not in kwargs and 'undict' not in kwargs:
				raise ValueError('bidict and undict is required')
			bidict = kwargs['bidict']
			undict = kwargs['undict']
			if args in bidict and args[0] in undict:
				return float(bidict[args] - discount_value) / undict[args[0]]
			else:
				return 0
		elif len(args) == 3:
			if 'tridict' not in kwargs and 'bidict' not in kwargs:
				raise ValueError('tridict and bidict is required')
			tridict = kwargs['tridict']
			bidict = kwargs['bidict']
			bitup = (args[0], args[1])
			if args in tridict and bitup in bidict:
				return float(tridict[args] - discount_value) / bidict[bitup]
			else:
				return 0
		else:
			return 0

#main  主运行程序
train_dataset = open('./corpus/toy/train.txt', encoding='utf-8'). \
        read().strip().split('\n')
test_dataset = open('./corpus/toy/test.txt', encoding='utf-8'). \
        read().strip().split('\n')


###################### ungram start ######################


model_unsmooth = UnGram(train_dataset)
model_smooth = UnGram(train_dataset, Smooth.discounting)

vocabs = model_unsmooth.sort_vocab()

print("- ungram unsmooth -")
GramUtil.print_ungram_probs(model_unsmooth, vocabs)

print("- ungram smooth -")
GramUtil.print_ungram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_ungram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_ungram_count))

###################### ungram end ######################


###################### bigram start ######################

model_unsmooth = BiGram(train_dataset)  #非平滑化模型
model_smooth = BiGram(train_dataset, Smooth.discounting) #平滑化模型

vocabs = model_unsmooth.sort_vocab()

print("- bigram unsmooth -")
GramUtil.print_bigram_probs(model_unsmooth, vocabs)

print("- bigram smooth -")
GramUtil.print_bigram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_bigram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_bigram_count))

###################### bigram  end ######################


###################### trigram start ######################

model_unsmooth = TriGram(train_dataset)
model_smooth = TriGram(train_dataset, Smooth.discounting)

vocabs = model_unsmooth.sort_vocab()

print("- ungram unsmooth -")
GramUtil.print_trigram_probs(model_unsmooth, vocabs)

print("- ungram smooth -")
GramUtil.print_trigram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_bigram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_bigram_count))

###################### trigram end ######################