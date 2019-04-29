#是用来定义一些固定的操作参数
MAX_LENGTH = 10
eng_prefixes = (   #是用来简化训练难度的
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

SOS_token = 0  #是用来标注语言的开始
EOS_token = 1  #是用来标注语言的结束