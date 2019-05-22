a=open('LETA-lv-en/small_vocab_en',encoding='utf-8')
b=open('LETA-lv-en/small_vocab_fr',encoding='utf-8')
c=open('data/eng-fra.txt',encoding='utf-8')
a1=a.read().split("\n")
b1=b.read().split("\n")
f1=c.read().split("\n")
print(a1)
a.close()
b.close()
c.close()
c1=[]
d1=[]
for j in range (len(f1)):
    if j%2==0:
        print('o{}'.format(f1[j]))
        c1.append(f1[j])
    elif j%2!=0:
        print('j{}'.format(f1[j]))
        d1.append(f1[j])
print(len(a1),len(b1))
for i in range(len(a1)):
    # print(c1[i])
    if i >=(1/2)*len(a1):
        d1.append((a1[i]+'\t'+b1[i]))
    else:
        c1.append((a1[i]+'\t'+b1[i]))

print(len(d1))
print(len(c1))
f=open("LETA-lv-en/eng-fra1.txt","w",encoding='utf-8')
f.write('\n'.join(c1))
f.close()
g=open("LETA-lv-en/1eng-fra1.txt","w",encoding='utf-8')
g.write('\n'.join(d1))
g.close()