#!/usr/bin/env python
# coding: utf-8

# # Pythonの基礎

# ## 半角と全角 

# * コード（空白も含めて）を書く際は**半角**を使うこと。
# 
# 以下に全角を使っても構わない場合を簡単に説明するが、それ以外で全角を使うとエラーが発生する。特に、空白は見えないので要注意！

# （１）１行のうち`#`の後に続くコードは無視されるので、全角でも構わない。改行すると新たな行になるので注意すること。
# 
# 例えば、次のコード・セルを実行しても何も出力されない。またエラーも出ない。

# In[1]:


# この行は無視されます。
# この行も無視されます。
10   # ここも無視されます。
# この行も無視されますよ。


# （２）`"`と`"`もしくは`'`と`'`で挟まれた文字は、以下で説明する**文字列型**のデータ型となり、全角でも構わない。`"`と`'`は半角であることに注意しよう。

# In[2]:


"Pythonは楽しい(^o^)/"


# In[3]:


'Pythonは楽しい(^o^)/'


# （３）`"""`と`"""`もしくは`'''`と`'''`で挟まれた複数行は以下で説明する**文字列型**なので全角でも構わない。`"""`と`'''`は半角であることに注意しよう。

# In[4]:


"""Pythonは楽しい(^o^)/
Pythonはもっと楽しい(^o^)/
Pythonは一番楽しい(^o^)/"""


# In[5]:


'''Pythonは楽しい(^o^)/
Pythonはもっと楽しい(^o^)/
Pythonは一番楽しい(^o^)/'''


# `\n`は改行を表す記号であり、以下で説明する`print()`関数を使うと`\n`で改行されて表示される。

# ## ４つのデータ型

# `Python`には無数のデータ型があるが，まず基本となる3つを考える。
# * 整数型（`int`と表す）
# * 浮動小数点型（`float`と表す）
# * 文字列型（`str`と表す）
# * ブール型（`bool`と表す）
# 
# 整数型とは文字通り`1`や`100`などの整数のことであり，浮動小数点型とは小数を指す。`Python`では整数と小数を異なるデータ型として区別して扱うので，`1`と`1.0`はデータ型が異なるのである。

# In[6]:


1 


# In[7]:


1.0


# データ型を確認するために`type()`という関数が用意されているので，それを使ってみよう（関数に関しては後述する）。

# In[8]:


type(1)


# In[9]:


type(1.0)


# 整数型は`int`（integer），浮動小数点型は`float`と表示されている。文字列は

# In[10]:


'国内総生産'


# のように`'`で挟まれて定義される。`'`の代わりに`"`を使っても良い。どちらを使っても同じだが，クォートの中でクォートを使う必要がある場合は，それぞれ違ったものを使う。例えば，

# In[11]:


'He said "Python is very useful!" to me.'


# データ型を確認してみよう。

# In[12]:


type('神戸大学')


# `str`はstringの略である。注意が必要な点は次も文字列型となる。

# In[13]:


'0.1'


# In[14]:


type('0.1')


# 従って，後述する加算や除算などはできない`0.1`である。
# 
# ブール型には`True`（真）, `False`（偽）がある。名前が示すように「真偽」を示すのがブール型である。例えば，

# In[15]:


1==10


# 後ほど説明するが，`==`は「等しい」ということを意味する。また，`True`は`1`, `False`は`0`として計算される。

# In[16]:


True==1 # Trueは１なのでTrueを返す


# In[17]:


False==0 # Falseは0なのでTrueを返す


# In[18]:


True + True


# またそれぞれのデータ型を他のデータ系に変換することも可能である。
# 
# `float()`：浮動小数点数型への変換

# In[19]:


float(10)


# `int()`：整数型への変換

# In[20]:


int(10.0)


# `str()`：文字列型への変換

# In[21]:


str(10.0)


# 例：文字列型 $\rightarrow$ 浮動小数点数型

# In[22]:


float('10.0')


# 例：文字列型 $\rightarrow$ 浮動小数点数型 $\rightarrow$ 整数型

# In[23]:


int(float('10.0'))


# ちなみに`bool()`はブール型に変換するのではなく、`()`の中のオブジェクト（以下の説明を参照）の truth value （`True`か`False`のブール値）と呼ばれる値を返す。あるオブジェクトが「空」かどうかを確かめるの使われるが、この授業では使わない。

# In[24]:


bool(100)


# In[25]:


bool(0)


# ## 計算機としての`Python`

# ### 算術演算子

# 基本的な算術演算子として以下を挙げることができる。
# * `+`（加算）
# * `-`（減算）
# * `*`（乗算）
# * `/`（除算）
# * `**`（累乗）

# In[26]:


100 + 1


# 整数型と整数型の加算なので整数型が返される。一方で，整数型と浮動小数点型で計算すると浮動小数点型が返される。

# In[27]:


100 - 1.0


# In[28]:


100 * 10.0


# 除算の場合は整数を使っても浮動小数点が返される。

# In[29]:


10 / 5


# In[30]:


10 ** 5.0


# 数学では`()`の中が先に計算されるが，`Python`でも同じである。

# In[31]:


(100 - 50) / 2


# 注意が必要な点は文字列型は上で説明した計算に使うとエラーが発生する。

# In[32]:


1 + '1.0'


# 浮動小数点型に変換する関数`float()`を使えばエラーは出ない。

# In[33]:


1+float('1.0')


# ちなみに文字列型`1`は関数`int()`整数型に変換できる。

# In[34]:


1+int('1')


# 最後に，文字列型`+`文字列型とすると結合される。

# In[35]:


'1.0'+'1.00'


# In[36]:


'一人当たり' + 'GDP'


# ### 関係演算子

# 以下の関係演算子を使うことにより，変数の真偽を確認することができる。
# 
# * `==`（等号）
# * `!=`（等号不成立）
# * `<`（小なり）
# * `>`（大なり）
# * `<=`（小なりイコール）
# * `>=`（大なりイコール）
# 
# `==`はブール型を説明した際に使ったものと同じであり，値が同じかを確認する。

# In[37]:


10 == 10.0


# In[38]:


10 != 10


# In[39]:


10 > 5


# In[40]:


10 >= 10


# ## `=`による変数の割り当て

# ### `=`の役割

# 関係演算子に`=`を他の記号と一緒に使ったが，単独で使う場合は「変数の割り当て」に使う。次の例では値`10`を変数$x$に割り当てている。

# In[41]:


x = 10


# $x$の値を表示すには、$x$を書いたセルを評価するだけである。

# In[42]:


x


# ここで「代入する」と説明せずに「割り当てる」という表現を使ったが，その理由を説明する。簡単にいうと，`x`と`10`は全く別物であるためである。これを理解するために、多くの品物が保管されている大きな倉庫を考えてみよう。倉庫の管理者はどの品物がどこに保管されているかを記録する在庫リストを作成し、そこに品物が保管されている棚を示す記号を記入しているとしよう。この例を使うと、
# * `10`　→　倉庫の棚に保管されている品物
# * `x`　→　在庫リストに記載されている棚の記号
# 
# となる。品物と棚の記号は別物である。`Python`では、品物である`10`がコンピューター内のメモリーの所定の場所に保存され、その場所を示すのが変数`x`となる。即ち、`x`は品物`10`の実態とは異なる単なる「参照記号」なのである。
# * `10`　→　PCのメモリーに保存されている情報
# * `x`　→　参照記号
# 
# この点を明確にするために、上のコードは「`10`を記号`x`に**割り当てる**」と考える。ここで、式を**右から左に**読んでいることに注意しよう。`=`を右から左に読む（考える）ことを習慣づけることが、今後`Python`を勉強する上で重要となるからである。この点を示すために次のコードを考えてみよう。

# In[43]:


x = x + 1


# 「？」と思うかもしれない。暗に方程式として考えるためであろう（私がそうだった）。これを右から左に読むとスッキリする。
# 1. 上で`10`を`x`に割り当てたが、問題のコードの右辺の`x`がその`10`である。`10`に`1`を加えたものが`11`であり、それが右辺である。
# 1. `=`を使い右辺の`11`を左辺の`x`に割り当てている。この時点で、`10`の参照記号であった`x`は`11`の参照記号に変更される。
# 
# 実際に`x`を表示してみよう。

# In[44]:


x


# 「品物と参照記号の関係」は今の段階ではそれ程重要ではないが，先に進むにつれて重要性が増してくるので，今のうちにこのようなイメージを持つと良いだろう。

# ### 累算代入演算子

# 上で`x = x + 1`が出てきたが、これを短縮して`x += x`

# In[45]:


x = 10
x += 1
x += 1
x += 1
x += 1
x += 1
x


# 同様に次の様に書くことができる。
# * `x = x-1` $\Rightarrow$ `x -= 1`
# * `x = x*10` $\Rightarrow$ `x *= 10`
# * `x = x/10` $\Rightarrow$ `x /= 10`
# * `x = x**2` $\Rightarrow$ `x **= 2`

# In[46]:


x=2
x **= 2
x **= 2
x


# ### 複数の変数の割り当て

# `=`を使い１つの変数への割り当てを考えたが、実は複数の変数への割り当ても可能である。次の例を考えよう。

# In[47]:


a, b, c = 10, False, 'Python'


# 左辺の`1`は右辺の`a`に、同様に`10`は`b`、`神戸大学`は`c`に割り当てている。確認してみる。

# In[48]:


a


# In[49]:


b


# In[50]:


c


# ### 予約語

# アルファベットで分かりやすい変数名にするのが可読性が高いコードを書くコツである。しかし，変数の名前を作る上で守らなくてはならないルールがある。
# 
# * `(a-z, A-Z)`もしくは`_`（アンダースコア）で始める
# * 最初の文字以外であれば`(a-z, A-Z)`と`_`に加え数字も可
# * 長さに制限はない
# * 小文字と大文字は異なる記号としてあつかう
# * 次の単語は特定の目的のために事前に定義されている「予約語」なため，変数名としては使えない（使うとエラーが出る）。

# In[51]:


import keyword
keyword.kwlist


# これらに加え，
# 
# * 変数の頭文字は小文字とする
# 
# というのが慣例（エラーにはならない）であり，大文字で始まる変数は`class`と呼ばれるオブジェクトに使う。

# ## コレクション系データ型

# 上で説明した基本データ型の集まりとなるコレクション系データ型には次の４種類が用意されている。
# * リスト（list）
# * タプル（tuple）
# * 辞書（dict）
# * 集合（set）
# 
# 授業では集合を使わないので，そ例外の３つだけを説明する。

# ### リストとタプル

# #### リスト

# タプルは`()`を使って作成する。また何かをリストに変換する場合は`list()`を使う。

# In[52]:


list0 = [10, 3 , 2]
list0


# 要素のデータ型が違っても構わない。

# In[53]:


list1 = ['A', True, 100]
type(list1)


# 要素が１つのリストの生成も可能。

# In[54]:


[100]


# 次に説明するタプルをリストに変換したい場合は`list()`を使うことができる。

# #### タプル

# タプルは`()`を使って作成する。また何かをタプルに変換する場合は`tuple()`を使う。

# In[55]:


tuple0 = ('A', True, 100)
print(tuple0)


# リストと変わりないように見えるが，大きな違いは要素を変更できるかできないかという点である。
# 
# * リストの要素は変更可能
# * タプルの要素は変更不可能
# 
# リストの要素の変更方法は以下で説明する。
# 
# 上で通常タプルは`(`と`)`を使って作成できると説明したが、実は、コンマ`,`によってタプルは定義されるため`(`と`)`は必須ではない。例えば、次のコードでもタプルとなる。従って、`(`と`)`はタプルを明確にするためと考えて良い。

# In[56]:


'1', 'GDP', '消費'


# 要素が１つのタプルも生成可能だが、その場合も必ず`,`を付ける事を忘れないように。

# In[57]:


('1',)


# 上で定義した`list1`をタプルに変換したい場合は`tuple()`を使う。

# In[58]:


tuple(list1)


# #### 要素のアクセス方法

# リストもタプルも要素のインデックス（位置）を考える場合，次の図のように左から`0`，`1`，`2`...，右からは`-1`，`-2`，`-3`と数える。
# ```
#    0   1   2   3   4   5  （左から数える） 
#  +---+---+---+---+---+---+
#  | P | y | t | h | o | n |
#  +---+---+---+---+---+---+
#   -6  -5  -4  -3  -2  -1　（右から数える）
# ```
# 例えば，次のリストを考えよう。

# In[59]:


my_list = ['A', 'B', 'C', 'D', 'E', 'F']
my_tuple = (100, 200, 300, 400, 500, 600)


# `'A'`は０番目，`'B'`１番目，`'C'`は２番目と数える。例えば，`A`を抽出したい場合，

# In[60]:


my_list[0]


# In[61]:


my_tuple[0]


# 連続する複数の要素を選択する場合（スライシング）は`:`を使う。`:`の左側が選択する最初の要素で，`:`の右側が選択する最後の次の番号である（即ち，`:`の右側の番号の要素は含まれない。

# In[62]:


my_list[1:4]


# In[63]:


my_tuple[1:4]


# `:`の左側の番号を省略すると`0`と解釈され，`:`右側を省略すると`-1`と解釈される。

# In[64]:


my_list[:4]


# In[65]:


my_tuple[3:]


# `+`を使い，複数のリストを結合することができる。

# In[66]:


my_list + [1,2,3]


# タプルも同様である。

# In[67]:


my_tuple[4:] + ('GDP','GNP')


# この場合，元のタプルを変更しているのではなく，新しいタプルを生成している。

# ### 辞書

# 辞書（dictionary）はキー（key）と値（value）のペアとなって定義され，`:`を挟んで１つのペアとなる。全てを`{}`で囲み辞書を定義する。もしくは`dict()`でも生成することができる。

# In[68]:


dict0 = {'a':10, 'b':'失業率'}


# もしくは

# In[69]:


dict0 = dict(a=10, b='失業率')


# `dict0`には２つのペアがある。`a`のキーには値`10`が対応しており，`b`には`'失業率'`が設定されている。今の段階では辞書を使う目的が不明確でしっくりこないと思うが，勉強を進めるととてもパワフルなツールだと気づくだろう。

# In[70]:


print(type(dict0))


# 値にアクセスするにはキーを次のように使う。

# In[71]:


dict0['a']


# In[72]:


dict0['b']


# ## 関数

# コードを書くうえで関数は非常に重要な役割を果たすのが関数である。主に２つのタイプに分けることができる。
# * 組み込み関数
# * ユーザー定義の関数（ユーザー自身が作成する関数）

# ### 組み込み関数

# 組み込み関数（built-in functions）とは，ユーザーが使えるように事前に準備された関数である。データ型を調べる際に使った`type()`もその１つである。頻繁に使う関数を３つ紹介する。

# #### `sum()`

# `sum()`は合計を計算する関数である。

# In[73]:


gdp_component = [10, 20, 30, 40]

sum(gdp_component)


# #### `print()`

# `print()`は表示するための関数だが，Jupyter Notebookでは`print()`を使わなくとも出力が表示される。例えば，

# In[74]:


10


# しかし複数行の場合は最後の行しか表示されない。

# In[75]:


10
200


# `print()`を使うと両方を表示することができる。

# In[76]:


print(10)
print(200)


# 異なるオブジェクトを表示するには`,`を使う。

# In[77]:


print('2019年の実質GDP：約', 500+34, '兆円')


# 文字列の中で`\n`は改行を示す。

# In[78]:


print('マクロ\n経済学')


# 次に`f-string`を紹介する。文字列の前に`f`を書き加え，文字列の中で`{}`を使うことにより，割り当てた変数の値や計算結果などを表示することが可能となる。次の例を考えよう。

# In[79]:


x = 1/3

print(f'3分の1は{x}です。')


# 小数点第二位まで表示する場合は，`x`の後に`:.2f`を付け加える。

# In[80]:


print(f'10割る3は{x:.2f}です。')


# `2`を`5`にすると，小数点第五位までの表示となる。

# #### `range()`

# `range()`は連続する整数を用意する関数である。例えば，`0`から`9`までの`10`の整数を生成するには 

# In[81]:


range(10)


# とする。用途が明確ではないと感じるかもしれないが，後述する`for`ループで多用することになる。またリストとして表示するには`list()`を使う。

# In[82]:


x_list = list(range(10))
x_list


# #### `len()`

# 要素の個数などを返す関数。

# In[83]:


len(x_list)


# ### ユーザー定義の関数

# `def`を使って関数を定義し，引数が設定される（省略される場合もある）。ここでは基本となる引数のみを考えるが，引数の**位置**と`=`が重要な役割を果たすことになる。例を使いながら説明する。最初の例は数字の2乗を計算する関数である。

# In[84]:


def func_0(x):
    return x**2


# * １行目：
#     * `def`で始まり（`def`はdefinitionの省略形）`:`で終わる。
#     * `func_0`が関数名，`x`が第１引数（ひきすう）であり唯一の引数である。
# * ２行目：
#     * `return`は評価した値を「返す」という意味。必ず`return`の前には4つの半角スペースが必要であり，ないとエラーになる。
#     * `x**2`という返り値（戻り値）の設定をする
# 
# 関数を評価するには，引数に数字を入れて実行する。

# In[85]:


func_0(2)


# 引数が無い関数を定義することを可能である。

# In[86]:


def func_100():
    return 10**2

func_100()


# `print()`関数を追加することもできる。

# In[87]:


def func_kobe():
    print('経済学はおもしろい(^o^)/')
    return 10**2

func_kobe()


# #### 引数の位置が重要

# 引数が複数ある場合にその位置が重要になってくる。次の例を考えよう。

# In[88]:


def func_1(a, b, c):
    return a / b + c


# In[89]:


func_1(10, 2, 10)


# In[90]:


func_1(2, 10, 10)


# #### 実行する際に`=`を使う

# 関数を実行する際、引数に`=`を使って値を指定することも可能である。この場合，引数の順番を変えることが可能となる。

# In[91]:


func_1(10, c=10, b=2)


# `=`が付いていない引数は該当する位置に書く必要があり，`10`が最初に来ないとエラーとなる。一般的なルールとして，`=`を使う引数は全て`()`の右端にまとめる。

# #### 関数を定義する際に`=`を使う

# 関数を定義する際に`=`を使って引数のデフォルトの値を設定することができる。即ち，引数を入力すると入力された数値を使うが，引数を入力しない場合は引数に予め設定した値（デフォルトの値）が使われて評価される。次の例では`c`のデフォルトの値が`10`に設定されている。

# In[92]:


def func_2(a, b, c=10):
    return sum([a,b])*c


# `c`の値を設定しない場合とする場合を比較してみる。

# In[93]:


func_2(2, 3), func_2(2, 3, 100)


# * 関数を実行する際に`=`無しで関数に渡される引数は，その位置が重要であるため「位置引数」と呼ばれる。
# * 関数を実行する際に`=`付きで関数に渡される引数は「キーワード引数」と呼ばれる。

# #### 返り値が複数の場合

# In[94]:


def func_3(x):
    return x, x**2, x**3

func_3(3)


# この場合、返り値はタプルとして返される。これを利用して、関数を実行する際に変数にに返り値のを割り当てることができる。

# In[95]:


a, b, c = func_3(4)
print(a)
print(b)
print(c)


# ````
# 返り値はタプルなので、それに合わせて次のように書くこともできる。
# ```
# (a, b, c) = func_3(4)
# ```
# しかしタプルは`,`で定義されるので、`()`を省いても問題ない。
# ````

# また次のコードでも同じ結果となる。

# In[96]:


d = func_3(4)
print(d[0])
print(d[1])
print(d[2])


# ここでは返り値のタプル自体を`d`に割り当て、その各要素を`print()`関数で表示している。

# In[97]:


d


# #### `lambda`関数

# `def`を使い複雑な関数を定義することができるが，単純な関数の場合，より簡単な方法がある。それが`lambda`関数である。例として，$x^2$を計算する関数を考えよう。

# In[98]:


func_3 = lambda x: x**2


# In[99]:


func_3(2)


# ### 経済学を考える

# #### 将来価値

# * `t`：時間（年; 0,1,2,...）
# * `i`：名目利子率（例えば，0.02）
# * `r`：実質利子率（例えば，0.05）
# * `pi`：インフレ率（例えば，0.03）
# 
# 次の式が成立する。
# $$
# 1+r=\dfrac{1+i}{1+pi}
# $$
# 
# `x`万円を年率`i`%の利息を得る金融商品に投資し，`t`年後に現金化するとしよう。その間のインフレ率は`pi`%とした場合の`x`万円の実質将来価値を計算する関数を考える。

# In[100]:


def future_value(x, i, pi, t):
    r = (1+i)/(1+pi)-1
    return x*(1+r)**t


# In[101]:


future_value(100, 0.05, 0.03, 10)


# #### 現在価値

# `t`年後に`x`万円をもらえるとしよう。`x`万円の現在価値を計算する関数を考える。

# In[102]:


def current_value(x, i, inf, t):
    r = (1+i)/(1+inf)-1
    return x/(1+r)**t


# In[103]:


current_value(100, 0.05, 0.03, 10)


# #### 複利計算

# * `y`：元金
# * `t`：投資期間
# * `r`：実質利子率（年率）
# * `m`：複利の周期（年間の利息発生回数）
# * `y_total`：`t`年後の元利合計
# $$
# \text{y_total}=y\left( 1+\dfrac{r}{m}  \right)^{mt}
# $$
# 
# `t`年後の元利合計を計算する関数を考えよう。

# In[104]:


def y_total(y=100, r=0.05, m=1, t=10):
    return y*( 1+r/m )**(m*t)


# In[105]:


y_total()


# In[106]:


y_total(m=12)


# ## オブジェクトと属性

# `Python`を習うと「オブジェクト」という単語が必ず出てくる。今の内にイメージをつかむために自転車をオブジェクトの例として考えてみよう。通常の自転車には車輪が２つあり、サドルが１つあり、左右にペダルが２つある。これらの数字が自転車に関する**データ**である。またペダルを踏むことにより前に動き、ハンドルを右にきると右方向に進むことになる。即ち、あることを実行すると、ある結果が返されるのである。これは数学の**関数**と同じように理解できる。$y=x^2$の場合、$x$が`2`であれば$y$の値として`4`が返される。このように自転車はデータと関数が備わっているオブジェクトとして考えることができる。また、車輪の数やペダルを踏むことは自転車特有のデータと関数であり、他のオブジェクト（例えば、冷蔵庫）にはない。即ち、世の中の「オブジェクト」にはそれぞれ異なるデータと関数が存在していると考えることができる。
# 
# `Python`の世界でも「すべて」をこれと同じように考える。上のコードの`10`にもデータと関数が備わっており、それらを**属性**（attributes）と呼ぶ。`10`は単なる数字に見えるが、実は様々な属性から構成されるオブジェクトなのである。上の例の自転車のように、`Python`の「属性」は次の２つに分類される。`10`を例にとると、
# 1. `10`が持つ様々な**データ（属性）**（data attributes）（例えば、`10`という値や整数という情報）
# 1. `10`特有の関数である**メソッド（属性）**（method attributes）（例えば、加算、除算のように`10`というデータに働きかける関数）
# 
# を指す。自転車と冷蔵庫は異なるデータと関数を持つように、整数`10`と文字列`マクロ経済学`は異なるデータと関数を備えるオブジェクトなのである。この考え方は`Python`のすべてに当てはまる。即ち、Everything is an object in Python.

# データ属性の例として浮動小数点型`10.0`を`y`に割り当てよう。

# In[107]:


y = 10.0


# `y`の属性は`dir()`という組み込み関数を使うことにより表示できる。

# In[108]:


dir(y)


# この中の最後にある`real`は数字の実部を表し，実数である`10.0`の実部は`10.0`である。一方，`imag`は複素数の虚部を表すが，`10.0`は複素数ではないので`0.0`になっている。（上で紹介しなかったが，データ型に複素数型もある。）

# In[109]:


y.real, y.imag


# 上の属性のリストにある`_`はアンダースコア（underscore）と呼ぶが，２つ連続した場合`__`となりダブル・アンダースコア（double underscore）と呼ぶ。長いのでダンダー（dunder）と省略する場合が多々ある。ダンダーが付いている属性は`Python`が裏で使うものでありユーザーが直接使う属性ではない。
# 
# 次にメソッドを考えるために次のリストを例に挙げる。

# In[110]:


my_list = [1,2,3]

dir(my_list)


# この中に`append`があるが，`my_list`に要素を追加するメソッドである。

# In[111]:


my_list.append(100)

my_list


# ```{note}
# * データ属性と異なりメソッドは`()`が必要となる。これは関数の`()`に対応している。
# 
# ```

# ## `for`ループ

# ### 説明

# `for`ループは同じコードを複数回リピートして実行したい場合に有効な方法である。例えば，次のリストにある名前を表示したいとしよう。

# In[112]:


gdp_components = ['消費', '投資', '政府支出', '純輸出']


# In[113]:


for com in gdp_components:
    print(com)


# ＜説明と注意点＞
# * 1行目
#     * `for`で始まり`:`で終わる。
#     * `com`は`gdp_components`にあるそれぞれの要素を捉えるダミー記号。`com`ではなく`i`や`s`など使いやすい記号を使えば良い。
# * 2行目
#     * 4つの半角スペースのインデント後に実行したいコードを書く。
#     * `print()`関数が実行される。
# * `gdp_components`にある要素を最初から一つずつ実行する。
# 
# ---
# 次に上で紹介した，リストに要素を追加するメソッドである`.append()`を使って新たなリストを作成する。まず次のリストを定義する。

# In[114]:


var_list = [1,2,3,4,5]


# それぞれの要素の10倍からなるリストを作成するとしよう。

# In[115]:


my_list = []              # 説明１

for i in var_list:        # 説明２
    x = 10*i              # 説明３
    my_list.append(x)     # 説明４

my_list                   # 説明５


# このループの説明：
# 
# 1. 空のリストの作成（ここに10倍にした数字を格納する）
# 1. ここから`for`ループの始まり。`i`はリスト`[1,2,3,4,5]`の要素のダミー変数であり，`var_list`の左から一つずつ次の行の`i`に該当する要素を代入して評価する。
# 1. `2*i`を計算し`x`に割り当てる。
# 1. `.append()`を使い`x`を`my_list`に追加する
# 1. `my_list`の表示する。

# ### 経済学を考える

# #### 消費関数

# 同じ方法を使い，所得によって消費がどのように変化するかを考えてみよう。まず消費関数を次のように仮定する。

# In[116]:


def consumption(y):
    return 100 + 0.7 * y


# * `100`：自発的消費（autonomous consumption）
# * `0.7`：限界消費性向（marginal propensity to consume）
# 
# 所得は次のリストで与えられるとする。

# In[117]:


income_list = [1000, 1100, 1500, 2000, 2300, 3000] 


# In[118]:


c_list = []

for y in income_list:
    con = consumption(y)
    c_list.append(con)

c_list


# #### 将来価値の時系列

# まず，`1`から`11`までの数字から構成されるリストを作成する。より簡単なコードで同じ結果を得ることができるが，ここでの目的は動学的な変数の変化を考える基礎作りである。一度のループを一年と考えて変数`y`の時系列的な変化を捉えていると考えよう。

# In[119]:


my_list = [1]           # １行目

for i in range(10):     # ２行目
    y = my_list[i] + 1  # ３行目
    my_list.append(y)   # ４行目
    
my_list                 # ５行目


# ＜このループを時系列的に説明する＞
# * （１行目）初期時点の`y`の値からなるリストの作成する。
# * （２行目）1回目のループの実行開始。
#     * `i`は３行目だけしかない。これにより`range(10)`指定された`10`回ループ計算を行う。
# * （３行目）`my_list`の`0`番目の要素に`1`を足し右辺は`2`となり，それを`y`に割り当てる。
# * （４行目）`y`の値`2`を`my_list`に追加する。
# * （２行目）2回目のループの実行開始
# * （３行目）`my_list`の`1`番目の要素に`1`を足し右辺は`3`となり，それを`y`に割り当てる。
# * （４行目）`y`の値`3`を`my_list`に追加する。
# * （２〜４行目）3回目から最後までのループを繰り返す。
# * （５行目）`my_list`を表示する。

# もしくは次のコードでも同じ結果となる。

# In[120]:


y = 1                   # １行目

my_list = [y]           # ２行目

for i in range(10):     # ３行目
    y = y + 1           # ４行目
    my_list.append(y)   # ５行目
    
my_list                 # ６行目


# ＜このループを時系列的に説明する＞
# * （１行目）初期時点での`y`の値を設定する。
# * （２行目）初期時点の`y`の値からなるリストの作成する。
# * （３行目）1回目のループの実行開始。
#     * `i`は３行目だけしかない。これにより`range(10)`指定された`10`回ループ計算を行う。
# * （４行目）右辺の`y`の値は初期時点の値である`1`。それに`1`を足し右辺は`2`となり，それを`y`に割り当てる。即ち、`2`が１行目の`y`に割り当てられる（`y`が更新される）。
# * （５行目）`y`の値`2`を`my_list`に追加する。
# * （３行目）2回目のループの実行開始
# * （４行目）前回の計算により右辺の`y`の値は`2`。それに`1`を足し右辺は`3`となり，それを`y`に割り当てる。即ち、`3`が１行目の`y`に割り当てられる（`y`が更新される）。
# * （５行目）`y`の値`3`を`my_list`に追加する。
# * （３〜５行目）3回目から最後までのループを繰り返す。
# * （６行目）`my_list`を表示する。

# １行目の`y`値を確認してみる。

# In[121]:


y


# どちらを使っても良い。最初の方法は`i`を書く必要があるので面倒だが、明示的に感じるかも知れない。2つ目の方法は`i`を書く必要がない。しかし、初期値の変数`x`を`for`ループの外に作る必要があるが、次の例で説明するように、関数の中で使う場合はこの問題が解決できる。以下では、主に２つ目の手法を多用する。

# このコードを利用して，`x`万円を実質年率`r`%の利息を得る金融商品に投資し，`t`年間の将来価値（期首の値）をリストで示す関数を作成する。

# In[122]:


def calculate_futre_value(x, r, t):
    
    value_list = [x]            # 初期値が含まれている
    
    for year in range(1,t+1):   # 1からtまでの整数
        x = x*(1+r)             # 来期は(1+r)倍に増加
        value_list.append(x)
    
    return value_list


# In[123]:


values = calculate_futre_value(100, 0.02, 5)
values


# 関数`calculate_futre_value()`は、上の2つ目のコードの形で書いているが、初期値を設定する１行目`y=1`に対応する行がない。その代わりになっているのが関数の引数として使われている`x`である。`calculate_futre_value(100, 0.02, 5)`を実行した時点で、`x=100`が実行され関数内のみで有効な変数`x`が作成される。

# 次に`for`ループを使って見易くする。

# In[124]:


for i, v in enumerate(values):  # 以下の説明を参照
    print(f'{i}期の始め：{v:.1f}万円')


# 次のコードが示すように，`enumerate()`はリストの要素のインデックと要素をタプルとして返す。それを`i`と`v`のダミー変数に割り当てている。

# In[125]:


list(enumerate(['A','B']))


# #### 複利計算

# * `y`：元金
# * `t`：投資期間
# * `r`：実質利子率（年率）
# * `m`：複利の周期（年間の利息発生回数）
# * `y_total`：`t`年後の元利合計
# $$
# \text{y_total}=y\left( 1+\dfrac{r}{m}  \right)^{mt}
# $$
# 
# `t`年後の元利合計の時系列を計算する関数。

# In[126]:


def calculate_futre_value_m(y, r, t, m=1):
    
    value_list = [y]            # 初期値が含まれている
    
    for year in range(1,t+1):   # 1からtまでの整数
        y = y*(1+r/m)**m        # 来期は(1+r/m)^m倍に増加する
        value_list.append(y)
    
    return value_list


# In[127]:


calculate_futre_value_m(100, 0.02, 5, m=12)


# ## 内包表記

# `for`ループの考えを使い、リストや辞書などを簡単に１行で生成する方法があり、それを内包表記と呼ぶ。リストの場合、次のような書き方となる。
# ```
# [＜何かをする＞ for x in sequence]
# ```
# 
# * `for x in sequence`の箇所は`for`ループの１行目と同じとなる。
#     * `sequence`はリストや`array`を指す。
#     * `x`は`sequence`の要素を指す変数。
# * `＜何かをする＞`の箇所でしたい事を指定する。
#     * 以下で例を考えよう。
#     
# `sequence`の例として`range(5)`を考えよう。まず`for`ループを使って、それぞれの要素を２乗にしたリストを生成するとしよう。

# In[128]:


lst = []

for x in range(5):
    lst.append(x**2)

lst


# `for`ループを使うと３行のコードとなるが、内包表記を使うと１行で済む。

# In[129]:


[x**2 for x in range(5)]


# `lst`の要素を全て文字列に変換してみよう。

# In[130]:


[str(x) for x in lst]


# `def`を使って作成した自前の関数を使うことも可能である。試してみよう。

# ## `if`文

# `if`文を使うと，あるブール型（真偽）の条件のもとでコードを実行することが可能となる。例えば，`change_in_gdp`をGDPの変化だとしよう。`change_in_gdp`の値が正の場合，
# 
# `print('GDPは増加しています。')`
# 
# を実行したいとしよう。

# In[131]:


change_in_gdp = 1000


# In[132]:


if change_in_gdp > 0:              # 説明１
    print('GDPは増加しています。')    # 説明２
    
else:                              # 説明３
    pass                           # 説明４


# ＜説明＞
# 1. `if`で始まり`:`で終わる。`change_in_gdp>0`は`change_in_gdp`が正であれば`True`を，そうでなければ`False`を返す。
# 1. ４つの半角スペースのインデントが入り，`change_in_gdp>0`が`True`の場合に実行される。
# 1. `else`とは「`change_in_gdp>0`以外の場合」という意味であり，`change_in_gdp>0`が`False`の場合に続く行が実行される。`else`の後には必ず`:`が入る。
# 1. `pass`は「何もしない」という意味。
# 
# ここで`else`以下を省略してもエラーにはならない（結果も変わらない）。即ち，`else`以下がない場合は，それが省略されていると考えれば良い。

# 次に，複数の条件を導入するために次の関数を考える。
# 
# 1. `print('GDPは変化していません。)`
# 1. `print('GDPは増加しています。)`
# 1. `print('GDPは減少加しています。)`
# 
# `change_in_gdp`の値がゼロの場合は`1`を，正の場合は`2`を，負の場合は`3`を表示したいとしよう。

# In[133]:


change_in_gdp = -200


# In[134]:


if change_in_gdp == 0:
    print('GDPは変化していません。')
    
elif change_in_gdp > 0:
    print('GDPは増加しています。')
    
else:
    print('GDPは減少しています。')


# ＜説明＞
# * `if`が１つ目の条件, `elif`が２つ目の条件, `else`が３つ目の条件を指定している。
# * `if`, `elif`, `else` で始まる行の最後は`:`となる。
# * `print()`の行は４つの半角スペースのインデントが入る。
# * `else`の行に`change_in_gdp<0`は不要（残りの可能性は`X<0`しかないため）
# * `elif`は`else if`の省略形であり，２つ目の条件を定義する。
# * `elif`は`if`と`else`の間に複数入れることが可能である。

# ---
# 次の例として一般的な生産関数を考えよう。
# $$
# Y=F(K,L)
# $$
# 要素間の代替の弾力性は次のように定義される。
# $$
# \sigma = 
# \dfrac{\log(L/K)}{\log\left(\frac{dF}{dK}/\frac{dF}{dL}\right)}
# $$
# 
# $\sigma$が一定な生産関数はCES生産関数（Constant Elasticity of Substitution）と呼ばれ，次の式で与えられる。
# $$
# Y = a\left[\alpha (bK)^\rho+(1-\alpha)(cL)^\rho\right]^{\frac{1}{\rho}}
# $$
# ここで
# * $\sigma=\dfrac{1}{1-\rho}$
# * $\rho\leq 1$：要素の代替の程度を決定する。
# * $0<\alpha<1$：要素の貢献度のシェアを決定する。
# * $b>0,c>$：要素の単位に依存する。
# * $a>0$：生産の単位に依存する。
# 
# また，$\rho$の値によって次のような生産関数となる。
# $$
# Y = 
# \begin{cases}
#     &a\left[\alpha bK+(1-\alpha)cL\right],\quad\rho=1\quad \text{（完全代替型）}\\
#     &aK^\alpha L^{1-\alpha},\quad\rho=0\quad\text{（コブ・ダグラス型）}\\
#     &a\cdot\text{min}\left\{bK, cL\right\},\quad\rho=-\infty\quad\text{（完全補完型またはレオンティエフ型）}
# \end{cases}
# $$
# 
# 次のコードで$\rho=-\infty$以外のケースを関数で表している。

# In[135]:


def ces_production(k, l, rho=0, alpha=0.3, a=1, b=1, c=1):
    
    if rho > 1:
        print('rhoには１以下の数字を入力してください。')

    elif rho == 1:
        return a*( alpha*b*k + (1-alpha)*c*l )
    
    elif rho == 0:
        return a*k**alpha * l**(1-alpha)
    
    else:
        return a*( alpha*(b*k)**rho + (1-alpha)*(c*l)**rho )**(1/rho)


# In[136]:


ces_production(10,3,rho=-1)


# ## ヘルプ

# 組み込み関数`help()`を使うと関数やモジュールなど説明を表示させることができる。例えば，`print()`を例として挙げる。

# In[137]:


help(print)


# 引数は関数名であり`()`は付いていないことに留意しよう。`()`を付けると`print()`を評価した結果に対しての説明が表示されることになる。英語での説明だがパターンを理解すればこれだけでも有用に感じることだろう。 
# 
# `help()`の代わりに`?`を使うこともできる。

# In[138]:


get_ipython().run_line_magic('pinfo', 'print')


# ## パッケージとモジュール

# Pythonには組み込み関数が多く用意されているが，Jupyter Notebookで`Python`のセッションを開始しても全ての関数が使える状態になっていない。使えるようにするためには，モジュール（modules）やパッケージ（package）と呼ばれるものを読み込む必要がある。２つの違いを簡単にいうと
# * モジュールは１つのファイル（.py）にまとめられた関数群であり，
# * パッケージは複数つのファイル（.py）で構成されている（フォルダーにまとめられている）
# 
# となる。従って，モジュール全体を読み込んだり，あるパッケージの１つのモジュールだけを読み込むということも可能である。授業では，`NumPy`，`SciPy`，`Pandas`を使うが，ここでは例として数学用の`math`モジュールを考える。
# 
# 含まれる関数を使うためには`import`を使って読み込む必要がある。モジュールの全てを読み込むとモジュール内の全ての関数が使用可能となる。

# In[139]:


import math


# In[140]:


math.sqrt(4)    # sqrt（）とはルート


# `math.sqrt`とは「`math`モジュールの`sqrt`」という意味であり，`math`をつけるのは他のモジュール等の`sqrt`とバッティングしないようにするためである。モジュール名が長い場合は，短い名前で読み込むことも可能である。

# In[141]:


import math as m


# In[142]:


m.sqrt(9)


# モジュール内の特定の関数だけを読み込むことも可能である。

# In[143]:


from math import sqrt, log   # logは自然対数で, sqrtの両方を読み込む


# このコードは「`math`モジュールから`sqrt`と`log`を読み込む」と読むことができる。

# In[144]:


sqrt(10), log(10)


# ただ、この場合は他のモジュールやパッケージ、もしくは自分が定義した関数をバッティングしないように注意する必要がある。

# ## スコープ

# ### 説明

# スコープとは、変数が所属し直接アクセスできるコードの中の「領域」を示す。類似する概念に名前空間（Namespace）もあるが、スコープの情報を記す「表」のことであり、スコープ（Scope）と同義と理解すれば良い。
# 
# ここでは基本的に以下のように理解すれば良いであろう。
# 
# * Jupyter Notebookを開始した時点からglobalスコープが始まる。
# * 関数を定義すると、その関数の範囲内でlocalスコープが生成される。
# * globalスコープで定義された変数は、localスコープからアクセスできるが、globalスコープからlocalスコープの変数にはアクセスできない。
# * 関数を実行すると次の順番で変数を探す。
#     1. 関数のローカス・スコープ
#     2. グローバル・スコープ
# 
# 次の例を考えよう。

# In[145]:


s = "Kobe University"  # globalスコープ

def scope_0():
    s = "神戸大学"  # localスコープ
    return s

scope_0()


# この関数を実行すると、Pythonはまず関数`scope_0`のローカル・スコープ内で変数`s`を探すことになる。ローカル・スコープに`s`があるので、それを返している。

# In[146]:


def scope_1():
    return s

scope_1()


# この例では、まず`Python`はローカル・スコープに`s`があるかを確かめる。ローカル・スコープにないため、次にグローバル・スコープに`s`がないかを確かめている。グローバル・スコープに`s`があったので、それを返している（ないとエラーが出る）。
# 
# 次の例では、グローバル・スコープからローカル・スコープの変数へのアクセスを考える。

# In[147]:


def scope_2():
    s_local = 'Pythonは楽しい(^o^)/'
    return s_local

scope_2()


# `s_local`は関数`scope_2`のローカル・スコープで定義されている。グローバル・スコープからアクセスしようとするとエラーが発生する。

# In[148]:


s_local


# ### 教訓１

# > * 関数内で使う変数は、可能な限り関数内で定義する方が意図しない結果につながるリスクを軽減できる。
# > * グローバル・スコープで定義した変数を関数に使いたい場合は、引数として同じ変数を使う。
# 
# 次の例を考えよう。`a`は既に定義されている。

# In[149]:


a


# この値とは知らずに`10`だったと勘違いして次の関数を定義したとしよう。

# In[150]:


def scope_3(x):
    return x + a


# `scope_3(10)`は`20`を返すと思って実行すると意図しない結果になる。

# In[151]:


scope_3(10)


# このような場合は`a`を引数に使うことにより問題を回避できる。`a`は`10`として関数を実行すると意図した結果となる。

# In[152]:


def scope_4(x,a):
    return x + a

scope_4(10,10)


# この場合、関数`scope_4(x,a)`の`a`はローカス・スコープで定義され、グルーバル・スコープの`a`とは異なる。実際、グルーバル・スコープの`a`の値を確認してみると依然と変わらないままである。

# In[153]:


a


# ちなみに、グローバル・スコープの変数名や関数名は`%who`もしくは`%whos`のコマンドで確認できる。

# In[154]:


get_ipython().run_line_magic('who', '')


# このリストにある`s`はグローバル・スコープの`s`である。またローカル・スコープにある`s_local`はこのリストには含まれていない。

# ### 教訓２

# > `for`ループの１行目に使う変数は上書きされても構わない変数を使おう。
# 
# 例を使って説明しよう。

# In[155]:


for i in range(5):
    print(i)


# この`for`ループの`i`は`range(5)`の連番`0`、`1`、`2`、`3`、`4`を指す変数として使われるが、グローバル・スコープの変数として存在し、ループの最後の値が割り当てられている。確認してみよう。

# In[156]:


i


# `for`ループで使う変数は、ループ用の変数（例えば，`i`，`j`，`k`など）を使うのが良いだろう。
