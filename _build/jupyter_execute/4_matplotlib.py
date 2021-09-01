#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# ## 説明

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# `Matplotlib`（「マットプロットリブ」と読む）はプロットのための代表的なパッケージであり、ここではその使い方を解説する。プロットには`Matplotlib`のモジュールである`pyplot`を使うことになる。慣例に沿って`plt`としてインポートする。<br>

# ```{margin}
# <div name="html-admonition">
# Do you want to read in a differnt language? Open the 
# <input type="button" onclick="location.href='https://translate.google.com/translate?hl=&sl=ja&tl=en&u='+window.location;" value="Google translated version" style="color:#ffffff;background-color:#008080;" onmouseover="this.style.background='#99ccff'" onmouseout="this.style.background='#008080'"/>
# in English or the language of your choice.
# </div>
# ```

# `pyplot`モジュールを使ってプロットする場合、主に３つのコードの書き方がる。
# 
# **書き方１**：（オブジェクト指向）
# ```
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(...)
# ```
# **書き方２**：（オブジェクト指向）
# ```
# fig, ax = plt.subplots()
# ax.plot(...)
# ```
# **書き方３**： `Matlab`という有料ソフトのコードに沿った書き方
# ```
# plt.plot(...)
# ```
# 
# 本サイトでは「書き方２」に沿って`Matplotlib`の使い方を説明する。一方で，その裏にある考え方を理解するには、「書き方１」から始めた方が良いと思われるので，「書き方１」の簡単な説明をすることにする。理解する上で，特に重要な点は「キャンバス」と「軸」の違いである。
# 1. 図を描くためのキャンバスを用意する。
#     * それを行なっているのが書き方１の１行目である。右辺の`plt.figure()`でキャンバスを作成し、それを変数`fig`に割り当てている。キャンバスは実際に表示されないので、「透明のキャンバス」と理解すれば良いだろう。
# 1. キャンバス上に「軸」を描く。
#     * ここでの「軸」とは縦軸・横軸をイメージしても良いし，パイ・チャートであれば図の骨格と理解すれば良いだろう。
#     * 「軸」を描いているのが書き方１の２行目である。右辺では、キャンバス`fig`のメソッド`add_subplot()`を使いキャンバスに軸を描き、それを左辺の変数`ax`に割り当てている。実際に、書き方１の２行のコードを実行すると透明のキャンバス上に軸が表示される。（試してみよう。）
#     * 作成された軸`ax`にメソッド`plot()`などを使い直線・曲線・点など付け加えたり、タイトルなどの「飾り付け」をおこない作図することになる。
#     * １つのキャンバスには複数の軸を描くことが可能である。
# 
# この説明でわかることは、`Matplotlib`で表示される「図」とは透明のキャンバス上に描かれ装飾された「軸」（英語で "axis"）ということである。以下の説明では、「図」という単語はキャンバス上の全ての「軸」を指す場合もあれば，一つの「軸」を表す場合もあるので，文脈に従って解釈してほしい。
# 
# さて「書き方２」に話を戻すと、既に察している読者もいるかも知れないが、１行目は「書き方１」の最初の２行を１行に省略したものである。右辺の`plt.subplots()`は透明のキャンバスと軸の２つをタプルとして返す。最初の要素が「キャンバス」でありそれを左辺の`fig`に、２つ目の要素が軸であり`ax`に割り当てている。実際、１行目だけを実行すると軸が表示される。後は「書き方１」同様、`ax`のメソッド`plot()`などを使い軸を飾り付けして作図することになる。
# 
# ちなみに、書き方３は書き方２の３行を１行に省略した形と解釈して良いだろう。本来は`Matlab`ユーザーの`Python`への移行を促す為に用意された書き方である。

# 次に，`plot()`の引数について説明するが，主に２つの設定方法がある。
# 
# **引数の書き方１**：
# ```
# plot(＜横軸のデータ＞, ＜縦軸のデータ＞)
# ```
# **引数の書き方２**：変数のラベル名が使える場合（`Pandas`の`DataFrame`）
# ```
# plot(＜横軸の変数名＞, ＜縦軸の変数名＞, data=＜データ名＞)
# ```
# 
# それぞれの引数の書き方について解説する。

# ## 引数の書き方１

# ### `array`を使って

# プロットするデータを生成しよう。変数`x`に横軸に使うデータを次のように割り当てる。

# In[2]:


x = np.linspace(-2, 2, 3)
x


# ３つの要素からなる`array`である。縦軸のデータとして`y0`を用意する。

# In[3]:


y0 = x**2
y0


# 同じように`y0`も３つの要素からなる`array`である。プロットしてみよう。

# In[4]:


fig, ax = plt.subplots()
ax.plot(x, y0, marker='o')


# ２行目に`marker='o'`が追加されているが，「●」を表示するために使っている。このような引数の使い方は後で詳しく説明するので，ここでは気にしないで読み進めて欲しい。
# 
# 「●」のマーカーがある点が`x`と`y0`の`array`要素の組み合わせとして表示されている。`plot()`はデフォルトでそれらの点を直線で結んでいる。この場合，データの組み合わせが３点しかないため$y=x^2$の形を明確に表示できていない。言い換えると，データの組み合わせの数を増やすことにより$y=x^2$をより正確に表示されることになる。もう一度，`x`と`y0`のデータを生成しなおす。

# In[5]:


x = np.linspace(-2, 2, 100)
y0 = x**2


# `x`と`y0`にはそれぞれ100の要素があり，これらを使うことにより100のデータの組み合わせが成立することになる。プロットしてみよう。

# In[6]:


fig, ax = plt.subplots()
ax.plot(x, y0, marker='o')


# より$y=x^2$の図に近づいたことがわかる。
# 
# （注意点）
# * 上の２つの図の上に文字が表示されているが，表示したくない場合は最後に`;`を加えるか次の行に`pass`と書くと表示されなくなる。
# * `x`を省いて`ax.plot(y0)`としても同じような図が表示されるが，横軸の値がデータの数に対応することになり，意図した図と異なることになる。
# 
# 次に縦軸のデータとして，２つを用意する。

# In[7]:


y1 = x+1
y2 = np.exp(x)


# `y0`，`y1`，`y2`を同じ軸にプロットしてみよう。コードは簡単で`ax.plot()`をリピートするだけである。

# In[8]:


fig, ax = plt.subplots()
ax.plot(x, y0)           
ax.plot(x, y1)
ax.plot(x, y2)
pass


# それぞれの`ax.plot()`を「軸`ax`にデータの組み合わせをプロットする」と読めば理解しやすいと思う。また同じようなコードが続いているので，次のように`for`ループを使うことより短いコードで同じ図を表示することができる。

# In[9]:


fig, ax = plt.subplots()

for y in [y0, y1, y2]:
    ax.plot(x, y)           

pass


# ### `DataFrame`を使って

# まず`DataFrame`を作成しよう。

# In[10]:


dic = {'X':x, 'Y0':y0, 'Y1':y1, 'Y2':y2}  # 辞書の作成

df0 = pd.DataFrame(dic)      # DataFrameの作成

df0.head()


# 横軸に`X`，縦軸に`Y0`をプロットしてみる。

# In[11]:


fig, ax = plt.subplots()
ax.plot(df0['X'],df0['Y0'])
pass


# 横軸に`X`，縦軸に`Y0`，`Y1`，`Y2`をプロットしてみる。

# In[12]:


fig, ax = plt.subplots()
ax.plot(df0['X'],df0['Y0'])
ax.plot(df0['X'],df0['Y1'])
ax.plot(df0['X'],df0['Y2'])
pass


# 縦軸の変数だけを指定した場合はどうなるかを考えてみよう。

# In[13]:


fig, ax = plt.subplots()
ax.plot(df0['Y0'])
pass


# 上の図との違いは横軸の値である。この場合，横軸に`df0`の行インデックスが自動的に使われている。

# ## 引数の書き方２

# 次に引数`data`を使う書き方を考えてみよる。

# ### 同じ`DataFrame`を使う

# 横軸に`X`，縦軸に`Y0`を指定してプロットしてみる。

# In[14]:


fig, ax = plt.subplots()
ax.plot('X', 'Y0', data=df0)
pass


# 変数`Y1`と`Y2`も同じ軸`ax`にプロットしてみよう。

# In[15]:


fig, ax = plt.subplots()
ax.plot('X', 'Y0', data=df0)
ax.plot('X', 'Y1', data=df0)
ax.plot('X', 'Y2', data=df0)
pass


# 横軸の`X`を指定せずにプロットしてみよう。

# In[16]:


fig, ax = plt.subplots()
ax.plot('Y0', data=df0)
pass


# 行インデックスが横軸の値として使われている。

# ### 異なる`DataFrame`の変数を図示する

# 別の`DataFrame`を作成する。

# In[17]:


dic = {'X':x, 'Y3':-x-1, 'Y4':-x**2}
df1 = pd.DataFrame(dic)
df1.head()


# `X`を横軸として`df0`の`Y0`，`Y3`を横軸として`df1`の`Y3`を同じ図に表示してみよう。

# In[18]:


fig, ax = plt.subplots()
ax.plot('X', 'Y0', data=df0)
ax.plot('Y3', 'Y4', data=df1)
pass


# ## その他の引数とメソッド

# 上で説明したコードでは単純なプロットしかできない。一方で様々な引数やメソッドが用意されており，それらを駆使することにより様々な「飾り付け」をすることが可能となる。ここでは主なものを紹介するが，他にも数多く用意されているので`Matplotlib`の[サイト](https://matplotlib.org/stable/index.html)を参考にしてほしい。また，これらの引数・メソッドは，上述の「引数の書き方１」と「引数の書き方２」の両方に有効であることも覚えておこう。
# 
# 引数・メソッドがもたらす違いを際立たせるために新たな`DataFrame`を使おう。

# In[19]:


dic = {'X':[10, 20, 30],
       'Y':[5.0, 30.0, 15.0],
       'Z':[7.0, 2.0, 10.0]}
df2 = pd.DataFrame(dic)
df2.head()


# 以下のプロットでは，行インデックが横軸に使われるので注意しよう。

# ### `subplots()`の引数

# `subplots()`はキャンバスと軸を設定する`plt`のメソッドとなるが、主に３つの引数を紹介する。
# * `figsize`：図の大きさ
#     * `figsize=(キャンバスの横幅、キャンバスの縦の長さ)`
# * `nrows`（デフォルトは`1`）：軸の数を指定するために使う引数（「図を並べる」のセクションで説明する）
# * `ncols`（デフォルトは`1`）：軸の数を指定するために使う引数（「図を並べる」のセクションで説明する）
# * `sharex`（デフォルトは`False`）：複数の軸がある場合に使う引数（「図を並べる」のセクションで説明する）
# * `sharey`（デフォルトは`False`）：複数の軸がある場合に使う引数（「図を並べる」のセクションで説明する）
# * `tight_layout`（デフォルトは`False`）：複数の軸がある場合に使う引数（「図を並べる」のセクションで説明する）

# In[20]:


fix, ax = plt.subplots(figsize=(8,4))
ax.plot('X', data=df2)
pass


# ### `plot()`の基本的な引数

# `plot()`は軸`ax`にデータを描くメソッドだが、引数を使うことによりデータの表示方法を指定できる。詳しくは[このリンク](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)を参照することにして，ここでは基本的な引数だけを紹介する。
# * `linestyle`：線のスタイル（リストにして列の順番で指定する;`-`，`--`，`-.`，`:`などがある）
# * `linewidth` or `lw`：線の幅
# * `color` or `c`：色（[参照サイト](https://matplotlib.org/stable/gallery/color/named_colors.html)）
#     * `r`は赤
#     * `k`は黒
#     * `g`はグリーン
# * `marker`：観測値のマーカー（`o`，`.`，`>`，`^`などがある; [参照サイト](https://matplotlib.org/stable/api/markers_api.html)）
# * `markersize`：マーカーの大きさ
# * `label`：以下で説明する`ax.legend()`がある場合に有効となる

# In[21]:


fix, ax = plt.subplots()
ax.plot('X'
           , data=df2
           , linestyle=':'
           , linewidth=2
           , color='red'
           , marker='o'
           , markersize=10
           , label='X series')
ax.plot('Y'
           , data=df2
           , linestyle='-'
           , linewidth=2
           , color='k'
           , marker='^'
           , markersize=10
           , label='X series')
pass


# 引数をいちいち書くのが面倒な場合、次の３つを簡略して一緒に指定できる。
# * `linestyle`
# * `color`
# * `marker`
# 
# 例えば、
# * `linestyle=':'`
# * `color='red'`
# * `marker='o'`
# 
# の場合、`:ro`と書くことができる。

# In[22]:


fix, ax = plt.subplots()
ax.plot('Y', ':ro', data=df2)
pass


# （注意点）
# * `:ro`は文字列
# * `:`、`r`、`o`の順番を変えても良い。
# * `:`や`:o`のように１つもしくは２つだけを指定しても良い。
# * `:ro`は`=`を使う引数の前に置く。
# 
# 詳細は[参考サイト（英語）](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html)を参照。

# ### `ax`の基本的なメソッド

# `ax`は透明なキャンバス上にある軸を指しているが、そのメソッドを使うことにより、軸周辺を「飾り付け」することができる。
# * `.set_title()`：タイトルを設定する。
#     * 文字列で指定し、大きさは引数`size`で指定する。
# * `.set_xlabel()`：横軸ラベル
#     * 文字列で指定し、大きさは引数`size`で指定する。
# * `.set_ylabel()`：縦軸ラベル
#     * 文字列で指定し、大きさは引数`size`で指定する。
# * `.legend()`：凡例を表示する。
#     * `plot()`の引数`label`がなければ、データ・フレームの列ラベルが使われる。
#     * `plot()`の引数`label`があれば、それが使われる。
# * `.grid()`：グリッド線が表示される。

# In[23]:


fig, ax = plt.subplots()
ax.plot('X', data=df2, label='X series')
ax.set_title('A Large Title', size= 30)     # タイトルの設定
ax.set_xlabel('Horizontal Axis', size=20)   # 横軸ラベルの設定
ax.set_ylabel('Vertical Axis', size=20)     # 縦軸ラベルの設定
ax.legend()
ax.grid()
pass


# ### 図を並べる

# 複数の図を並べたいとしよう。考え方としては、透明のキャンバスに複数の軸を設定し、それぞれの軸を使って作図していけば良いのである。キャンバスは四角なので偶数個の軸が設定できる。例として、次のように軸を配置したいとしよう。
# ```
# +---+---+---+
# | 1 | 2 | 3 |
# +---+---+---+
# | 4 | 5 | 6 |
# +---+---+---+
# ```
# ２行・３列の配置になっており、左上から軸の番号が振られている。このような場合、`subplots()`の２つの引数が必須であり、別の２つの引数が有用である。
# ```
# subplots(行の数, 列の数, sharex=False, sharey=False, tight_layout=False)
# ```
# * 行の数（デフォルトは`1`）：上の例では`2`
# * 列の数（デフォルトは`1`）：上の例では`3`
# * `sharex`（デフォルトは`False`）：`True`にすると、全ての図で横軸が同じになり、不要な横軸の数字などを非表示になる。
# * `sharey`（デフォルトは`False`）：`Ture`にすると、全ての図で縦軸が同じになり、不要な縦軸の数字などを非表示にする
# * `tight_layout`（デフォルトは`False`）：`Ture`にすると、全ての図の間隔を調整して見やすくなる。
# 
# 上の例を使うと次のようになる。

# In[24]:


fig, ax = plt.subplots(2,3)


# ６つの軸が表示されているが、それらは`ax`に割り当てられている。`ax`を表示してみよう。

# In[25]:


ax


# `<AxesSubplot:>`が軸を示しており、それぞれの軸に対応している。

# In[26]:


ax.shape


# 軸は`(2,3)`の`array`に格納されているのが確認できる。従って、それぞれの軸は`array`の要素を抽出することによりアクセスできる。
# * `ax[0,0]`：軸１を抽出
# * `ax[0,1]`：軸２を抽出
# * `ax[0,2]`：軸３を抽出
# * `ax[1,0]`：軸４を抽出
# * `ax[1,1]`：軸５を抽出
# * `ax[1,2]`：軸６を抽出
# 
# これを使い次のように６つのを図示できる。

# In[27]:


fig, ax = plt.subplots(2,3,figsize=(8,3))
ax[0,0].plot('X', data=df2)
ax[0,1].plot('Y', data=df2)
ax[0,2].plot('Z', data=df2)
ax[1,0].plot('X', data=df2)
ax[1,1].plot('Y', data=df2)
ax[1,2].plot('Z', data=df2)
pass


# 横軸と縦軸を共有するには次のようにする。

# In[28]:


fig, ax = plt.subplots(2,3,figsize=(8,3),sharex=True,sharey=True)
ax[0,0].plot('X', data=df2)
ax[0,1].plot('Y', data=df2)
ax[0,2].plot('Z', data=df2)
ax[1,0].plot('X', data=df2)
ax[1,1].plot('Y', data=df2)
ax[1,2].plot('Z', data=df2)
pass


# `ax`のメソッド（例えば、軸のタイトル）はそれぞれ設定することができる。

# 次に、キャンバス全体のタイトルを設定したいとしよう。キャンバスは`fig`に割り当てられているので、そのメソッド`suptitle`を使い表示することができる。

# In[29]:


fig, ax = plt.subplots(2, 3, figsize=(8,3),
                       sharex=True, sharey=True,
                       tight_layout=True)
ax[0,0].plot('X', data=df2)
ax[0,0].set_title('Figure 1')

ax[0,1].plot('Y', data=df2)
ax[0,1].set_title('Figure 2')

ax[0,2].plot('Z', data=df2)
ax[0,2].set_title('Figure 3')

ax[1,0].plot('X', data=df2)
ax[1,0].set_title('Figure 4')

ax[1,1].plot('Y', data=df2)
ax[1,1].set_title('Figure 5')

ax[1,2].plot('Z', data=df2)
ax[1,2].set_title('Figure 6')

fig.suptitle("Grand Title", fontsize=20)
pass


# ### ２軸グラフ

# 複数のデータを表示する際、右の縦軸を使いデータを表示したい場合がある。例を使って説明することにする。

# In[30]:


fig, ax1 = plt.subplots()  # (1)

ax2 = ax1.twinx()          # (2)

ax1.plot('X', data=df2)    # (3)
ax2.plot('Z', data=df2)    # (4)
pass


# コードの説明
# 
# > 1. キャンバスと軸を作成し、軸を`ax1`に割り当てる。
# > 2. `ax1`のメソッド`twinx()`を使い、右の縦軸を準備し`ax2`に割り当てる。
# > 3. `ax1`に`X`をプロットする。
# > 4. `ax2`に`Z`をプロットする。

# また上で説明した方法で様々な「飾り付け」をすることができる。ただ、凡例については少し追加的なコードが必要となるので、それについても簡単に以下で説明する。

# In[31]:


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot('X', 'k-o', data=df2, label='X series')
ax2.plot('Z', 'r:', data=df2, label='Z series')

ax1.set_title('Title', size=20)    # (1)
ax1.set_ylabel('X data', size=15)  # (2)
ax2.set_ylabel('Z data', size=15)  # (3)

ax1.legend()                       # (4)
ax2.legend(loc=(0.015,0.81))        # (5)
pass


# コードの説明
# 
# > 1. 軸`ax1`にタイトルを設定しているが、`ax2`でも同じ図となる。
# > 2. 軸`ax1`の縦軸（左）のラベルを設定する。
# > 3. 軸`ax2`の縦軸（右）のラベルを設定する。
# > 4. 軸`ax1`に描いた`X`の凡例を設定する。
# >    * `legend()`には引数がないので`Matplotlib`が自動で凡例の位置を決める。この場合は図の左上に表示されている。
# > 5. 軸`ax2`に描いた`Z`の凡例を設定する。
# >    * `legend(loc=(0.015,0.8))`には引数`loc`があり、凡例の位置を指定している。`(0.015,0.8)`の数字は図の原点を`(0,0)`とて左の数字はx軸、右の数字はy軸の位置を指定している。もし引数を設定しないと、`X`の凡例を上書きすることになる。`(0.015,0.8)`のような指定方法だけではなく、他の方法もあるので[参照サイト](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)を参考にして欲しい。

# ## 日本語

# ２つ方法を紹介するが、`japanize_matplotlib`を使う方法がより簡単であろう。

# ### `japanize_matplotlib`

# 使い方は到って簡単で、`Pandas`と同様にインポートするだけである。

# In[32]:


import japanize_matplotlib


# In[33]:


fix, ax = plt.subplots()
ax.plot('X', 'Y', 'r-o', data=df2)
ax.set_title('縦横タイトル', size= 30)
ax.set_xlabel('横軸', size=20)
ax.set_ylabel('縦軸', size=20)
ax.legend()
pass


# ### フォントを指定する

# 2つの方法：
# 1. フォントはインストールせず、PC内にあるフォントを指定する。
# 1. フォントをインストールする方法

# 方法１の場合、以下で説明に使う変数`jfont`にフォントを指定する。
#     * Macの場合、例えば`AppleGothic`
#     * Windowsの場合、例えば`Yu Gothic`
#     * この方法では一部の日本語が文字化けする場合がある。
# 
# 方法２の場合：
# * [このサイト](https://ipafont.ipa.go.jp/node193#jp)から次の内の１つをダウンロードする。
#     * 2書体パック(IPAex明朝(Ver.xxx)、IPAexゴシック(Ver.xxx))
#     * IPAex明朝 (Ver.xxx)
#     * IPAexゴシック(Ver.xxx)
# * [このサイト](https://ipafont.ipa.go.jp/node72#jp)に従ってインストールする。
# * 次の両方もしくは１つがPCにインストールされる
#     * IPAexMincho（IPAex明朝）
#     * IPAexGothic（IPAexゴシック）
# 
# 上の例を使い、設定方法の例を示す。

# In[34]:


jfont = 'IPAexGothic'    # (1)

fix, ax = plt.subplots()
ax.plot('X', 'Y', 'ro-', data=df2)
ax.set_title('縦横タイトル', size= 30, fontname=jfont)   # (2)
ax.set_xlabel('横軸', size=20, fontname=jfont)          # (3)
ax.set_ylabel('縦軸', size=20, fontname=jfont)          # (4)
ax.legend(prop={'family':jfont, 'size':17})            # (5)
pass


# >* (1) 使用するフォントを`jfont`に割り当てる。
# >* (2) 引数`fontname`で`jfont`を指定する。タイトルのフォントが変更される。
# >* (3) 引数`fontname`で`jfont`を指定する。横軸名のフォントが変更される。
# >* (4) 引数`fontname`で`jfont`を指定する。縦軸名のフォントが変更される。
# >* (5) `legend`は他と設定方法が異なる。
#     * `prop`はフォントのプロパティを設定する引数であり、辞書で指定する。
#     * キー`family`に値`jfont`を指定する。凡例のフォントが変更される。
#     * キー`size`に数値を設定してフォントの大きさが変更される。

# この例では個別にフォントを設定したが、一括で全てのフォントを変更する方法もあるが説明は割愛する。

# ## マクロ経済学の例

# ### 投資関数

# 実質利子率`r`によって投資がどのように変化するかを考えてみよう。まず投資関数を次のように仮定する。

# In[35]:


def investment(y):
    return 100/(1+r)


# * `100`：実質利子率が`0`の場合の投資
# 
# 実質利子率は次のリストで与えられるとする。

# In[36]:


r_list = list(range(0, 50))   # %表示の利子率 


# In[37]:


i_list = []   # 空のリスト

for r in r_list:
    inv = investment(r)   # 投資の計算
    i_list.append(inv)    # リストに追加

df_inv = pd.DataFrame({'Investment':i_list})   # DataFrameの作成


# 最初の5行を表示する。

# In[38]:


df_inv.head()


# 最後の5行を表示する。

# In[39]:


df_inv.tail()


# In[40]:


fix, ax = plt.subplots()
ax.plot(df_inv)
pass


# `df_inv`はデータ・フレームだが、変数が１つしかないため`plot()`の引数に`df_inv`を指定するだけで`investment`のデータが描かれている。

# ### 将来価値

# `x`万円を実質年率`r`%の利息を得る金融商品に投資し，`t`年間の将来価値（期首の値）をリストで示す関数は以下で与えられた。

# In[41]:


def calculate_futre_value(x, r, t):
    
    value_list = [x]           # 初期値が入ったリスト
    
    for year in range(1,t+1):  # 1からtまでの期間
        x = x*(1+r)            # 来期のxの値の計算
        value_list.append(x)   # リストに追加
    
    return value_list          # リストを返す


# これを使い，
# * `x`=`100`
# * `t`=`30`
# 
# の下で実質利子率が次のリストで与えられる値を取る場合の将来価値を図示する。

# In[42]:


r_list = [0.01, 0.03, 0.06]   # 実質利子率のリスト
dic = {}                      # 空の辞書

for r in r_list:
    dic['r='+str(r)] = calculate_futre_value(100, r, 30)  # 辞書に追加

df_future = pd.DataFrame(dic) # DataFrameの作成


# `dic['r='+str(r)]`の説明：
# * `str(r)`：`r_list`の要素が割り当てられる`r`は浮動小数点型なので関数`str()`を使って文字列型に変換する。
# * `'r='+str(r)`：文字列型の`r=`と文字列型の`str(r)`を`+`で結合する。
# * `dic['r='+str(r)]`：辞書`dic`にキー・値のペアを作成する。
#     * キー：`'r='+str(r)`（文字列）
#     * 値：`calculate_futre_value(100, r, 30)`の返り値

# 最初の5行を表示する。

# In[43]:


df_future.head()


# 最後の5行を表示する。

# In[44]:


df_future.tail()


# In[45]:


fix, ax = plt.subplots()
ax.plot('r=0.01', data=df_future)
ax.plot('r=0.03', data=df_future)
ax.plot('r=0.06', data=df_future)
ax.legend()
pass


# `ax.plot()`が続いているので、`for`ループを使う事を推奨する。

# In[46]:


fix, ax = plt.subplots()

for col in df_future.columns:     # (1)
    ax.plot(col, data=df_future)

ax.legend()
pass


# （１）の`df_future.columns`について説明すると、`.columns`はデータ・フレーム`df_future`の属性であり列ラベルを返す。

# ## `Pandas`のメソッド`plot()`

# この章で付け加えたいもう一つのプロット方法がある。それが`Pandas`の`DataFrame`と`Series`のメソッド`plot()`である。これを使うと簡単にプロットできる場合が多々ある。詳細については[Pandas:プロット](chap:pandas_plot)もしくは[参考サイト（英語）](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)を参照して欲しいが，裏で動いているのは`Matplotlib`であり，より複雑な図を作成する場合は、やはり`Matplotlib`のコードを直接書くことが必要になるだろう。ちなみに，`pyplot`をインポートしなくとも`Pandas`の`plot()`のみでプロットすることが可能である。

# ### `plot()`の使い方

# `df2`を使ってプロットする方法を説明するが，方法は簡単で，メソッド`.plot()`を使うと全ての列が同じ軸にプロットされる。

# In[47]:


df2.plot()
pass


# 気づいたと思うが，列ラベルが自動的に凡例に表示される。
# 
# 次に，列`X`と`Y`だけをプロットしたいとしよう。その場合は，表示したい列だけを抽出して`plot()`を使えば良い。

# In[48]:


df2.loc[:,['X','Y']].plot()
pass


# もしくは2行に分けても同じ図となる。

# In[49]:


df2.loc[:,'X'].plot()
df2.loc[:,'Z'].plot()
pass


# デフォルトでは横軸に行インデックが使われるが，横軸を指定したい場合は，次の引数を使う。
# * `x`：横軸に使う列ラベル
# * `y`：縦軸に使う列ラベル（複数の場合はリスト）

# In[50]:


df2.plot(x='X',y=['Y','Z'])
pass


# ### 複数の図を並べる

# 図を縦に並べるには`subplots=True`を指定する。

# In[51]:


df2.plot(subplots=True)
pass


# 図を横に並べるには`layout=(1,3)`を付け加える。`layout`は図の配置を行列のように考えて指定し、`1`は行の数であり、`3`は列の数。
# ```
# layout(行の数、列の数)
# ```

# In[52]:


df2.plot(subplots=True, layout=(1,3), figsize=(8,3))
pass


# ### ２軸グラフ

# 2つの`df0`を並べて、右の縦軸を使う`df0`に`secondary_y=True`を設定する。

# In[53]:


df2.loc[:,['X','Y','Z']].plot(secondary_y='Z')
pass


# 別々の飾り付けをする場合は次のようにする。

# In[54]:


df2.loc[:,['X','Y']].plot()
df2.loc[:,'Z'].plot(marker='x',
                    secondary_y=True,
                    markersize=10,
                    linestyle=':',
                    legend=True)
pass


# ここで使った引数については，３.参考にある[Pandas:プロット](chap:pandas_plot)を参照してください。
