#!/usr/bin/env python
# coding: utf-8

# (chap:pandas_plot)=
# # Pandas：プロット

# ## 説明

# In[1]:


import pandas as pd


# プロット用のパッケージ`Matplotlib`を紹介したが，実は`Pandas`の`DataFrame`と`Series`にはメソッド`plot()`が備えられており，それを使えば基本的なプロットをより簡単んコードで実現できる。裏で動いているのは`Matplotlib`であり，より複雑な図を作成する場合は、`Matplotlib`のコードを直接書くことが必要になるだろうが，手っ取り早くプロットしたい場合には重宝する手法である。詳細は[参考サイト（英語）](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)を参考にして欲しいが，ここでは基本的な使い方を紹介する。
# 
# 次の`df0`を使って説明する。

# In[2]:


dic = {'X':[10, 20, 30],
       'Y':[5.0, 30.0, 15.0],
       'Z':[3.0, 2.0, 5.0]}
df0 = pd.DataFrame(dic)
df0


# ## プロット方法

# プロット方法は簡単で，メソッド`.plot()`を使うと全ての列がプロットされる。

# In[3]:


df0.plot()


# * デフォルトでは横軸に行インデックスが使われ、`float`として扱われて表示されている。
# * 凡例は自動的に表示され、列ラベルが使われる。
# * 図の上に文字が表示されるが，表示したくない場合は最後に`;`を加えるか次の行に`pass`と書くと良いだろう。
# 
# 次にある列だけをプロットしたい場合を考えよう。その場合は，プロットしたい列を抽出しメソッド`plot()`を使う。選択には`[]`を使い，その中に列ラベルをリストとして指定する。まず１つの列を選択する場合を考えよう。例えば，`X`を選択するとしよう。この場合、２つの方法がある。
# 
# * `X`を`Series`として抽出する場合
#     ```
#     df0['X']
#     ```
# * `X`を`DataFrame`として抽出する場合
#     ```
#     df0[['X']]
#     ```
#     ここでは`X`をリスト`['X']`として使っている。
#     
# 列ラベルは文字列`X`で指定していることに留意しよう。
# 
# まず`X`の`Series`として抽出しよう。

# In[4]:


df0['X']


# 次に`X`を`DataFrame`として抽出する。

# In[5]:


df0[['X']]


# 表示が少し異なることが確認できるが，メソッド`plot()`でも少しだけ違いがある。まず`X`を`Series`としてプロットしてみる。

# In[6]:


df0['X'].plot()
pass


# `X`を`DataFrame`としてプロットすると凡例も表示される。

# In[7]:


df0[['X']].plot()
pass


# この場合の凡例は列ラベルがそのまま使われている。
# 
# 次に`X`と`Z`をプロットしたいとしよう。２つの列を抽出するためには，以下のように列ラベルをリストとして書く。

# In[8]:


df0[['X','Z']]


# メソッド`plot()`を使いプロットする。

# In[9]:


df0[['X','Z']].plot()
pass


# もしくは`Series`を2行に分けても同じ図となるが，凡例は表示されない。

# In[10]:


df0['X'].plot()
df0['Z'].plot()
pass


# デフォルトでは横軸に行インデックが使われるが，横軸を指定したい場合は，次の引数を使う。
# * `x`：横軸に使う列ラベル
# * `y`：縦軸に使う列ラベル（複数の場合はリスト）

# In[11]:


df0.plot(x='X',y=['Y','Z'])
pass


# ## 引数とメソッド

# ### 基本的な引数

# `plot()`には様々な引数があり図に「飾り付け」をすることができる。詳しくは[このリンク](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)を参照することにして，ここでは基本的な引数だけを紹介する。
# * `title`：図のタイトル（文字列型で指定）
# * `style`：線のスタイル（リストにして列の順番で指定する;`-``--``-.``:`）
# * `linewidth` or `lw`：線の幅
# * `color`：色（リストにして列の順番で指定する; [参照サイト](https://matplotlib.org/3.1.0/gallery/color/named_colors.html)）
#     * `r`は赤
#     * `k`は黒
#     * `g`はグリーン
# * `marker`：観測値のマーカー（`o`，`.`，`>`，`^`などがある; [参照サイト](https://matplotlib.org/3.2.2/api/markers_api.html)）
# * `markersize`：マーカーの大きさ
# * `fontsize`：横軸・縦軸の数字のフォントサイズの設定
# * `figsize`：図の大きさ
#     * `figsize=(キャンバスの横幅、キャンバスの縦の長さ)`
# * `legend`：凡例（デフォルトは`True`）
# * `grid`：グリッド表示（ブール型;デフォルトは`False`)

# In[12]:


df0.plot(title='This Is a Title'
        ,style=[':','--','-']
        ,linewidth=2
        ,color=['r','k','g']
        ,marker='o'
        ,markersize=10
        ,fontsize=15
        ,figsize=(8, 4)   # 8は横軸、4は縦軸のサイズ
        ,legend=False
        ,grid=True
        )
pass


# ### タイトルとラベルのサイズの調整

# タイトルのフォント・サイズの指定，横軸と縦軸のラベルとフォント・サイズの指定をおこなう場合は、`plot()`の引数ではなく下で説明する方法でおこなう。この方法を理解するために、`Python`は次の手順で描画していることをイメージして欲しい。ここで重要なのは「キャンバス」と「軸」の違いである。
# 1. 図を描くためのキャンバスを用意する。
#     * `figure`や`fig`などの変数名があれば、キャンバスを指していると理解すれば良いだろう。
# 1. キャンバスのうえに実際に図を描く。
#     * `ax`や`axes`などの変数名があれば、「図」を表していると理解すれば良いだろう。
#     * ここでの「図」はキャンバスと異なるが混同しやすいので、以下では「図」の代わりに「軸」（英語で "axis"）という言葉を使うことにする。
#     * キャンバスには複数の軸を描くことが可能である。
# 
# では実際に手順を説明する。
# * `df0.plot()`を変数（例えば，`ax`）に割り当てる。
#     * ここで割り当てたオブジェクト（実体）は「軸」である。
# * `ax`のメソッドを使って以下を設定する。
#     * タイトル：`.set_title()`
#     * 横軸ラベル：`.set_xlabel()`    
#     * 縦軸ラベル：`.set_ylabel()`

# In[13]:


ax = df0.plot(grid=True                     # title=の引数は使わない
             ,style=[':','--','-']
             ,marker='o'
             ,fontsize=15
             )
ax.set_title('A Large Title', size= 30)     # タイトルの設定
ax.set_xlabel('Horizontal Axis', size=20)   # 横軸ラベルの設定
ax.set_ylabel('Vertical Axis', size=20)     # 縦軸ラベルの設定
pass


# ### 図を並べる

# 図を縦に並べるには`subplots=True`を指定する。

# In[14]:


df0.plot(subplots=True)
pass


# 図を横に並べるには`layout=(1,3)`を付け加える。`layout`は図の配置を行列のように考えて指定し、`1`は行の数であり、`3`は列の数。
# ```
# layout(行の数、列の数)
# ```

# In[15]:


df0.plot(subplots=True, layout=(1,3), figsize=(8,3))
pass


# ### ２軸グラフ

# 2つの`df0`を並べて、右の縦軸を使う`df0`に`secondary_y=True`を設定する。

# In[16]:


df0[['X','Y','Z']].plot(secondary_y='Z')
pass


# 別々の飾り付けをする場合は次のようにする。

# In[17]:


df0[['X','Y']].plot()
df0['Z'].plot(marker='x', markersize=10, linestyle=':', legend=True, secondary_y=True)
pass


# 凡例が表示されるように`legend=True`も追加してある。

# ## 日本語

# ２つ方法を紹介するが、`japanize_matplotlib`を使う方法がより簡単であろう。

# ### `japanize_matplotlib`

# 使い方は到って簡単で、`Pandas`と同様にインポートするだけである。

# In[18]:


import japanize_matplotlib


# In[19]:


ax = df0.plot(grid=True
             ,style=[':','--','-']
             ,marker='o'
             ,fontsize=15
             )
ax.set_title('縦横タイトル', size= 30)
ax.set_xlabel('横軸', size=20)
ax.set_ylabel('縦軸', size=20)
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

# In[20]:


jfont = 'IPAexGothic'    # (1)

ax = df0.plot(grid=True
             ,style=[':','--','-']
             ,marker='o'
             ,fontsize=15
             )
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

# In[21]:


def investment(y):
    return 100/(1+r)


# * `100`：実質利子率が`0`の場合の投資
# 
# 実質利子率は次のリストで与えられるとする。

# In[22]:


r_list = list(range(0, 50))   # %表示の利子率 


# In[23]:


i_list = []   # 空のリスト

for r in r_list:
    inv = investment(r)   # 投資の計算
    i_list.append(inv)    # リストに追加

df_inv = pd.DataFrame({'Investment':i_list})   # DataFrameの作成


# 最初の5行を表示する。

# In[24]:


df_inv.head()


# 最後の5行を表示する。

# In[25]:


df_inv.tail()


# In[26]:


df_inv.plot()
pass


# ### 将来価値

# `x`万円を実質年率`r`%の利息を得る金融商品に投資し，`t`年間の将来価値（期首の値）をリストで示す関数は以下で与えられた。

# In[27]:


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

# In[28]:


r_list = [0.01, 0.03, 0.06]   # 実質利子率のリスト
dic = {}                      # 空の辞書

for r in r_list:
    dic['r='+str(r)] = calculate_futre_value(100, r, 30)  # 辞書に追加

df_future = pd.DataFrame(dic) # DataFrameの作成


# `dic['r='+str(r)]`の説明：
# * `str(r)`：`r_list`の要素のダミーである`r`は浮動小数点型なので関数`str()`を使って文字列型に変換する。
# * `'r='+str(r)`：文字列型の`r=`と文字列型の`str(r)`を`+`で結合する。
# * `dic['r='+str(r)]`：辞書`dic`にキー・値のペアを作成する。
#     * キー：`'r='+str(r)`
#     * 値：`calculate_futre_value(100, r, 30)`の返り値

# 最初の5行を表示する。

# In[29]:


df_future.head()


# 最後の5行を表示する。

# In[30]:


df_future.tail()


# In[31]:


df_future.plot()
pass

