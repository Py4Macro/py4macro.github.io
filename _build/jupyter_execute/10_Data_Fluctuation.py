#!/usr/bin/env python
# coding: utf-8

# (chap:10-gdp)=
# # GDPと構成要素

# <div name="html-admonition" style="font-size: 0.8em">
# <input type="button" onclick="location.href='https://translate.google.com/translate?hl=&sl=ja&tl=en&u='+window.location;" value="Google translation" style="color:#ffffff;background-color:#008080; height:25px" onmouseover="this.style.background='#99ccff'" onmouseout="this.style.background='#008080'"/> in English or the language of your choice.
# </div><br>

# In[1]:


import japanize_matplotlib
import numpy as np
import pandas as pd
import py4macro


# ## はじめに

# 人の心の中を知ることはできない。友人，同僚，パートナー，親族もそうである。また何十年も連れ添った夫婦でさえも完全にお互いの心の中を分かっているとは言い難いだろう。しかし話し言葉やその内容，行動から心の中を推測することは可能である。実際，我々は日常そうしており，人間関係を豊かにする為には必須でる。経済も同じである。経済の動きの裏にある「真のメカニズム」は完全に把握されていない。アダム・スミスの時代に比べると，「熟練夫婦」に匹敵するほど知見は蓄積されたかも知れないが，今でも「真のメカニズム」が100％解明されたとは言い難い。経済学者にとってできることは，経済の「言動」であるデータを観察し「真のメカニズム」について想像を巡らすことである。人の言動に注意を払わずに人の心の中を想像するのは「妄想」となり，データを無視して経済を考えるのは「空想」となってしまう危険性がある。
# > 人の心の中　$\Longleftrightarrow$　経済の真のメカニズム
# >
# > 人の言動　$\Longleftrightarrow$　経済データ

# この章はマクロ・データの特徴を理解することを目的とする。統計量やプロットを駆使して異なるマクロ・データの特性を炙り出し，経済の「言動」に注目しようということである。特に，以下では国内総生産（GDP）とその構成要素に着目し議論を進める。
# 
# 使うデータは`py4macro`に含まれる日本の四半期データである。1980年から1993年のデータと1994年から2021年までの２つのデータ・セットを組み合わせて作成した。データを読み込むには次のようにする。
# ```
# py4macro.data('jpn-q')
# ```
# ここでjpnはJapan，qはquerterlyを表している。変数の定義を表示するには`description=True`の引数を追加する。

# In[2]:


py4macro.data('jpn-q', description=True)


# このデータ・セットにはGDPとその構成要素以外も含まれているが，それらは次章で使うことになる。
# 
# まずデータを読み込んで最初の５行を表示してみよう。

# In[3]:


df = py4macro.data('jpn-q')
df.head()


# 列には変数が並んでいるのが確認できる。行はインデックスではなくラベルとなっており，年・四半期を示している。例えば，`1980-3-31`は1980年第１四半期であり，その最終日がラベルとして使われている。`.info()`を使って確認してみる。

# In[4]:


df.info()


# 表示の上から２行目に`DatetimeIndex`とあるが，行ラベルのことを表しており，時系列データ用となっている。具体的な説明は割愛するが，時系列データが扱いやすくなりプロットする際にも便利である。

# ## GDPとその変化

# まずGDPを図示しよう。

# In[5]:


df['gdp'].plot(title='GDP')
pass


# 長期的には上昇トレンドであることが分かる。これが経済成長である。一方で，よく観察するとギザギザに動いていることも確認できる。景気変動である。小さい上下の動きもあれば，より大きな動きもある。例えば，2008年のリーマン・ショック，そして2020年のコロナ禍の影響でGDPは大きく下落している。
# 
# > (問４）なぜ景気変動は起こるのか？
# 
# これがマクロ経済学のもう一つのBig Questionである。この章以降は問４を中心にマクロ経済を考察することになる。
# 
# 上の図には2022年第1四半期以降のデータは含まれていない。参考に最新のデータを含む図を下にに表示する。

# In[6]:


from IPython.display import IFrame
IFrame(src='https://fred.stlouisfed.org/graph/graph-landing.php?g=BOLN&width=420&height=320',
       width=450, height=330)


# この図は[米国セントルイス連邦準備銀行が管理するFederal Reserve Economic Data (FRED)](https://fred.stlouisfed.org)と呼ばれるデータ・ベースからライブでダウンロードして表示している。
# 
# `py4macro`を使った上の図もしくはFREDの図にあるGDPは水準を表しているため，GDPの成長率を直接読み取り異なる時期の成長率を比べることは難しい。しかし，それを可能にするのが対数化したGDPのプロットである。まず，なぜそうなのかを簡単に説明しよう。
# 
# 変数`x`の成長率$g_x$は次の式で定義される。
# 
# $$
# 1+g_x=\frac{x_{t+1}}{x_{t}}
# $$
# 
# ここで$g_x$の値は，例えば`0.02`（`2％`）のような値となる。この式の両辺に対数を取り，近似を使うと次のように書き直すことができる。
# 
# $$
# \log(1+g_x)=\log(x_{t+1})-\log(x_{t})
# $$
# 
# $$
# \qquad\qquad\quad\Downarrow\text{左辺を近似}\;\log(1+g_x)\approx g_x
# $$
# 
# $$
# g_x\approx\log(x_{t+1})-\log(x_{t})
# $$ (eq:10-log-growth)
# 
# `x`の成長率である$g_x$は，`x`の対数の差分ということになる。即ち，縦軸に`x`の対数，横軸に時間をおけば，プロットの傾きは`x`の成長率と等しいことになる。
# 
# では実際にプロットしよう。

# In[7]:


df['gdp_log'] = np.log(df['gdp'])

df['gdp_log'].plot(title='Log of GDP')
pass


# プロットの傾きは四半期ごとのGDPの成長率であり，成長率が変化しているのがわかる。また，2008年のリーマン・ショックとコロナ禍により成長率が大きく下落していることが確認できる。景気循環により成長率が小刻みに，また時には大きく変化している。
# 
# 次に，長期的な視点で図を眺めてみよう。1980年から1990年までの間に直線のトレンドを描くことができる。その傾きが平均成長率である。同様に，1990年から2020年までの直線トレンドを描くこともできるが，傾きが緩やかになっていることに気づくのではないだろうか。1991年頃のバブル崩壊の影響により，その後のGDPの伸びが鈍化していることがわかる。「失われた10年」，「失われた20年」と言われていたが，今では「失われた30年」と言われる所以である。
# 
# 次に式[](eq:10-log-growth)を使い四半期成長率を計算してみよう。

# In[8]:


df['gdp_growth_quarter'] = 100*df['gdp_log'].diff()   # 1

# 2
ax_ = df['gdp_growth_quarter'].plot(title='Quarterly Rate of GDP Growth (%)')
ax_.grid()   # 3
pass


# ```{admonition} コードの説明
# :class: dropdown
# 1. `.diff()`は差分を計算するメソッドであり，引数の数（デフォルトは`1`）は何期前の値と差分を計算するかを指定する。
# 2. `Pandas`のメソッド`plot`は図を表示し，同時に図の「軸（axis）」を返すが，それを変数`ax_`に割り当てている。
# 3. 軸の変数`ax_`には様々なメソッドが用意されており，その１つがグリッド線を表示する`.grid()`である。
# ```

# 新聞などで「前年同期比成長率」という表現をみたことがあると思うが，`diff()`の引数を`4`にすることにより，前年同期比の成長率を計算することができる。それを図示してみよう。

# In[9]:


df['gdp_growth_annual'] =  100*df['gdp_log'].diff(4)

ax_ = df['gdp_growth_annual'].plot(title=
      'Annual GDP Growth Rate over \nthe Corresponding Quarter of a Previous Year')
ax_.grid()
pass


# 上の図と比べると縦軸の幅の違いに気づくだろう。最高値と最小値を計算してみよう。

# In[10]:


df['gdp_growth_annual'].max(), df['gdp_growth_annual'].min()


# 1980年代後半ののバブル景気には約`7.6`％増加しているが，コロナ禍では約`10`％以上下落している。
# 
# 次の点を知っておくのも有用だろう。
# * 次の式を使うと四半期成長率を年率換算することもできる。
# 
#     $$
#     g_{\text{年率換算成長率}}=\left(1+g_{\text{四半期成長率}}\right)^4-1
#     $$
# 
#     $g_{\text{四半期成長率}}$が４期続くことを想定した成長率である。
# * 年次データを使うとGDPの値は均されるので変化率は低めに出る。

# ## GDPの構成要素

# 次式は所得恒等式である。
# ```
#     GDP=消費＋投資＋政府支出＋純貿易（輸出ー輸入）
# ```
# GDPに対してそれぞれ構成要素は何％を占め，どのように推移したかを図示する為に，まず変数を計算する。

# In[11]:


# 消費の割合
con_gdp_ratio = 100 * df['consumption'] / df['gdp']

# 投資の割合
inv_gdp_ratio = 100 * df['investment'] / df['gdp']

# 政府支出の割合
gov_gdp_ratio = 100 * df['government'] / df['gdp']

# 純輸出の割合
net_exp_gdp_ratio = 100 * ( df['exports']-df['imports'] ) / df['gdp']


# それぞれの平均を`for`ループで計算し表示してみよう。

# In[12]:


ratio_list = [con_gdp_ratio,inv_gdp_ratio,
              gov_gdp_ratio,net_exp_gdp_ratio]

label_list = ['Consumption','Investment','Gov Exp','Net Exports']

for r, l in zip(ratio_list, label_list):
    avr = r.mean()
    print(l, f'\t{avr:.1f}')


# 消費はGDPの`60`％近くあり，GDPの約1/4が政府支出となっている。消費とは対照的に，投資は約`20`％であり消費の約３分の１である。（小数点第一位までしか表示していないため合計は100にならない。）
# 
# 次に推移を確認する。同じように`for`ループを使うと簡単に表示できる。

# In[13]:


for r, l in zip(ratio_list,label_list):
    r.plot(label=l, legend=True)


# ````{admonition} コードの説明
# :class: dropdown
# 
# * `zip()`について説明する。`zip()`はループによく使われる便利な関数である。以下の単純な`for`ループ
#     ```
#     for r in ratio_list:
#         r.plot()
#     ```
#     にはループ用の変数が`r`の１種類しかない。これでも図示することは可能である。しかし凡例を追加するためには`label_list`も`for`ループの中で同時に使う必要がある。そのような場合に便利な関数が`zip()`である。引数に`ratio_list`と`label_list`を入れると，同時に順番に要素にアクセスできる。それぞれの要素を割り当てる変数として`r`と`l`を使っている。`r`と`l`の間に`,`を入れることにより，それぞれ別のリストの要素を捉えることができる。
# * この`for`ループは以下と等しい。
#     ```
#     con_gdp_ratio.plot(label='Consumption', legend=True)
#     inv_gdp_ratio.plot(label='Investment', legend=True)
#     gov_gdp_ratio.plot(label='Gov Exp', legend=True)
#     net_exp_gdp_ratio.plot(label='Net Exports', legend=True)
#     ```
#     もちろんこれでも良いが，コードを書く際は極力同じような行を繰り返すのではなくループを使ってまとめる方が良い。第一の理由は，簡単なエラーの可能性を軽減することができることだ。リピートして書く場合，1行をコピペしその後に1行ずつ修正をするパターンが多いが，最初の１行目が間違っている場合，全ての行を修正する必要が発生する。その際に修正し忘れることがある。第二の理由は，コードの修正も簡単になるためである。例えば，`linewidth=2`を設定したいとしよう。`for`ループの場合は一箇所に加えるだけで済むことになる。
# ````

# 景気の動向によって上下することがわかる。例えば，リマン・ショック後には投資が大きく下落し少しずつしか上昇しない。一方，景気悪化に反応し政府支出の割合は上昇している。またバブル景気（1986年から1991年まで）ではその逆が起こっており，順位が逆転する程である。コロナ禍の影響もプロットにハッキリと現れている。

# ## トレンドと変動

# ### 説明

# 上の分析から明らかなことは，マクロ変数は変動するということであり，正しくそれが景気循環を表している。ここでは景気循環について，もう一歩踏み込んで考えてみる。
# 
# マクロ変数はトレンドと変動（サイクル）に分解することができる。例えば，$Y$をGDPとすると次式のように２つの項で表すことができる。
# 
# $$
# Y=Y_t^{\text{trend}}Y_t^{\text{cycle}}
# $$ (eq:10-decompose)
# 
# $Y_t^{\text{trend}}$は長期的な傾向を表し，GDPであれば経済成長を捉えている。また，トレンドが右上がりでないものありえる。例えば，失業率の場合，自然失業率が一定であればトレンドはないということになる。一方，$Y_t^{\text{cycle}}$は短期的な景気循環を捉えている。内閣府は景気循環を「山」と「谷」の２つに分けて[景気基準日付](https://www.esri.cao.go.jp/jp/stat/di/hiduke.html)を発表している。トレンドを上回っている期間を「山」，下回っている期間を「谷」と考えて良いだろう。図示すると次のようになる。

# In[14]:


def trend_cycle():
    
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    
    # フォントのサイズ
    font_largest=30
    font_large=20

    # GDPのプロット
    xlow = 0
    xhigh = 3*np.pi
    x = np.arange(xlow, xhigh, 0.1)

    def yfunc(x):
        return np.cos(x+np.pi/2)+0.1*x

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, yfunc(x), linewidth=7, label='GDP')

    # トレンド線のプロット
    xx = np.linspace(xlow-1., xhigh+1)

    def trend_line(x):
        return ( yfunc(3*np.pi)-yfunc(0) )/(3*np.pi) * (x-0)+yfunc(0)

    ax.plot(xx, trend_line(xx), lw=4, color='orange', label='トレンド')

    # 縦点線のプロット
    ylow = -1.68
    
    tani0 = mlines.Line2D([np.pi/2, np.pi/2], [ylow, yfunc(np.pi/2)],  linestyle='--')
    yama = mlines.Line2D([np.pi*3/2, np.pi*3/2], [ylow+0.4, yfunc(np.pi*3/2)],  linestyle='--')
    tani1 = mlines.Line2D([np.pi*5/2, np.pi*5/2], [ylow, yfunc(np.pi*5/2)],  linestyle='--')
    
    ax.add_line(tani0)
    ax.add_line(yama)
    ax.add_line(tani1)

    # 矢印のプロット（拡張と後退）
    for dx in [np.pi/2, -np.pi/2]:
        ax.arrow(x=np.pi, y=-1.2, dx=dx, dy=0,
                 width=0.01, head_width=0.1,head_length=0.2,
                 length_includes_head=True,color='k')

    for dx in [np.pi/2, -np.pi/2]:
        ax.arrow(x=2*np.pi, y=-1.2, dx=dx, dy=0,
                 width=0.01, head_width=0.1,head_length=0.2,
                 length_includes_head=True,color='k')

    # 矢印のプロット（全循環）
    start_arrow_shift = 0.8
    ax.arrow(x=np.pi*3/2-start_arrow_shift, y=-1.5,
             dx=-(np.pi-start_arrow_shift), dy=0,
             width=0.01, head_width=0.1,head_length=0.2,
             length_includes_head=True,color='k')
    ax.arrow(x=np.pi*3/2+start_arrow_shift, y=-1.5,
             dx=np.pi-start_arrow_shift, dy=0,
             width=0.01, head_width=0.1,head_length=0.2,
             length_includes_head=True,color='k')

    # 注釈の挿入
    x0 = np.pi/2
    x1 = np.pi*3/2
    x2 = np.pi*5/2
    xshift = 0.33
    yshift = 0.2

    ax.annotate('谷', xy=(x0-xshift, yfunc(x0)+yshift),size=font_largest)
    ax.annotate('山', xy=(x1-xshift, yfunc(x1)+yshift), annotation_clip=False, size=font_largest)
    ax.annotate('谷', xy=(x2-xshift, yfunc(x2)+yshift),size=font_largest)

#     ax.annotate('好況', xy=(xlow-1.1,0.2), annotation_clip=False, size=font_large, color='green')
#     ax.annotate('不況', xy=(xlow-1.1,-0.5), annotation_clip=False, size=font_large, color='red')
#     ax.annotate('好況', xy=(xhigh+0.3,1.25), annotation_clip=False, size=font_large, color='green')
#     ax.annotate('不況', xy=(xhigh+0.3,0.55), annotation_clip=False, size=font_large, color='red')

    ax.annotate('拡張', xy=(np.pi-0.2,-1.05), size=font_large, color='green')
    ax.annotate('後退', xy=(2*np.pi-0.5,-1.05), size=font_large, color='red')
    ax.annotate('全循環', xy=(np.pi*3/2-0.65,-1.6), size=font_large)

    # 横軸のラベル，凡例を追加，縦軸の表示範囲の設定
    ax.set_xlabel('時間', size=font_large)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_ylim(ymax=2.3)

    # 縦軸・横軸のラベルと目盛の削除
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # 枠を削除
    for s in ['top', 'right', 'left','bottom']:
        ax.spines[s].set_visible(False)

trend_cycle()


# この図のトレンドは右上がりとなっているが，一定もしくは右下がりになる場合もありえる。この図から次のことが言える。好況・不況とは，GDPがトレンドを上回るか・下回るかの状態を表している。一方で，拡張・後退はGDPの変化を捉えている。また，このプロットは`NumPy`のコサイン関数を使って描画しているので，「谷」と「山」が発生した時期を簡単に見分けることができる。しかし，上のGDPデータのプロットを見直すと，「谷」と「山」がいつだったかを決めるのは簡単な問題ではないと想像できるだろう。実際，内閣府は単にGDPのデータを眺めるだけではなく[景気動向指数](https://www.esri.cao.go.jp/jp/stat/di/menu_di.html)を含め様々なデータに基づき，「谷」と「山」の時期を確定している。
# 
# 景気循環を考える上で重要な役割を果たすのがトレンドである。トレンドはどのように計算するのだろうか。実は，決まった計算方法があるわけではなく，いろいろな考え方がある。実際，内閣府と日本銀行は独自の方法でGDPトレンドを算出している。
# * [内閣府のデータ](https://www5.cao.go.jp/keizai3/getsurei/getsurei-index.html)
#     * 「潜在GDP」がトレンドに対応しており，「GDPギャップ」はトレンドからの％乖離を表している。
# * [日本銀行のデータ](https://www.boj.or.jp/research/research_data/gap/index.htm/)
#     * 「潜在成長率」はトレンドの成長率に対応しており，「需給ギャップ」がトレンドからの％乖離を表している。
# 
# いずれにしろ，どのようにトレンドを決めたとしても，景気循環が人々の厚生に大きな影響を及ぼすことには変わりはない。不況は失業や様々な社会問題（例えば，犯罪や自殺）につながる。好況でも問題が無いわけではない。例えば，高いインフレが発生し資産価値（例えば，貨幣）が暴落し通常の生活に支障が出ることもある。またバブル景気が示すように「山」は次の「谷」の芽を育む期間となりえる。
# 
# 以下では，データを使い景気循環$Y_t^{\text{cycle}}$の特徴を調べるが，まず上の式[](eq:10-decompose)を対数化し次式に書き換える。
# 
# $$
# y_t^{\text{cycle}}=y_t-y_t^{\text{trend}}
# $$ (eq:10-decompose_log)
# 
# ここで小文字は大文字の変数を対数化した値である（例えば，$y_t\equiv\log(Y_t)$）。$y_t^{\text{cycle}}$をより直感的に解釈するために式[](eq:10-decompose_log)の右辺を次のように近似しよう。
# 
# $$
# y_t^{\text{cycle}}\approx\dfrac{Y_t-Y_t^{\text{trend}}}{Y_t^{\text{trend}}}
# $$ (eq:10-decompose_log_approx)
# 
# 即ち，$y_t^{\text{cycle}}$は変数$Y_t$のトレンドからの乖離をパーセンテージで表している。
# 
# ここまでの説明から，景気循環を捉える項$y_t^{\text{cycle}}$は変数の値とトレンドとの残差によって決まることがわかる。換言すると，景気循環の特徴はトレンドをどのように考えるかに依存しており，トレンドの算出方法をどうするか決めるかが重要なポイントとなる。ここでは，内閣府や日本銀行の複雑な手法ではなく，マクロ経済学研究でスタンダードなツールとなっている[Hodrick–Prescottフィルター](https://www.google.co.jp/search?q=Hodrick%E2%80%93Prescott%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC)と呼ばれる手法を使うことにする。詳細についての説明は控え，単に`py4macro`に含まれる関数`trend`を使ってトレンド抽出を行うことにする。使い方は次のコードで確認できる。

# In[15]:


help(py4macro.trend)


# 使い方は簡単で，トレンドを計算したい変数の`Series`もしくは１列の`DataFrame`を引数に設定し実行するとトレンドが返される。

# まず次の変数のトレンドを計算し対数化した後に`df`に追加する。

# In[16]:


var_list = ['gdp',
            'consumption',
            'investment',
            'government',
            'exports',
            'imports']


# In[17]:


for v in var_list:
    df[v+'_trend'] = py4macro.trend(df.loc[:,v])  # 1


# ```{admonition} コードの説明
# :class: dropdown
# 
# 1. 右辺の`py4macro.trend()`はトレンドを`Series`として返し，それを左辺では新たな列として追加している。ここで列ラベルを`v+'_trend'`としているのは元々の変数名の後に`_trend`を追加するためである。例えば，１回目のループの新たな列ラベルは`gdp_trend`となり，２回目のループでは`consumption_trend`となる。
# ```
# 
# `df`の属性`.columns`を使い，列ラベルを表示して確認してみよう。

# In[18]:


df.columns


# `_trend`が使いされた変数が６つあることが確認できる。

# ### GDP

# 対数化したGDPの変数を作成しトレンドと重ねて図示してみる。

# In[19]:


df[['gdp','gdp_trend']].plot()
pass


# トレンドは直線ではなくスムーズな曲線となっている。上下に動く変数を平滑化したものがトレンドなので直線になるとは限らないのである。
# 
# 次に景気循環の変数を作成するが，トレンドからの乖離を％で表すために次の関係を利用する。
# 
# $$
# \log\left(\frac{X}{Z}\right)
# =\log\left(1+\frac{X-Z}{Z}\right)
# \approx\frac{X-Z}{Z}
# $$ (eq:10-devaition)

# In[20]:


df['gdp_cycle'] = 100 * np.log( df['gdp']/df['gdp_trend'] )


# `0`に横線を引いてプロットしてみよう。

# In[21]:


ax_ = df['gdp_cycle'].plot(marker='.',    # 1
                           linestyle=':',
                           title='% Deviation from GDP Trend')
ax_.axhline(0, color='red')               # 2
pass


# ```{admonition} コード説明
# :class: dropdown
# 
# 1. `Pandas`のメソッド`plot`は図を表示すると共に図の「軸（axis）」を返すが，それを変数`ax`に割り当てている。
#     * `marker`はデータのマーカーを指定する引数であり，`'.'`は小さな点を指定している。
#     * `linestyle`は線のスタイルを指定する引数であり，`':'`は点線を指定している。
# 2. `ax_`のメソッド`.axhline()`を使うと横線を描くことができる。
#     * 縦軸の値`0`は必須の引数
#     * `color=red`は色を指定する引数（無くても良い）。
# ```

# 図の景気循環はトレンドからの乖離である。乖離の典型的な特徴として持続性がある。これは同じ方向の乖離が続く傾向があるという意味である。即ち，今期景気が良ければ（悪ければ）来期も景気が良い（悪い）可能性が高いということである。もちろん，いずれは乖離が狭まり「山」から「谷」，そして「山」へと循環して行く。持続性を示す数値として自己相関係数がある。
# 
# 説明するために２つの変数$Y$と$Z$を考え次の係数を定義しよう。
# 
# $$
# R(Y,Z)=
# \frac{
#     \text{YとZの標本共分散}
#     }{
#     \sqrt{\text{Yの標本分散}}
#     \sqrt{\text{Xの標本分散}}
#     }
# $$ (eq:10-YZ)
# 
# * 分子の標本共分散が符号を決定する。
# * 分母は$R(Y,Z)$の値が$-1\leq R(Y,Z)\leq 1$となるよう標準化している。
# 
# $t$期のGDPを$x_t$としよう。式[](eq:10-YZ)を使うと，$Y$と$Z$が次の場合に$x_t$の自己相関係数となる。
# 
# $$
# Y=x_t,\quad Z=x_{t-1}
# $$ (eq:10-autocorrelation)
# 
# 直感的に説明すると，自己相関とは今期の値が前期の値に依存する程度をいう。前期の影響が強ければ，自己相関係数の絶対値は大きくなる。
# 
# 式[](eq:10-autocorrelation)として式[](eq:10-YZ)を計算するが，`Series`にはそれを計算するメソッド`.autocorr()`が用意されている。

# In[22]:


df['gdp_cycle'].autocorr()


# 正の値であり，今期に高（低）ければ来期も高い（低い）可能性が非常に大きいことが分かる。第二の特徴として，「山」と「谷」の形，幅（期間），高さ（深さ）はまちまちであり，不確実な要素の働きが大きい。言い換えると，景気循環の予測は非常に難しい。この点は確率変数として捉えることができるが，このアプローチは後のトピックで扱う。

# ### GDPの構成要素

# #### トレンドからの乖離

# 次にGDPの構成要素について考えるが，まずトレンドからの乖離の変数を作成する（単位：％）。

# In[23]:


for v in var_list[1:]:   # gdp以外の変数    
    df[v+'_cycle'] = 100 * np.log( df[v]/df[v+'_trend'] )


# 図示する変数のリスを作成するために，列ラベルを確認しよう。

# In[24]:


df.columns


# 列ラベルに`_cycle`がついている変数だけを選ぶために`cycle_list`を作成する。

# In[25]:


cycle_list = df.columns[-6:]   # 最後の6変数

df.loc[:,cycle_list].plot(subplots=True, figsize=(8,10))
pass


# この図に基づいて３つの点について考える。
# 1. 持続性を示す自己相関
# 1. GDPと順循環的（pro-cyclical）か反循環的（counter- cyclical）かどうか
# 1. 変動の大きさの比較

# #### 自己相関

# `for`ループを使って自己相関係数を計算しよう。

# In[26]:


for v in cycle_list:
    autocorr = df[v].autocorr()           # (1)
    print(f'{v:<19}{autocorr:>5.2f}')     # (2)


# ```{admonition} コードの説明
# :class: dropdown
# 
# 1. `.autocorr()`は自己相関係数を計算するメソッド。
# 2. `f-string`を使っている。
#     * `<19`は`v`の文字列の長さを空白を足して`19`にし左詰めにする。
#     * `>5`は`corr`の文字列の長さを空白を足して`5`にし右詰めにする。
#     * `.2f`は小数点第二位までの表示を設定している。
#         * `>5`と`.2f`の順番を逆にするとエラーが発生する。
# ```

# 全て`0.45`以上であり，強い持続性が確認できる。

# #### GDPとの相関度

# GDPと他の変数との相関度を確認するために２変数の相関係数を考えよう。式[](eq:10-YZ)を使うと$Y$と$Z$は次のようになる。
# 
# $$
# Y=\text{GDP},\quad
# Z=\text{GDPの構成要素の変数}
# $$ (eq:10-correlation)
# 
# 式[](eq:10-correlation)として式[](eq:10-YZ)を計算するが，`Series`にはそれを計算するメソッド`.corr()`が用意されている。`for`ループを使って計算しよう。

# In[27]:


print('GDPの変動との相関係数\n------------------------')

for v in cycle_list:
    corr = df[['gdp_cycle', v]].corr().iloc[0,1]  # 1
    print(f'{v:<19}{corr:>5.2f}')                 # 2


# ```{admonition} コードの説明
# :class: dropdown
# 
# 1. `.corr()`は相関係数を計算するメソッドであり，2x2の行列を返す。
#     * `iloc[0,1]`は相関係数の値を抽出している。`iloc[1,0]`でも同じ結果となる。
# 2. `f-string`を使っている。
#     * `<19`は`v`の文字列の長さを空白を足して`19`にし左詰めにする。
#     * `>5`は`corr`の文字列の長さを空白を足して`5`にし右詰めにする。
#     * `.2f`は小数点第二位までの表示を設定している。
#         * `>5`と`.2f`の順番を逆にするとエラーが発生する。
# ```

# 政府支出以外は全て相関係数は正の値であり，値も大きい。即ち，順循環的である（裏にあるメカニズムを考えてみよう）。下の図はGDPと投資の散布図であり，正の相関を確認できる。

# In[28]:


df.plot(x='gdp_cycle', y='investment_cycle', kind='scatter')
pass


# 一方，政府支出の値は負であり，景気循環をコントロールしようとする政府の政策の現れと解釈できる。値が小さいのは，時間的なラグがあるためだと思われる。景気に関するデータを集計するには数ヶ月かかり，国会審議や支出の実行にも時間を取られることになる。時間的なラグを捉えるために，`Series`のメソッド`.shift()`を使って相関係数を再計算してみよう。

# In[29]:


print('  GDPとの相関係数\n-----------------------')
for n in range(1,12):
    df_temp = df     # 1
    df_temp['gov_cycle_shift'] = df_temp['government_cycle'].shift(-n)  # 2
    corr = df_temp[['gdp_cycle', 'gov_cycle_shift']].corr().iloc[0,1]
    print(f'{n:>3}期先の政府支出: {corr:>6.3f}')


# ```{admonition} コードの説明
# :class: dropdown
# 
# 1. `df`を変更しないように`df_temp`を新しく作成する。
# 2. `.shift()`の引数は値を先に何期ずらすかを指定する。比べたいのは$t$期のGDPと$t+n$期の政府支出なので，後にずらす必要があるためマイナス符号を付けて`.shift(-t)`としている。
# ```

# GDPの乖離は2四半期先の政府支出との相関係数が最大となっている。景気循環に対する政府の姿勢が確認できる一方，政府の対応には長いの時間が掛かることを示す結果となっている。

# #### 変動の大きさ

# 次に変動の大きさを考えるために，GDPの標準偏差に対するそれぞれの構成要素の標準偏差の比率を計算する。

# In[30]:


for v in cycle_list:
    var = df[v].std() / df['gdp_cycle'].std()
    print(f'{v:<19}{var:>5.2f}')


# 投資，輸出，輸入の値はGDPの3倍以上であり，政府支出の値はGDPより低く，消費の変動は更に小さい。これは消費者の不確実性を嫌う姿勢が反映されていると解釈できる。GDPの構成要素を対GDP比率で検討した際，消費は投資よりも比率が大きかったことを思い出そう。変動に関しては，順位が逆転し投資が大きく上回っている。この点を図示して視覚的に確認してみよう。

# In[31]:


df.loc[:,cycle_list[:3]].plot()
pass


# これらの結果を説明するために，効用最大化に基づく消費者理論があり，利潤最大化に基づく企業行動に関するモデルが使われている。
