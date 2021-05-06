---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Pythonで学ぶ中級マクロ経済学

```{epigraph}
[春山 鉄源](https://t-haruyama.github.io)

神戸大学経済学研究科
```

```{code-cell} python3
import datetime
dt = datetime.datetime.now()
print('Version:',dt.strftime('%Y年%m月%d日'))
```

<!---
%H:%M:%S
dt = datetime.datetime.now()
dt = datetime.datetime.today()
-->

本サイトに関するコメント等はGitHubの[Discussions](https://translate.google.com/translate?hl=&sl=ja&tl=en&u=https%3A%2F%2Fpy4macro.github.io%2F)もしくは<haruyama@econ.kobe-u.ac.jp>にご連絡ください。

---

If you come here without expecting Japanese, please click [Google translated version](https://translate.google.co.jp/translate?hl=ja&sl=ja&tl=en&u=https%3A%2F%2Fpy4macro.github.io&sandbox=1) in English or the language of your choice. Note that my name is Tetsu HARUYAMA, not  "Haruyama Iron Source" as Google claims. The title of this site may be more appropriately translated as "Learning Intermediate Macroeconomics with Python."

---

<br>

姉妹サイト：[「Pythonで学ぶ入門計量経済学」](https://py4etrics.github.io) <font size="+0">🐍</font>


## はじめに

本サイトの目的は２つある。第一に，中級レベルのマクロ経済学をとおして`Python`を学び，`Python`をとおして中級レベルのマクロ経済学を学ぶ（復習する）ことである。大学でのマクロ経済学教育は主に講義でおこなわれる。モデルの展開と解説，数値計算の結果やデータが紹介されるが，私もそうだったように，「そういうものなんだ」と納得はするが，なんとなく「距離感」を感じる学生が多いのではないだろうか。その距離感を縮めるために，演習や数値例を使った宿題があり，その役目はある程度果たしている。その距離感を更に縮めようというのが，本サイトの目的である。`Python`を使うことにより，異なる政策がもたらす均衡への影響を数値化し，簡単に計算することができる。また複雑なマクロモデルの均衡の動学的安定性を視覚的に確認し，政策などのパラメータにどのように反応するかも一瞬で確認することができる。即ち，マクロ経済学のハンズオン（[hands-on](https://eow.alc.co.jp/search?q=hands-on)）が可能となる。ハンズオンこそが今の授業で足りないものであり，学生のマクロ経済学の理解と興味を深め「距離感」を更に縮めることが期待できるのではないだろうか。

第二に，経済学部の学生にプログラミングの授業をとおして，今後変わりゆく社会に少しでも対応できる[transferable skill](https://www.google.co.jp/search?q=transferable+skills&spell=1&sa=X&ved=2ahUKEwj68fqc7LPwAhWKfd4KHT_xC64QBSgAegQIARA1&biw=1440&bih=767)を身につける機会を提供することである。新聞，雑誌やインターネット上で「AI」や「機械学習」などプログラミングに関するキーワードを頻繁に見聞きする。これは一過性の流行りではなく，社会全体がデジタル化する大きなうねりの「大音」である。実際，政府もプログラミングの重要性を強く認識している。2020年度からは小学校でプログラミング的思考を育成する学習指導要領が実施され，続いて中高教育でもプログラミングに関する内容・科目が充実される予定である。このようにプログラミングのスキルの重要性は益々大きくなると思われる。一方，今の経済学部の学生は，デジタル化による社会のうねりとプログラミング教育の盛り上がりの狭間にあり，プログラミングの「プの字」も知らずにデジタル化社会へ飛び込むことになりかねない。学生にとって卒業後の社会は「人生の本番」であり，その準備を少しでも手助けするのが教育の役割ではないだろうか。もちろん，近年一般教養科目としてプログラミング科目が導入されている大学も多くある。しかし，専門科目として提供することにより専門性とプログラミングの「いいとこ取り」を提供できる機会を利用しないのは，経済学でいう「非効率的」な教育になってしまう。

では，なぜ`Python`なのか？プログラミング言語は無数に存在し，それぞれ様々な特徴があり，お互いに影響し合い進化している。その過程で，広く使われ出す言語もあれば廃れていく言語もある。その中で`Python`は，近年注目を集める言語となっている。それを示す相対的な人気指標として[Stack Overflow Trends]( https://insights.stackoverflow.com/trends?tags=java%2Cc%2Cc%2B%2B%2Cpython%2Cc%23%2Cvb.net%2Cjavascript%2Cassembly%2Cphp%2Cperl%2Cruby%2Cvb%2Cswift%2Cr%2Cobjective-c)がある。
```{figure} /images/many_languages.png
:scale: 35%
```
[Stack Overflow](https://stackoverflow.com)とは（[日本語版はこちら](https://ja.stackoverflow.com)），プログラミングに関する質問をすると参加者が回答するフォーラムであり，質の高い回答で定評があるサイトである。その英語版で，ある言語に関する質問が何％を占めるのかを示しているのが上の図である。2012年頃から`Python`は急上昇しているが（右上がりの赤色の線），過去５年間でみると下降トレンドの言語が多い印象である。

では，`Python`の人気はどこにあるのか？まず最初の理由は無料ということである。経済学研究でよく使われる数十万円するソフトと比べると，その人気の理由は理解できる。しかし計量経済学で広く使われる`R`を含めて他の多くの言語も無料であり，それだけが理由ではない。人気の第２の理由は，汎用性である。`Python`はデータ分析や科学的数値計算だけではなく，ゲーム（ゲーム理論ではない），画像処理や顔認識にも使われている。またPCやスマートフォンの様々な作業の自動化に使うことも可能なのである。第３の理由は，学習コストが比較的に低いことである。`Python`のコードは英語を読む・書く感覚と近いため，他の言語と比較して可読性の高さが大きな特徴である（日本語にも近い点もある）。もちろん，`Python`の文法や基本的な関数を覚える必要があるが，相対的に最も初心者に易しい言語と言われる程である。他にも理由はあるが，`Python`はIT産業だけではなく金融・コンサルティング・保険・医療などの幅広い分野で使われており，データ分析の重要性が増すごとにより多くの産業で使われると思われる。経済学部の大多数の卒業生は幅広い産業で働くことになることを考えると，社会全体で注目され，今後より多くの産業で使われることが予測される言語を学ぶことは有意義ではないだろうか。

本サイトは３部構成となっている。第１部では，経済学の例を交えて`Python`の基礎について解説する。第２部では，実際に`Python`コードを書きマクロ経済分析をおこなう。`Python`の特性を生かし動学的な景気循環モデルも解説している。IS-MP-PCモデルは標準的な学部レベルの教科書で扱われているが，実物的景気循環モデルやニューケインジアン・モデルは「中級レベル」から少し逸脱するかもしれないが，できる限り学部生を意識して解説している。従って，大学院生には物足りない内容となっている。第３部では，参考になるトピックを集める計画である。

## 本サイトで使うPythonとパッケージのバージョン
```{code-cell} python3
import gapminder, linearmodels, lmdiag, matplotlib, numba, numpy, pandas, py4etrics, scipy, see, statsmodels, wooldridge
from platform import python_version

packages = ['Python','gapminder', 'matplotlib', 'numba', 'numpy','pandas', 'scipy','see', 'statsmodels', 'wooldridge']
versions = [python_version(),gapminder.__version__, matplotlib.__version__, numba.__version__, numpy.__version__, pandas.__version__,  scipy.__version__, see.__version__, statsmodels.__version__, wooldridge.__version__]

for pack, ver in zip(packages, versions):
    print('{0:14}{1}'.format(pack,ver))
```

---

[Economists（経済学を研究もしくは勉強(?)する人）と付き合わない方が良い２１＋$\alpha$の理由]( http://inesad.edu.bo/developmentroast/2012/10/21-reasons-why-you-should-never-date-an-economist/)

---

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Haruyama-KobeU/for_binder/main?filepath=for_binder.ipynb) for an interactive Jupyter Notebook session with empty code cells.






































