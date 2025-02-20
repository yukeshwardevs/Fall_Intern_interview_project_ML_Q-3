import matplotlib.pyplot as plt
import pandas as pd

clm = ['frame_number', 'gender', 'emotion', 'year', 'movie_name']
tdat = pd.read_csv(r"C:\Users\yukes\PycharmProjects\FIintern_project\Data\Bollywood-Data-master\trailer-data\complete-data.csv", sep = ',')

tdat.head()

tdat_woman = tdat.loc[tdat['gender'] == 'woman']
tdat_man = tdat.loc[tdat['gender'] == 'man']
tdat_woman = tdat_woman.loc[tdat['emotion'] != 'neutral']
tdat_man = tdat_man.loc[tdat['emotion'] != 'neutral']

res_woman = tdat_woman.groupby(by=['emotion'])['frame_number'].count()
res_woman=res_woman.to_frame(name = 'count').reset_index()
labels=res_woman['emotion'].tolist()

res_man = tdat_man.groupby(by=['emotion'])['frame_number'].count()
res_man=res_man.to_frame(name = 'count').reset_index()
labels=res_man['emotion'].tolist()

colors = ['#BEF480', '#80F4F1', '#B780F4', '#F48084', '#F10EBB', '#F1B60E', '#0E49F1']

plt.pie(res_woman['count'],labels=res_woman['emotion'], shadow=False, colors=colors,
    autopct='%1.1f%%')

plt.axis('equal')
plt.tight_layout()
plt.savefig('pie_chart_emotions_woman.png')
