import pandas as pd
from collections import defaultdict
import glob
import numpy as np
records = defaultdict(list)
files = glob.glob("CSV/*.CSV")
files.sort()
j = 1
columns = []
flag = 1
for i in range(0,len(files), 8):
    req_files = files[i:i+8]
    for p in range(0,len(req_files)):
        df_temp = pd.read_csv(req_files[p])
        for k in range(0,7):
            if k >= 1:
                df_temp.iloc[:,k] = np.exp(df_temp.iloc[:,k])
                records[j].append(df_temp.iloc[:, k].mean())
                if flag:
                    columns.append("F{}-Mean-C{}".format(p, k))

                records[j].append(df_temp.iloc[:, k].std())
                if flag:
                    columns.append("F{}-STD-C{}".format(p, k))

                df_temp.iloc[:,k] = (df_temp.iloc[:,k] - df_temp.iloc[:,k].mean()) / df_temp.iloc[:,k].std()
                counts, bins = np.histogram(df_temp.iloc[:, k], bins=30)
                density = counts / sum(counts)
                density = list(density)
                bins = list(bins)
                records[j].append(max(density))
                if flag:
                    columns.append("F{}-Density1-C{}".format(p, k))
                records[j].append(bins[np.argmax(density)])
                if flag:
                    columns.append("F{}-XCoord1-C{}".format(p, k))
            else:
                records[j].append(df_temp.iloc[:,k].mean())
                if flag:
                    columns.append("F{}-Mean-C{}".format(p,k))

                records[j].append(df_temp.iloc[:,k].std())
                if flag:
                    columns.append("F{}-STD-C{}".format(p,k))

                df_temp.iloc[:, k] = (df_temp.iloc[:, k] - df_temp.iloc[:, k].mean()) / df_temp.iloc[:, k].std()
                counts, bins = np.histogram(df_temp.iloc[:,k], bins=30)
                density = counts / sum(counts)
                density = list(density)
                bins = list(bins)
                records[j].append(max(density))
                if flag:
                    columns.append("F{}-Density1-C{}".format(p, k))
                records[j].append(bins[np.argmax(density)])
                if flag:
                    columns.append("F{}-XCoord1-C{}".format(p, k))

    flag = 0
    j = j + 1

ans = []
for i in range(1,360):
    ans.append(records[i])

final_df = pd.DataFrame(ans, columns=columns)
labels = pd.read_csv("AMLTraining.csv")
l = []
for i in range(0,labels.shape[0], 8):
    if labels.iloc[i,-1] == 'normal':
        l.append(0)
    elif labels.iloc[i,-1] == 'aml':
        l.append(1)
    else:
        l.append(-1)

final_df['labels'] = l
final_df.to_csv("AML_Classification_Data.csv")