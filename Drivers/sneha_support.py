# # -*- coding: utf-8 -*-


# import pandas as pd
# import gc

# from math import log2
# Sample = []
# result_col = 'result'  # input("enter name of result column")


# def entropy(class0, class1):
#     epsilon = 1e-10  # small value to avoid taking the logarithm of zero
#     class0 = max(epsilon, class0)
#     class1 = max(epsilon, class1)
#     return -(class0 * log2(class0) + class1 * log2(class1))


# # find score and best feature
# def find_score(x, best_score):
#     # print("feature checked =",x,best_score)
#     score = 0
#     for s in Sample:
#         # print(s)
#         p0 = 0
#         n0 = 0
#         p1 = 0
#         n1 = 0
#         for i in range(len(s[x])):
#             if (s[x][i] == 0 and s[len(s)-1][i] == 1):
#                 p0 = p0 + 1
#             elif (s[x][i] == 0 and s[len(s)-1][i] == 0):
#                 n0 = n0 + 1
#             elif (s[x][i] == 1 and s[len(s)-1][i] == 1):
#                 p1 = p1+1
#             elif (s[x][i] == 1 and s[len(s)-1][i] == 0):
#                 n1 = n1+1
#         t0 = p0+n0  # split via value x =0
#         t1 = p1+n1  # split via value x =1

#         total = p0+p1+n0+n1  # total observations
#         # split the dataset based on class result
#         class1 = (p0+p1)/total  # positive observations
#         class0 = (n0+n1)/total  # negative observations

#         # calculate entropy before the change
#         s_entropy = entropy(class0, class1)
#         # print(s_entropy)
#         # split 1 (split via value x =0)
#         if (p0 == 0 or n0 == 0):
#             s1_class1 = 0
#             s1_class0 = 0
#             s1_entropy = 0
#         else:
#             s1_class1 = p0/t0
#             s1_class0 = n0/t0
#             # print(s1_class1,s1_class0)
#             # calculate the entropy of the first group
#             s1_entropy = entropy(s1_class0, s1_class1)

#         # print('Group1 Entropy: %.3f bits' % s1_entropy)

#         # split 2 (split via value x =1)
#         if (p1 == 0 or n1 == 0):
#             s2_class1 = 0
#             s2_class0 = 0
#             s2_entropy = 0
#         else:
#             s2_class1 = p1/t1
#             s2_class0 = n1/t1
#             # calculate the entropy of the first group
#             s2_entropy = entropy(s2_class0, s2_class1)

#         # print('Group2 Entropy: %.3f bits' % s2_entropy)

#         # calculate the information gain
#         gain = s_entropy - (t0/total * s1_entropy + t1/total * s2_entropy)
#         # print('Information Gain: %.3f bits' % gain)

#         # print(t0, t1, total)
#         if (t0 != 0 and t1 != 0):
#             split = - ((t0/total) * log2(t0/total)) - \
#                 ((t1/total) * log2(t1/total))
#         elif (t0 == 0):
#             split = - ((t1/total) * log2(t1/total))
#         else:
#             split = - ((t0/total) * log2(t0/total))
#         if (split > 0):
#             gainratio = gain / split
#         else:
#             gainratio = 0

#         score = score + gainratio
#         # print("SCORE =",score," Gainratio ",gainratio)
#     if (score > best_score):
#         # print("socre = ",score,"  best = ",best_score)
#         global best_feature
#         best_feature = x
#         # global best_score
#         best_score = score
#         # print("BEST F ",best_feature)
#     # else:

#         # print("BEST F ",best_feature)
#     return (best_score)

# # find_score = jit(find_score)

# # parttion samplke based on best featue


# def partition(s, bestf, newSample):

#     df = pd.DataFrame(s)
#     df = df.transpose()
#     df = df.astype("int8")
#     # print(len(df))
#     df1 = df[df[bestf] >= 1]
#     df1 = df1.drop(columns=[bestf])
#     df1 = df1.astype("int8")
#     # print(len(df1))
#     df0 = df[df[bestf] <= 0]
#     df0 = df0.drop(columns=[bestf])
#     df0 = df0.astype("int8")
#     # print(len(df),len(df0),len(df1))
#     S0_list = []
#     S1_list = []
#     if (df1.empty == False):
#         # print(df1)
#         l1 = df1[len(df1.columns)].values.tolist()
#         # print("l1=",l1)
#         if (len(set(l1)) == 1):
#             ''
#             # print("All elements in list are same",bestf)
#         else:
#             # print("All elements in list are not same")
#             S1_list = df1.values.T.tolist()
#             # print("S1 =",S1_list)
#     if (df0.empty == False):
#         l2 = df0[len(df0.columns)].values.tolist()
#         # print("l2=",l2)
#         if (len(set(l2)) == 1):
#             ''
#             # print("All elements in list are same",bestf)
#         else:
#             # print("All elements in list are not same")

#             S0_list = df0.values.T.tolist()
#             # print("S0 =",S0_list)

#     if (len(S1_list) > 0):
#         # print("S1 = ",S1_list)
#         newSample.append(S1_list)
#     if (len(S0_list) > 0):
#         # print("S0 = ",S0_list)
#         newSample.append(S0_list)
#     # print("new sample = ",newSample)
#     del [[df, df1, df0]]
#     gc.collect()


# # df=pd.read_csv(r'F:\SNEHA PHD 15 OCT 2021\PhD work JAN 2022\Compare code\info gain ratio on kdd\bin-kdd125k-lev-interval.csv')
# # df=pd.read_csv(r'E:\OneDrive - iitr.ac.in\lad\Lad paper info gain 5 aug 2021\paper-examplebin.csv')
# # df=pd.read_csv(r'C:\Users\user\Dropbox\lad\final lad code\checked  result\KDDTRAIN_CSV_numerical2-bin-out-28july2020.csv')

# # print(df)
# def ss(df, path):
#     global best_feature
#     global Sample
#     df = df.astype("int8")
#     head_list = []
#     for col in df.columns:
#         head_list.append(col)
#     # print(head_list)

#     '''Sample_list=[]
#     for col in head_list:
#         Slist=df[col].tolist()
#         print(Slist)
#         Sample_list.append(Slist)
#     print(Sample_list)'''

#     Sample_list = df.values.T.tolist()

#     Sample.append(Sample_list)
#     # print(Sample)
#     rows = len(df.axes[0])  # count rows
#     # print(rows)
#     cols = len(head_list)
#     # print(cols)

#     # find_score = jit(find_score)
#     # partition =jit(partition)

#     h_list = head_list.copy()
#     h_list.pop()
#     # print(h_list)
#     features = []
#     best_feature = -1
#     ctr = 0
#     b_score = 1
#     while (len(Sample) > 0 and b_score != 0 and len(features) < 41):
#         ctr = ctr+1
#         # print("BEGIN")
#         best_score = 0
#         for x in range(len(h_list)):
#             best_score = find_score(x, best_score)
#             # print("best score ", best_score)
#         # if(best_score==0):
#         #     h_list.pop(best_feature)
#         #     continue
#         # print("BEST SCORE ",best_score)
#         print("BEST FEATURE ", best_feature, "BEST SCORE ", best_score)
#         newSample = []
#         for s in Sample:
#             partition(s, best_feature, newSample)
#         # print("main new sample = ",newSample)
#         Sample = newSample
#         f = h_list.pop(best_feature)
#         # print(f)
#         features.append(f)
#         b_score = best_score

#     print("FEATURES SELECTED =", features)
#     # print(df)
#     support_set = df[features]

#     print(support_set)
#     result_list = df[result_col].tolist()
#     support_set['result'] = result_list
#     df_supportset = support_set.copy()

#     print("total sub samples =", ctr)
#     df_supportset.to_csv(path, index=None)
#     return df_supportset


# # df = pd.read_csv(
# #     r'Dataset/bank_additional_string/bank-additional-full_bin.csv').head(1000)
# # path = "Dataset/bank_additional_string/bank-additional-full_ss_1000.csv"
# # ss(df, path)

# -*- coding: utf-8 -*-


import pandas as pd
import gc
import sys

from math import log2
# Sample = []
result_col = 'label'  # input("enter name of result column")


def entropy(class0, class1):
    epsilon = 1e-10  # small value to avoid taking the logarithm of zero
    class0 = max(epsilon, class0)
    class1 = max(epsilon, class1)
    return -(class0 * log2(class0) + class1 * log2(class1))


# find score and best feature
def find_score(x, best_score,best_feature,Sample):
    # print("feature checked =",x,best_score)
    score = 0
    for s in Sample:
        # print(s)
        p0 = 0
        n0 = 0
        p1 = 0
        n1 = 0
        # print(s[x])
        # print(len(s[x]))
        for i in range(len(s[x])):
            if (s[x][i] == 0 and s[len(s)-1][i] == 1):
                p0 = p0 + 1
            elif (s[x][i] == 0 and s[len(s)-1][i] == 0):
                n0 = n0 + 1
            elif (s[x][i] == 1 and s[len(s)-1][i] == 1):
                p1 = p1+1
            elif (s[x][i] == 1 and s[len(s)-1][i] == 0):
                n1 = n1+1
        t0 = p0+n0  # split via value x =0
        t1 = p1+n1  # split via value x =1

        total = p0+p1+n0+n1  # total observations
        # split the dataset based on class label
        class1 = (p0+p1)/total  # positive observations
        class0 = (n0+n1)/total  # negative observations

        # calculate entropy before the change
        s_entropy = entropy(class0, class1)
        # print(s_entropy)
        # split 1 (split via value x =0)
        if (p0 == 0 or n0 == 0):
            s1_class1 = 0
            s1_class0 = 0
            s1_entropy = 0
        else:
            s1_class1 = p0/t0
            s1_class0 = n0/t0
            # print(s1_class1,s1_class0)
            # calculate the entropy of the first group
            s1_entropy = entropy(s1_class0, s1_class1)

        # print('Group1 Entropy: %.3f bits' % s1_entropy)

        # split 2 (split via value x =1)
        if (p1 == 0 or n1 == 0):
            s2_class1 = 0
            s2_class0 = 0
            s2_entropy = 0
        else:
            s2_class1 = p1/t1
            s2_class0 = n1/t1
            # calculate the entropy of the first group
            s2_entropy = entropy(s2_class0, s2_class1)

        # print('Group2 Entropy: %.3f bits' % s2_entropy)

        # calculate the information gain
        gain = s_entropy - (t0/total * s1_entropy + t1/total * s2_entropy)
        # print('Information Gain: %.3f bits' % gain)

        # print(t0, t1, total)
        if (t0 != 0 and t1 != 0):
            split = - ((t0/total) * log2(t0/total)) - \
                ((t1/total) * log2(t1/total))
        elif (t0 == 0):
            split = - ((t1/total) * log2(t1/total))
        else:
            split = - ((t0/total) * log2(t0/total))
        if (split > 0):
            gainratio = gain / split
        else:
            gainratio = 0

        score = score + gainratio
        # print("SCORE =",score," Gainratio ",gainratio)
    if (score > best_score):
        # print("socre = ",score,"  best = ",best_score)
        # global best_feature
        best_feature = x
        # global best_score
        best_score = score
        # print("BEST F ",best_feature)
    # else:

        # print("BEST F ",best_feature)
    return (best_score, best_feature)

# find_score = jit(find_score)

# parttion samplke based on best featue


def partition(s, bestf, newSample):

    df = pd.DataFrame(s)
    df = df.transpose()
    df = df.astype("int8")
    # print(len(df))
    df1 = df[df[bestf] >= 1]
    df1 = df1.drop(columns=[bestf])
    df1 = df1.astype("int8")
    # print(len(df1))
    df0 = df[df[bestf] <= 0]
    df0 = df0.drop(columns=[bestf])
    df0 = df0.astype("int8")
    # print(len(df),len(df0),len(df1))
    S0_list = []
    S1_list = []
    if (df1.empty == False):
        # print(df1)
        l1 = df1[len(df1.columns)].values.tolist()
        # print("l1=",l1)
        if (len(set(l1)) == 1):
            ''
            # print("All elements in list are same",bestf)
        else:
            # print("All elements in list are not same")
            S1_list = df1.values.T.tolist()
            # print("S1 =",S1_list)
    if (df0.empty == False):
        l2 = df0[len(df0.columns)].values.tolist()
        # print("l2=",l2)
        if (len(set(l2)) == 1):
            ''
            # print("All elements in list are same",bestf)
        else:
            # print("All elements in list are not same")

            S0_list = df0.values.T.tolist()
            # print("S0 =",S0_list)

    if (len(S1_list) > 0):
        # print("S1 = ",S1_list)
        newSample.append(S1_list)
    if (len(S0_list) > 0):
        # print("S0 = ",S0_list)
        newSample.append(S0_list)
    # print("new sample = ",newSample)
    del [[df, df1, df0]]
    gc.collect()


# df=pd.read_csv(r'F:\SNEHA PHD 15 OCT 2021\PhD work JAN 2022\Compare code\info gain ratio on kdd\bin-kdd125k-lev-interval.csv')
# df=pd.read_csv(r'E:\OneDrive - iitr.ac.in\lad\Lad paper info gain 5 aug 2021\paper-examplebin.csv')
# df=pd.read_csv(r'C:\Users\user\Dropbox\lad\final lad code\checked  result\KDDTRAIN_CSV_numerical2-bin-out-28july2020.csv')

# print(df)
def ss(df, path):
    Sample = []
    
    df = df.astype("int8")
    head_list = []
    for col in df.columns:
        head_list.append(col)

    Sample_list = df.values.T.tolist()

    Sample.append(Sample_list)

    h_list = head_list.copy()
    h_list.pop()
    # print(h_list)
    features = []
    best_feature = -1
    ctr = 0
    b_score = 1
    # and len(features) < 41
    while (len(Sample) > 0 and b_score != 0 ):
        ctr = ctr+1
        best_score = 0
        for x in range(len(h_list)):
            best_score,best_feature = find_score(x, best_score,best_feature,Sample)
        
        print("BEST FEATURE ", best_feature, "BEST SCORE ", best_score)
        newSample = []
        for s in Sample:
            partition(s, best_feature, newSample)
        # print("main new sample = ",newSample)
        Sample = newSample
        f = h_list.pop(best_feature)
        # print(f)
        features.append(f)
        b_score = best_score

    print("FEATURES SELECTED =", features)
    # print(df)
    support_set = df[features]

    print(support_set)
    result_list = df[result_col].tolist()
    support_set['label'] = result_list
    df_supportset = support_set.copy()

    print("total sub samples =", ctr)
    df_supportset.to_csv(path, index=None)
    return df_supportset

    # y = "UNSW_NB15_training-set_ss.csv"  # input("enter path name")


df = pd.read_csv(
   f'../Dataset/{sys.argv[1]}/Binarized/{sys.argv[1]}_train.csv')
path = f"../Dataset/{sys.argv[1]}/Support_Set/{sys.argv[1]}_train.csv"
ss(df, path)
print("Support Set Generation completed.")