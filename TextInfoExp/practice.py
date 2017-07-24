import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

rounds = input("Your Round: ")


record = pd.DataFrame(columns = ('human','computer'),index=None)


def judge(your, computer):
    your = int(your)
    print(your,computer)
    if your == computer:
        print("draw")
    elif your - computer == 1 or your - computer == -2:
        print("win")
    else:
        print("lost")

def com(rounds):
    for i in range(int(rounds)):
        computer = 0
        your = 0
        your = input("Your Choose(1 for Rock, 2 for Paper, 3 for Scissors): ")
        computer = random.randint(1,3)
        record.loc[i] = [int(your), computer]
        if computer == 1:
            print("Computer: Rock")
            judge(your, computer)
        elif computer == 2:
            print("Computer: Paper")
            judge(your, computer)
        else:
            print("Computer: Scissors")
            judge(your, computer)

com(rounds)

rec = record
reg.fit(rec.values,rec.index.values)
coef = reg.coef_[0]

record.to_csv("record.csv", index=False)