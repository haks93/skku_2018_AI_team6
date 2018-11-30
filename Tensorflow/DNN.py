import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

for dateset in combine:
    dateset['Title'] = dateset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df["Title"], train_df["Sex"])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({"female": 1, "male": 0}).astype(int)

guess_ages = np.zeros((2, 3))
guess_ages

# sex, Pclass를 이용해서 비어있는 age를 채우기
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset["Pclass"] == j + 1)]["Age"].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age == 가장 가까운 0.5 단위의 수로 바꾸는 것
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]

    dataset["Age"] = dataset["Age"].astype(int)

# Survived변수와 상관관계를 파악하여 AgeBand변수를 생성
train_df["AgeBand"] = pd.cut(train_df["Age"], 5)
train_df[["AgeBand", "Survived"]].groupby(["AgeBand"], as_index=False).mean().sort_values(by="AgeBand", ascending=True)

# 앞서 생성한 AgeBand를 이용해서 Age를 분류
for dataset in combine:
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"]

train_df = train_df.drop(["AgeBand"], axis=1)
combine = [train_df, test_df]

# familySize 변수 생성
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

train_df[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by="Survived",
                                                                                                ascending=False)
# IsAlone 변수 생성
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

train_df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()

# Parch, SibSp, FamilySize를 IsAlone 변수로 모두 합치기
# 그르니까, 혼자면 1, 아니면 0
train_df = train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
test_df = test_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
combine = [train_df, test_df]

# Embarkation변수의 비어 있는 부분을 가장 많이 나왔던 것으로 채워넣기
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)

train_df[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean().sort_values(by="Survived",
                                                                                            ascending=False)
# Embarkation의 카테고리컬 변수를 숫자로 변환하기
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

# Fare변수에서 가장 자주 나오는 값으로 빈칸을 채워넣기
test_df["Fare"].fillna(test_df["Fare"].dropna().median(), inplace=True)

train_df["FareBand"] = pd.qcut(train_df["Fare"], 4)
train_df[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean().sort_values(by="FareBand",
                                                                                            ascending=True)
# 앞서 나눠둔 Fareband를 이용해서 Fare을 4가지 카테고리로 분류
for dataset in combine:
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31.0), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

train_df = train_df.drop(["FareBand"], axis=1)
combine = [train_df, test_df]

xtrain = train_df.drop("Survived", axis=1).copy()
ytrain = train_df.iloc[:, 0]
xtest = test_df.drop("PassengerId", axis=1).copy()
ytest = pd.read_csv('../input/gender_submission.csv')
ytest = ytest.iloc[:, 1]

n = xtrain.shape[1]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
# model.add(tf.keras.layers.Dense(n, input_shape=(n,), activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

hist = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=500)

test = model.predict(xtest)
plt.figure(figsize=(12, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()

test = (test >= 0.5)
test = pd.Series(test[:, 0])
test = test.map({True: 1, False: 0})
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test})
submission.to_csv('submission.csv', index=False)

