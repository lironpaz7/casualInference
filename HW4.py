import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, desc=None):
        if desc is not None:
            print(desc)

    def load_data(self, path1, path2):
        """
        Loads the given paths into a pandas dataframe
        :param path1: path for first dataset
        :param path2: path for second dataset
        :return: dataframe for each given path
        """
        return pd.read_csv(path1, index_col=0), pd.read_csv(path2, index_col=0)

    def preprocess(self, data):
        """
        Preprocessing the data using scaler and columns reduction
        :param data: dataset
        :return:
        """
        categories = ['x_2', 'x_21', 'x_24']
        data = data.join(pd.get_dummies(data[categories]))
        data = data[data.columns.difference(categories)]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data.drop(['T', 'Y'], axis=1))
        T, y = data['T'], data['Y']
        X_treated = X[data['T'] == 1]
        y_treated = y[data['T'] == 1]
        X_control = X[data['T'] == 0]
        y_control = y[data['T'] == 0]
        return X, T, y, X_treated, y_treated, X_control, y_control

    def show_stats(self, model, X, T):
        """
        Showing model accuracy, brier and plots Roc curve
        :param model: given model
        :param X: test data
        :param T: true data
        """
        preds = model.predict(X)
        print('Accuracy', model.score(X, T))
        print('Brier', np.mean((preds - T) ** 2))
        fpr, tpr, _ = roc_curve(T, preds)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr)})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.legend()
        plt.show()

    def ipw(self, data, statistics=False):
        """
        Calculates the inverse propensity score weighting (IPW)
        :param data: dataset
        :param statistics: if true then model accuracy and graphs are presented
        :return: att and propensity scores
        """
        X, T, y, X_treated, y_treated, X_control, y_control = self.preprocess(data)
        model = RandomForestClassifier(max_depth=20, n_estimators=500)
        model.fit(X, T)
        if statistics:
            self.show_stats(model, X, T)
        control_e = model.predict_proba(X_control)[:, 0]
        e_ratio = control_e / (1 - control_e)
        return np.mean(y_treated) - np.sum(y_control * e_ratio) / np.sum(e_ratio), model.predict_proba(X)[:, -1]

    def s_learner(self, data):
        """
        Perform S-Learner on the given dataset
        :param data: dataset
        :return: the mean distance of X_treated in the linear regression model
        """
        X, T, y, X_treated, _, _, _ = self.preprocess(data)
        model = LinearRegression()
        model.fit(np.c_[X, T], y)
        return np.mean(model.predict(np.c_[X_treated, np.ones(X_treated.shape[0])]) -
                       model.predict(np.c_[X_treated, np.zeros(X_treated.shape[0])]))

    def t_learner(self, data):
        """
        Perform T-Learner on the given dataset
        :param data: dataset
        :return: the mean distance between X_treated of model 1 and X_treated on model 0
        """
        _, _, _, X_treated, y_treated, X_control, y_control = self.preprocess(data)
        model1 = LinearRegression()
        model0 = LinearRegression()
        model1.fit(X_treated, y_treated)
        model0.fit(X_control, y_control)
        return np.mean(model1.predict(X_treated) - model0.predict(X_treated))

    def matching(self, data):
        """
        Calculates the mean distance between y_treated and the predicted treating
        :param data: dataset
        :return: the mean distance between y_treated and predicted treating
        """
        _, _, _, X_treated, y_treated, X_control, y_control = self.preprocess(data)
        knn = KNeighborsRegressor(1)
        knn.fit(X_control, y_control)
        return np.mean(y_treated - knn.predict(X_treated))

    def estimate_att(self, data, statistics=False):
        """
        Estimating the att and propensity scores for the given data set
        :param data: dataset
        :param statistics: if true then model accuracy and graphs are presented
        :return: att score, propensity score
        """
        att_ipw, propensity_scores = self.ipw(data, statistics)
        att = [att_ipw, self.s_learner(data), self.t_learner(data), self.matching(data)]
        return att + [np.mean(att)], propensity_scores


def show_results(att1, att2):
    print('Showing results:')
    print('\t\tData1:')
    print(f'ipw: {att1[0]}\n'
          f's_learner: {att1[1]}\n'
          f't_learner: {att1[2]}\n'
          f'matching: {att1[3]}\n'
          f'competitive: {att1[4]}')
    print('-' * 30)
    print('\t\tData2:')
    print(f'ipw: {att2[0]}\n'
          f's_learner: {att2[1]}\n'
          f't_learner: {att2[2]}\n'
          f'matching: {att2[3]}\n'
          f'competitive: {att2[4]}')
    print('-' * 30)


if __name__ == '__main__':
    predictor = Predictor('Casual Inference - HW4')
    print('Loading data...')
    data1, data2 = predictor.load_data('data1.csv', 'data2.csv')

    print('Calculating att & propensity...')
    att1, propensity1 = predictor.estimate_att(data1, statistics=False)
    att2, propensity2 = predictor.estimate_att(data2, statistics=False)
    att = pd.DataFrame(data=[np.arange(5) + 1, att1, att2], index=['Type', 'data1', 'data2']).transpose()
    propensity = pd.DataFrame(data=[propensity1, propensity2], index=['data1', 'data2'])
    att['Type'] = att['Type'].astype('int')

    show_results(att1, att2)

    print('Exporting to csv...')
    att.to_csv('ATT_results.csv', index=False)
    propensity.to_csv('models_propensity.csv', header=False)
