from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import statsmodels.api as sm

import cleantools as clnt


class myRegressor:

    """This class is to create a logistic regression."""

    def __init__(
        self,
        column_list=[
            "Contract Date",
            "Contract ID",
            "Has BoContract",
            "Has Flee",
            "Has Single Salary",
            "Has Master Paying Salary",
            "Has Whole Period",
            "Has Ducati",
            "Master Paid Accom",
            "Master Paid Clothing",
            "Quondam1",
            "Master Paid Gen. Exp.",
            "Master Paid Per. Care",
            "Apprentice Age",
            "Apprentice Male",
            "Parent Label",
            "S",
            "T",
            "P",
            "Duration - Merged",
            "Paid Salary in Goods",
            "Total Payment Received",
            "From Venezia",
        ],
    ):
        """This it the constructor of the class """
        self.data = None
        self.x_data = None
        self.y_data = None
        self.x_train = None
        self.y_train = None
        self.target = None
        self.selected_features = None
        self.coef_table = None
        self.column_list = column_list
        self.regressor = LogisticRegression()

    def loadData(
        self,
        df: pd.DataFrame,
        target: str,
        seperateLabels=[
            "prod. e lav. di tessuti e filati",
            "commercio all'ingrosso e al minuto",
            "prod. e lav. di generi alimentari",
            "fabbr. e lav. di oggetti in vetro",
            "prod. di calzature",
            "lav. di metalli comuni",
        ],
        has_date="Contract Date",
        outlier_list=[
            "Apprentice Age",
            "Duration - Merged",
            "Total Payment Received",
        ],
    ):
        """Takes in a dataframe, loads it to itself

        """
        venezia_encoding = clnt.hotEncode(
            df, df["Apprentice Province"], operation="max"
        )[["Contract ID", "Venezia"]]
        venezia_encoding.columns = ["Contract ID", "From Venezia"]
        merged = df.merge(venezia_encoding, on="Contract ID",
                          how="left").copy()
        del merged["Apprentice Province"]
        # Take the necessary part only
        merged = merged[self.column_list]
        merged = merged.set_index("Contract ID")
        # Lose na cells in data
        print("Before drop na columns", len(merged))
        merged = merged.dropna()
        if has_date:
            merged = clnt.normalizeTime(merged, has_date)
        if seperateLabels:
            merged = self.sepLabels(merged, seperateLabels)
        else:
            del merged["Parent Label"]

        self.target = target
        self.loadDataHelper(merged.copy(), outlier_list=outlier_list)

        print("Total Size", len(self.data))
        print("Training Size", len(self.x_train))
        print("Number of 1s in target", np.sum(self.y_train))
        print(
            "Number of 0s in target", len(self.x_train) - np.sum(self.y_train)
        )

    def loadDataHelper(self, data, outlier_list=[]):
        """Takes in a data set, divides it to x, y, train, test
        :returns: TODO

        """
        self.data = data
        print("Before outlier clean", len(data))
        if outlier_list:
            self.cleanOutliers(loC=outlier_list)
        # Get the X data
        x_data = self.data.copy()
        del x_data[self.target]
        self.x_data = pd.DataFrame(
            scale(x_data), columns=x_data.keys(), index=x_data.index.tolist()
        )

        # Get the Y data
        self.y_data = self.data[self.target].copy()

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.x_data, self.y_data, test_size=0.25, random_state=42
        )

    def sepLabels(self, df, dummy_list):
        """Separates the Parent Column to different job columns of encoding

        """
        dummy_jobs = df["Parent Label"].str.get_dummies()[dummy_list]
        df = df.reset_index().merge(
            dummy_jobs.reset_index(), on="Contract ID", how="left"
        )
        df["Other Label"] = 1 - np.max(df[dummy_list], axis=1,)
        del df["Parent Label"]
        df = df.set_index("Contract ID")
        return df

    def cleanOutliers(
        self, loC, lower=0.02, higher=0.97,
    ):
        """Takes in a list of outliers, and lower and upper quantiles, then
        eliminates those quantiles from the data

        """
        df = self.data
        for a in loC:
            print(df[a].quantile([lower, higher]))
            df = df[
                (
                    (df[a] == df[a].quantile(higher))
                    | (df[a] < df[a].quantile(higher))
                )
                & (
                    (df[a] == df[a].quantile(lower))
                    | (df[a] > df[a].quantile(lower))
                )
            ]
        self.data = df.copy()

    def getReport(self, getGraph=True, getMarginals=True):
        """Gets a regression model report

        """
        logit_model = sm.Logit(self.y_data, self.x_data)
        result = logit_model.fit(method="bfgs")
        print(result.summary2())
        if getMarginals:
            print(result.get_margeff().summary())
        if getGraph:
            coef = pd.DataFrame(result.params, columns=["Coef"]).reset_index()
            error = result.conf_int().reset_index()
            error.columns = ["Index", "L", "H"]
            error["Final"] = (error["H"] - error["L"]) / 2
            coef = coef.merge(
                error[["Index", "Final"]], left_on="index", right_on="Index"
            )[["Index", "Coef", "Final"]]
            coef.columns = ["Feature", "Coefficient", "Error"]
            self.coef_table = coef.copy()
            _, ax = plt.subplots(figsize=(14, 10))
            ax.plot(0)
            ax.axvline(x=0, color="black")
            ax.set_title(
                "Coefficients with Confidence Intervals\nSmaller X range",
                fontsize=15,
            )
            coef.plot.scatter(
                x="Coefficient",
                y="Feature",
                ax=ax,
                xerr="Error",
                grid=True,
                xlim=(-0.35, 0.35),
            )
            _, ax2 = plt.subplots(figsize=(14, 10))
            ax2.plot(0)
            ax2.axvline(x=0, color="black")
            ax2.set_title(
                "Coefficients with Confidence Intervals\nGreater X range",
                fontsize=15,
            )
            coef.plot.scatter(
                x="Coefficient", y="Feature", ax=ax2, xerr="Error", grid=True
            )
            plt.show()

    def applyLogistic(self):
        """Apply logistic regression to the data set we created
        :returns: TODO

        """
        logisticReg = self.regressor
        logisticReg.fit(self.x_train, self.y_train)
        predictions = logisticReg.predict(self.x_test)
        score = logisticReg.score(self.x_test, self.y_test)
        print(metrics.classification_report(self.y_test, predictions))

        # Confusion matriself.x
        self.confusionMatrix(predictions, score)

    def confusionMatrix(self, y_predictions, score):
        cm = metrics.confusion_matrix(self.y_test, y_predictions)
        plt.figure(figsize=(9, 9))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            square=True,
            cmap="Blues_r",
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        all_sample_title = "Accuracy Score: {0}".format(score)
        plt.title(all_sample_title, size=15)
        plt.show()

    def overSample(self):
        """Oversamples the test data set
        :returns: TODO

        """
        # Use sampling to close the gap between classes
        os = SMOTE(random_state=0)
        x_train, y_train = os.fit_sample(self.x_train, self.y_train)

        # we can Check the numbers of our data
        print("Oversample Size", len(x_train))
        print("Number of 1s in target", np.sum(y_train))
        print("Number of 0s in target", len(x_train) - np.sum(y_train))

        self.x_train = x_train
        self.y_train = y_train

    def assessFeatures(self):
        # Use feature selection to assess which columns matter more
        rfe = RFE(self.regressor)
        rfe = rfe.fit(self.x_train, self.y_train)
        useful_features = list()
        for b, f in zip(rfe.support_, self.x_train.keys()):
            if b:
                useful_features.append(f)
        print(useful_features)
        self.selected_features = useful_features

    def selectFeatures(self):
        """Eliminates the features selected, and re applies the algorithm
        :returns: TODO

        """
        if not self.selected_features:
            self.assessFeatures()

        self.loadDataHelper(self.data[self.selected_features + [self.target]])
        self.applyLogistic()
