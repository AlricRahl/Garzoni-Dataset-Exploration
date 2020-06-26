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
from statsmodels.formula.api import logit

import cleantools as clnt


def dummyLabels(
    df,
    dummy_list=[
        "prod. e lav. di tessuti e filati",
        "commercio all'ingrosso e al minuto",
        "prod. e lav. di generi alimentari",
        "fabbr. e lav. di oggetti in vetro",
        "prod. di calzature",
        "lav. di metalli comuni",
    ],
    other=False,
):
    dummy_jobs = df["Parent Label"].str.get_dummies()[dummy_list]
    df = df.join(dummy_jobs, how="left")
    if other:
        df["Other Label"] = 1 - np.max(df[dummy_list], axis=1)
    return df


def labelConvert(x, label_columns):
    """Convert labels to other if not in label_columns"""
    if x in label_columns:
        return x
    else:
        return "Other_Label"


class myRegressor:

    """This class is to create a logistic regression."""

    def __init__(
        self,
        column_list=[
            "Contract Date",
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
        """This is the constructor of the class """
        self.x_data = None
        self.y_data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.target = None
        self.coef_table = None
        self.formula = None
        self.column_list = column_list
        self.regressor = LogisticRegression()

    def loadData(
        self,
        df: pd.DataFrame,
        target: str,
        has_date="Contract Date",
        outlier_list=[
            "Apprentice Age",
            "Duration - Merged",
            "Total Payment Received",
        ],
    ):
        """Takes in a data frame, a target column, and optionally a list of
        jobs to encode from the parent label column, a date column name, and an
        outlier columns list. Trims the data to features in self.column_list.
        Cleans the na data. If given the optional arguments:
            - Normalizes time column
            - Separates Parent Label to given job labels as dummy columns
            - Cleans numeric outliers from given columns
        Then saves these data to the object attributes for easy access

        """
        # Take the necessary part only
        df = df[self.column_list]

        # Lose na cells in data
        print("Before drop na columns", len(df))
        df = df.dropna()

        # Normalize time data
        if has_date:
            clnt.normalizeTime(df, has_date)

        # Load to necessary objects
        self.target = target
        self.loadDataHelper(df.copy(), outlier_list=outlier_list)

        # Report numbers
        print("Total Size", len(self.x_data))
        print("Training Size", len(self.x_train))
        print("Number of 1s in target", np.sum(self.y_train))
        print(
            "Number of 0s in target", len(self.x_train) - np.sum(self.y_train)
        )

    def loadDataHelper(self, data, outlier_list=[]):
        """Takes in a data set, divides it to x, y, train, test
        :returns: TODO

        """
        # Clean outliers if wanted
        print("Before outlier clean", len(data))
        if outlier_list:
            data = self.cleanOutliers(data, loC=outlier_list)

        # Extract and reformat the X data
        x_data = data.copy()
        del x_data[self.target]
        self.x_data = pd.DataFrame(
            scale(x_data), columns=x_data.keys(), index=x_data.index.tolist()
        )

        # Get the Y data
        self.y_data = data[self.target].copy()
        # Load the division of data
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.x_data, self.y_data, test_size=0.25, random_state=42
        )

    def cleanOutliers(
        self, df, loC, lower=0.02, higher=0.97,
    ):
        """Takes in a list of outliers, and lower and upper quantiles, then
        eliminates those quantiles from the data

        """
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
        return df.copy()

    def getReport(self, getGraph=True, getMarginals=True):
        """Gets a regression model report

        """
        # Load to the Logit model, and assess results
        if not self.formula:
            logit_model = sm.Logit(self.y_data, self.x_data)
            result = logit_model.fit(method="bfgs")
        else:
            logit_model = logit(
                self.formula, data=self.x_data.join(self.y_data)
            )
            result = logit_model.fit()

        # Get summary
        print(result.summary2())

        # Get marginal differences summary
        if getMarginals:
            print(result.get_margeff().summary())

        # Get coefficients as a graph
        if getGraph:
            # Load coefficients and format columns
            coef = pd.DataFrame(result.params, columns=["Coef"])
            error = result.conf_int()
            error.columns = ["L", "H"]

            # Acquire an approximate size of error
            error["Final"] = (error["H"] - error["L"]) / 2
            coef = coef.join(error[["Final"]]).reset_index()
            coef.columns = ["Feature", "Coefficient", "Error"]

            # Save the table
            self.coef_table = coef.copy()

            # Plot two graphs for reference
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
        # Do the sklearn regression
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

        # Save values to attributes
        self.x_train = x_train
        self.y_train = y_train

    def assessFeatures(self):
        # Use feature selection to assess which columns matter more
        rfe = RFE(self.regressor)
        rfe = rfe.fit(self.x_train, self.y_train)
        useful_features = list()

        # Save the features accordingly, b for if selected, f for feature
        for b, f in zip(rfe.support_, self.x_train.keys()):
            if b:
                useful_features.append(f)
        print(useful_features)
        return useful_features

    def selectFeatures(self):
        """Eliminates the features selected, and re applies the algorithm
        :returns: TODO

        """
        selecteds = self.assessFeatures()

        self.loadDataHelper(self.x_data[selecteds].join(self.y_data).copy())
        self.applyLogistic()
