#!/usr/bin/env python3

import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import sklearn
import xgboost
import xgboost.core
from carla import MLModel
from carla.data.catalog.csv_catalog import CsvCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.catalog.parse_xgboost import parse_booster
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import check_counterfactuals
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


def _L1_cost_func(a, b):
    """The 1-norm ||a-b||_1"""
    return np.linalg.norm(a - b, ord=1)


def _L2_cost_func(a, b):
    """The 2-norm ||a-b||_2"""
    return np.linalg.norm(a - b, ord=2)


def search_path(tree, class_labels):
    def parse_tree(tree):
        if isinstance(tree, sklearn.tree.DecisionTreeClassifier):
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            # leaf nodes ID
            leaf_nodes = np.where(children_left == -1)[0]

            # outcomes of leaf nodes
            leaf_values = values[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
            leaf_classes = np.argmax(leaf_values, axis=-1)
            leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

            return children_left, children_right, feature, threshold, leaf_nodes
        elif isinstance(tree, xgboost.core.Booster):
            children_left, children_right, threshold, feature, scores = parse_booster(
                tree
            )
            leaf_nodes = np.where(children_left == -1)[0]
            leaf_classes = scores[leaf_nodes] > 0.5
            leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

            return children_left, children_right, feature, threshold, leaf_nodes
        else:
            raise ValueError("tree is not of a supported Class")

    children_left, children_right, feature, threshold, leaf_nodes = parse_tree(tree)

    paths = {}
    for leaf_node in leaf_nodes:
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = []
        parents_right = []
        while parent_node != 0:
            if np.where(children_left == child_node)[0].shape == (0,):
                parent_left = -1
                parent_right = np.where(children_right == child_node)[0][0]
                parent_node = parent_right
            elif np.where(children_right == child_node)[0].shape == (0,):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            child_node = parent_node
        paths[leaf_node] = (parents_left, parents_right)

    path_info = get_path_info(paths, threshold, feature)
    return path_info


def get_path_info(paths, threshold, feature):
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        inequality_symbols = []  # inequality symbols used in the current node
        thresholds = []  # thresholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]

        for idx in range(len(parents_left)):

            def do_appends(node_id):
                """helper function to reduce duplicate code"""
                node_ids.append(node_id)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])

            if parents_left[idx] != -1:
                """the child node is the left child of the parent"""
                node_id = parents_left[idx]  # node id
                inequality_symbols.append(0)
                do_appends(node_id)
            elif parents_right[idx] != -1:
                """the child node is the right child of the parent"""
                node_id = parents_right[idx]
                inequality_symbols.append(1)
                do_appends(node_id)

            path_info[i] = {
                "node_id": node_ids,
                "inequality_symbol": inequality_symbols,
                "threshold": thresholds,
                "feature": features,
            }
    return path_info


class FeatureTweak(RecourseMethod):

    def __init__(
        self,
        mlmodel: MLModelCatalog,
        hyperparams: Dict,
        cost_func=_L2_cost_func,
    ):

        super().__init__(mlmodel)

        self.model = mlmodel
        self.data = mlmodel.data
        self.eps = hyperparams["eps"]
        self.target_col = self.data.target
        self.cost_func = cost_func

    def esatisfactory_instance(self, x: np.ndarray, path_info):
        esatisfactory = copy.deepcopy(x)
        for i in range(len(path_info["feature"])):
            feature_idx = path_info["feature"][i]  # feature index

            if isinstance(feature_idx, str):
                feature_idx = np.where(
                    np.array(self.model.feature_input_order) == feature_idx
                )

            threshold_value = path_info["threshold"][i]  # threshold in current node
            inequality_symbol = path_info["inequality_symbol"][i]  # inequality symbol
            if inequality_symbol == 0:
                esatisfactory[feature_idx] = threshold_value - self.eps
            elif inequality_symbol == 1:
                esatisfactory[feature_idx] = threshold_value + self.eps
            else:
                print("something wrong")
        return esatisfactory

    def feature_tweaking(self, x: np.ndarray, class_labels: List[int], cf_label: int):

        def predict(classifier, x):
            if isinstance(
                classifier,
                (sklearn.tree.DecisionTreeClassifier, MLModel),
            ):
                # need to reshape x as it's not a batch
                return classifier.predict(x.reshape(1, -1))
            elif isinstance(classifier, xgboost.core.Booster):
                threshold = 0.5
                return (
                    classifier.predict(
                        xgboost.DMatrix(
                            x.reshape(1, -1),
                            feature_names=self.model.feature_input_order,
                        )
                    )
                    > threshold
                )
            raise ValueError("tree is not of a supported Class")

        x_out = copy.deepcopy(x)  # initialize output
        delta_mini = 10**3  # initialize cost
        for tree in self.model.tree_iterator:  # loop over individual trees

            estimator_prediction = predict(tree, x)
            if (
                predict(self.model, x) == estimator_prediction
                and estimator_prediction != cf_label
            ):
                paths_info = search_path(tree, class_labels)
                for key in paths_info:
                    """generate epsilon-satisfactory instance"""
                    path_info = paths_info[key]
                    es_instance = self.esatisfactory_instance(x, path_info)
                    if (
                        predict(tree, es_instance) == cf_label
                        and self.cost_func(x, es_instance) < delta_mini
                    ):
                        x_out = es_instance
                        delta_mini = self.cost_func(x, es_instance)
                else:
                    continue
        return x_out

    def get_counterfactuals(self, factuals: pd.DataFrame):

        # drop targets
        instances = factuals.copy()
        instances = instances.reset_index(drop=True)

        # only works for continuous data
        instances = self.model.get_ordered_features(instances)

        class_labels = [0, 1]

        counterfactuals = []
        for i, row in instances.iterrows():
            cf_label = 1  # flipped target label
            counterfactual = self.feature_tweaking(
                row.to_numpy(), class_labels, cf_label
            )
            counterfactuals.append(counterfactual)

        counterfactuals_df = check_counterfactuals(
            self._mlmodel, counterfactuals, factuals.index
        )
        counterfactuals_df = self._mlmodel.get_ordered_features(counterfactuals_df)
        return counterfactuals_df


class RandomForestModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        df_train = self.data.df_train
        df_test = self.data.df_test

        x_train = df_train[self.data.continuous]
        y_train = df_train[self.data.target]
        x_test = df_test[self.data.continuous]
        y_test = df_test[self.data.target]

        self._feature_input_order = self.data.continuous
        self._mymodel = RandomForestClassifier(random_state=42)
        self._mymodel.fit(
            x_train,
            y_train
        )
        print("Accuracy on test data: {:.2f}".format(self._mymodel.score(x_test, y_test)))

    @property
    def feature_input_order(self):
        return self._feature_input_order

    @property
    def backend(self):
        return "sklearn"

    @property
    def raw_model(self):
        return self._mymodel

    @property
    def tree_iterator(self):
        return self.raw_model.estimators_

    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))

    def update_train_data(self, cf):
        df_train_with_cf = pd.concat([self.data.df_train, cf], ignore_index=True)
        df_test = self.data.df_test

        x_train_cf = df_train_with_cf[self.data.continuous]
        y_train_cf = df_train_with_cf[self.data.target]
        x_test = df_test[self.data.continuous]
        y_test = df_test[self.data.target]

        skf = StratifiedKFold(n_splits=5, random_state=43)
        acc_score = 0
        f_score = 0
        for train_index, test_index in skf.split(self.data.df_train[self.data.continuous], self.data.df_train[self.data.target]):
            train = self.data.df_train[self.data.continuous].iloc[train_index,:]
            train_target = self.data.df_train[self.data.target].iloc[train_index]
            sm = SMOTE(sampling_strategy=1.0, random_state=42)
            X_train_oversampled, y_train_oversampled = sm.fit_sample(train, train_target)
            model_smote = RandomForestClassifier(random_state=42)
            model_smote.fit(X_train_oversampled, y_train_oversampled )  
            y_pred = model_smote.predict(x_test)
            acc_score += model_smote.score(x_test, y_test)
            f_score += f1_score(y_test, y_pred)
        
        print(f'\nAvg. Accuracy with smote: {acc_score/5}')
        print(f'Avg. f-score with smote: {f_score/5}')

        acc_score = 0
        f_score = 0
        skf = StratifiedKFold(n_splits=5, random_state=42)
        for train_index, test_index in skf.split(x_train_cf, y_train_cf):
            model_cf = RandomForestClassifier(random_state=42)
            model_cf.fit(x_train_cf.iloc[train_index,:], y_train_cf.iloc[train_index])
            y_pred = model_cf.predict(x_test)
            acc_score += model_cf.score(x_test, y_test)
            f_score += f1_score(y_test, y_pred)
    
        print(f'\nAvg. Accuracy with CF: {acc_score/5}')
        print(f'Avg. f-score with CF: {f_score/5}')


model_type = "forest"
dataset = CsvCatalog('LLPs_mono.csv', categorical=[], continuous = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"], immutables=[], target="label")

ml_model = RandomForestModel(dataset)
hyperparams = {
    "eps": 0.05 
}

recourse_method = FeatureTweak(ml_model, hyperparams)
sample_df = ml_model.data.df_train[ml_model.data.df_train['label'] == 0].sample(295)
sample_df.to_csv('original.csv', index = False)
#sample_df.drop('label', inplace=True, axis=1)
counterfactuals = recourse_method.get_counterfactuals(sample_df)
counterfactuals = counterfactuals.dropna()
#counterfactuals['label'] = 1
counterfactuals.to_csv('generated.csv', index = False)
ml_model.update_train_data(counterfactuals)