import sys
import math
import numpy as np

sys.path.insert(1, "../EXPLAN/LORE")
from pyyadt import *


def mdi_gain_ratio(dt, X, y, features, discrete, features_type):
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X = X.iloc[:, 1:]
        X = X.values
    # get edge labels, node labels, is leaf
    edge_labels = get_edge_labels(dt)
    node_labels = get_node_labels(dt)
    node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt, 'shape').items()}

    root = "n0"

    # mdi-array und anzahl klassen als variablen anlegen
    importances = {}
    for att in features:
        importances[att] = 0
    classes = np.unique(y)

    # rekursive funktion als in-funktion-definition:
    def recursive_computation(node,indexes):
        n_data = len(indexes)
        entropy = 0.0 # impurity

        for c in classes:
            count = sum([1 if y[i] == c else 0 for i in indexes])
            if count > 0:
                count /= n_data
                entropy -= count * math.log(count, 2)

        if not node_isleaf[node]:
            child_indexes = {}
            impurities = {}
            for child in dt.neighbors(node):
                child_indexes[child] = []

            att = node_labels[node]
            for idx in indexes:
                val = X[idx][features.index(att)]
                for child in dt.neighbors(node):
                    edge_val = edge_labels[(node, child)]
                    if att in discrete:
                        val = val.strip() if isinstance(val, str) else val
                        if yadt_value2type(edge_val, att, features_type) == yadt_value2type(val, att, features_type):
                            child_indexes[child].append(idx)
                            break
                        elif math.isclose(yadt_value2type(edge_val, att, features_type),val,abs_tol=1e-6):
                            print("it was the not equal but close problem")
                            child_indexes[child].append(idx)
                            break
                    else:
                        pyval = yadt_value2type(val, att, features_type)
                        if '>' in edge_val:
                            thr = yadt_value2type(edge_val.replace('>', ''), att, features_type)
                            if pyval > thr:
                                child_indexes[child].append(idx)
                                break
                        elif '<=' in edge_val:
                            thr = yadt_value2type(edge_val.replace('<=', ''), att, features_type)
                            if pyval <= thr:
                                child_indexes[child].append(idx)
                                break

            split_info = 0.0
            for child in child_indexes:
                # Abfangen, dass ein Pfad keine Daten hat
                if len(child_indexes[child]) > 0:
                    impurities[child] = recursive_computation(child,child_indexes[child])
                    ratio = len(child_indexes[child])/n_data
                    split_info -= ratio * math.log(ratio, 2)
                else:
                    print("branch without indexes", child)
            # berechne mittels längen der Datenlisten + gespeicherten entropie-Werten und berechneter split info die mdi
            importances[att] += (n_data * entropy - sum([len(child_indexes[child]) * impurities[child] for child in impurities]))/split_info

        # return eigene impurity
        return entropy

    # rekusive funktion starten mit root
    recursive_computation(root,range(len(X)))

    # importances durch gesamtzahl daten teilen
    for att in importances:
        importances[att] /= len(X)

    # print("edge_labels", edge_labels)
    # dict von (n,m) : "</>= x"
    # print("node_labels", node_labels)
    # dict von n : "f" für innere Knoten bzw n : "p(x,y)" für Blätter
    # print("node_isleaf", node_isleaf)
    # dict von n : True/False

    return list(importances.items())
