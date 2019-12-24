import numpy as np


# Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sumVsEntropy = []
    # calculate the entropies of each of the branch..
    for attribute in branches:
        sum_attribute = sum(attribute)
        entropy_attribute = 0
        if sum_attribute != 0:  # means some data is present
            for count in attribute:
                if count != 0:
                    class_entropy = - ((count / sum_attribute) * np.log2(count / sum_attribute))
                    entropy_attribute += class_entropy
        sumVsEntropy.append((sum_attribute, entropy_attribute))

    entropy = 0
    total_count = sum([pair[0] for pair in sumVsEntropy])
    if total_count != 0:
        # weighted entropy calculation
        for (sum_attribute, entropy_attribute) in sumVsEntropy:
            entropy_branch = (sum_attribute / total_count) * entropy_attribute
            entropy += entropy_branch
    return S - entropy


# get the accuacy of the system
def getAccuracy(y_predicted, y_test):
    if len(y_test) == 0:
        return -1
    count = 0
    for i in range(0, len(y_predicted)):
        if y_predicted[i] == y_test[i]:
            count += 1
    return float(count / len(y_test))  # no.of correct labels/total no.of labels


# get all the parent child relationship and store it in the map
def getParentChild(root):
    parentChild = {}
    parentChild[root] = []
    i = 0
    while i < len(parentChild.keys()):
        currNode = list(parentChild.keys())[i]
        children = []
        for child in currNode.children:
            if child.splittable and len(child.children) > 0:  # if there is a child then add it to the map.
                parentChild[child] = []
                children.append(child)  # append it to the children
        parentChild[currNode] = children
        i = i + 1
    return parentChild


# implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List

    # base conditions
    if decisionTree.root_node is None or len(X_test) == 0 or len(y_test) == 0 or len(X_test) != len(y_test):
        return

    y_predicted = decisionTree.predict(X_test)
    max_accuracy = getAccuracy(y_predicted, y_test)
    if max_accuracy == -1:
        return
    root = decisionTree.root_node
    # get all the parent child relationships
    parentAndchildren = getParentChild(root)
    # do this until we have a parent to check
    while len(parentAndchildren.keys()) > 0:
        localAccuracyMax = 0
        localMaxnode = None
        nodes = list(parentAndchildren.keys())
        for i in range(len(nodes)):
            currNode = nodes[i]
            # make the splittable of current node to false and check the accuracy
            currNode.splittable = False

            # predict now after pruning a branch
            y_predicted = decisionTree.predict(X_test)

            # get the accuracy
            accuracy = getAccuracy(y_predicted, y_test)

            # update local accuracy if its big
            if accuracy > localAccuracyMax:
                localAccuracyMax = accuracy
                localMaxnode = currNode

            # reset the current node and continue to prune other nodes
            currNode.splittable = True
        # if the local accuracy is same or greater then prune that branch
        if localAccuracyMax >= max_accuracy:
            children = parentAndchildren[localMaxnode]
            # remove all its children fron consideration as its no longer needed
            for child in children:
                parentAndchildren.pop(child, None)
            localMaxnode.children = None
            localMaxnode.splittable = False
            # remove that node also..
            parentAndchildren.pop(localMaxnode, None)
            max_accuracy = localAccuracyMax
        else:
            break


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t',
                       deep=deep + 1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
