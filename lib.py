
'''
Here’s a high-level overview of the classes:

    [Element Class]     This class represents the basic unit of your data. 
                        Each Element has a class label and a set of features.

    [Set Class]         This class represents a set of Element objects. 
                        It includes methods for adding elements, calculating 
                        the Gini impurity, and splitting the set based on a feature and value.

    [Node Class]        This class represents a node in your decision tree. 
                        Each Node has an ID, depth, associated Set, parent, 
                        left child, right child, feature, and condition.

    [Tree Class]        This class represents a decision tree. It includes 
                        methods for growing the tree and traversing the tree to make predictions.

    [Traverse Method]   This method in the Tree class is used to navigate 
                        through the tree based on the features of a given 
                        Element and returns the class distribution of the 
                        final node.

And a detailed overview of the methods implemented so far:                     

    [Element Class]
    - __init__:         is the constructor for the Element class. 
                        It initializes an Element with a class label and a set of features.

    [Set Class]
    - __init__:         is the constructor for the Set class. It initializes 
                        a Set with a list of classes and a dictionary of features.
    - addElement:       adds an Element to the Set. It checks if the Element's 
                        class and features are known before adding it.
    - addElements:      adds multiple Element objects to the Set using the addElement method.
    - impurity:         calculates the Gini impurity of the Set based on 
                        the class distribution of its elements.
    - prob:             calculates the estimated probability for a class 
                        from relative occurrences.
    - split:            splits the Set at a specific value based on a feature.

    [Node Class]
    - __init__:         is the constructor for the Node class. It initializes a 
                        Node with an ID, depth, associated Set, parent, left 
                        child, right child, feature, and condition.

    [Tree Class]
    - __init__:         is the constructor for the Tree class. It initializes 
                        a Tree with a Set and a depth.
    - pureNode:         finds the best split by minimizing the impurity of a 
                        node with respect to a provided feature. 
    - grow:             grows the tree to a specified depth. It selects a random 
                        subset of features and finds the best feature that 
                        minimizes the impurity at each node.

    [Traverse Method]
    - traverse:         This method in the Tree class is used to navigate through 
                        the tree based on the features of a given Element. 
                        It returns the class distribution of the final node ("leaf").
'''

from __future__ import annotations
from typing import Any 
import random

# multiprocessing might be a better choice than threading 
# as it can bypass the GIL and utilize multiple cores effectively
# with seperated memory spaces.
from multiprocessing import Process


class Element:

    '''
    Elements are the lowest level objects which are contained in a set.
    Every Element has a specific class, and arbitrary features, both of
    them are either int|float|str types.  
    '''

    def __init__ (self, _class: int|float|str, **features: int|float|str) -> None:

        self._class = _class
        self.features = features

class Set:

    '''
    A set only respects elements whose class is contained in a predefined
    list of possible classes and possible features and their corr. types.
    Features can be either types or list of categories (same type formats).
    Example for a cat-dog classifier:
        classes=['cat', 'dog']
        feature={ # feature domain
            'color': ['white', 'brown', 'grey', 'black'],
            'height': float,
            'age': int
        }
    '''

    def __init__ (self, classes: list[int|float|str], features: dict[str, list | float | int]) -> None:
        
        self.elements = set()
        self.classes = classes
        self.features = features
        self.formatTypes = [int, float, str]

        # create a frequency distribution for all class occurrences
        self.classFrequencies = {}
        self.size = 0
    
    def addElement (self, elem: Element) -> None:

        # check if provided element is in a known class
        if not elem._class in self.classes:
            raise ValueError(f'Class {elem._class} not known.')

        # check if the features are known
        for k, v in elem.features.items():
            if not k in self.features:
                raise ValueError(f'Feature {k} not known.')
            if self.features[k] in self.formatTypes and not type(v) == self.features[k]:
                raise TypeError(f'Feature {k}={v} ({type(v)}) is not a valid type, expected {self.features[k]}.')
            elif self.features[k] not in self.formatTypes and v not in self.features[k]:
                raise TypeError(f'Feature {k} must be in {self.features[k]} ')
        
        # count class for distribution
        if not elem._class in self.classFrequencies:
            self.classFrequencies[elem._class] = 1
        else:
            self.classFrequencies[elem._class] = self.classFrequencies[elem._class] + 1
        self.size += 1
        
        self.elements.add(elem)
    
    def addElements (self, elements: list[Element]) -> None:

        '''
        Adds multiple elements in sequence using self.addElement
        '''

        for e in elements:
            self.addElement(e)
    
    def impurity (self) -> float:

        '''
        Current Gini-impurity of the contained elements
        to their class distribution.
        Metric of entropy.  

        Returns a float with impurity metric
        0 means the set is pure (consists of elements of one class)
        the closer the value is to 1 the higher the class diversity.
        '''

        val = 1

        for c in self.classes:

            val -= self.prob(c) ** 2

        return val 

    def prob (self, _class: int|float|str) -> float:

        '''
        Returns the estimated probability for a class from rel. occurrences.
        '''

        # catch key errors
        if not _class in self.classFrequencies:
            return 0

        return self.classFrequencies[_class] / self.size

    def split (self, feature: str, value: str|float|int, ignoreCasing: bool=False) -> tuple[Set, Set]:

        '''
        Splits the set at a specific value.

        If the value is a float it will be treated as a threshold,
        first returned set will yield values lower or equal than threshold 
        while the second will contain elements whose feature values are 
        higher values.

        Otherwise, if a categorical string value is detected,
        the first Set will contain the matches, while the 2nd
        everything else.

        Left Set: matching Elements
        Right Set: all other Elements
        '''

        l, r = [], []

        expectedFeatureType = self.features[feature]

        for e in self.elements:

            if expectedFeatureType in [int, float]:

                if e.features[feature] <= value:

                    l.append(e)
                
                else:

                    r.append(e)


            else: # list of strings

                if e.features[feature] == value or (ignoreCasing and e.features[feature].lower() == value.lower()):

                    l.append(e)
                
                else:

                    r.append(e)

        # initialize new splitted Set
        LS = Set(self.classes, self.features)
        RS = Set(self.classes, self.features)

        LS.addElements(l)
        RS.addElements(r)

        return LS, RS

class Node:

    '''
    Node blueprint.
    Every node is an object which holds:

     - id (starting at 0 and ascending in increments of 1)
     - depth (root is 0)
     - the underlying set
     - pointer to parent (None if root)
     - pointer to left child
     - pointer to right child
     - feature
     - split condition

    Every initialized node will auto-assign an nonce-id.
    '''

    count = 0

    def __init__(self, depth: int, _Set: Set, parent: Node=None, leftChild: Node=None, rightChild: Node=None,
                feature: str|None=None, condition: int|float|str=None) -> None:
        
        self.id = Node.count
        Node.count += 1 # increment global class count
        self.depth = depth
        self.Set = _Set
        self.parent = parent
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.feature = feature
        self.condition = condition

class Tree:

    '''
    A base decision tree.
    '''

    def __init__ (self, _Set: Set, depth: int) -> None:

        self.Set = _Set

        # Nodes are structured to point to their closest peers.
        # Initialize first (root) node which has no parent by definition
        # and keep children undefined yet.
        self.nodes = [Node(0, _Set, None, None, None)]

        self.depth = depth

    def pureNode (self, node: Node, feature: str, ignoreCasing: bool=False) -> float:
        
        '''
        Finds the best split by minimizing the impurity  (in regards of a 
        provided feature) of a node.
        '''

        fType = type(self.features[feature])

        L, R = None, None
        condition = None
        impurity = float('inf')

        # split the parent set depending on feature type into L, R childs
        if isinstance(fType, (int, float)):

            # float and integers demand continuous splits

            # first determine search domain by accumulating all 
            # observed feature values
            domain = []
            for e in node.Set:
                domain.append(e.features[feature])
        
        elif isinstance(fType, list):

            # string features are discrete, use all 
            # possible string features as domain.
            domain = node.Set.features[feature]

        # sort the domain in ascending order
        domain.sort() 

        # minimize impurity
        for value in domain:
            
            # compute candidate split sets and impurity
            cL, cR = node.Set.split(feature, value, ignoreCasing)

            # Sum both Gini impurities (L,R) as figure of merit - which needs to be minimized
            cImpurity = cL.impurity() + cR.impurity()

            if cImpurity < impurity:

                impurity = cImpurity
                condition = value
                L, R = cL, cR
        
        # override node values
        node.condition = condition 
        node.feature = feature
        node.leftChild = Node(node.depth + 1, L, node)
        node.rightChild = Node(node.depth + 1, R, node)

        return impurity

    def grow (self, currentDepth: int=0, featureSubset: int=None, ignoreCasing: bool=False) -> None:

        '''
        depth: total depth (provided persistently)
        currentDepth: pass argument in the iterative process
        featureSubset: random subset size of the features selected for impurity search
        '''

        # terminate if leaf is hit i.e. when depth extincts
        if currentDepth == self.depth:
            return
        
        # filter all nodes at current depth
        for node in self.nodes:

            if node.depth != currentDepth:
                continue

            # -- node is at current depth --

            # select random but distinct features
            if not featureSubset:
                featureSubset = len(self.Set.features)
            randomFeatures = self.Set.features
            random.shuffle(randomFeatures)
            randomFeatures = randomFeatures[:featureSubset]
            
            # find best feature which minimizes the impurity
            bestFeature = None
            impurity = float('inf')
            for feat in randomFeatures:
                probe = self.pureNode(node, feat, ignoreCasing)
                if probe < impurity:
                    bestFeature = feat
                    impurity = probe
            
            # pure the node by best feature
            self.pureNode(node, bestFeature, ignoreCasing)

            # add the produced children to node pool
            node.leftChild.depth = node.depth + 1
            node.rightChild.depth = node.depth + 1
            self.nodes.append(node.leftChild)
            self.nodes.append(node.rightChild)        

            # Call grow function recursively with incremented currentDepth
            self.grow(currentDepth + 1, featureSubset, ignoreCasing)

    def traverse (self, element: Element, ignoreCasing: bool=False) -> dict[int|float|str, float]:

        '''
        Main decision making method. 
        Used for propagating decision tasks on a Set, and should
        be used after the tree has grown.
        Returns class distribution in dict format.
        '''

        # start from root
        pointer = self.nodes[0]

        # simulate decision path
        while pointer.condition:
            
            expectedFeatureType = self.features[pointer.feature]

            # decide if evaluation continues in left or right child branch
            if (expectedFeatureType in [int, float] and element.features[pointer.feature] <= pointer.condition) or (element.features[pointer.feature] == pointer.condition or (ignoreCasing and element.features[pointer.feature].lower() == pointer.condition.lower())):
                pointer = pointer.leftChild
            else:
                pointer = pointer.rightChild

        # extract the set from final pointer (leaf)
        leafSet = pointer.Set

        # determine class distribution
        classDist = {}
        for c in self.Set.classes:
            classDist[c] = leafSet.prob(c)

        return classDist

class Forest:

    '''
    Arguments:
        seeds:      number of trees
        depth:      the maximum depth of the trees
        minDepth:   the min. depth for varying depth method
                    (default=None which locks the depth to 
                    constant across all trees)
        featureSubset:  Random subset size for features to pick during training. 
    '''

    def __init__ (self, _Set: Set, seeds: int, depth: int, minDepth: int=None, featureSubset: int=None, ignoreCasing: bool=False) -> None:

        self.Set = _Set
        self.seeds = seeds
        self.featureSubset = featureSubset
        self.ignoreCasing = ignoreCasing

        self.trained = False

        # tree buffer
        self.trees = []

        # plant trees from seeds
        for _ in range(seeds):

            # sample a depth
            if minDepth:
                if minDepth > depth:
                    ValueError('minDepth cannot exceed depth! Choose minDepth <= depth.')
                sampleDepth = random.randint(minDepth, depth)
            else:
                sampleDepth = depth

            # spawn Tree
            self.trees.append(
                Tree(self.Set, sampleDepth)
            )
    
    def classify (self, element: Element, normalize: bool=True) -> dict[int|float|str, float]:

        if not self.trained:
            Warning('The forrest has not been trained yet. First train the forrest by calling the Forrest.grow method.')

        finalDist = {}
        for c in self.Set.classes:
            finalDist[c] = 0

        for t in self.trees:
            dist = t.traverse(element, self.ignoreCasing)
            for c in self.Set.classes:
                finalDist[c] = finalDist[c] + dist[c]

        if normalize:
            for k, v in finalDist.items():
                finalDist[k] = v/self.seeds
        
        return finalDist

    def grow (self, threads: int=4) -> None:

        '''
        Grows the forest by splitting the n trees accross <=n threads.
        Each process will run in seperated mem space and is scheduled
        by the OS to run on different physical or logical cores.
        '''

        if threads > self.seeds:
            Warning(f'The number of threads ({threads}) exceeds the number of trees ({self.seeds}) in the forest.')
        
        bunchSize = int(self.seeds / threads) + 1 # the +1 ensures that all trees will be used
        processes = [] # accumulate threads
        for t in range(threads):
            bunch = self.trees[t*bunchSize:(t+1)*bunchSize]
            prc = Process(target=self.growBunch, kwargs={'trees': bunch})
            processes.append(prc)
            prc.start()

        # await all processes to finish
        for prc in processes:
            prc.join()

        # flip trained label
        self.trained = True

    def growBunch (self, trees: list[Tree]):

        '''
        Grows a bunch of provided trees procedually in a single thread.
        This routine is designed to be spawned in threaded environments.
        '''

        for t in trees:

            t.grow(0, self.featureSubset, self.ignoreCasing)

if __name__ == '__main__':

    s = Set(classes=['Good', 'Bad'], features={
        'Savings': ['Low', 'Medium', 'High'],
        'Assets': ['Low', 'Medium', 'High'],
        'Income': int
    })

    # add data to the set
    s.addElement( Element(_class='Good', Savings='Medium', Assets='High', Income=75) )
    s.addElement( Element(_class='Bad', Savings='Low', Assets='Low', Income=50) )
    s.addElement( Element(_class='Bad', Savings='High', Assets='Medium', Income=25) )
    s.addElement( Element(_class='Good', Savings='Medium', Assets='Medium', Income=75) )
    s.addElement( Element(_class='Good', Savings='Low', Assets='Medium', Income=75) )
    s.addElement( Element(_class='Good', Savings='High', Assets='High', Income=25) )
    s.addElement( Element(_class='Bad', Savings='Low', Assets='Low', Income=25) )
    s.addElement( Element(_class='Good', Savings='Medium', Assets='Medium', Income=75) )

    print(s.impurity())