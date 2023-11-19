# randomForest

A vanilla implementation of [random forest](https://en.wikipedia.org/wiki/Random_forest) coming with a python framework.

# Intro
*In a 1906 fair, 800 attendees guessed an ox’s weight. Astonishingly, the crowd’s median guess was within 1% of the actual weight, outperforming all individual estimates. [[source](https://databasecamp.de/en/ml/random-forests)]*

Imagine the crowd at the fair as a "Random" Forest, and each person acts as a Decision Tree. Each person (Decision Tree) makes an independent guess (decision) based on their own rules (criteria), just like how each tree in a Random Forest makes a decision based on a subset of features. The final estimate of the ox’s weight (the Random Forest’s prediction) is then determined by combining all these individual guesses (decisions), often by taking the mode (for classification) or mean (for regression). This ensemble method leverages the wisdom of the crowd (multiple Decision Trees), making the final prediction more robust and accurate than any single guess (Decision Tree). Just like the crowd outperformed the individuals in guessing the ox’s weight, a Random Forest often outperforms individual Decision Trees in machine learning tasks.

In a fruit garden full of apples and pears. Each fruit in this garden has two properties: color and size. The color can range from green to red, and the size can range from small to large.
Now, you want to build a machine that can predict whether a fruit is an apple or a pear based on these two properties. This is where Random Forest comes in!
A Random Forest is like a team of decision trees where each tree is like a fruit expert who specializes in judging fruits based on certain rules. One tree might be really good at classifying fruits based on color, while another might excel at classifying based on size.
When a new fruit comes along, each tree in the Random Forest makes a vote: “I think this is an apple!” or “I think this is a pear!”. The majority vote decides the final prediction.

So, in essence, a Random Forest is a powerful way to make predictions by combining the knowledge of multiple ‘experts’ or decision trees. It’s like having a panel of fruit experts in your garden, helping you classify your fruits! Luckily you do not have to trust a single one, since you can weigh all decisions together.

# Getting Started

## Loading Credit Risk CSV Data 
The credit test data is provided as a csv-file outlined in the table below.

|Customer|Savings|Assets|Income|Credit_Risk|
|-|-|-|-|-|
|1|Medium|High|75|Good|
|2|Low|Low|50|Bad|
|3|High|Medium|25|Bad|
|4|Medium|Medium|50|Good|
|5|Low|Medium|100|Good|
|6|High|High|25|Good|
|7|Low|Low|25|Bad|
|8|Medium|Medium|75|Good|

*Note:* the last column is used as default for classes - what needs to be classified "the classes". All other columns are "features" with sample values. This means: one could classify the credit risk (into good or bad), based on the e.g. Savings and Income. Columns for pure identification, like the very first column, do not provide useful information and can (should) be ignored. BlackForest has the ignoreColumns flag to accomplish this.

The csv loader can be used to parse csv files into `Set`-type object:

```python
from blackForest import Forest, loadSetFromCsv

# the loadSetFromCsv parser will turn a csv file to a set 
# initliaze all elements etc. 
# Ignore the first id column.
root = loadSetFromCsv ('./credit_testdata.csv', ignoreColumns=[0])
```

Now the set only needs to be passed while initializing a `Forest` object. The number `seeds` determines the numbers of trees seeded.

``` python
# blueprint
class Forest(
    _Set: Set,
    seeds: int,
    depth: int,
    minDepth: int = None,
    featureSubset: int = None,
    ignoreCasing: bool = False
)
```

Initialize a forest with 3 decision trees, each exactly 2 nodes, derived from two random features (out of four). Also we ignore upper and lower casing of features.

```python
f = Forest(root, 3, 2, featureSubset=2, ignoreCasing=True)
```

Subsequently, let the forest grow trees, since the task is short this can happen in a single thread.

```python
f.grow(thread=1)
```

While the forest grows, it trains each decision node of his trees, searching for patterns.

### Quick Test Sample

Once the forest has grown, it needs to be tested. For this we can use the root `Set`-object to get a randomly sampled element for classification:

```python
# get a single sample element
testSample = root.sample()[0]

# classify
distribution = f.classify(testSample)
```

The result could be displayed as such:

```python
print(f'Test Data Element Features: {str(testSample.features)}')
print('Expected/Actual Evaluation:', testSample._class)
print('Forest Prediction:', dist)
```
output:
```
Test Element Features: {'Savings': 'Low', 'Assets': 'Medium', 'Income': 100}
Expected/Actual: Good
Forest Prediction: {'Good': 0.625, 'Bad': 0.375}
```
**Good** was correctly predicted with **62.5%** certainty, from only 8 data entries.