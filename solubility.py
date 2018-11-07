import tensorflow as tf
import pandas as pd
import deepchem as dc
import numpy as np
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from deepchem.utils.evaluate import Evaluator
import numpy.random
from deepchem.utils.save import load_from_disk


dataset_file= "datasets/delaney-processed.csv"
dataset = load_from_disk(dataset_file) #Compound ID', 'ESOL predicted log solubility in mols per litre',
    #    'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
    #    'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area',
    #    'measured log solubility in mols per litre', 'smiles']が入っている


#//Erorrが出てしまうためコメントアウト
def display_images(filenames):
    imagesList=''.join(["<img style='width: 140px; margin: 0px; float: left; border: 1px solid black;' src='%s' />"% str(s) for s in sorted(filenames)])
    display(HTML(imagesList))


def mols_to_pngs(mols, basename="test"):
    filenames = []
    for i, mol in enumerate(mols):
        filename = "%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename)
        filenames.append(filename)
    return filenames

num_to_display = 14
molecules = []
for _, data in islice(dataset.iterrows(), num_to_display):#_は数値、dataは構造式
    molecules.append(Chem.MolFromSmiles(data["smiles"]))
display_images(mols_to_pngs(molecules))



solubilities = np.array(dataset["measured log solubility in mols per litre"])
n, bins, patches = plt.hist(solubilities, 50, facecolor='green', alpha=0.75)
plt.xlabel('Measured log-solubility in mols/liter')
plt.ylabel('Number of compounds')
plt.title(r'Histogram of solubilities')
plt.grid(True)
plt.show()

featurizer = dc.feat.CircularFingerprint(size=1024)

loader = dc.data.CSVLoader(tasks=['measured log solubility in mols per litre'], smiles_field="smiles", featurizer=featurizer)

dataset = loader.featurize(dataset_file)

splitter = dc.splits.ScaffoldSplitter(dataset_file)
train_data, valid_data, test_data = splitter.train_valid_test_split(dataset)

train_mol = [Chem.MolFromSmiles(compound) for compound in train_data.ids]
displayimages(mols_to_pngs(train_mol, basename="train"))


transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_data)]

for dataset in [train_data, valid_data, test_data]:
  for transformer in transformers:
      dataset = transformer.transform(dataset)




sklearn_model = RandomForestRegressor(n_estimators=100) #ランダムフォレストを使っていry
model = dc.models.SklearnModel(sklearn_model)
model.fit(train_data)


metric = dc.metrics.Metric(dc.metrics.r2_score) #決定係数を使って、モデルの精度を確かめる
evaluator = Evaluator(model, valid_data, transformers) #Evaluatorはモデル評価メソッド
r2score = evaluator.compute_model_performance([metric]) #metricの計算（今回はr2_score）を出して、csvファイルを作り出す
print(r2score)



def rf_model_builder(model_params, model_dir):
  sklearn_model = RandomForestRegressor(**model_params)
  return dc.models.SklearnModel(sklearn_model, model_dir)

params_dict = {
    "n_estimators": [10, 100],
    "max_features": ["auto", "sqrt", "log2", None],
}

metric = dc.metrics.Metric(dc.metrics.r2_score)
optimizer = dc.hyper.HyperparamOpt(rf_model_builder)
best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
    params_dict, train_data, valid_data, transformers,
    metric=metric)



task = "measured log solubility in mols per litre"
predicted_test = best_rf.predict(test_data)
true_test = test_data.y
plt.scatter(predicted_test, true_test)
plt.xlabel('Predicted log-solubility in mols/liter')
plt.ylabel('True log-solubility in mols/liter')
plt.title(r'RF- predicted vs. true log-solubilities')
plt.show()