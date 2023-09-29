'''
Notes:
1. Install the following libraries before running codes
    !pip install git+https://github.com/Jingkang50/OpenOOD
    !pip install pytorch_lightning
    !pip install gpytorch
    !pip install --upgrade git+https://github.com/y0ast/DUE.git
    !pip install nflows

2. Place the preprocessed CSV files in the same directory as this file
'''

import argparse
import torch
import torch.nn as nn

from reading_files.csv_read import read, check
from reading_files.feature_selection import get_eICU_selected_features, get_mimic_selected_features
from ood_experiment.experiments import get_params_data
from ood_experiment.validate_difference import validate_ood_data
from models.predictive_models import get_params
from training.utils import set_all_seeds
from training.data_handler import normalization, split_data
from training.train import train_predictive_model
from ood_measures.ood_score import get_ood_score
from ood_measures.detection_methods_posthoc import detection_method

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_version', default='iv', type=str, choices={'iv', 'iii'})
    parser.add_argument('--in_distribution', default='mimic', type=str, choices={'mimic', 'eicu'})

    parser.add_argument('--ood_type', default='other_domain', type=str, choices={'other_domain', 'multiplication', 'feature_seperation'})
    parser.add_argument('--feature_to_seperate', default='age', type=str, choices={'age', 'gender', 'ethnicity', 'admission_type', 'first_careunit'}) #only for 'feature_separation'
    parser.add_argument('--threshold', default='70', type=str) #threshold for dividing data in 'feature_separation'

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--d_out', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--handle_imbalance_data', default=0, type=int, choices={0, 1})
    parser.add_argument('--train_model', default=1, type=int, choices={0, 1})
    parser.add_argument('--architecture', default='ResNet', type=str, choices={'MLP', 'ResNet', 'FTTransformer'})

    parser.add_argument("--detectors", nargs='+', default=['MDS', 'AE'])
    #Post-hoc: 'MSP', 'KNN', 'OpenMax', 'MDS', 'RMDS', 'temp_scaling', 'odin', 'gram', 'ebo', 'gradnorm', 'react', 'mls', 'klm', 'vim', 'dice', 'ash', 'she_euclidean', 'she_inner'
    #Density: 'HiVAE', 'AE', 'VAE', 'Flow', 'ppca', 'lof', 'due'

    return parser.parse_args()

#get args
args = get_args()
mimic_version = args.mimic_version
in_distribution = args.in_distribution
ood_type = args.ood_type
feature_to_seperate = args.feature_to_seperate
try:
  threshold = float(args.threshold)
except:
  threshold = args.threshold

seed=args.seed
d_out = args.d_out
batch_size = args.batch_size
n_epochs = args.n_epochs
lr = args.lr
weight_decay = args.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
handle_imbalance = args.handle_imbalance_data
architecture = args.architecture
train_model = args.train_model
detectors = args.detectors

#reading preprocessed csv file
eICU_features, mimic_features = read(mimic_version=mimic_version)

# Select features and label
eICU_features_selected, eICU_label = get_eICU_selected_features(eICU_features)
mimic_features_selected, mimic_label = get_mimic_selected_features(mimic_features)


# Define in/out data in ood experiments
print('\nPreparing in/out data for the experiment ...')
print('Type of experiment:', ood_type)

if in_distribution == 'mimic':
  in_features_df, in_label_df = mimic_features_selected, mimic_label
  ood_features_df = eICU_features_selected
elif in_distribution == 'eicu':
  in_features_df, in_label_df = eICU_features_selected, eICU_label
  ood_features_df = mimic_features_selected


if ood_type == 'feature_seperation':
    in_features_np, ood_features_np, in_label_np = get_params_data(in_distribution=in_distribution, in_features_df=in_features_df, in_label_df=in_label_df,
                                                                ood_type=ood_type, ood_features_df=None, feature_to_seperate=feature_to_seperate, threshold=threshold, mimic_version=mimic_version)
    print(in_features_np.shape, ood_features_np.shape, in_label_np.shape)

elif ood_type  == 'other_domain':
    in_features_np, ood_features_np, in_label_np = get_params_data(in_distribution=in_distribution, in_features_df=in_features_df, in_label_df=in_label_df,
                                                                ood_type=ood_type, ood_features_df=ood_features_df, feature_to_seperate=None, threshold=None)
    print(in_features_np.shape, ood_features_np.shape, in_label_np.shape)

elif ood_type == 'multiplication':
    in_features_np, scales, random_sample, in_label_np = get_params_data(in_distribution=in_distribution, in_features_df=in_features_df, in_label_df=in_label_df,
                                                                ood_type=ood_type, ood_features_df=None, feature_to_seperate=None, threshold=None)
    print(in_features_np.shape, in_label_np.shape)

#set random seed
set_all_seeds(seed)

#split and normalize data
X, y = split_data(in_features_np, in_label_np, handle_imbalance_data=handle_imbalance, random_state=seed)
X, y, preprocess = normalization(X, y, device)

report_frequency = len(X['train']) // batch_size // 5
if not ood_type == 'multiplication':
    ood_features_tensor = torch.tensor(preprocess.transform(ood_features_np), device=device)
else:
    ood_features_tensor = None

#define and train prediction model for posthoc methods
print('\nPreparing prediction model for the experiment ...')
model, optimizer = get_params(architecture, d_out, lr, weight_decay, X['train'].shape[1])
model.to(device)
criterion = nn.CrossEntropyLoss()
if train_model:
  train_predictive_model(model, optimizer, criterion, X, y, batch_size, n_epochs, device, report_frequency)

print('\nStart detection experiments ...\n')
# OOD performance
for detector in detectors:
    print(detector)
    score_function = detection_method(detector=detector, model=model, device=device, k_knn=5, x_train=X['train'], y_train=y['train'], batch_size=128, n_classes=d_out, x_val=X['val'], y_val=y['val'], vim_dim=64, lr=lr, n_epochs=n_epochs)
    if ood_type in ['other_domain', 'feature_seperation']:
          get_ood_score(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, batch_size=batch_size, device=device, preprocess=preprocess, random_sample=None, scales=None, out_features=ood_features_tensor, missclass_as_ood=False)
    elif ood_type == 'multiplication':
          get_ood_score(model=model, in_test_features=X['test'], in_test_labels=y['test'], ood_type=ood_type, score_function=score_function, batch_size=batch_size, device=device, preprocess=preprocess, random_sample=random_sample, scales=scales, out_features=None, missclass_as_ood=False)
    print('\n')