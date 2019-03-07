"""Script which starts from timeseries extracted on ABIDE. Timeseries
   can be downloaded from "https://osf.io/hc4md/download" (1.7GB).

   phenotypes: if not downloaded, it should be downloaded from
   https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
   Before, please read the data usage agreements and related material at
   http://preprocessed-connectomes-project.org/abide/index.html

   Prediction task is named as column "DX_GROUP".

   The timeseries are pre-extracted using several atlases
   AAL, Harvard Oxford, BASC, Power, MODL on ABIDE rs-fMRI datasets.

   After downloading, each folder should appear with name of the atlas and
   sub-folders, if necessary. For example, using BASC atlas, we have extracted
   timeseries signals with networks and regions. Regions implies while
   applying post-processing method to extract the biggest connected networks
   into separate regions. For MODL, we have extracted timeseries with
   dimensions 64 and 128 components.

   Dimensions of each atlas:
       AAL - 116
       BASC - 122
       Power - 264
       Harvard Oxford (cortical and sub-cortical) - 118
       MODL - 64 and 128

   The timeseries extraction process was done using Nilearn
   (http://nilearn.github.io/).

   Note: To run this script Nilearn is required to be installed.
"""
import warnings
import os
from os.path import join
import numpy as np
import pandas as pd

from downloader import fetch_abide


def _get_paths(phenotypic, atlas, timeseries_dir):
    """
    """
    timeseries = []
    IDs_subject = []
    diagnosis = []
    subject_ids = phenotypic['SUB_ID']
    mean_fd = phenotypic['func_mean_fd']
    num_fd = phenotypic['func_num_fd']
    perc_fd = phenotypic['func_perc_fd']
    for index, subject_id in enumerate(subject_ids):
        this_pheno = phenotypic[phenotypic['SUB_ID'] == subject_id]
        this_timeseries = join(timeseries_dir, atlas,
                               str(subject_id) + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            IDs_subject.append(subject_id)
            diagnosis.append(this_pheno['DX_GROUP'].values[0])
    return timeseries, diagnosis, IDs_subject, mean_fd, num_fd, perc_fd


# Path to data directory where timeseries are downloaded. If not
# provided this script will automatically download timeseries in the
# current directory.

timeseries_dir = None

# If provided, then the directory should contain folders of each atlas name
if timeseries_dir is not None:
    if not os.path.exists(timeseries_dir):
        warnings.warn('The timeseries data directory you provided, could '
                      'not be located. Downloading in current directory.',
                      stacklevel=2)
        timeseries_dir = fetch_abide(data_dir='./ABIDE')
else:
    # Checks if there is such folder in current directory. Otherwise,
    # downloads in current directory
    timeseries_dir = './ABIDE'
    if not os.path.exists(timeseries_dir):
        timeseries_dir = fetch_abide(data_dir='./ABIDE')

# Path to data directory where predictions results should be saved.
predictions_dir = None

if predictions_dir is not None:
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
else:
    predictions_dir = './ABIDE/predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

atlases = ['AAL', 'HarvardOxford', 'BASC/networks', 'BASC/regions',
           'Power', 'MODL/64', 'MODL/128']

atlases = ['BASC/networks']

dimensions = {'AAL': 116,
              'HarvardOxford': 118,
              'BASC/networks': 122,
              'BASC/regions': 122,
              'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

# prepare dictionary for saving results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'covariance_estimator', 'dimensionality', 'cor_fd_mean', 'cor_fd_num', 'cor_fd_perc']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

pheno_dir = 'Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(pheno_dir)

# Connectomes per measure
from connectome_matrices import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
measures = ['correlation', 'partial correlation', 'tangent']
# tspisak
#measures = ['correlation', 'partial correlation']

from my_estimators import sklearn_classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

#cv = StratifiedShuffleSplit(n_splits=100, test_size=0.25,
#                            random_state=0)

# tspisak
cv = StratifiedShuffleSplit(n_splits=100, test_size=0.25,
                            random_state=0)

for atlas in atlases:
    print("Running predictions: with atlas: {0}".format(atlas))
    timeseries, diagnosis, IDs_subject, mean_fd, num_fd, perc_fd = _get_paths(phenotypic, atlas, timeseries_dir)

    _, classes = np.unique(diagnosis, return_inverse=True)
    iter_for_prediction = cv.split(timeseries, classes)

    for index, (train_index, test_index) in enumerate(iter_for_prediction):
        print("[Cross-validation] Running fold: {0}".format(index))
        for measure in measures:
            print("[Connectivity measure] kind='{0}'".format(measure))
            connections = ConnectivityMeasure(
                cov_estimator=LedoitWolf(assume_centered=True),
                kind=measure)
            conn_coefs = connections.fit_transform(timeseries)

            for est_key in sklearn_classifiers.keys():
                print('Supervised learning: classification {0}'.format(est_key))
                estimator = sklearn_classifiers[est_key]
                score = cross_val_score(estimator, conn_coefs,
                                        classes, scoring='roc_auc',
                                        cv=[(train_index, test_index)])

                est_fit_train = estimator.fit(conn_coefs[train_index], classes[train_index])
                prediction_test = est_fit_train.predict_proba(conn_coefs[test_index])

                tmp=pd.DataFrame({'pred': prediction_test[:,1],
                                          'fd': mean_fd[test_index]})
                results['cor_fd_mean'].append(tmp.corr().values[0,1])

                tmp = pd.DataFrame({'pred': prediction_test[:, 1],
                                    'fd': num_fd[test_index]})
                results['cor_fd_num'].append(tmp.corr().values[0, 1])

                tmp = pd.DataFrame({'pred': prediction_test[:, 1],
                                    'fd': perc_fd[test_index]})
                results['cor_fd_perc'].append(tmp.corr().values[0, 1])

                results['atlas'].append(atlas)
                results['iter_shuffle_split'].append(index)
                results['measure'].append(measure)
                results['classifier'].append(est_key)
                results['dataset'].append('ABIDE')
                results['dimensionality'].append(dimensions[atlas])
                results['scores'].append(float(score))
                results['covariance_estimator'].append('LedoitWolf')
        all_results = pd.DataFrame(results)
        print(all_results[['classifier', 'measure', 'scores']].groupby(['measure', 'classifier']).mean())
        print(all_results[['classifier', 'measure', 'cor_fd_mean']].groupby(['classifier', 'measure']).mean())
    res = pd.DataFrame(results)
    # save classification scores per atlas
    this_atlas_dir = join(predictions_dir, atlas)
    if not os.path.exists(this_atlas_dir):
        os.makedirs(this_atlas_dir)
    res.to_csv(join(this_atlas_dir, 'scores.csv'))
all_results = pd.DataFrame(results)
all_results.to_csv('predictions_on_abide.csv')

# tspisak
print( all_results[['classifier', 'measure', 'scores']].groupby(['measure', 'classifier']).mean() )
