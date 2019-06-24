"""
This is a modified version of the scrip "run_prediction_on_abide.py" from Dadi et al. 2019.
https://github.com/KamalakerDadi/benchmark_rsfMRI_prediction

Modifications are (mostly) labeled with #tspisak.
"""


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

#tspisak
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def _get_paths(phenotypic, atlas, timeseries_dir):
    """
    """
    timeseries = []
    IDs_subject = []
    diagnosis = []
    subject_ids = phenotypic['SUB_ID']
    # tspisak
    mean_fd = []
    num_fd = []
    perc_fd = []
    for index, subject_id in enumerate(subject_ids):
        this_pheno = phenotypic[phenotypic['SUB_ID'] == subject_id]
        this_timeseries = join(timeseries_dir, atlas,
                               str(subject_id) + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            IDs_subject.append(subject_id)
            diagnosis.append(this_pheno['DX_GROUP'].values[0])
            # tspisak
            mean_fd.append(this_pheno['func_mean_fd'].values[0])
            num_fd.append(this_pheno['func_num_fd'].values[0])
            perc_fd.append(this_pheno['func_perc_fd'].values[0])
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

#tspisak
#atlases = ['AAL', 'HarvardOxford', 'BASC/networks', 'BASC/regions',
#           'Power', 'MODL/64', 'MODL/128']
atlases = ['BASC/networks']

dimensions = {'AAL': 116,
              'HarvardOxford': 118,
              'BASC/networks': 122,
              'BASC/regions': 122,
              'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

#tspisak
# prepare dictionary for saving results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'covariance_estimator', 'dimensionality', 'cor_fd_mean', 'cor_fd_num', 'cor_fd_perc',
           'disccor_fd_mean', 'disccor_fd_num', 'disccor_fd_perc', 'diff_fd_mean', 'diff_fd_num', 'diff_fd_perc'
           ]
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

pheno_dir = 'Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(pheno_dir)

# Connectomes per measure
from connectome_matrices import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
measures = ['correlation', 'partial correlation', 'tangent']

from my_estimators import sklearn_classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.5,
                            random_state=0)

for atlas in atlases:
    print("Running predictions: with atlas: {0}".format(atlas))
    # tspisak

    timeseries, diagnosis, IDs_subject, mean_fd, num_fd, perc_fd = _get_paths(phenotypic, atlas, timeseries_dir)


    _, classes = np.unique(diagnosis, return_inverse=True)
    #print(classes)
    #print(mean_fd)
    print(roc_auc_score(classes, [i * -1.0 for i in mean_fd]))

    out=pd.DataFrame({
        "diagnosis" : classes,
        "mean_fd" : mean_fd
        }
    )

    out.to_csv("pheno_abide.csv")

    _, classes = np.unique(diagnosis, return_inverse=True)
    iter_for_prediction = cv.split(timeseries, classes)

    # tspisak
    print("Correlation of Class and meanFD, numFD, percFD:")
    print(len(diagnosis))
    tmp = pd.DataFrame({'class': classes,
                        'fd': mean_fd})
    print(tmp.corr().values[0, 1])

    tmp = pd.DataFrame({'class': classes,
                        'fd': num_fd})
    print(tmp.corr().values[0, 1])

    tmp = pd.DataFrame({'class': classes,
                        'fd': perc_fd})
    print(tmp.corr().values[0, 1])

    print("MeanFD, numFD, percFD class differences:")
    cls=np.array(classes)

    mean_fd = np.array(mean_fd)
    print(np.mean(mean_fd[cls==1])-np.mean(mean_fd[cls==0]))

    num_fd = np.array(num_fd)
    print(np.mean(num_fd[cls == 1]) - np.mean(num_fd[cls == 0]))

    perc_fd = np.array(perc_fd)
    print(np.mean(perc_fd[cls == 1]) - np.mean(perc_fd[cls == 0]))

    ##################

    for index, (train_index, test_index) in enumerate(iter_for_prediction):
        print("[Cross-validation] Running fold: {0}".format(index))
        for measure in measures:
            print("[Connectivity measure] kind='{0}'".format(measure))
            connections = ConnectivityMeasure(
                cov_estimator=LedoitWolf(assume_centered=True),
                kind=measure)
            conn_coefs = connections.fit_transform(timeseries)
            # tspisak ToDo: pipeline here?

            for est_key in sklearn_classifiers.keys():
                print('Supervised learning: classification {0}'.format(est_key))
                estimator = sklearn_classifiers[est_key]
                # tspisak
                #score = cross_val_score(estimator, conn_coefs,
                #                        classes, scoring='roc_auc',
                #                        cv=[(train_index, test_index)])

                ##################
                # tspisak
                est_fit_train = estimator.fit(conn_coefs[train_index], classes[train_index])
                prediction_test = est_fit_train.predict_proba(conn_coefs[test_index])

                # equivalent to that commented out above
                score = roc_auc_score(classes[test_index], prediction_test[:,1])

                tmp=pd.DataFrame({'pred': prediction_test[:,0],
                                          'fd': mean_fd[test_index]})
                results['cor_fd_mean'].append(tmp.corr().values[0,1])

                tmp = pd.DataFrame({'pred': prediction_test[:, 0],
                                    'fd': num_fd[test_index]})
                results['cor_fd_num'].append(tmp.corr().values[0, 1])

                tmp = pd.DataFrame({'pred': prediction_test[:, 0],
                                    'fd': perc_fd[test_index]})
                results['cor_fd_perc'].append(tmp.corr().values[0, 1])

                #################
                # tspisak
                prediction_test = est_fit_train.predict(conn_coefs[test_index])

                tmp = pd.DataFrame({'pred': prediction_test,
                                    'fd': mean_fd[test_index]})
                results['disccor_fd_mean'].append(tmp.corr().values[0, 1])

                tmp = pd.DataFrame({'pred': prediction_test,
                                    'fd': num_fd[test_index]})
                results['disccor_fd_num'].append(tmp.corr().values[0, 1])

                tmp = pd.DataFrame({'pred': prediction_test,
                                    'fd': perc_fd[test_index]})
                results['disccor_fd_perc'].append(tmp.corr().values[0, 1])

                ###################
                # tspisak
                prediction_test = np.array(prediction_test)

                mean_fd_test = np.array(mean_fd[test_index])
                results['diff_fd_mean'].append(
                    np.mean(mean_fd_test[prediction_test==1])-np.mean(mean_fd_test[prediction_test==0]))

                num_fd_test = np.array(num_fd[test_index])
                results['diff_fd_num'].append(
                    np.mean(num_fd_test[prediction_test == 1]) - np.mean(num_fd_test[prediction_test == 0]))

                perc_fd_test = np.array(perc_fd[test_index])
                results['diff_fd_perc'].append(
                    np.mean(perc_fd_test[prediction_test == 1]) - np.mean(perc_fd_test[prediction_test == 0]))

                ###################

                results['atlas'].append(atlas)
                results['iter_shuffle_split'].append(index)
                results['measure'].append(measure)
                results['classifier'].append(est_key)
                results['dataset'].append('ABIDE')
                results['dimensionality'].append(dimensions[atlas])
                results['scores'].append(float(score))
                results['covariance_estimator'].append('LedoitWolf')
        all_results = pd.DataFrame(results)
        # tspisak (print in all iteration to monitor progress)
        print(all_results[['classifier', 'measure', 'scores']].groupby(['measure', 'classifier']).mean())
        print(all_results[['classifier', 'measure', 'cor_fd_mean']].groupby(['classifier', 'measure']).mean())
    res = pd.DataFrame(results)
    # save classification scores per atlas
    this_atlas_dir = join(predictions_dir, atlas)
    # tspisak
    if not os.path.exists(this_atlas_dir):
        os.makedirs(this_atlas_dir)
    res.to_csv(join(this_atlas_dir, 'scores_nopipe_noreg.csv'))
all_results = pd.DataFrame(results)
all_results.to_csv('predictions_on_abide-test0.5.csv')

#tspisak
print( all_results[['classifier', 'measure', 'scores']].groupby(['measure', 'classifier']).mean() )
