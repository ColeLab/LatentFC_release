import os,sys
import numpy as np
import pandas as pd

## Define Functions
def get_data(dat, subj=None, subjs=None, loo_state=None, states=None, connection=None, task_only=False):
    n_subjs, n_states, n_connections = dat.shape
    if type(subj) != type(None):
        i = [subjs.index(subj)]
    else:
        i = np.arange(n_subjs).tolist()
    j = np.arange(n_states, dtype=int).tolist()
    if loo_state:
        loo_ind = states.index(loo_state)
        j.remove(loo_ind)
        if len(j) != n_states - 1:
            raise InputError('Failed to remove %s' % (loo_state))
    if task_only:
        rest_ind = states.index('rest')
        j.remove(rest_ind)
    if type(connection) == int:
        k = [connection]
    else:
        k = np.arange(n_connections).tolist()
    return np.squeeze(dat[np.ix_(i,j,k)])

def calc_weighted_sum_of_factor_load(load_df, dat, subj, subjs, loo_state=None, states=None, connection=None, task_only=False):
    subj_dat = get_data(dat=dat, subj=subj, subjs=subjs, loo_state=loo_state, states=states, connection=connection, task_only=task_only)
    if type(connection) == int:
        n_states = len(subj_dat)
        weights = np.multiply(load_df.loc[connection], subj_dat)
        #return np.sum(weights)/n_states
        return np.array(sums)/np.sum(load_df.loc[k])
    else:
        sums = []
        sums2 = []
        n_states, n_connections = subj_dat.shape
        for k in np.arange(n_connections):
            weights = np.multiply(load_df.loc[k], subj_dat[:,k])
            sums.append(np.nansum(weights) / np.nansum(load_df.loc[k]))
            sums2.append(np.nansum(weights) / n_states)
        #return np.array(sums)/n_states
        return np.array(sums), np.array(sums2)

def run_all(load_df, score_df, avg_df, dat, subjs, loo_state, states, connection, task_only, savedir):
    for subj in subjs:        
        tmp_sums, tmp_sums2 = calc_weighted_sum_of_factor_load(load_df=load_df, dat=dat, subj=subj, subjs=subjs, loo_state=loo_state, states=states, connection=connection, task_only=task_only)
        out_mat = np.zeros(shape=(360,360))
        out_mat[np.triu_indices(360,1)] = tmp_sums
        outf = '%s/%s_latent_connectivity_mat.npy' % (savedir, subj)
        np.save(outf, out_mat)
        out_mat = np.zeros(shape=(360,360))
        out_mat[np.triu_indices(360,1)] = tmp_sums2
        outf = '%s/%s_latent_connectivity_mat2.npy' % (savedir, subj)
        np.save(outf, out_mat)
        print('Wrote component latent fc %s' % (outf))
        
        tmp_sums = score_df.loc[int(subj)]
        out_mat = np.zeros(shape=(360,360))
        out_mat[np.triu_indices(360,1)] = tmp_sums
        outf = '%s/%s_score_connectivity_mat.npy' % (savedir, subj)
        np.save(outf, out_mat)
        print('Wrote score fc %s' % (outf))
        
        tmp_sums = avg_df.loc[int(subj)]
        out_mat = np.zeros(shape=(360,360))
        out_mat[np.triu_indices(360,1)] = tmp_sums
        outf = '%s/%s_avg_connectivity_mat.npy' % (savedir, subj)
        np.save(outf, out_mat)
        print('Wrote avg fc %s' % (outf))


## Create Matrices
subjs_group = sys.argv[1]
data_group = sys.argv[2]
control_tp = sys.argv[3]
task_only = False
connection = None

# Setup directories
homedir = '/projects/f_mc1689_1/LatentFC/data/hcpPostProcCiric'
adjmatdir = '%s/Results/adjmat/correlation' % (homedir)
datadir = '%s/Results/fa/correlation/%s/%s' % (homedir, subjs_group, data_group)
if control_tp == '1':
    adjmatdir = '%s/control_tp' % (adjmatdir)
    datadir = '%s/control_tp' % (datadir)


# Load data
subjsf = '%s/%s_subjects.npy' % (homedir, subjs_group)
subjs = np.array(np.load(subjsf), dtype='U').tolist()
states = np.array(np.load('%s/states.npy' % (homedir)), dtype='U').tolist()
dat = np.load('%s/%s_data_summary_subj_by_state_by_connectivity.npy' % (adjmatdir, subjs_group))
fshr_dat = np.arctanh(dat)

if data_group == 'overall':
    loo_state = None
    load_df = pd.read_csv('%s/factor_loadings_minres_tenBerge.csv' % (datadir), index_col=0)
    score_df = pd.read_csv('%s/factor_scores_minres_tenBerge.csv' % (datadir), index_col=0)
    avg_df = pd.read_csv('%s/avg_val.csv' % (datadir), index_col=0)
    savedir = '%s/latent_fc' % (datadir)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    print('Saving files to %s' % (savedir))
    run_all(load_df=load_df, score_df=score_df, avg_df=avg_df, dat=fshr_dat, subjs=subjs, loo_state=loo_state, states=states, connection=connection, task_only=task_only, savedir=savedir)

elif data_group == 'loo':
    for loo_state in states:
        if task_only:
            if loo_state == 'rest':
                continue
        load_df = pd.read_csv('%s/%s/factor_loadings_minres_tenBerge.csv' % (datadir, loo_state), index_col=0)
        score_df = pd.read_csv('%s/%s/factor_scores_minres_tenBerge.csv' % (datadir, loo_state), index_col=0)
        avg_df = pd.read_csv('%s/%s/avg_val.csv' % (datadir, loo_state), index_col=0)
        savedir = '%s/latent_fc/%s' % (datadir, loo_state)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        print('Saving files to %s' % (savedir))
        run_all(load_df=load_df, score_df=score_df, avg_df=avg_df, dat=fshr_dat, subjs=subjs, loo_state=loo_state, states=states, connection=connection, task_only=task_only, savedir=savedir)