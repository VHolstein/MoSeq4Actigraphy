from ActigraphyLoader.MoSeqDataLoader import ActigraphyDataLoader
import jax, jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from keypoint_moseq.util import *
from keypoint_moseq.gibbs import *
from keypoint_moseq.initialize import *

# load data
loader = ActigraphyDataLoader(base_dir='/data/sbdp/PHOENIX/GENERAL/MCLEAN_TEST')
x, mask, keys = loader.load_data(subjects=['HRG20'], big_array=True)
x, mask = jnp.array(x), jnp.array(mask)  # jax device arrays

# set hyper-params
latent_dim = x.shape[-1] # number of PCs
num_states = 100         # max number of states
nlags = 3                # number of lags for AR dynamics

trans_hypparams = {
    'gamma': 1e3,
    'alpha': 5.7,
    'kappa': 2e5,
    'num_states':num_states}

init_hypparams = {
    'alpha': 5.7,
    'kappa': 2e5,
    'num_states':num_states}

ar_hypparams = {
    'nu_0': latent_dim+200,
    'S_0': 10*jnp.eye(latent_dim),
    'M_0': jnp.pad(jnp.eye(latent_dim),((0,0),((nlags-1)*latent_dim,1))),
    'K_0': 0.1*jnp.eye(latent_dim*nlags+1),
    'num_states':num_states,
    'nlags':nlags}

# initialize model
key = jr.PRNGKey(0)
data = {'mask':mask}
states = {'x':x}
params = {}

params['pi'] = initial_transitions(key, **init_hypparams)
params['Ab'],params['Q']= initial_ar_params(key, **ar_hypparams)
states['z'] = resample_stateseqs(key, **data, **states, **params)

num_iters = 500
plot_iters = 10
keys = jr.split(key, num_iters)

for i in tqdm.trange(num_iters):
    params['betas'], params['pi'] = resample_hdp_transitions(keys[i], **data, **states, **params, **trans_hypparams)
    params['Ab'], params['Q'] = resample_ar_params(keys[i], **data, **states, **params, **ar_hypparams)
    states['z'] = resample_stateseqs(keys[i], **data, **states, **params)

    if i % plot_iters == 0:
        usage, durations = stateseq_stats(states['z'], mask)
        fig, axs = plt.subplots(1, 2)
        axs[0].bar(range(len(usage)), sorted(usage, reverse=True))
        axs[0].set_ylabel('Syllable usage')
        axs[0].set_xlabel('Syllable rank')
        axs[1].hist(durations, range=(0, 30), bins=30, density=True)
        axs[1].axvline(np.median(durations), linestyle='--', c='k')
        axs[1].set_xlabel('Syllable duration (frames)')
        axs[1].set_ylabel('Probability density')
        fig.set_size_inches((8, 2))
        plt.suptitle('Iteration {}, Median duration = {}'.format(i, np.median(durations)))
        plt.show()
