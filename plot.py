import matplotlib.pyplot as plt
from matplotlib import interactive
from scipy.interpolate import make_interp_spline, BSpline
import csv
import numpy as np
import sys

interactive(True)

moving_average_window = 100


def moving_average(a, n=moving_average_window):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plot_metadata = {
    'Hopper': {
        'baseline_fname': "env_Hopper-v2_model_mlp_baseline",
        'baseline_line_title': 'mlp',
        'location': './data/Hopper'
    },
    'Swimmer': {
        'baseline_fname': "env_Swimmer-v2_model_mlp_base",
        'baseline_line_title': 'mlp',
        'location': './data/Swimmer'
    },
    'Walker': {
        'baseline_fname': "env_Walker2d-v2_model_mlp",
        'baseline_line_title': 'mlp',
        'location': './data/Walker'
    },
    'HalfCheetah': {
        'baseline_fname': "env_HalfCheetah-v2_model_mlp",
        'baseline_line_title': 'mlp',
        'location': './data/HalfCheetah'
    }
}

plot_data_hopper = [
    {
        'title': 'Hopper Deep-SCN Depth Comparison',
        'filenames_no_csv': ["env_Hopper-v2_model_dscn_layers_1616_tanh_noplan_resid_blocks_2",
                             "env_Hopper-v2_model_dscn_layers_1616_tanh_blocks_4",
                             "env_Hopper-v2_model_dscn_layers_1616_tanh_noplan_blocks_8"],
        'line_legends': ['dscn_size_2', 'dscn_size_4', 'dscn_size_8'],
        'out_filename': 'Hopper_dscn_depth_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper Deep-SCN Residual Comparison',
        'filenames_no_csv': [
            "env_Hopper-v2_model_dscn_layers_1616_tanh_blocks_4",
            "env_Hopper-v2_model_dscn_size_1616_tanh_noplan_noresid_blocks_4"],
        'line_legends': ['dscn_size_4 w/ residual', 'dscn_size_4 w/o residual'],
        'out_filename': 'Hopper_dscn_residual_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper GRU Hidden Size Comparisons',
        'filenames_no_csv': [
            "env_Hopper-v2_model_gru_size_16_tanh_sigmoidgate_bias",
            "env_Hopper-v2_model_gru_size_32_tanh_sigmoidgate_bias",
            "env_Hopper-v2_model_gru_size_64_tanh_sigmoidgate_bias"],
        'line_legends': ['GRU_16', 'GRU_32', 'GRU_64'],
        'out_filename': 'Hopper_GRU_hidden_size_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper Locomotor Size Comparison',
        'filenames_no_csv': ["env_Hopper-v2_model_ln_size_8_sin",
                             "env_Hopper-v2_model_ln_size_16_sin",
                             "env_Hopper-v2_model_ln_size_32_sin"],
        'line_legends': ['ln_size_8', 'ln_size_16 (baseline)', 'ln_size_32'],
        'out_filename': 'Hopper_loco_size_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper Locomotor Function Comparison',
        'filenames_no_csv': ["env_Hopper-v2_model_ln_size_16_cos",
                             "env_Hopper-v2_model_ln_size_16_sin",
                             "env_Hopper-v2_model_ln_size_16_tan"],
        'line_legends': ['ln_cos', 'ln_sin (baseline)', 'ln_tan'],
        'out_filename': 'Hopper_loco_func_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper LSTM',
        'filenames_no_csv': ["env_Hopper-v2_model_lstm_tanh_sigmoidgate_bias"],
        'line_legends': ['lstm'],
        'out_filename': 'Hopper_lstm',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper MLP',
        'filenames_no_csv': ["env_Hopper-v2_model_mlp_baseline"],
        'line_legends': ['mlp'],
        'out_filename': 'Hopper_mlp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper RNN Hidden Units Comparison',
        'filenames_no_csv': ["env_Hopper-v2_model_rnn_size_16_tanh_biased",
                             "env_Hopper-v2_model_rnn_size_32_tanh_bias",
                             "env_Hopper-v2_model_rnn_hidden_64_tanh_bias"],
        'line_legends': ['rnn-16', 'rnn-32', 'rnn-64'],
        'out_filename': 'Hopper_rnn_units_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper Parallel-MLP',
        'filenames_no_csv': ["env_Hopper-v2_model_pmlp_layers_1616_tanh"],
        'line_legends': ['pmlp-16x16'],
        'out_filename': 'Hopper_pmlp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper SCN Activations Comparison',
        'filenames_no_csv': [
            "env_Hopper-v2_model_scn_layers_1616_sinc_noplan",
            "env_Hopper-v2_model_scn_layers_1616_tanh_noplan",
            "env_Hopper-v2_model_scn_layer_1616_sin_noplan"
        ],
        'line_legends': ['scn_sinc', 'scn_tanh (baseline)', 'scn_sin'],
        'out_filename': 'Hopper_scn_activations_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper SCN Hidden Size Comparison',
        'filenames_no_csv': [
            "env_Hopper-v2_model_scn_layers_1616_tanh_noplan",
            "env_Hopper-v2_model_scn_layers_3232_tanh_noplan",
            "env_Hopper-v2_model_scn_layers_6464_tanh_noplan"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_32x32', 'scn_tanh_64x64'],
        'out_filename': 'Hopper_scn_hidden_size_comp',
        'metadata':plot_metadata['Hopper']
    },
    {
        'title': 'Hopper SCN Depth Comparison',
        'filenames_no_csv': [
            "env_Hopper-v2_model_scn_layers_1616_tanh_noplan",
            "env_Hopper-v2_model_scn_layers_16161616_tanh_noplan"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_16x16x16x16'],
        'out_filename': 'Hopper_scn_depth_comp',
        'metadata':plot_metadata['Hopper']
    },
]

plot_data_swimmer = [
    {
        'title': 'Swimmer Deep-SCN Depth Comparison',
        'filenames_no_csv': ["env_Swimmer-v2_model_dscn_2",
                             "env_Swimmer-v2_model_dscn_base",
                             "env_Swimmer-v2_model_dscn_8"],
        'line_legends': ['dscn_size_2', 'dscn_size_4', 'dscn_size_8'],
        'out_filename': 'Swimmer_dscn_depth_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer Deep-SCN Residual Comparison',
        'filenames_no_csv': ["env_Swimmer-v2_model_dscn_base",
                             "env_Swimmer-v2_model_dscn_nores"],
        'line_legends': ['dscn_size_4 w/ residual', 'dscn_size_4 w/o residual'],
        'out_filename': 'Swimmer_dscn_residual_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer GRU Hidden Size Comparisons',
        'filenames_no_csv': [
            "env_Swimmer-v2_model_gru_16",
            "env_Swimmer-v2_model_gru_32",
            "env_Swimmer-v2_model_gru_base"],
        'line_legends': ['GRU_16', 'GRU_32', 'GRU_64'],
        'out_filename': 'Swimmer_GRU_hidden_size_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer Locomotor Size Comparison',
        'filenames_no_csv': ["env_Swimmer-v2_model_ln_8",
                             "env_Swimmer-v2_model_ln_base",
                             "env_Swimmer-v2_model_ln_32"],
        'line_legends': ['ln_size_8', 'ln_size_16 (baseline)', 'ln_size_32'],
        'out_filename': 'Swimmer_loco_size_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer Locomotor Function Comparison',
        'filenames_no_csv': ["env_Swimmer-v2_model_ln_cos",
                             "env_Swimmer-v2_model_ln_base",
                             "env_Swimmer-v2_model_ln_tan"],
        'line_legends': ['ln_cos', 'ln_sin (baseline)', 'ln_tan'],
        'out_filename': 'Swimmer_loco_func_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer LSTM',
        'filenames_no_csv': ["env_Swimmer-v2_model_lstm_base"],
        'line_legends': ['lstm'],
        'out_filename': 'Swimmer_lstm',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer MLP',
        'filenames_no_csv': ["env_Swimmer-v2_model_mlp_base"],
        'line_legends': ['mlp'],
        'out_filename': 'Swimmer_mlp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer RNN Hidden Units Comparison',
        'filenames_no_csv': ["env_Swimmer-v2_model_rnn_16",
                             "env_Swimmer-v2_model_rnn_32",
                             "env_Swimmer-v2_model_rnn_base"],
        'line_legends': ['rnn-16', 'rnn-32', 'rnn-64'],
        'out_filename': 'Swimmer_rnn_units_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer Parallel-MLP',
        'filenames_no_csv': ["env_Swimmer-v2_model_pmlp_base"],
        'line_legends': ['pmlp-16x16'],
        'out_filename': 'Swimmer_pmlp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer SCN Activations Comparison',
        'filenames_no_csv': [
            "env_Swimmer-v2_model_scn_sinc",
            "env_Swimmer-v2_model_scn_base",
            "env_Swimmer-v2_model_scn_sin"
        ],
        'line_legends': ['scn_sinc', 'scn_tanh (baseline)', 'scn_sin'],
        'out_filename': 'Swimmer_scn_activations_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer SCN Hidden Size Comparison',
        'filenames_no_csv': [
            "env_Swimmer-v2_model_scn_base",
            "env_Swimmer-v2_model_scn_32",
            "env_Swimmer-v2_model_scn_64"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_32x32', 'scn_tanh_64x64'],
        'out_filename': 'Swimmer_scn_hidden_size_comp',
        'metadata':plot_metadata['Swimmer']
    },
    {
        'title': 'Swimmer SCN Depth Comparison',
        'filenames_no_csv': [
            "env_Swimmer-v2_model_scn_base",
            "env_Swimmer-v2_model_scn_4x16"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_16x16x16x16'],
        'out_filename': 'Swimmer_scn_depth_comp',
        'metadata':plot_metadata['Swimmer']
    },
]

plot_data_walker = [
    {
        'title': 'Walker Deep-SCN Depth Comparison',
        'filenames_no_csv': ["env_Walker2d-v2_model_dscn_2",
                             "env_Walker2d-v2_model_dscn_base",
                             "env_Walker2d-v2_model_dscn_8"],
        'line_legends': ['dscn_size_2', 'dscn_size_4', 'dscn_size_8'],
        'out_filename': 'Walker_dscn_depth_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker Deep-SCN Residual Comparison',
        'filenames_no_csv': ["env_Walker2d-v2_model_dscn_base",
                             "env_Walker2d-v2_model_dscn_nores"],
        'line_legends': ['dscn_size_4 w/ residual', 'dscn_size_4 w/o residual'],
        'out_filename': 'Walker_dscn_residual_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker GRU Hidden Size Comparisons',
        'filenames_no_csv': [
            "env_Walker2d-v2_model_gru_16",
            "env_Walker2d-v2_model_gru_32",
            "env_Walker2d-v2_model_gru_base"],
        'line_legends': ['GRU_16', 'GRU_32', 'GRU_64'],
        'out_filename': 'Walker_GRU_hidden_size_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker Locomotor Size Comparison',
        'filenames_no_csv': ["env_Walker2d-v2_model_ln_8",
                             "env_Walker2d-v2_model_ln_base",
                             "env_Walker2d-v2_model_ln_32"],
        'line_legends': ['ln_size_8', 'ln_size_16 (baseline)', 'ln_size_32'],
        'out_filename': 'Walker_loco_size_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker Locomotor Function Comparison',
        'filenames_no_csv': ["env_Walker2d-v2_model_ln_cos",
                             "env_Walker2d-v2_model_ln_base",
                             "env_Walker2d-v2_model_ln_tan"],
        'line_legends': ['ln_cos', 'ln_sin (baseline)', 'ln_tan'],
        'out_filename': 'Walker_loco_func_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker LSTM',
        'filenames_no_csv': ["env_Walker2d-v2_model_lstm_base"],
        'line_legends': ['lstm'],
        'out_filename': 'Walker_lstm',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker MLP',
        'filenames_no_csv': ["env_Walker2d-v2_model_mlp"],
        'line_legends': ['mlp'],
        'out_filename': 'Walker_mlp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker RNN Hidden Units Comparison',
        'filenames_no_csv': ["env_Walker2d-v2_model_rnn_16",
                             "env_Walker2d-v2_model_rnn_32",
                             "env_Walker2d-v2_model_rnn_base"],
        'line_legends': ['rnn-16', 'rnn-32', 'rnn-64'],
        'out_filename': 'Walker_rnn_units_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker Parallel-MLP',
        'filenames_no_csv': ["env_Walker2d-v2_model_pmlp_base"],
        'line_legends': ['pmlp-16x16'],
        'out_filename': 'Walker_pmlp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker SCN Activations Comparison',
        'filenames_no_csv': [
            "env_Walker2d-v2_model_scn_sinc",
            "env_Walker2d-v2_model_scn_base",
            "env_Walker2d-v2_model_scn_sin"
        ],
        'line_legends': ['scn_sinc', 'scn_tanh (baseline)', 'scn_sin'],
        'out_filename': 'Walker_scn_activations_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker SCN Hidden Size Comparison',
        'filenames_no_csv': [
            "env_Walker2d-v2_model_scn_base",
            "env_Walker2d-v2_model_scn_32",
            "env_Walker2d-v2_model_scn_64"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_32x32', 'scn_tanh_64x64'],
        'out_filename': 'Walker_scn_hidden_size_comp',
        'metadata':plot_metadata['Walker']
    },
    {
        'title': 'Walker SCN Depth Comparison',
        'filenames_no_csv': [
            "env_Walker2d-v2_model_scn_base",
            "env_Walker2d-v2_model_scn_4x16"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_16x16x16x16'],
        'out_filename': 'Walker_scn_depth_comp',
        'metadata':plot_metadata['Walker']
    },
]

plot_data_halfcheetah = [
    {
        'title': 'HalfCheetah Deep-SCN Depth Comparison',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_dscn_16x16_tanh_2blk",
                             "env_HalfCheetah-v2_model_dscn_16x16_tanh_4blk",
                             "env_HalfCheetah-v2_model_dscn_16x16_tanh_8blk"],
        'line_legends': ['dscn_size_2', 'dscn_size_4', 'dscn_size_8'],
        'out_filename': 'HalfCheetah_dscn_depth_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah Deep-SCN Residual Comparison',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_dscn_16x16_tanh_4blk",
                             "env_HalfCheetah-v2_model_dscn_16x16_tanh_4_False"],
        'line_legends': ['dscn_size_4 w/ residual', 'dscn_size_4 w/o residual'],
        'out_filename': 'HalfCheetah_dscn_residual_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah GRU Hidden Size Comparisons',
        'filenames_no_csv': [
            "env_HalfCheetah-v2_model_gru_16_tanh",
            "env_HalfCheetah-v2_model_gru_32_tanh",
            "env_HalfCheetah-v2_model_gru_64_tanh"],
        'line_legends': ['GRU_16', 'GRU_32', 'GRU_64'],
        'out_filename': 'HalfCheetah_GRU_hidden_size_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah Locomotor Size Comparison',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_ln_8_sin",
                             "env_HalfCheetah-v2_model_ln_16_sin",
                             "env_HalfCheetah-v2_model_ln_32_sin"],
        'line_legends': ['ln_size_8', 'ln_size_16 (baseline)', 'ln_size_32'],
        'out_filename': 'HalfCheetah_loco_size_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah Locomotor Function Comparison',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_ln_16_cos",
                             "env_HalfCheetah-v2_model_ln_16_sin",
                             "env_HalfCheetah-v2_model_ln_16_tan"],
        'line_legends': ['ln_cos', 'ln_sin (baseline)', 'ln_tan'],
        'out_filename': 'HalfCheetah_loco_func_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah LSTM',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_lstm_tanh_tanh"],
        'line_legends': ['lstm'],
        'out_filename': 'HalfCheetah_lstm',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah MLP',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_mlp"],
        'line_legends': ['mlp'],
        'out_filename': 'HalfCheetah_mlp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah RNN Hidden Units Comparison',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_rnn_16_tanh",
                             "env_HalfCheetah-v2_model_rnn_32_tanh",
                             "env_HalfCheetah-v2_model_rnn_64_tanh"],
        'line_legends': ['rnn-16', 'rnn-32', 'rnn-64'],
        'out_filename': 'HalfCheetah_rnn_units_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah Parallel-MLP',
        'filenames_no_csv': ["env_HalfCheetah-v2_model_pmlp_64x64_tanh_4blk"],
        'line_legends': ['pmlp-16x16'],
        'out_filename': 'HalfCheetah_pmlp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah SCN Activations Comparison',
        'filenames_no_csv': [
            "env_HalfCheetah-v2_model_scn_16x16_sinc",
            "env_HalfCheetah-v2_model_scn",
            "env_HalfCheetah-v2_model_scn_16x16_sin"
        ],
        'line_legends': ['scn_sinc', 'scn_tanh (baseline)', 'scn_sin'],
        'out_filename': 'HalfCheetah_scn_activations_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah SCN Hidden Size Comparison',
        'filenames_no_csv': [
            "env_HalfCheetah-v2_model_scn",
            "env_HalfCheetah-v2_model_scn_32x32",
            "env_HalfCheetah-v2_model_scn_64x64_tanh"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_32x32', 'scn_tanh_64x64'],
        'out_filename': 'HalfCheetah_scn_hidden_size_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
    {
        'title': 'HalfCheetah SCN Depth Comparison',
        'filenames_no_csv': [
            "env_HalfCheetah-v2_model_scn",
            "env_HalfCheetah-v2_model_scn_16x16x16x16_tanh"
        ],
        'line_legends': ['scn_tanh_16x16 (baseline)', 'scn_tanh_16x16x16x16'],
        'out_filename': 'HalfCheetah_scn_depth_comp',
        'metadata':plot_metadata['HalfCheetah']
    },
]

plot_data = []
plot_data.extend(plot_data_hopper)
plot_data.extend(plot_data_swimmer)
plot_data.extend(plot_data_walker)
plot_data.extend(plot_data_halfcheetah)

for pdata in plot_data:
    title = pdata['title']
    filenames_no_csv = pdata['filenames_no_csv']
    line_legends = pdata['line_legends']
    out_filename = pdata['out_filename']
    location = pdata['metadata']['location']
    baseline_fname = pdata['metadata']['baseline_fname']
    baseline_line_title = pdata['metadata']['baseline_line_title']

    print('Generating plot for %s' % title)
    plt.figure()
    plt.xlim(0, 2000000)
    plt.xticks([x for x in range(0, 2000001, 250000)], [
               str(x/1000000) + 'M' for x in range(0, 2000001, 250000)])
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Reward')
    plt.title(title)

    if baseline_fname != '' and baseline_line_title != '':
        filenames_no_csv.insert(0, baseline_fname)
        line_legends.insert(0, baseline_line_title)

    for fname, line_title in zip(filenames_no_csv, line_legends):
        with open(location + '/' + fname + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x = []
            y = []
            for row in csv_reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
            x = np.array(x)
            y = np.array(y)
            print('lines read', len(x))
            y_moveavg = moving_average(y, moving_average_window)
            y_moveavg = np.append(
                y_moveavg, [y_moveavg[-1]] * (moving_average_window - 1))
            plt.plot(x, y_moveavg, label=line_title)

    print('Saving...')
    plt.legend()
    plt.savefig('{}.png'.format(out_filename))
    plt.close()
