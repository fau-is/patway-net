import torch
import os
import src.data as data
from src.main import time_step_blow_up
import copy
import numpy as np
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
baseline = "lstm_all_feat_seq"
model = torch.load(os.path.join("../model", f"model_{baseline}_{name}"), map_location=torch.device('cpu'))

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)
case = -1
file_format = "pdf"  # pdf, png

x_seqs_final = torch.from_numpy(x_seqs_final)
x_statics_final = torch.from_numpy(x_statics_final)
y_final = torch.from_numpy(y_final)

def similarity_kernel(original_input, perturbed_input, perturbed_interpretable_input, **kwargs):
    # kernel_width will be provided to attribute as a kwarg
    kernel_width = kwargs["kernel_width"]

    input_seq_flat = torch.reshape(original_input[0], (-1, 50 * 16))
    input_stat = original_input[1]
    input_all = torch.cat((input_seq_flat, input_stat), dim=1)

    l2_dist = torch.norm(input_all - perturbed_input)
    return torch.exp(- (l2_dist ** 2) / (kernel_width ** 2))


# Define sampling function
# This function samples in original input space
def perturb_func(original_input, **kwargs):
    # input_seq_flat = torch.reshape(original_input[0], (-1, 50 * 16))
    # input_stat = original_input[1]
    # input_all = torch.cat((input_seq_flat, input_stat), dim=1)

    # input_all + randn_like(input_all)
    # torch.randint(0, 1, (1, 823))
    # input_all + torch.rand((1, 823))
    # torch.rand((1, 823))

    rnd = torch.randint(0, x_seqs_final.shape[0], (1,))
    input_seq_flat = torch.reshape(x_seqs_final[rnd], (1, 50 * 16))
    input_all = torch.cat((input_seq_flat, x_statics_final[rnd]), dim=1)

    return input_all


# For this example, we are setting the interpretable input to
# match the model input, so the to_interp_rep_transform
# function simply returns the input. In most cases, the interpretable
# input will be different and may have a smaller feature set, so
# an appropriate transformation function should be provided.
def to_interp_transform(curr_sample, original_inp, **kwargs):
    return curr_sample

def wrapper(input, tupel_original_input):
    return model(tupel_original_input[0], tupel_original_input[1])

lime_attr = LimeBase(wrapper,
                     SkLearnLinearModel("linear_model.Ridge"),  # https://captum.ai/api/_modules/captum/_utils/models/linear_model/model.html
                     similarity_func=similarity_kernel,
                     perturb_func=perturb_func,
                     perturb_interpretable_space=False,
                     from_interp_rep_transform=None,
                     to_interp_rep_transform=to_interp_transform)

inputs = (torch.reshape(x_seqs_final[case], (1, 50, 16)), torch.reshape(x_statics_final[case], (1, 23)))
# inputs = (x_seqs_final, x_statics_final)

attr_coefs = lime_attr.attribute(inputs,
                                 baselines=None,
                                 target=None,  # as model returns a scalar value, no target index is necessary
                                 n_samples=50,
                                 additional_forward_args=(inputs,),
                                 kernel_width=1.1)

# todo: aggregate attributions
# todo: like global plot



print(0)
