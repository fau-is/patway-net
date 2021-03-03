import shap
import epph.utils as utils
import numpy


def calc_relevance_score_prefix(args, preprocessor, event_log, case, prefix_size, background, model, pred_act_str,
                                prefix_words):
    """
    Calculates relevance scores for all event attributes in a subsequence/prefix of a case.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    case : dict
        A case from the event log.
    prefix_size : int
        Size of a prefix to be cropped out of a whole case.
    background : ndarray with shape [num training cases, max case length, num features]
        Background examples used to plug in to approximate a feature being missing.
    model : tensorflow.python.keras.engine.functional.Functional object
        Deep neural network used for next activity prediction - to be explained.
    pred_act_str : str
        Predicted next activity.
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.

    Returns
    -------
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    """
    subseq_case = case[:prefix_size]
    features_tensor = preprocessor.get_features_tensor(args, event_log, [subseq_case])

    # Perform shap
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(features_tensor)

    pred_act_id = preprocessor.activity['strings_to_ids'][pred_act_str]
    R_words, R_words_context = create_heatmap_data(preprocessor, event_log, subseq_case, shap_values[pred_act_id][0],
                                                   prefix_words, print_relevance_scores=True)

    return R_words, R_words_context


def get_background_data(args, event_log, preprocessor, train_indices):
    """
    Selects neighborhood data to be used for

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    train_indices : list of ints
        Indices of training cases from event log.

    Returns
    -------
    ndarray with shape [num training cases, max case length, num features]
            Background examples used to plug in to approximate a feature being missing.

    """
    training_cases = preprocessor.get_subset_cases(args, event_log, train_indices)
    training_cases_sample = training_cases[:args.shap_num_samples]
    sample_subseqs = preprocessor.get_subsequences_of_cases(training_cases_sample)
    background_tensor = preprocessor.get_features_tensor(args, event_log, sample_subseqs)
    return background_tensor


def create_heatmap_data(preprocessor, event_log, subseq_case, shap_values, prefix_words, print_relevance_scores=False):
    """
    Prepares explanation data for heatmap visualization.

    Parameters
    ----------
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    subseq_case : list of dicts, where single dict represents an event
        Subsequence / subset of a case whose length is prefix_size.
    shap_values : ndarray with shape [max case length, num features]
        Computed shap values for a prefix.
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.
    print_relevance_scores : bool
        If set to "True", prints relevance scores.

    Returns
    -------
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    """
    max_case_len = preprocessor.get_max_case_length(event_log)
    act_enc_length = preprocessor.activity['label_length']
    context_enc_lengths = preprocessor.context['encoding_lengths']

    R_act = {}
    R_context = {}

    prefix_size = len(subseq_case)
    shap_prefix = shap_values[-prefix_size:]
    for timestep, shap_event in enumerate(shap_prefix):
        i_val = 0
        act_relevance = 0
        context_relevance = 0
        # Activity
        while i_val < act_enc_length:
            act_relevance += shap_event[i_val]
            i_val += 1
        R_act[timestep] = act_relevance
        # Context
        for context_name, context_length in context_enc_lengths.items():
            i_end = i_val + context_length
            while i_val < i_end:
                context_relevance += shap_event[i_val]
                i_val += 1
            if timestep not in R_context:
                R_context[timestep] = {}
            R_context[timestep][context_name] = context_relevance

    # R_words
    R_words = numpy.zeros(max_case_len)
    for timestep, rel_scr in R_act.items():
        idx = (max_case_len - 1) - timestep
        R_words[idx] = rel_scr

    # R_words_context
    R_words_context = {}
    context_attributes = preprocessor.get_context_attributes()
    for attr in context_attributes:
        R_words_context[attr] = numpy.zeros(max_case_len)

    for timestep, t_dict in R_context.items():
        idx = (max_case_len - 1) - timestep
        for attr in t_dict.keys():
            R_words_context[attr][idx] = t_dict[attr]

    if print_relevance_scores:
        print("Relevance scores:")
        for timestep, event_attributes in enumerate(prefix_words):
            # activity
            utils.ll_print("'%s' : %s, " % (event_attributes[0], R_act[timestep]))
            # context
            for i_context, context_attr in enumerate(context_attributes):
                if i_context == len(context_attributes)-1 and timestep == len(prefix_words)-1:
                    utils.ll_print("'%s' : %s " % (event_attributes[i_context + 1], R_context[timestep][context_attr]))
                else:
                    utils.ll_print("'%s' : %s, " % (event_attributes[i_context + 1], R_context[timestep][context_attr]))
        print("\n")

    return R_words, R_words_context
