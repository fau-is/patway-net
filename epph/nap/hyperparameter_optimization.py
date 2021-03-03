from sklearn.model_selection import train_test_split


def create_data(args, event_log, preprocessor, training_cases):
    """
    Generates data to train and test/evaluate a model during hyper-parameter optimization (hpo) with Optuna.
    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    event_log : list of dicts, where single dict represents a case
        pm4py.objects.log.log.EventLog object representing an event log.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    cases_of_fold : list of dicts, where single dict represents a case
        Cases of the current fold.

    Returns
    -------
    x_train : ndarray, shape[S, T, F], S is number of samples, T is number of time steps, F is number of features.
        The features of the training set.
    y_train : ndarray, shape[S, T], S is number of samples, T is number of time steps.
        The labels of the training set.
    x_test : ndarray, shape[S, T, F], S is number of samples, T is number of time steps, F is number of features.
        The features of the test set.
    y_test : ndarray, shape[S, T], S is number of samples, T is number of time steps.
        The labels of the test set.

    """

    train_indices, test_indices = train_test_split_for_hyperparameter_optimization(args, training_cases)
    train_cases, test_cases = retrieve_train_test_instances(training_cases, train_indices, test_indices)

    train_subseqs = preprocessor.get_subsequences_of_cases(train_cases)
    test_subseqs = preprocessor.get_subsequences_of_cases(test_cases)

    x_train = preprocessor.get_features_tensor(args, event_log, train_subseqs)
    x_test = preprocessor.get_features_tensor(args, event_log, test_subseqs)
    y_train = preprocessor.get_labels_tensor(args, train_cases)
    y_test = preprocessor.get_labels_tensor(args, test_cases)

    return x_train, x_test, y_train, y_test


def train_test_split_for_hyperparameter_optimization(args, cases):
    """
    Executes a split-validation and retrieves indices of training and test cases for hpo.
    Parameters
    ----------
    cases : list of dicts, where single dict represents a case
        Cases of the training set.
    Returns
    -------
    hpo_train_indices[0] : list of ints
        Indices of training cases for hpo.
    hpo_test_indices[0] : list of ints
        Indices of test cases for hpo.
    """

    indices = [index for index in range(0, len(cases))]

    if args.shuffle:
        if args.seed:
            seed = args.seed_val
        else:
            seed = None

        x_train_indices, x_test_indices, y_train_indices, y_test_indices = train_test_split(
                indices, indices,
                train_size=args.split_rate_train_hpo,
                shuffle=args.shuffle,
                random_state=seed)

        train_indices = x_train_indices
        test_indices = x_test_indices
    else:
        train_indices = indices[:int(len(indices) * args.split_rate_train_hpo)]
        test_indices = indices[int(len(indices) * args.split_rate_train_hpo):]

    return train_indices, test_indices


def retrieve_train_test_instances(cases, train_indices, test_indices):
    """
    Retrieves training and test cases from indices for hpo.
    Parameters
    ----------
    cases : list of dicts, where single dict represents a case
        Cases of the training set of the event log.
    train_indices : list of ints
        Indices of training cases for hpo.
    test_indices : list of ints
        Indices of test cases for hpo.
    Returns
    -------
    train_cases : list of dicts, where single dict represents a case
        Training cases for hpo.
    test_cases : list of dicts, where single dict represents a case
        Test cases for hpo.
    """

    train_cases = []
    test_cases = []

    for idx in train_indices:
        train_cases.append(cases[idx])

    for idx in test_indices:
        test_cases.append(cases[idx])

    return train_cases, test_cases
