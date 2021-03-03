from __future__ import print_function, division
import tensorflow as tf
from datetime import datetime
import epph.utils as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import optuna
from optuna.integration import KerasPruningCallback
import epph.nap.hyperparameter_optimization as hpo
import os


# used access during hpo
global train_cases_
global event_log_
global preprocessor_
global args_


def train(args, preprocessor, event_log, train_indices, measures):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # eager execution to avoid retracing with tensorflow
    tf.config.run_functions_eagerly(args.eager_execution)

    train_cases = preprocessor.get_subset_cases(args, event_log, train_indices)

    best_model_id = -1
    print('Create machine learning model ... \n')

    if args.hpo:

        # set global variables
        global train_cases_
        global event_log_
        global preprocessor_
        global args_
        train_cases_ = train_cases
        event_log_ = event_log
        preprocessor_ = preprocessor
        args_ = args

        if args.seed:
            sampler = optuna.samplers.TPESampler(seed=args.seed_val)  # make the sampler behave in a deterministic way.
        else:
            sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)

        start_training_time = datetime.now()

        if args.classifier == "LSTM":
            # Deep Neural Network
            study.optimize(train_lstm_hpo, n_trials=args.hpo_eval_runs)

        if args.classifier == "RF":
            # Random Forest
            study.optimize(train_rf_hpo, n_trials=args.hpo_eval_runs)

        if args.classifier == "DT":
            # Decision Tree
            study.optimize(train_dt_hpo, n_trials=args.hpo_eval_runs)

        if args.classifier == "RIP":
            study.optimize(train_rip_hpo, n_trials=args.hpo_eval_runs)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        utils.add_to_file(args, "hyper_params", trial)
        best_model_id = study.best_trial.number

        # Remove all models except the best performing one
        for model_id in range(args.hpo_eval_runs):
            if model_id == best_model_id:
                continue
            else:
                
                try:  # if model is not stored
                    os.remove(utils.get_model_dir(args, model_id))
                except ValueError:
                    pass

        # Set best model to default
        os.rename(utils.get_model_dir(args, best_model_id), utils.get_model_dir(args, 0))
        best_model_id = 0

    else:

        # prepare data
        train_subseq_cases = preprocessor.get_subsequences_of_cases(train_cases)
        features_tensor = preprocessor.get_features_tensor(args, event_log, train_subseq_cases)
        labels_tensor = preprocessor.get_labels_tensor(args, train_cases)

        start_training_time = datetime.now()

        if args.classifier == "LSTM":
            # LSTM
            train_lstm(args, preprocessor, event_log, features_tensor, labels_tensor)

        if args.classifier == "RF":
            # Random Forest
            train_rf(args, features_tensor, labels_tensor)

        if args.classifier == "DT":
            # Decision Tree
            train_dt(args, features_tensor, labels_tensor)




    training_time = datetime.now() - start_training_time
    measures["training_time_seconds"] = training_time.total_seconds()
    return best_model_id


def train_lstm_hpo(trial):
    x_train, x_test, y_train, y_test = hpo.create_data(args_, event_log_, preprocessor_, train_cases_)

    max_case_len = preprocessor_.get_max_case_length(event_log_)
    num_features = preprocessor_.get_num_features()
    num_activities = preprocessor_.get_num_activities()

    # Bidirectional LSTM
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(max_case_len, num_features), name='input_layer')

    # Hidden layer
    hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=100,
        activation=trial.suggest_categorical('hl_activation', args_.hpo_activation),
        kernel_initializer=trial.suggest_categorical('hl_kernel_initializer', args_.hpo_kernel_initializer),
        return_sequences=False,
        dropout=trial.suggest_discrete_uniform('hl_drop_out', 0.1, 0.5, 0.1)))(input_layer)

    # Output layer
    output_layer = tf.keras.layers.Dense(num_activities,
                                         activation='softmax',
                                         name='output_layer',
                                         kernel_initializer=trial.suggest_categorical('ol_kernel_initializer',
                                                                                      args_.hpo_kernel_initializer))(
        hidden_layer)

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(loss={'output_layer': 'categorical_crossentropy'},
                  optimizer=trial.suggest_categorical('optimizer', args_.hpo_optimizer),
                  metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(utils.get_model_dir(args_, trial.number),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)
    model.summary()
    model.fit(x_train, {'output_layer': y_train},
              validation_split=args_.val_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer, KerasPruningCallback(trial, "val_accuracy")],
              batch_size=args_.batch_size_train,
              epochs=args_.dnn_num_epochs)

    # optimize parameters based on accuracy (currently no other metrics considered)
    score = model.evaluate(x_test, y_test, verbose=0)

    if score[1] > 1.0:
        return 0.0
    else:
        return score[1]


def train_rf_hpo(trial):
    x_train, x_test, y_train, y_test = hpo.create_data(args_, event_log_, preprocessor_, train_cases_)

    model = RandomForestClassifier(n_jobs=-1,  # use all processors
                                   random_state=args_.seed,
                                   n_estimators=trial.suggest_categorical('n_estimators', args_.hpo_n_estimators),
                                   criterion=trial.suggest_categorical('criterion', args_.hpo_criterion),
                                   max_depth=trial.suggest_categorical('max_depth', args_.hpo_max_depth),
                                   min_samples_split=trial.suggest_categorical('min_samples_split',
                                                                               args_.hpo_min_samples_split),
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features="auto",
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   bootstrap=True,
                                   oob_score=False,
                                   warm_start=False,
                                   class_weight=None)

    model.fit(x_train, y_train)
    joblib.dump(model, utils.get_model_dir(args_, trial.number))
    # optimize parameters based on accuracy
    score = model.score(x_test, y_test, sample_weight=None)
    return score


def train_dt_hpo(trial):
    x_train, x_test, y_train, y_test = hpo.create_data(args_, event_log_, preprocessor_, train_cases_)

    model = DecisionTreeClassifier(
        criterion=trial.suggest_categorical('criterion', args_.hpo_criterion),
        splitter='best',
        max_depth=trial.suggest_categorical('max_depth', args_.hpo_max_depth),
        min_samples_split=trial.suggest_categorical('min_samples_split', args_.hpo_min_samples_split),
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=args_.seed,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        ccp_alpha=0.0)

    model.fit(x_train, y_train)
    joblib.dump(model, utils.get_model_dir(args_, trial.number))
    # optimize parameters based on accuracy
    score = model.score(x_test, y_test, sample_weight=None)
    return score



def train_lstm(args, preprocessor, event_log, features_tensor, labels_tensor):
    max_case_len = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features()
    num_activities = preprocessor.get_num_activities()

    # Bidirectional LSTM
    # Input layer
    main_input = tf.keras.layers.Input(shape=(max_case_len, num_features), name='main_input')

    # Hidden layer
    b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,
                                                            activation="tanh",
                                                            kernel_initializer='glorot_uniform',
                                                            return_sequences=False,
                                                            dropout=0.2))(main_input)

    # Output layer
    act_output = tf.keras.layers.Dense(num_activities,
                                       activation='softmax',
                                       name='act_output',
                                       kernel_initializer='glorot_uniform')(b1)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[act_output])

    optimizer = tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                          schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(utils.get_model_dir(args),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)
    model.summary()
    model.fit(features_tensor, {'act_output': labels_tensor},
              validation_split=args.val_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)


def train_rf(args, features_tensor_flattened, labels_tensor):
    model = RandomForestClassifier(n_jobs=-1,  # use all processors
                                   random_state=0,
                                   n_estimators=100,
                                   criterion="gini",
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features="auto",
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   bootstrap=True,
                                   oob_score=False,
                                   warm_start=False,
                                   class_weight=None)

    model.fit(features_tensor_flattened, labels_tensor)
    joblib.dump(model, utils.get_model_dir(args))


def train_dt(args, features_tensor_flattened, labels_tensor):
    model = DecisionTreeClassifier(criterion='gini',
                                   splitter='best',
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features=None,
                                   random_state=0,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   min_impurity_split=None,
                                   class_weight=None,
                                   ccp_alpha=0.0)

    model.fit(features_tensor_flattened, labels_tensor)
    joblib.dump(model, utils.get_model_dir(args))
