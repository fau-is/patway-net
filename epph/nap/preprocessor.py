import numpy
import pandas
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.log import Event
import epph.utils as utils
import category_encoders
from sklearn.model_selection import train_test_split


class Preprocessor(object):

    iteration_cross_validation = 0
    activity = {}
    context = {}

    def __init__(self):
        self.activity = {
            'end_char': '!',
            'chars_to_labels': {},  # label = (e.g. one-hot) encoded activity
            'labels_to_chars': {},
            'ids_to_labels': {},
            'labels_to_ids': {},
            'strings_to_ids': {},
            'ids_to_strings': {},
            'label_length': 0,
        }
        self.context = {
            'attributes': [],
            # 'strings_to_one_hot': {},
            'one_hot_to_strings': {},  # can be accessed by attribute name
            'encoding_lengths': {}
        }

    def get_event_log(self, args):
        """ Constructs an event log from a csv file using PM4PY """

        # load data with pandas and encode
        df = pandas.read_csv(args.data_dir + args.data_set, sep=';')
        df_enc = self.encode_data(args, df)

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: args.case_id_key}
        event_log = log_converter.apply(df_enc, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        event_log = self.add_end_event_to_event_log_cases(args, event_log)

        self.get_context_attributes(df)
        return event_log

    def get_context_attributes(self, df=None):
        """ Retrieves names of context attributes """

        if df is not None:
            attributes = []
            column_names = df.columns
            for i in range(len(column_names)):
                if i > 2:
                    attributes.append(column_names[i])
            self.context['attributes'] = attributes
        else:
            return self.context['attributes']

    def encode_context_attribute(self, args, df, column_name):
        """ Encodes values of a context attribute for all events in an event log """

        data_type = self.get_attribute_data_type(df[column_name])
        encoding_mode = self.get_encoding_mode(args, data_type)

        encoding_columns = self.encode_column(args, df, column_name, encoding_mode)

        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_context_encoding(column_name, len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_context_encoding(column_name, 1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        return df[column_name]

    def add_end_event_to_event_log_cases(self, args, event_log):
        """ Adds an end event at the end of each case in an event log """

        end_label = self.char_to_label(self.get_end_char())
        end_event = Event()
        if args.encoding_cat == 'int':
            end_event[args.activity_key] = end_label[0]
        else:
            end_event[args.activity_key] = list(end_label)

        for case in event_log:
            case.append(end_event)

        return event_log

    def encode_data(self, args, df):
        """ Encodes an event log represented by a data frame """

        utils.ll_print('Encode data ... \n')

        encoded_df = pandas.DataFrame(df.iloc[:, 0])

        if args.classifier == "RF" or args.classifier == "DT" or args.classifier == "RIP":
            args.encoding_cat = 'int'

        for column_name in df:
            column_index = df.columns.get_loc(column_name)

            if column_index == 0 or column_index == 2:
                # no encoding of case id and timestamp
                encoded_df[column_name] = df[column_name]
            else:
                if column_index == 1:
                    # activity
                    encoded_column = self.encode_activities(args, df, column_name)
                else:
                    # context attributes
                    encoded_column = self.encode_context_attribute(args, df.copy(), column_name)
                    data_type = self.get_attribute_data_type(df[column_name])
                    encoding_mode = self.get_encoding_mode(args, data_type)
                    if encoding_mode == args.encoding_cat:
                        self.save_context_mapping_one_hot_to_id(column_name, df[column_name], encoded_column)

                encoded_df = encoded_df.join(encoded_column)

        return encoded_df

    def get_attribute_data_type(self, attribute_column):
        """ Returns the data type of the passed attribute column 'num' for numerical and 'cat' for categorical """

        column_type = str(attribute_column.dtype)

        if column_type.startswith('float'):
            attribute_type = 'num'
        else:
            attribute_type = 'cat'

        return attribute_type

    def get_encoding_mode(self, args, data_type):
        """ Returns the encoding method to be used for a given data type """

        if data_type == 'num':
            mode = args.encoding_num
        elif data_type == 'cat':
            mode = args.encoding_cat

        return mode

    def encode_activities(self, args, df, column_name):
        """ Encodes activities for all events in an event log """

        encoding_mode = args.encoding_cat
        if encoding_mode == 'hash':
            encoding_mode = 'onehot'

        df = self.add_end_char_to_activity_column(df, column_name)
        encoding_columns = self.encode_column(args, df, column_name, encoding_mode)
        self.save_mapping_of_activities(df[column_name], encoding_columns)
        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_activity_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_activity_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)
        df = self.remove_end_char_from_activity_column(df)

        return df[column_name]

    def add_end_char_to_activity_column(self, df, column_name):
        """ Adds single row to dataframe containing the end char (activity) representing an artificial end event """

        df_columns = df.columns
        new_row = []
        for column in df_columns:
            if column == column_name:
                new_row.append(self.get_end_char())
            else:
                new_row.append(0)

        row_df = pandas.DataFrame([new_row], columns=df.columns)
        df = df.append(row_df, ignore_index=True)

        return df

    def remove_end_char_from_activity_column(self, df):
        """ Removes last row of dataframe containing the end char (activity) representing an artificial end event """

        orig_rows = df.drop(len(df) - 1)
        return orig_rows

    def encode_column(self, args, df, column_name, mode):
        """ Returns columns containing encoded values for a given attribute column """

        if mode == 'min_max_norm':
            encoding_columns = self.apply_min_max_normalization(df, column_name)

        elif mode == 'onehot':
            encoding_columns = self.apply_one_hot_encoding(df, column_name)

        elif mode == 'hash':
            encoding_columns = self.apply_hash_encoding(args, df, column_name)

        elif mode == 'int':
            encoding_columns = self.apply_integer_mapping(df, column_name)

        else:
            # no encoding
            encoding_columns = df[column_name]

        return encoding_columns

    def save_mapping_of_activities(self, column, encoded_column):
        """ Creates a mapping for activities (chars + labels (= encoded activity)) """

        activity_chars, activity_ids = self.convert_activities_to_chars(column.values.tolist())

        activities_encoded = []
        for entry in encoded_column.values.tolist():
            if type(entry) != list:
                activities_encoded.append((entry,))
            else:
                activities_encoded.append(tuple(entry))

        tuples_chars_labels = list(zip(activity_chars, activities_encoded))
        tuples_ids_labels = list(zip(activity_ids, activities_encoded))
        tuples_ids_strings = list(zip(activity_ids, column.values.tolist()))

        unique_chars_labels = []
        unique_ids_labels = []
        unique_ids_strings = []
        for tup_char_label, tup_id_label, tup_id_str in zip(tuples_chars_labels, tuples_ids_labels, tuples_ids_strings):
            if tup_char_label not in unique_chars_labels:
                unique_chars_labels.append(tup_char_label)
            if tup_id_label not in unique_ids_labels:
                unique_ids_labels.append(tup_id_label)
            if tup_id_str not in unique_ids_strings:
                unique_ids_strings.append(tup_id_str)

        self.activity['chars_to_labels'] = dict(unique_chars_labels)
        self.activity['labels_to_chars'] = dict([(t[1], t[0]) for t in unique_chars_labels])

        self.activity['ids_to_labels'] = dict(unique_ids_labels)
        self.activity['labels_to_ids'] = dict([(t[1], t[0]) for t in unique_ids_labels])

        self.activity['ids_to_strings'] = dict(unique_ids_strings)
        self.activity['strings_to_ids'] = dict([(t[1], t[0]) for t in unique_ids_strings])

    def save_context_mapping_one_hot_to_id(self, column_name, activity_column, encoded_column):
        """ Saves the mapping from one hot to its id/name """

        activity_ids = activity_column.values.tolist()

        encoded_column_tuples = []
        for entry in encoded_column.values.tolist():
            if type(entry) != list:
                encoded_column_tuples.append((entry,))
            else:
                encoded_column_tuples.append(tuple(entry))

        tuple_all_rows = list(zip(activity_ids, encoded_column_tuples))

        tuple_unique_rows = []
        for tuple_row in tuple_all_rows:
            if tuple_row not in tuple_unique_rows:
                tuple_unique_rows.append(tuple_row)

        self.context['one_hot_to_strings'][column_name] = dict([(t[1], t[0]) for t in tuple_unique_rows])

    def transform_encoded_attribute_columns_to_single_column(self, encoded_columns, df, column_name):
        """ Transforms multiple columns (repr. encoded attribute) to a single column in a data frame """

        encoded_values_list = encoded_columns.values.tolist()
        df[column_name] = encoded_values_list

        return df

    def convert_activity_to_char(self, activity):
        """ Convert initial activities' representations to chars """

        if activity != self.get_end_char():
            # 161 below is ascii offset
            activity = chr(int(activity) + 161)

        return activity

    def convert_activities_to_chars(self, activities):
        """ Convert initial activity's representation to char """
        unique_activities = []
        for activity in activities:
            if activity not in unique_activities:
                unique_activities.append(activity)

        activity_ids = list(range(len(unique_activities)))
        str_to_id = dict(zip(unique_activities, activity_ids))

        chars = []
        ids = []
        for activity in activities:
            act_id = str_to_id[activity]
            if activity != self.get_end_char():
                act_char = self.convert_activity_to_char(act_id)
            else:
                act_char = activity
            chars.append(act_char)
            ids.append(act_id)

        return chars, ids

    def apply_min_max_normalization(self, df, column_name):
        """ Normalizes a data frame column with min max normalization """

        column = df[column_name].fillna(df[column_name].mean())
        encoded_column = (column - column.min()) / (column.max() - column.min())

        return encoded_column

    def apply_one_hot_encoding(self, df, column_name):
        """ Encodes a data frame column with one hot encoding """

        onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
        encoded_df = onehot_encoder.fit_transform(df)

        encoded_column = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        return encoded_column

    def apply_integer_mapping(self, df, column_name):
        """ Encodes a data frame column with one hot encoding """

        column = df[column_name].fillna("missing")
        unique_values = column.unique().tolist()
        int_mapping = dict(zip(unique_values, range(1, len(unique_values)+1)))
        encoded_column = column.map(int_mapping)

        return encoded_column

    def apply_hash_encoding(self, args, df, column_name):
        """ Encodes a data frame column with hash encoding """

        hash_encoder = category_encoders.HashingEncoder(cols=[column_name],
                                                        n_components=args.num_hash_output,
                                                        hash_method='md5')
        encoded_df = hash_encoder.fit_transform(df)
        encoded_column = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith('col_')]]

        new_column_names = []
        for number in range(len(encoded_column.columns)):
            new_column_names.append(column_name + "_%d" % number)

        encoded_column = encoded_column.rename(columns=dict(zip(encoded_column.columns.tolist(), new_column_names)))

        return encoded_column

    def set_length_of_activity_encoding(self, num_columns):
        """ Save number of columns representing an encoded activity """
        self.activity['label_length'] = num_columns

    def set_length_of_context_encoding(self, column_name, num_columns):
        """ Save number of columns representing an encoded context attribute """
        self.context['encoding_lengths'][column_name] = num_columns

    def get_length_of_activity_label(self):
        """ Returns number of columns representing an encoded activity """
        return self.activity['label_length']

    def get_lengths_of_context_encoding(self):
        """ Returns number of columns representing all encoded context attribute """
        return list(self.context['encoding_lengths'].values())

    def get_length_of_context_encoding(self, attribute_name):
        """
        Returns number of columns representing an encoded context attribute

        Parameters
        ----------
        attribute_name : Name of context attribute.

        Returns
        -------
        int : Number of encoding values representing a context attribute's original value.
        """
        return self.context['encoding_lengths'][attribute_name]

    def get_activity_labels(self):
        """ Returns labels representing encoded activities of an event log """
        return list(self.activity['labels_to_chars'].keys())

    def char_to_label(self, char):
        """ Maps a char to a label (= encoded activity) """
        return self.activity['chars_to_labels'][char]

    def label_to_char(self, label):
        """ Maps a label (= encoded activity) to a char """
        return self.activity['labels_to_chars'][label]

    def get_end_char(self):
        """ Returns a char symbolizing the end (activity) of a case """
        return self.activity['end_char']

    def get_num_activities(self):
        """ Returns the number of activities (incl. artificial end activity) occurring in the event log """

        return len(self.get_activity_labels())

    def context_exists(self):
        """ Checks whether context attributes exist """

        return len(self.get_context_attributes(df=None)) > 0

    def get_num_features(self):
        """ Returns the number of features used to train and test the model """

        num_features = self.get_length_of_activity_label()
        num_features += sum(self.get_lengths_of_context_encoding())

        return num_features

    def get_max_case_length(self, event_log):
        """ Returns the length of the longest case in an event log """

        max_case_length = 0
        for case in event_log:
            if case.__len__() > max_case_length:
                max_case_length = case.__len__()

        return max_case_length

    def get_indices_split_validation(self, args, event_log):
        """ Produces indices for training and test set of a split-validation """

        indices_ = [index for index in range(0, len(event_log))]  # Get number of cases from data

        if args.shuffle:
            if args.seed:
                train_indices, test_indices, train_indices_, test_indices_ = train_test_split(indices_, indices_,
                                                                                              train_size=args.split_rate_train,
                                                                                              shuffle=args.shuffle,
                                                                                              random_state=args.seed_val)
                return train_indices, test_indices

            else:
                train_indices, test_indices, train_indices_, test_indices_ = train_test_split(indices_, indices_,
                                                                                              train_size=args.split_rate_train,
                                                                                              shuffle=args.shuffle,
                                                                                              random_state=None)
                return train_indices, test_indices
        else:
            return indices_[:int(len(indices_) * args.split_rate_train)], \
                   indices_[int(len(indices_) * args.split_rate_train):]


    def get_subset_cases(self, args, event_log, indices):
        """ Retrieves cases of a fold """
        subset_cases = []

        for index in indices:
            subset_cases.append(event_log[index])

        return subset_cases

    def get_subsequences_of_cases(self, cases):
        """ Creates subsequences of cases representing increasing prefix sizes """

        subseq = []

        for case in cases:
            for idx_event in range(1, len(case)):
                subseq.append(case[0:idx_event])

        return subseq

    def get_next_events_of_subsequences_of_cases(self, cases):
        """ Retrieves next events (= suffix) following a subsequence of a case (= prefix) """

        next_events = []

        for case in cases:
            for idx_event in range(0, len(case)):
                if idx_event == 0:
                    continue
                else:
                    next_events.append(case[idx_event])

        return next_events

    def get_features_tensor(self, args, event_log, subseq_cases):
        """ Produces a vector-oriented representation of feature data as a 3-dimensional tensor """

        num_features = self.get_num_features()
        max_case_length = self.get_max_case_length(event_log)

        features_tensor = numpy.zeros((len(subseq_cases),
                                       max_case_length,
                                       num_features), dtype=numpy.float64)

        for idx_subseq, subseq in enumerate(subseq_cases):
            left_pad = max_case_length - len(subseq)

            for timestep, event in enumerate(subseq):

                # activity
                activity_values = event.get(args.activity_key)

                if args.encoding_cat == 'int':
                    features_tensor[idx_subseq, timestep + left_pad, 0] = activity_values
                else:
                    for idx, val in enumerate(activity_values):
                        features_tensor[idx_subseq, timestep + left_pad, idx] = val

                # context
                if self.context_exists():
                    start_idx = 0

                    for attribute_idx, attribute_key in enumerate(self.context['attributes']):
                        attribute_values = event.get(attribute_key)

                        if not isinstance(attribute_values, list):
                            features_tensor[idx_subseq, timestep + left_pad, start_idx +
                                            self.get_length_of_activity_label()] = attribute_values
                            start_idx += 1
                        else:
                            for idx, val in enumerate(attribute_values, start=start_idx):
                                features_tensor[
                                    idx_subseq, timestep + left_pad, idx + self.get_length_of_activity_label()] = val
                            start_idx += len(attribute_values)

        if args.classifier == 'RF' or args.classifier == "DT" or args.classifier == "RIP":
            features_tensor_flattened = features_tensor.reshape(len(features_tensor), max_case_length * num_features)
            return features_tensor_flattened

        return features_tensor

    def get_labels_tensor(self, args, cases_of_fold):
        """ Produces a vector-oriented representation of labels as a 2-dimensional tensor """

        subseq_cases = self.get_subsequences_of_cases(cases_of_fold)
        next_events = self.get_next_events_of_subsequences_of_cases(cases_of_fold)
        num_event_labels = self.get_num_activities()
        activity_labels = self.get_activity_labels()

        if args.encoding_cat == 'int':
            labels_tensor = numpy.zeros((len(subseq_cases)), dtype=numpy.int64)

            for idx_subseq, next_event in enumerate(next_events):
                next_act = next_event[args.activity_key]
                labels_tensor[idx_subseq] = next_act

        else:
            labels_tensor = numpy.zeros((len(subseq_cases), num_event_labels), dtype=numpy.float64)

            for idx_subseq, subseq in enumerate(subseq_cases):
                for label_tuple in activity_labels:
                    if list(label_tuple) == next_events[idx_subseq][args.activity_key]:
                        labels_tensor[idx_subseq, activity_labels.index(label_tuple)] = 1.0
                    else:
                        labels_tensor[idx_subseq, activity_labels.index(label_tuple)] = 0.0

        return labels_tensor

    def get_predicted_label(self, pred_probabilities):
        """ Returns label of a predicted activity """

        labels = self.get_activity_labels()
        max_probability = 0
        activity_index = 0

        for probability in pred_probabilities:
            if probability >= max_probability:
                max_probability = probability
                label = labels[activity_index]
            activity_index += 1

        return label


    def get_random_case(self, args, event_log, lower_bound, upper_bound):
        """
        Selects a random process instance from the complete event log.
        :param args:
        :param event_log:
        :param lower_bound:
        :param upper_bound:
        :return: case.
        """

        num_cases = len(event_log)
        num_train_cases = int(num_cases * args.split_rate_train)
        num_test_cases = num_cases - num_train_cases

        while True:

            rand = numpy.random.randint(num_test_cases)
            size = len(event_log[num_train_cases + rand])

            if lower_bound <= size <= upper_bound:
                case = event_log[num_train_cases + rand]

                if self.check_activity(case):
                    case_id = event_log[num_train_cases + rand].attributes["concept:name"]
                    print("Id of the selected case: %s" % case_id)
                    break

        return event_log[num_train_cases + rand]


    def check_activity(self, case):

        """
        for event in case:
            if self.activity['ids_to_strings'][self.activity['labels_to_ids'][tuple(event['activity'])]] == "Create SW anomaly":
                return True
        return False
        """

        return True



