'''
WindowGenerator adapted from https://www.tensorflow.org/tutorials/structured_data/time_series
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __init__(self, input_width, shift,
                train_df, val_df, test_df,
                label_columns=['c']):
        """initialises the window

        Args:
            input_width (int): width of input window
            shift (int): shift between label-value and inputs
            train_df (list, pandas dataframe): train dataset or list of train datasets
            val_df (list, pandas dataframe): validation dataset or list of val datasets
            test_df (list, pandas dataframe): test dataset or list of test datasets
            label_columns (str, optional): label of the window. Defaults to all labels.
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        if isinstance(train_df, list): # account for lists of datasets
            train_df = train_df[0]
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]


    """def plot(self, model=None, plot_col='c', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [d]')""" #TODO
    def create_label(self, inputs, end_row):
        #TODO get labels from shift
        return [0]*len(inputs)
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        end_row = features[:,-1,:] # get last 'c' value of 
        tf.print(features, features.shape ,end_row, end_row.shape)
        labels = self.create_label(inputs, end_row)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])

        return inputs,labels

    def make_dataset(self, data):
        if isinstance(data, list):
            ds_list = [self.make_dataset(d) for d in data]
            combined_ds= ds_list[0]
            for ds in ds_list[1:]: 
                combined_ds = combined_ds.concatenate(ds)
            return combined_ds
        else:
            data = np.array(data, dtype=np.float32)
            #ds = tf.data.Dataset.from_tensor_slices(data)

            # use total window_size with shift to not train on unavailable info
            ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)
            ds = ds.map(self.split_window)
        
        return ds