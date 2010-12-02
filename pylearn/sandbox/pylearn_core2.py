# Tentative implementation of a Theano-based Pylearn architecture.

import copy, sys

import numpy
import theano
from theano import tensor


def slen(dataset):
    """
    Return a symbolic Variable representing the length of a dataset.

    Should be replaced by `theano.len` when it exists.
    """
    return dataset.shape[0]


class ArrayData(object):

    """
    Data wrapper around a Numpy array.

    Current implementation is for a dataset that sees a Numpy matrix as
    underlying data (n_samples x n_features).
    """

    def __init__(self, array, fields=None):
        """
        Constructor.

        :type array: numpy.ndarray
        :param array: The array containing the data (should be a 2D matrix)

        :type fields: Dictionary
        :param fields: Maps a field's name to either:
                        - A column index
                        - Another field's name
                        - A list of either of the above
        """
        # The following code is basically just to be able to build the mapping
        # from a field's name to either its column index or its list of column
        # indices (i.e. get rid of strings in the fields definition).
        assert len(array.shape) == 2
        if fields is None:
            fields = {}
        all_fields = {}
        to_process = copy.copy(fields)
        shared_array = theano.shared(array)
        while to_process:
            to_delete = []
            for f_name, f_range in to_process.iteritems():
                found_range = None
                if isinstance(f_range, int):
                    found_range = f_range
                elif isinstance(f_range, str):
                    if f_range in all_fields:
                        found_range = all_fields[f_range]
                elif isinstance(f_range, list):
                    indices = []
                    for index in f_range:
                        if isinstance(index, int):
                            indices.append(index)
                        elif isinstance(index, str):
                            if index in all_fields:
                                indices.append(all_fields[index])
                            else:
                                break
                        else:
                            raise NotImplementedError(type(index))
                    if len(indices) == len(f_range):
                        found_range = indices
                else:
                    raise NotImplementedError(type(f_range))
                if found_range is not None:
                    all_fields[f_name] = found_range
                    to_delete.append(f_name)
            for f_name in to_delete:
                del to_process[f_name]
        # Replace lists by slices (lists do not seem to be currently supported
        # by Theano).
        for f_name, f_range in all_fields.iteritems():
            if isinstance(f_range, list):
                start = f_range[0]
                end = start + 1
                for i in f_range[1:]:
                    assert i == end
                    end += 1
                all_fields[f_name] = slice(start, end)
        # Make fields visible directly into this object's members.
        # A field is just a Theano variable (here a subset of the shared
        # variable that contains the Numpy array).
        self.__dict__.update(dict(
            (f_name, shared_array[:, f_range])
            for f_name, f_range in all_fields.iteritems()))


class datalearn(object):

    """
    Decorator to modify a Learner's compute_* methods.

    Applied to a function that takes as input a datapoint (sample) and returns
    a Theano Variable, returns a new function that takes as input a dataset
    and returns a dataset (corresponding to applying the same computation on
    each sample in the input dataset).
    """
    
    def __init__(self):
        pass

    def  __call__(self, f):
        def new_f(f_self, dataset):
            outputs, updates = theano.scan(
                    fn = lambda x: f(f_self, x),
                    sequences=dataset,
                    )
            assert not updates
            return outputs
        return new_f

 
class MeanLearner(object):

    """
    Compute training set's mean and substract it from its input.
    """

    def train_python(self, dataset):
        # `index` allows us to iterate over samples in the dataset.
        index = theano.tensor.iscalar()
        # `sum` will contain the sum of all samples' input parts.
        sum = None
        # Obtain the dataset's length (note: this would not be needed with a
        # way to iterate on data rather than use random access).
        n = theano.function([], slen(dataset))()
        # Compile a Theano function that returns a given sample's input.
        get_item = theano.function([index], dataset[index])
        # Iterate on dataset and compute sum.
        for i in xrange(n):
            if i == 0:
                sum = get_item(i).copy()
            else:
                sum += get_item(i)
        # Divide by the dataset's length to obtain the mean.
        self.mean = theano.shared(sum / float(n))

    def train_theano(self, dataset):
        sum = tensor.zeros(list(dataset.shape)[1:], dtype=dataset.dtype)
        total_sum, updates = theano.scan(
                fn=lambda sample, sum: sum + sample,
                outputs_info=sum,
                sequences=dataset,
                )
        assert not updates
        total_sum = total_sum[-1]
        mean = total_sum / slen(dataset)
        return mean

    def train(self, dataset):
        self.mean = theano.shared(theano.function([], self.train_theano(dataset))())

    def train_james(self, dataset):
        n = dataset['length']
        get_item = theano.function([dataset['index']], dataset['sample'])
        for i in xrange(n):
            if i == 0:
                sum = get_item(i).copy()
            else:
                sum += get_item(i)
        # Divide by the dataset's length to obtain the mean.
        self.mean = theano.shared(sum / float(n))

        
    @datalearn()
    def compute_output(self, datapoint):
        """Output of this learner: input shifted by global mean."""
        return datapoint - self.mean

    def compute_output_james(self, dataset):
        return {
                'index': dataset['index'],
                'length': dataset['length'],
                'sample': dataset['sample'] - self.mean,
                }


def main():

    if True:
        # Create some dummy data.
        data = ArrayData(
                array=numpy.arange(15).reshape((3, 5)),
                fields={
                    'x0': 0,
                    'x1': 1,
                    'x2': 2,
                    'y0': 3,
                    'y1': 4,
                    'input': ['x0', 'x1', 'x2'],
                    'target': ['y0', 'y1'],
                    })
        # Create learner.
        learner = MeanLearner()
        # Train on dataset to compute the mean on the input.
        index = theano.tensor.lscalar()
        dict_dataset =  {
                'index': index,
                'sample': data.input[index],
                'length': theano.function([], slen(data.input))(),
                }
        #learner.train_python(data.input)
        #learner.train(data.input)
        learner.train_james(dict_dataset)

        if False:
            # Compute the learner's output on this same dataset.
            out_dataset = learner.compute_output(data.input)
            # Iterate on the output dataset to see what it contains.
            index = theano.tensor.iscalar()
            get_item = theano.function([index], out_dataset[index])
            out_len = theano.function([], slen(out_dataset))()
            for i in xrange(out_len):
                print get_item(i)

        if True:
            out_dataset = learner.compute_output_james(dict_dataset)
            get_item = theano.function([out_dataset['index']], out_dataset['sample'])
            for i in xrange(out_dataset['length']):
                print get_item(i)

    return 0


if __name__ == '__main__':
    sys.exit(main())

