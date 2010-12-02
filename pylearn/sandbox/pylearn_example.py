# Example of a typical Pylearn experiment.

import copy, sys
from itertools import izip

import numpy
import theano
from theano import tensor


def slen(dataset):
    """
    Return a symbolic Variable representing the length of a dataset.

    Should be replaced by `theano.len` when it exists.
    """
    return dataset.shape[0]


class Data(object):

    def __init__(self, fields=None):
        if fields is None:
            fields = {}
        self._fields = fields
        self.__dict__.update(fields)

    def __iter__(self):
        return self._fields.iteritems()


class ArrayData(Data):

    """
    Data wrapper around a Numpy array.

    Current implementation is for a dataset that sees a Numpy matrix as
    underlying data (n_samples x n_features).
    """

    def __init__(self, array, fields=None, **kw):
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
        all_fields = dict(
                (f_name, shared_array[:, f_range])
                for f_name, f_range in all_fields.iteritems())
        super(ArrayData, self).__init__(fields=all_fields, **kw)


def transform_fields(data, transform):
    """
    Apply the same transformation to all fields in a dataset.

    The transformed dataset is returned.
    """
    return Data(fields=dict([
        (f_name, transform(f_var)) for f_name, f_var in data]))


class KeepSamplesOp(theano.Op):

    """
    Since advanced indexing does not seem to be well supported in Theano.

    This Op sees a subset of a tensor input.
    The subset is given in the 'to_keep' tensor, that must be such that Numpy
    can convert it to a boolean array indicating which samples must be kept.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, data, to_keep):
        data = tensor.as_tensor_variable(data)
        to_keep = tensor.as_tensor_variable(to_keep)
        return theano.Apply(
                self,
                inputs=[data, to_keep],
                outputs=[data.type()],
                )

    def perform(self, node, inputs, output_storage):
        data, to_keep = inputs
        output_storage[0][0] = data[numpy.asarray(to_keep, dtype=bool)]

keep_samples = KeepSamplesOp()

class ConcatColumnVectorsOp(theano.Op):

    """
    Concatenate column vectors into a matrix.

    Also takes as input a vector of boolean indicators that indicate whether
    each vector should be concatenated into the resulting matrix.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, to_concatenate, *vectors):
        assert len(vectors) >= 1
        vectors = [tensor.as_tensor_variable(v) for v in vectors]
        to_concatenate = tensor.as_tensor_variable(to_concatenate)
        return theano.Apply(
                self,
                inputs=[to_concatenate] + vectors,
                outputs=[theano.tensor.matrix(dtype=theano.scalar.upcast(*[v.dtype for v in vectors]))],
                )
 
    def perform(self, node, inputs, output_storage):
        to_concatenate = inputs[0]
        vectors = inputs[1:]
        assert all([len(v.shape) == 1 for v in vectors])
        output_storage[0][0] = numpy.vstack([
            v for v, c in izip(vectors, to_concatenate) if c]).T

concat_column_vectors = ConcatColumnVectorsOp()

class GetColumnVectorOp(theano.Op):

    """
    Extract a single vector out of a matrix (taking a column).

    Also takes as input a vector of boolean indicators whose i-th element
    indicates whether the i-th column is actually present in the matrix.
    If it is not, then a vector filled with NaNs is returned.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, index, to_concatenate, matrix):
        matrix = tensor.as_tensor_variable(matrix)
        to_concatenate = tensor.as_tensor_variable(to_concatenate)
        index = tensor.as_tensor_variable(index)
        return theano.Apply(
                self,
                inputs=[index, to_concatenate, matrix],
                outputs=[theano.tensor.vector(dtype=matrix.dtype)],
                )

    def perform(self, node, inputs, output_storage):
        index, to_concatenate, matrix = inputs
        if to_concatenate[index]:
            # Figure out the proper column index in the input matrix.
            i = numpy.sum(numpy.asarray(to_concatenate[0:index], dtype=bool))
            rval = matrix[:, i]
        else:
            rval = numpy.zeros(len(matrix))
            rval.fill(numpy.nan)
        output_storage[0][0] = rval

get_column_vector = GetColumnVectorOp()

class LinearRegression(object):

    """Learn linear regression weights between input and target by SGD."""

    def __init__(self, lr):
        # The only learning hyper-parameter is the learning rate.
        self.lr = lr

    def train(self, input, target):
        """Return the Variable that stores the model's parameters."""
        cost = lambda target, prediction: 0.5 * tensor.sum((target - prediction)**2)
        grad = lambda input, target, weights: tensor.grad(cost(target, self.compute_sample_prediction(weights, input)), weights)
        return theano.scan(
                fn=lambda input, target, weights: weights - self.lr * grad(input, target, weights),
                outputs_info=tensor.zeros((target.shape[1], input.shape[1])),
                sequences=[input, target],
                )[0][-1]

    def compute_sample_prediction(self, weights, input_sample):
        """Return the variable that represents the model's prediction on one sample."""
        return tensor.dot(weights, input_sample)

    def compute_prediction(self, weights, input):
        """Return the variable that represents the model's prediction on a dataset."""
        return theano.scan(
                fn=lambda input_sample, weights: self.compute_sample_prediction(weights, input_sample),
                sequences=input,
                non_sequences=weights,
                )[0]


def main():

    # Create data.
    fields_spec = {
                'x0': 0,
                'x1': 1,
                'x2': 2,
                'y0': 3,
                'y1': 4,
                'input': ['x0', 'x1', 'x2'],
                'target': ['y0', 'y1'],
                }
    data = ArrayData(
            array=numpy.random.RandomState(3439).uniform(low=0, high=1, size=(100, 5)),
            fields=fields_spec,
            )

    # Filter samples to only keep those for which x1 < 0.5.
    filter = data.x1 < 0.5
    filtered_data = transform_fields(
            data=data,
            transform=lambda field: keep_samples(data=field, to_keep=filter))

    # Filter input fields to only keep those for which the sum is at least 25.
    input_fields_vars = [getattr(filtered_data, f)
                         for f in fields_spec['input']]
    must_keep = [tensor.sum(v) > 25 for v in input_fields_vars]

    # Add a new target field, computed sample-wise.
    y2 = theano.scan(
            fn=lambda x0, x2, y0: (x0 + x2) * y0 * 10,
            sequences=[filtered_data.x0, filtered_data.x2, filtered_data.y0])[0]
    extended_target = tensor.concatenate(
            (filtered_data.target, y2.dimshuffle(0, 'x')), axis=1)

    # Modify an existing input field sample-wise.
    new_x0 = theano.scan(
            fn=lambda x0, target: x0 < tensor.sum(target[0:2]),
            sequences=[filtered_data.x0, extended_target])[0]

    # Learn input mean and standard deviation to normalize it sample-wise.
    new_input = concat_column_vectors(
            must_keep, *[new_x0, filtered_data.x1, filtered_data.x2])
    input_mean = tensor.mean(new_input, axis=0)
    input_std = tensor.std(new_input, axis=0)
    normalized_input = theano.scan(
            fn=lambda input, input_mean, input_std: (input - input_mean) / input_std,
            sequences=new_input,
            non_sequences=[input_mean, input_std],
            )[0]

    # Train a learner and compute its prediction.
    learner = LinearRegression(lr=0.01)
    weights = learner.train(input=normalized_input, target=extended_target)
    prediction = learner.compute_prediction(weights=weights,
                                            input=normalized_input)

    f = theano.function([], weights, )
    print f()

    # Compute Mean Squared Error.
    mse = tensor.mean((extended_target - prediction)**2)

    # Compile function that actually does everything and display output.
    f = theano.function([], mse)
    print f()


if __name__ == '__main__':
    sys.exit(main())

