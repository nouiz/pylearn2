# Variant with iterative datasets.

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


class DataStream(object):

    """Used to iterate on datasets."""

    def __init__(self):
        self.index = theano.shared(0)
        self.index.default_update = self.index + 1

    def __call__(self, dataset):
        return dataset[self.index]

    def seek(self, i):
        self.index.value = i


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

    def grad(self, inputs, output_gradients):
        data, to_keep = inputs
        return [keep_samples_grad(data, to_keep, output_gradients), None]

keep_samples = KeepSamplesOp()


class KeepSamplesGradOp(theano.Op):

    """Gradient of KeepSamplesOp."""

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, data, to_keep, output_gradients):
        data = tensor.as_tensor_variable(data)
        to_keep = tensor.as_tensor_variable(to_keep)
        output_gradients = tensor.as_tensor_variable(output_gradients)
        return theano.Apply(
                self,
                inputs=[data, to_keep, output_gradients],
                outputs=[data.type()],
                )

    def perform(self, node, inputs, output_storage):
        data, to_keep, output_gradients = inputs
        rval = numpy.zeros_like(data)
        rval[to_keep] = output_gradients
        output_storage[0] = rval

keep_samples_grad = KeepSamplesGradOp()


class ConcatScalarsOp(theano.Op):

    """
    Concatenate scalars into a vector.

    Also takes as input a vector of boolean indicators that indicate whether
    each scalar should be concatenated into the resulting vector.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, to_concatenate, *scalars):
        assert len(scalars) >= 1
        scalars = [tensor.as_tensor_variable(v) for v in scalars]
        to_concatenate = tensor.as_tensor_variable(to_concatenate)
        return theano.Apply(
                self,
                inputs=[to_concatenate] + scalars,
                outputs=[theano.tensor.vector(dtype=theano.scalar.upcast(*[v.dtype for v in scalars]))],
                )
 
    def perform(self, node, inputs, output_storage):
        to_concatenate = inputs[0]
        scalars = inputs[1:]
        assert all([len(v.shape) == 0 for v in scalars])
        output_storage[0][0] = numpy.array([
            v for v, c in izip(scalars, to_concatenate) if c])

    def grad(self, inputs, output_gradients):
        to_concatenate = inputs[0]
        assert all(i.ndim == 0 for i in inputs[1:])
        grads = concat_scalars(to_concatenate, *output_gradients)
        #return [None] * len(inputs)
        rval = [None] + [
                tensor.switch(to_concatenate[i],
                              grads[tensor.sum(tensor.cast(to_concatenate[0:i], dtype='int64'))],
                              tensor.constant(0, dtype=inputs[i + 1].dtype))
                for i in xrange(len(inputs) -1)]
        return rval


concat_scalars = ConcatScalarsOp()


class LinearRegression(object):

    """Learn linear regression weights between input and target by SGD."""

    def __init__(self, lr):
        # The only learning hyper-parameter is the learning rate.
        self.lr = lr

    def train(self, input, target, n_steps, non_sequences):
        """Return the Variable that stores the model's parameters."""
        cost = lambda weights: 0.5 * tensor.sum((target - self.compute_sample_prediction(weights, input))**2)
        grad = lambda weights: tensor.grad(cost(weights), weights)
        param, param_up =  theano.scan(
                fn=lambda weights, *lst: weights - self.lr * grad(weights),
                outputs_info=tensor.zeros((slen(target), slen(input))),
                n_steps=n_steps,
                non_sequences=non_sequences,
                )
        return param[-1]

    def compute_sample_prediction(self, weights, input_sample):
        """Return the variable that represents the model's prediction on one sample."""
        return tensor.dot(weights, input_sample)

    def compute_prediction(self, weights, input):
        """Return the variable that represents the model's prediction on a dataset."""
        pred, up = theano.scan(
                fn=lambda input_sample, weights: self.compute_sample_prediction(weights, input_sample),
                sequences=input,
                non_sequences=weights,
                )
        assert not up
        return pred

def dict_merge(*dictionaries):
    rval = copy.copy(dictionaries[0])
    for d in dictionaries[1:]:
        for k, v in d.iteritems():
            if k in rval and rval[k] != v:
                raise KeyError(k)
            rval[k] = v
    return rval


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
    # TODO Could there be a sensible "iterative" version of it?
    filter = data.x1 < 0.5
    filtered_data = transform_fields(
            data=data,
            transform=lambda field: keep_samples(data=field, to_keep=filter))

    # Filter input fields to only keep those for which the sum is at least 25.
    input_fields_vars = [getattr(filtered_data, f)
                         for f in fields_spec['input']]
    must_keep = [tensor.sum(v) > 25 for v in input_fields_vars]

    # Add a new target field, computed sample-wise.
    data_stream = DataStream()
    iter_x0 = data_stream(filtered_data.x0)
    iter_x2 = data_stream(filtered_data.x2)
    iter_y0 = data_stream(filtered_data.y0)
    y2_sample = (iter_x0 + iter_x2) * iter_y0 * 10
    iter_target = data_stream(filtered_data.target)
    extended_target_sample = tensor.concatenate(
            (iter_target, y2_sample.dimshuffle('x')), axis=0)

    # Modify an existing input field sample-wise.
    new_x0_sample = iter_x0 < tensor.sum(extended_target_sample[0:2])

    # Learn input mean and standard deviation to normalize it sample-wise.
    iter_x1 = data_stream(filtered_data.x1)
    new_input_sample = concat_scalars(
            must_keep, *[new_x0_sample, iter_x1, iter_x2])
    new_input, new_input_up = theano.scan(
            fn=lambda: new_input_sample,
            n_steps=slen(filtered_data.x0),
            )
    input_mean = tensor.mean(new_input, axis=0)
    input_std = tensor.std(new_input, axis=0)
    normalized_input = (new_input_sample - input_mean) / input_std

    # Train a learner and compute its prediction.
    learner = LinearRegression(lr=0.01)
    data_length = slen(filtered_data.x0)
    # Note how we need to provide the non sequences to the train method.
    weights = learner.train(
            input=normalized_input,
            target=extended_target_sample,
            n_steps=data_length,
            non_sequences=[input_mean, input_std])
    prediction = learner.compute_sample_prediction(
            weights=weights,
            input_sample=normalized_input)

    # Compute Mean Squared Error.
    mse_sample = tensor.mean((extended_target_sample - prediction)**2)

    mse_all, mse_all_up = theano.scan(
            fn=lambda *lst: mse_sample,
            # Need to remember all non sequences variables!
            non_sequences=[input_mean, input_std, weights],
            n_steps=data_length)

    mse = tensor.mean(mse_all)

    # Compile function that actually does everything and display output.
    f = theano.function([], mse, )
    print f()


if __name__ == '__main__':
    sys.exit(main())

