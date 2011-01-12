# Example of a typical Pylearn experiment.

import copy, sys

import numpy
import tables
import theano
from theano import tensor

from pylearn_example import concat_column_vectors, Data#, ArrayData, ConcatColumnVectorsOp, 
from pylearn_example import keep_samples, LinearRegression, transform_fields
#KeepSamplesOp, GetColumnVectorOp, get_column_vector

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
        assert len(array.shape) == 2

        shared_array = theano.shared(array)
        all_fields = self.process_fields(fields)
        # Make fields visible directly into this object's members.
        # A field is just a Theano variable (here a subset of the shared
        # variable that contains the Numpy array).
        all_fields = dict(
                (f_name, shared_array[:, f_range])
                for f_name, f_range in all_fields.iteritems())
        super(ArrayData, self).__init__(fields=all_fields, **kw)

    def process_fields(self, fields):
        # The following code is basically just to be able to build the mapping
        # from a field's name to either its column index or its list of column
        # indices (i.e. get rid of strings in the fields definition).
        if fields is None:
            fields = {}
        all_fields = {}
        to_process = copy.copy(fields)
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
        return all_fields

class PyTablesSubtensor(theano.tensor.Subtensor):
    """
    Allow to return a slice on a PyTables object in Theano.
    """

    def make_node(self, x, *inputs):
        sup = super(PyTablesSubtensor, self).make_node(
            theano.tensor.tensor(dtype=x.dtype,
                                 broadcastable=[False]*len(x.shape), *inputs))

        assert isinstance(x, tables.array.Array)
        shared_var = theano.shared(x)

        #shared_var.type.ndim = len(x.shape)
        shared_var.ndim = len(x.shape)

        return theano.Apply(
                sup.op,
                inputs=[shared_var]+sup.inputs[1:],
                outputs=[sup.outputs[0].type()],
                )

    def perform(self, node, inputs, output_storage):
        #currently we support only some case.
        assert len(inputs) == 1

        x = inputs[0]

        #import pdb;pdb.set_trace()
        #TODO, don't allow taking slice bigger then 100Mb
        output_storage[0][0] = x[tuple(self.idx_list)]
        

class PyTablesData(ArrayData):

    """
    Data wrapper around a PyTables object.

    Current implementation return a numpy ndarray at each iteration

    :note: support only PyTables whose data don't change.
    """

    def __init__(self, filename, fields=None, **kw):
        """
        Constructor.

        :type pytables: a hdf5 filename
        /:param array: The array containing the data (should be a 2D matrix)

        :type fields: Dictionary
        :param fields: Maps a field's name to either:
                        - A column index
                        - Another field's name
                        - A list of either of the above
        """
        ######shared_array = theano.shared(array)
        h5file = tables.openFile(filename, mode = "r")
        all_fields = {}
        # Make fields visible directly into this object's members.
        # A field is just a Theano variable (here the output of
        # Theano op PytableSlice that return a Numpy array).

        # Make fields visible directly into this object's members.
        # A field is just a Theano variable (here a subset of the shared
        # variable that contains the Numpy array).
        for group in h5file.walkGroups("/"):
            for array in h5file.listNodes(group, classname='Array'):
                #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, get the len! should be ":"
                assert isinstance(array, tables.array.Array)
                name = array.attrs._v__nodePath
                if name in fields:
                    name = fields[name]
                all_fields[name] = PyTablesSubtensor(
                    #[slice(0,array.shape[0])])(array)
                    [slice(0,100)])(array)
        super(ArrayData, self).__init__(fields=all_fields, **kw)
        return

def main():
    use_pytables = True

    # Create data.
    if not use_pytables:
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
    else:
        inputs = [('x'+str(i),i) for i in range(1024)]
        fields_spec = dict(inputs)
        fields_spec['input'] = [x for x,y in inputs]
        fields_spec['y0'] = 1024
        fields_spec['target']=['y0']
        fields_spec['/data/input'] = 'input'
        fields_spec['/data/target'] = 'target'
        data = PyTablesData("/Tmp/bastienf/PNIST07_train_data_maxfile=2_exprows=819200_chkshp=(128, 1024)_cmplevel=9_cmplib=zlib_shf=False.hdf5",
                            fields=fields_spec)


    # Filter samples to only keep those for which x1 < 0.5.
    if not use_pytables:#TODO enabled this.
        filter = data.x1 < 0.5
        filtered_data = transform_fields(
            data=data,
            transform=lambda field: keep_samples(data=field, to_keep=filter))
    else:
        filtered_data = data

    if not use_pytables: #TODO enabled this
        # Filter input fields to only keep those for which the sum is at least 25.
        input_fields_vars = [getattr(filtered_data, f)
                             for f in fields_spec['input']]
        must_keep = [tensor.sum(v) > 25 for v in input_fields_vars]
        
    # Add a new target field, computed sample-wise.
    y2 = theano.scan(
        fn=lambda x0, x2, y0: (x0 + x2) * y0 * 10,
        sequences=[filtered_data.input[:,0], filtered_data.input[:,2], filtered_data.target])[0]
    extended_target = tensor.concatenate(
        (filtered_data.target.dimshuffle(0,'x'), y2.dimshuffle(0, 'x')), axis=1)
        
    if not use_pytables: #TODO enabled this
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
    else:
        normalized_input = filtered_data.input

    # Train a learner and compute its prediction.
    learner = LinearRegression(lr=0.01)
    weights = learner.train(input=normalized_input, target=extended_target)
    prediction = learner.compute_prediction(weights=weights,
                                            input=normalized_input)

    f_train = theano.function([], weights, )
    # Compute Mean Squared Error.
    mse = tensor.mean((extended_target - prediction)**2)

    # Compile function that actually does everything and display output.
    f_mse = theano.function([], mse)

    #import pdb;pdb.set_trace()
    for i in range(2):
        print f_train()

        print f_mse()


if __name__ == '__main__':
    sys.exit(main())

