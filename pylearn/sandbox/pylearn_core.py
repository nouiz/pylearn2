# Tentative implementation of a Theano-based Pylearn architecture.

import copy, sys

import numpy
import theano


class PylearnException(Exception):
    """Base class for Pylearn exceptions."""


class LengthException(PylearnException):
    """Exception raised by datasets that cannot return a finite length."""


class InfiniteLengthException(LengthException):
    """Raised by infinite datasets."""


class UndefinedLengthException(LengthException):
    """Raised by datasets that do not know their own length."""


def slen(dataset):
    """
    Return a symbolic Variable representing the length of a dataset.

    Should be replaced by `theano.len` when it exists.
    """
    return dataset._symbolic_length()

class Extra(object):

    """
    Store additional information on a dataset, besides its own variable.

    Currently this additional information corresponds to the dataset's fields:
        - "base" fields map to Theano variables associated to each field
        - "concat" fields are defined by the concatenation of some base fields
    """

    def __init__(self, base_fields=None, concat_fields=None, concat_def=None):
        """
        Constructor.

        :type base_fields: Dictionary or `None`.
        :param base_fields: Maps a base field's name to its Theano Variable.
                            `None` is the same as an empty dictionary (i.e. no
                            base fields).

        :type concat_fields: Dictionary or `None`.
        :param concat_fields: Maps a concat field's name to its Thenao
                              Variable. `None` means this map should be built
                              based on `concat_def`, by creating new Theano
                              variables (currently we assume each base field
                              is a column vector, and thus concat fields become
                              matrices).

        :type concat_def: Dictionary or `None`.
        :param concat_def: Maps a concat field's name to the list of base
                           fields it is the concatenation of. `None` is the
                           same as an empty dictionary (i.e. no concat fields).
                           This parameter must be provided if there are concat
                           fields (while `concat_fields` may be omitted).
        """
        if base_fields is None:
            base_fields = {}
        self.base_fields = base_fields
        if concat_def is None:
            concat_def = {}
            # One cannot provided `concat_fields` if `concat_def` is not
            # available.
            assert concat_fields is None
        self.concat_def = concat_def
        if concat_fields is None and concat_def is not None:
            # Need to build Theano variables that concatenate base fields.
            concat_fields = dict([
                    (field_name, self.create_field_var(field_def))
                    for field_name, field_def in concat_def.iteritems()])
        self.concat_fields = concat_fields

    def __getitem__(self, item):
        # Return extra information corresponding to taking a subset of the
        # dataset: field names are the same, and are mapped to the same
        # subset of their corresponding Theano variable.
        return Extra(
                base_fields=dict((k, v[item]) for k, v in self.base_fields.iteritems()),
                concat_fields=dict((k, v[item]) for k, v in self.concat_fields.iteritems()),
                concat_def=self.concat_def)

    def create_field_var(self, field_def):
        """
        Return the Theano Variable corresponding to the given field definition.

        :type field_def: List of strings.
        :param field_def: List of base fields that should be concatenated. In
                          the current implementation, these fields are assumed
                          to map to vector variables.
        """
        return theano.tensor.horizontal_stack(*[
            self.base_fields[field_name].dimshuffle([0, 'x'])
            for field_name in field_def])

    def get_field(self, field):
        """
        Return the Extra information corresponding to a given field.

        :type field: String.
        :param field: Name of a field (base or concat).

        All fields that have a non-empty intersection with the base field(s)
        selected when keeping only `field` will be available in the Extra
        object being returned.
        """
        # What base fields are selected?
        if field in self.base_fields:
            new_base = [field]
        else:
            new_base = self.concat_def[field]
        base_kept = set(new_base)
        # Create map from base field to corresponding variable.
        new_base = dict((f, self.base_fields[f]) for f in new_base)
        new_concat_def = {}
        new_concat_fields = {}
        for f_name, f_def in self.concat_def.iteritems():
            # Figure out which concat fields should be kept.
            new_def = [f for f in f_def if f in base_kept]
            if len(new_def) > 0:
                # The field `f_name` should be kept since it contains at least
                # one of the base fields we selected.
                if len(new_def) == len(f_def):
                    # Field remains unchanged.
                    new_concat_def[f_name] = f_def
                    new_concat_fields[f_name] = self.concat_fields[f_name]
                else:
                    # We take a subset: need to redefine the corresponding
                    # Theano variable as a concatenation of fewer base fields.
                    new_concat_def[f_name] = new_def
                    new_concat_fields[f_name] = self.create_field_var(new_def)
        return Extra(base_fields=new_base, concat_fields=new_concat_fields,
                     concat_def=new_concat_def)

    def get_field_var(self, field):
        """Return the Theano variable corresponding to the given field.

        :type field: String.
        :param field: The field of interest (either base or concat).
        """
        if field in self.base_fields:
            return self.base_fields[field]
        else:
            return self.concat_fields[field]


class Data(object):

    """
    Store data (either a dataset, or a sample).

    A Data object contains:
        - A Theano variable that (symbolically) represents the data itself.
        - An Extra object, that holds additional information (i.e. field
          names currently).

    The field 'f' of a Data object `data` can be accessed through `data.f`,
    which is itself a Data object.
    """

    def __init__(self, variable, extra=None):
        """
        Constructor.

        :type variable: A Theano Variable.
        :param variable: The variable representing the data.

        :type extra: Extra or `None`.
        :param extra: Additional information about the Data (see `Extra`).
                      If `None`, then no additional information is available.
        """
        self._variable = variable
        if extra is None:
            extra = Extra()
        self._extra = extra

    def __getitem__(self, index):
        """Return the Data object that sees a subset of this Data."""
        return Data(variable=self._variable[index], extra=self._extra[index])
        
    def __len__(self):
        # The length of a data is usually not available directly: it should
        # be obtained through a symbolic variable.
        raise RuntimeError('Use slen(data) to obtain a symbolic '
                           'Variable representing the length of the data')

    def _symbolic_length(self):
        """Return the Theano Variable representing this Data's length."""
        if hasattr(self._variable, 'shape'):
            return self._variable.shape[0]
        else:
            raise UndefinedLengthException(self.__class__)

    def __call__(self):
        """Shortcut to gain access to the Data's variable."""
        return self._variable

    def __getattr__(self, attr):
        # Kind of hack to enable the `data.field` syntax.
        if attr.startswith('_'):
            return getattr(super(Data, self), attr)
        else:
            return self._get_field(attr)

    def _get_field(self, field):
        """Return the Data object corresponding to one field in this Data."""
        return Data(variable=self._extra.get_field_var(field),
                    extra=self._extra.get_field(field))


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
                        - A list of column indices (multi-dimensional fields)
                        - Another field's name
                        - A list of other fields' names
                       In the first two cases, the field is considered a base
                       field. In the other two cases, the field is considered
                       a 'concat' field (even if it concatenates only one
                       base field, which is not really a concatenation).
        """
            

        assert len(array.shape) == 2
        assert 'extra' not in kw
        if fields is None:
            fields = {}
        shared_array = theano.shared(array)
        # Analyze the fields' definition given in `fields` to extract the
        # base fields and the concat ones.
        base_fields, concat_def = self.analyze_fields(
                fields,
                width=array.shape[1],
                shared_array=shared_array)
        super(ArrayData, self).__init__(
                variable=theano.shared(array),
                extra=Extra(base_fields=base_fields, concat_def=concat_def),
                **kw)
 
    def analyze_fields(self, fields, width, shared_array):
        """
        Return base and concat fields given the fields' definition.

        This function is rather big and ugly, not fully tested, and contains
        known bugs (it was originally written with a slightly different
        definition of fields).
        No need to look into it too much: basically the idea is it looks at the
        fields' definition given by the user, and creates the corresponding
        arguments `base_fields` and `concat_def` that should be fed to the
        `Extra` class.
        """
        fields_var = {}
        field_to_indices = {}
        fields = copy.copy(fields)
        width_range = range(width)
        get_base_fields = True
        concat_def = {}
        index_to_base = {}
        while fields:
            to_delete = []
            for field, item in fields.iteritems():
                #print field
                direct = False
                if isinstance(item, int):
                    direct = True
                elif isinstance(item, str):
                    if item in field_to_indices:
                        field_to_indices[field] = field_to_indices[item]
                        to_delete.append(field)
                elif isinstance(item, slice):
                    direct = True
                elif isinstance(item, list):
                    direct = True
                    new_item = None
                    #print 'item = %s' % (item, )
                    for i, item_i in enumerate(item):
                        #print 'item_i = %s' % (item_i, )
                        if isinstance(item_i, str):
                            direct = False
                            if item_i in field_to_indices:
                                if new_item is None:
                                    new_item = item[0:i]
                                new_item.append(field_to_indices[item_i])
                            else:
                                #print '%s not found' % item_i
                                pass
                        elif new_item is not None:
                            new_item.append(item_i)
                    if direct:
                        total_item = []
                        for item_i in item:
                            if isinstance(item_i, list):
                                total_item += item_i
                            else:
                                total_item.append(item_i)
                        # Try to convert to slice if possible.
                        k = total_item[0]
                        is_slice = True
                        while k + 1 < len(total_item):
                            if total_item[k + 1] != total_item[k] + 1:
                                is_slice = False
                                break
                            k += 1
                        if is_slice:
                            total_item = slice(total_item[0],
                                                total_item[-1] + 1)
                        fields[field] = total_item
                    elif new_item is not None:
                        fields[field] = new_item
                else:
                    raise NotImplementedError(type(item))
                if direct:
                    item = fields[field]
                    if isinstance(item, int):
                        indices = [item]
                    elif isinstance(item, list):
                        indices = item
                    elif isinstance(item, slice):
                        indices = width_range[item]
                    else:
                        raise NotImplementedError(type(item))
                    #print '%s: %s' % (field, item)
                    field_to_indices[field] = indices
                    if get_base_fields:
                        fields_var[field] = shared_array[:, item]
                        assert isinstance(item, int)
                        index_to_base[item] = field
                    else:
                        concat_def[field] = [index_to_base[i] for i in indices]
                    #print 'field_to_indices[%s] = %s' % (field, indices)
                    to_delete.append(field)
            for field in to_delete:
                #print 'Deleting %s' % field
                del fields[field]
            get_base_fields = False
        return fields_var, concat_def


class datalearn(object):

    """
    Decorator to modify a Learner's compute_* methods.

    Applied to a function that takes as input a datapoint (sample) and returns
    a Theano Variable, returns a new function that takes as input a dataset
    and returns a dataset (corresponding to applying the same computation on
    each sample in the input dataset).
    """
    
    def __init__(self, name):
        self.name = name

    def  __call__(self, f):
        def new_f(f_self, something):
            if isinstance(something, Data):
                # We assume here that Data = dataset.
                outputs, updates = theano.scan(
                        fn=lambda i: f(f_self, something[i]),
                        sequences=theano.tensor.arange(slen(something)),
                        )
                assert not updates
                # Note that right now, the output dataset has a single base
                # field. This is probably not what we always want.
                return Data(variable=outputs, extra=Extra(base_fields={self.name: outputs}))
            else:
                # Currently can only handle a Data object that represents a
                # dataset. May be also extended to handle single datapoints?
                # What about direct numeric values?
                raise NotImplementedError()
        return new_f

        
class MeanLearner(object):

    """
    Compute data mean and subtract it from the input.

    Learner that at training time, learns the mean of its training set's input
    part. The output of the learner is then its input dataset's input part,
    shifted by said mean.
    """

    def train(self, dataset):
        # `index` allows us to iterate over samples in the dataset.
        index = theano.tensor.iscalar()
        # `sum` will contain the sum of all samples' input parts.
        sum = None
        # Extract the input part of the dataset.
        input_part = dataset.input
        # Obtain the dataset's length (note: this would not be needed with a
        # way to iterate on data rather than use random access).
        n = theano.function([], slen(input_part))()
        # Compile a Theano function that returns a given sample's input.
        get_item = theano.function([index], input_part[index]())
        # Iterate on dataset and compute sum.
        for i in xrange(n):
            if i == 0:
                sum = get_item(i).copy()
            else:
                sum += get_item(i)
        # Divide by the dataset's length to obtain the mean.
        self.mean = theano.shared(sum / float(n))

    @datalearn(name='output')
    def compute_output(self, datapoint):
        """Output of this learner: input part shifted by global mean."""
        return datapoint.input() - self.mean


def main():

    if True:
        # Create some dummy data.
        dataset = ArrayData(
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
        learner.train(dataset)
        # Compute the learner's output on this same dataset.
        out_dataset = learner.compute_output(dataset)
        # Iterate on the output dataset to see what it contains.
        index = theano.tensor.iscalar()
        get_item = theano.function([index], out_dataset[index]())
        out_len = theano.function([], slen(out_dataset))()
        for i in xrange(out_len):
            print get_item(i)
        # Note that in the current implementation, the output dataset only
        # contains one base field (named 'output'). Information about
        # individual input field names was lost (which is probably not what
        # someone would want).
        print out_dataset._extra.base_fields.keys()
        print out_dataset._extra.concat_fields.keys()

    if False:
        # Test stuff.
        dataset = ArrayData(array=numpy.arange(15).reshape((3, 5)))
        index = theano.tensor.iscalar()
        get_item = theano.function([index], dataset[index]())
        for i in xrange(len(dataset)):
            print get_item(i)
        print get_item(0)


    return 0


if __name__ == '__main__':
    sys.exit(main())

