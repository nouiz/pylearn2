# Tentative implementation of a Theano-based Pylearn architecture.

'''
 Some rumbling about what I think are the differences to pyleanr_core.py.
 Note that I'm not fully aware of the talk with James.

 My approach starts from fields. Fields (or basic fields in Olivier's
 terminology) are shared variables. This might not be ok all the time,
 since it does not allow chunking the data into pieces and load them
 from hdd, generate them on the fly .. etc. I think though these cases
 can be fixed by creating new types of (shared) variables that have this
 functionality. The difference between them and Dataset, is that thay
 are a uniform dtype so they are either a numpy.ndarray, or a scalar
 or a string (stored in generic), but not a mixture of those.
 You compose these fields using the MakeDataset op, into a Dataset variable
 which is just a symbolic dictionary/numpy.record.

 The difference is that a Dataset object also stores side_effects ( which is
 just a dictionary of updates rules that should be passed to the theano
 function.



'''

import copy, sys

import numpy
import theano


class MakeDataset(theano.gof.Op):
    ''' Constructs DataSet starting from Fields, which in this case are
    Shared Variables (?) '''
    def __init__(self):
        pass
    def make_node(self, **fields):
        _length = len(fields.values()[0].value)
        _data = {}
        _side_effects = {}
        _aliases = {}
        out = DataSet(  _data        = fields
                     , _length       = _length
                     , _aliases      = _aliases
                     , _side_effects = _side_effects )
        return theano.gof.Apply(
            self
            , fields.values()
            , [out])


make_dataset = MakeDataset()


def fromDict(fields):
    data = {}
    for k in fields:
        data[k] = theano.shared(fields[k])
    return make_dataset(**data)


class DataSetType(theano.gof.Type):
    pass


class DataSet(theano.gof.Variable):
    def __init__( self, _data, _length, _aliases, _side_effects):
        super(DataSet, self).__init__(DataSetType)
        self._data         = _data
        self._variables    = _data.values()
        self._keys         = _data.keys()
        self._length       = _length
        self._aliases      = _aliases
        self._side_effects = _side_effects
        self.owner         = None



    def set_alias(self, **aliases):
        for k in aliases:
            all_inputs = theano.gof.graph.inputs([aliases[k]])
            assert [ (x in self._variables or
                      isinstance(x, theano.gof.Constant) )
                                         for x in all_inputs]

            field_inputs = filter( lambda x: x in self._variables
                                   , all_inputs)
            field_map = [(x, self._variables.index(x)) for x
                          in field_inputs]
            constructor = None
            self._aliases[k] = (field_map, aliases[k], constructor)

    def __getitem__(self, index):
        _length = 0
        _data = dict([ (k, v[index]) for (k,v) in self._data.items()])
        new_side_effects = self._side_effects
        # Actually this should be the case:
        # new_side_effects = theano.clone( self.side_effects,
        #                   [(v, v[index]) for v in self._data.values()])
        # Where clone is a function I've implemented for scan but had not
        # pushed to trunk yet ( I'm currently debugging it )

        return DataSet(  _data         = _data
                       , _length       = _length
                       , _aliases      = self._aliases
                       , _side_effects = new_side_effects)



    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        elif attr in self._aliases:
            replace_map = [(x, self._variables[y]) for (x,y) in
                           self._aliases[attr][0]]
            constructor = self._aliases[attr][2]
            # out = theano.clone( self.aliases[attr][1], replace_map)
            return self.aliases[attr][1]
        else:
            return self.__getattribute__( attr)



def meanlearner(data):
    assert 'input' in data._aliases

    input_fields = [data._keys[x] for (_1,x) in data._aliases['input'][0]]
    # Generate shared variables for holding the means
    sizes = [ data._data[f].value.shape[1] for f in input_fields ]
    mean_vals = [theano.shared(numpy.zeros((sz,))) for sz in sizes]
    # updated the shared variables
    new_data =data._data
    for idx, f in enumerate(input_fields):
        new_data[f] = new_data[f] - mean_vals[idx]

    side_effects = {}
    for mv, f in zip(mean_vals, [ data._data[f] for f in input_fields]):
        side_effects[mv] = f.mean( axis=0 )

    return DataSet(  _data         = new_data
                   , _length       = data._length
                   , _aliases      = data._aliases
                   , _side_effects = side_effects )


def join(data, *fields):
    rval = {}
    rval['fields'] = fields
    rval['value']  = theano.tensor.join(0,*[data._data[f] for f in fields])
    return rval

def main():

    dataset = fromDict({ 'field1': numpy.arange(25).reshape(5,5),
                         'field2': numpy.arange(15).reshape(5,3),
                         'target': numpy.arange(5) } )
    dataset.set_alias(input = theano.tensor.join(1,
                                                 dataset.field1,
                                                 dataset.field2) )
    out_data = meanlearner(dataset)
    idx = theano.tensor.lscalar('idx')
    get_item = theano.function([idx], out_data[idx]._variables )
    train_fn = theano.function([], [], updates = out_data._side_effects)


    for i in xrange( out_data._length):
        print get_item(i)

    train_fn()

    for i in xrange( out_data._length):
        print get_item(i)


if __name__ == '__main__':
    sys.exit(main())

