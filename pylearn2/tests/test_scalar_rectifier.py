'''
This file has two purposes:
1. test pylearn2.scalar module (conducted in test_scalar_rectifier())
2. speed benchmark on pylearn2.scalar on CPU and GPU (conducted in benchmark_single_op)

Conclusion:
1. For pylearn2.scalar, both 'grad()' and 'c_code()' work as expected.
2. On CPU,
speed benchmark fprop old/new=  0.615451241318,
speed benchmark grad old/new=  2.8991003942

'''
import theano
import theano.tensor as T
from pylearn2.scalar import rectifier
import numpy
import time

floatX = 'float32'
relu = lambda x: T.maximum(0.0, x)

def test_scalar_rectifier():
    # verify the new op rectifier produces the same results as relu
    x = T.fmatrix('inputs')
    y1 = relu(x)
    y2 = rectifier(x)

    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_1_forward')
    f2 = theano.function(inputs=[x], outputs=y2, name='benchmark_2_forward')

    g1 = theano.function(inputs=[x], outputs=T.grad(y1.sum(),x), name='benchmark_1_grad')
    g2 = theano.function(inputs=[x], outputs=T.grad(y2.sum(),x), name='benchmark_2_grad')
    
    for i in range(10):
        value = numpy.random.uniform(size=(100,500)).astype(floatX)
        numpy.testing.assert_array_equal(f1(value), f2(value),
                                         err_msg='arrays not equal' )
        
        numpy.testing.assert_array_equal(g1(value), g2(value),
                                         err_msg='grad:arrays not equal' )
        

def benchmark_single_op():
    '''
    On CPU, the new scalar_rectifier is about 1.5 times faster.
    On GPU, they are almost of the same speed. 
    '''
    x = T.fmatrix('inputs')
    y1 = relu(x).sum()
    y2 = rectifier(x).sum()

    f1 = theano.function(inputs=[x], outputs=y1, name='benchmark_fprop_old')
    f2 = theano.function(inputs=[x], outputs=y2, name='benchmark_fprop_new')

    n_loops = 1000
    value = numpy.random.uniform(size=(100,5000)).astype(floatX)

    # benchmark forward computation
    t0 = time.time()
    for i in range(n_loops):
        f1(value)
    t1 = time.time()
    benchmark_1 = t1-t0
    
    t0 = time.time()
    for i in range(n_loops):
        f2(value)
    t1 = time.time()
    benchmark_2 = t1-t0

    print 'speed benchmark fprop old/new= ', benchmark_1/(benchmark_2+0.0)

    f1 = theano.function(inputs=[x], outputs=T.grad(y1, x), name='benchmark_grad_old')
    f2 = theano.function(inputs=[x], outputs=T.grad(y2, x), name='benchmark_grad_new')
    
    # benchmark grad computation
    t0 = time.time()
    for i in range(n_loops):
        f1(value)
    t1 = time.time()
    benchmark_1 = t1-t0
    
    t0 = time.time()
    for i in range(n_loops):
        f2(value)
    t1 = time.time()
    benchmark_2 = t1-t0

    print 'speed benchmark grad old/new= ', benchmark_1/(benchmark_2+0.0)

def benchmark_all():
    benchmark_single_op()

if __name__ == '__main__':
    benchmark_all()
    #test_scalar_rectifier()