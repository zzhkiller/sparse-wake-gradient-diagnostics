"""Checks scientific implementation invariants, not model-performance assertions."""
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
import numpy as np
from mat_v5 import loadmat
from pilot import MLP,dy


class NumericalChecks(unittest.TestCase):
    def test_derivative_axis_and_units(self):
        y=np.linspace(-3,3,31)
        u=np.stack([2*y+5,3*y-1])[None]
        out=dy(u,y[1]-y[0])
        np.testing.assert_allclose(out[0,0],2,atol=1e-12)
        np.testing.assert_allclose(out[0,1],3,atol=1e-12)

    def test_gradient_backprop_finite_difference(self):
        rng=np.random.default_rng(41)
        net=MLP(3,2,seed=9,hidden=4)
        x=rng.normal(size=(5,3));b0=rng.normal(size=(5,2));b1=rng.normal(size=(5,2))
        a=rng.normal(size=(4,2));q=a.T@a/4
        _,analytical=net.loss_grad(x,b0,b1,q,.3)
        for k,p in enumerate(net.p):
            for idx in list(np.ndindex(p.shape))[:7]:
                old=p[idx];eps=1e-6
                p[idx]=old+eps;plus,_=net.loss_grad(x,b0,b1,q,.3)
                p[idx]=old-eps;minus,_=net.loss_grad(x,b0,b1,q,.3)
                p[idx]=old
                self.assertAlmostEqual((plus-minus)/(2*eps),analytical[k][idx],places=6)

    def test_mat_reader_shape_and_fortran_order(self):
        def elt(kind,b):return struct.pack('<II',kind,len(b))+b+b'\0'*((-len(b))%8)
        a=np.array([[1.,2.,3.],[4.,5.,6.]])
        payload=elt(6,struct.pack('<II',6,0))+elt(5,struct.pack('<ii',2,3))+elt(1,b'test')+elt(9,a.tobytes(order='F'))
        matrix=elt(14,payload)
        compressed=zlib.compress(matrix)
        header=b'MATLAB 5.0 test fixture'.ljust(116,b' ')+b'\0'*8+b'\0\1IM'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.mat'
            p.write_bytes(header+struct.pack('<II',15,len(compressed))+compressed)
            np.testing.assert_array_equal(loadmat(p)['test'],a)

    def test_conditional_mean_weight_invariance(self):
        rng=np.random.default_rng(123)
        targets=rng.normal(size=(12,9));mean=targets.mean(0)
        g=rng.normal(size=(5,9))
        w=np.eye(9)+3*g.T@g
        fitted=np.linalg.solve(w,w@mean)
        np.testing.assert_allclose(fitted,mean,atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
