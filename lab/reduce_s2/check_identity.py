"""Regression for the observed stale-object-id family collision."""
import gc
from types import SimpleNamespace
import torch
import run

def main():
    b=torch.zeros(2,2,2)
    tree=SimpleNamespace(Bb=b,types=('slice211',))
    # Force precisely the previously ambiguous state: a tree's id appears in
    # the stale residual set. Family is now owned by the live tensor itself.
    assert run.family_of(tree,{id(b)})=='tree'
    b._reduce_s2_family='nontree'
    assert run.family_of(tree,set())=='nontree'
    for _ in range(100):
        temp=torch.empty(2,2,2)
        stale={id(temp)}
        del temp
        other=torch.empty(2,2,2)
        assert run.family_of(SimpleNamespace(Bb=other,types=('slice211',)),stale)=='tree'
    print('identity collision regression passed')
if __name__=='__main__':main()
