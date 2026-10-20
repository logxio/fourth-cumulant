"""One finite small-width chain or MC batch per process (under 290s)."""
import argparse, json, time, resource, signal
from pathlib import Path
from pair_read import torch, np, f, PairProbe


def weights(n, seed, layers):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((n, n))*(2/n)**.5).astype('f') for _ in range(layers)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['original','exact','tree','svd','rsvd','random','mc'])
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--start', type=int, default=4)
    p.add_argument('--prefix', choices=['full','empty'], default='full')
    p.add_argument('--rank', type=int, default=1)
    p.add_argument('--sketch-seed', type=int, default=111)
    p.add_argument('--samples', type=int, default=1048576)
    p.add_argument('--batch-seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    signal.alarm(285)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    start = time.perf_counter()
    W = weights(a.n, a.seed, a.layers)
    if a.mode == 'mc':
        # Independent antithetic pairs. Estimate variance of pair means,
        # rather than incorrectly treating the two samples as independent.
        rng = torch.Generator().manual_seed(55000+a.batch_seed)
        wt = [torch.from_numpy(w) for w in W]
        acc = torch.zeros(a.layers, a.n, dtype=torch.float64)
        sq = torch.zeros(a.n, dtype=torch.float64)
        count = 0
        for begin in range(0, a.samples//2, 8192):
            size = min(8192, a.samples//2-begin)
            x = torch.randn(size, a.n, generator=rng)
            h = torch.cat((x, -x))
            for layer, w in enumerate(wt):
                h = torch.relu(h @ w)
                pair = (h[:size].double()+h[size:].double())/2
                acc[layer] += pair.sum(0)
            sq += pair.square().sum(0)
            count += size
        mean = acc/count
        var = (sq-count*mean[-1].square())/(count-1)
        np.savez(a.out.with_suffix('.npz'), pred=mean.numpy(), pair_var=var.numpy(), pairs=count)
        result = dict(samples=2*count, pairs=count, floor=float(var.mean()/count))
    else:
        K = f.coerce_input({1:torch.zeros(a.n,dtype=torch.float64),
                            2:torch.eye(a.n,dtype=torch.float64)}, k_max=4, kind=f.Kind.SIMPLE)
        rec = None if a.mode=='original' else PairProbe(a.mode,a.rank,a.sketch_seed)
        preds, rows = [], []
        for layer, w in enumerate(W):
            if rec and layer==a.start: rec.install()
            if layer<a.start:
                f.SPEC=f.parse_spec('full') if a.prefix=='full' else []
            else:
                f.SPEC = f.parse_spec('slice4+slice31,!slice22+slice211') if layer<a.layers-1 else []
            f.LAST_LAYER = layer==a.layers-1
            if rec:rec.layer=layer
            t1=time.perf_counter()
            K=f.linear_kprop(K,torch.as_tensor(w.T.copy(),dtype=torch.float64),k_max=4,set_metric=2.)
            if rec and layer>=a.start:rec.start_layer()
            K=f.nonlin(K,nonlin_wick_coef=f.WICK_COEF_D['relu'])
            preds.append(K[1].to_tensor().numpy().copy())
            assert np.isfinite(preds[-1]).all(), f'nonfinite layer {layer}'
            rows.append(dict(layer=layer,seconds=time.perf_counter()-t1))
        np.save(a.out.with_suffix('.npy'),np.stack(preds))
        result=dict(layers=rows,births=rec.rows if rec else [],reads=rec.reads if rec else [])
        if rec:rec.uninstall()
    result.update(mode=a.mode,n=a.n,seed=a.seed,rank=a.rank,sketch_seed=a.sketch_seed,
                  start=a.start,prefix=a.prefix,depth=a.layers,seconds=time.perf_counter()-start,
                  rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('layers','births','reads')}),flush=True)


if __name__=='__main__':main()
