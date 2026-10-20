from upper_eval import *
from direction_probe import load_probe
n=64;seed=5;probe=FullPair(2);est=load_probe(probe);budget=2**55;est.setup(SetupContext(width=n,depth=16,flop_budget=budget,api_version='1',submission_dir=str(ROOT/'lab/cand_k4x/s11')))
rng=np.random.default_rng(seed);weights=[(rng.standard_normal((n,n))*(2/n)**.5).astype('f') for _ in range(16)]
mlp=MLP(width=n,depth=16,weights=[f.array(w) for w in weights],seed=seed,name='check')
ctx=flops.BudgetContext(flop_budget=budget,quiet=True);probe.ctx=ctx;signal.alarm(280)
with ctx:pred=np.asarray(est.predict(mlp,budget)).astype(float)
ref=np.load(ROOT/'outputs/research/k4s11/r4/chain_s5_k2_c0_power1.npy')
dict_=dict(max_abs=float(np.max(abs(pred-ref))),final_rel2=float(np.sum((pred[-1]-ref[-1])**2)/np.sum(ref[-1]**2)),births=probe.birth_rows,reference='outputs/research/k4s11/r4/chain_s5_k2_c0_power1.npy')
assert dict_['final_rel2']<1e-6,dict_
(ROOT/'outputs/research/k4s12/upper_check.json').write_text(json.dumps(dict_,indent=2));print(dict_)
