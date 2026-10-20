"""Read completed local evidence and make the cloud go/no-go estimate explicit."""
import json,csv,math,hashlib
from pathlib import Path
P=Path('outputs/research/k4s11/r4')
fixed=[json.loads((P/f'fixed_s{s}.json').read_text()) for s in (5,6,7)]
rows=sum([d['rows'] for d in fixed],[])
with (P/'fixed.csv').open('w') as fh:
 out=[]
 for r in rows:
  z={k:v for k,v in r.items() if not isinstance(v,list)}
  for k in ('read_rel2_full','read_rel2_selected','read_energy'):
   for ch,v in zip(('d4','d31','d22','d211'),r[k]):z[k+'_'+ch]=v
  out.append(z)
 w=csv.DictWriter(fh,fieldnames=out[0]);w.writeheader();w.writerows(out)
chains=[json.loads((P/f'chain_s{s}_k2.json').read_text()) for s in (5,6,7)]
assert all(len(x['rows'])==15 for x in chains)
with (P/'chain.csv').open('w') as fh:
 rr=sum([x['rows'] for x in chains],[]);w=csv.DictWriter(fh,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
meter=json.loads((P/'meter_s1_p1.json').read_text());old=json.loads(Path('outputs/research/k4s11/r3/summary.json').read_text())
local=[r for r in rows if r['layer']>=13]
summary=[]
for c in ('all',1,2):
 for mode in ('exact','rsvd','mean','power1','power2','svd'):
  rr=[r for r in local if r['sources']==c and r['direction']==mode];rat=[]
  for r in rr:
   best=next(t for t in local if t['seed']==r['seed'] and t['layer']==r['layer'] and t['sources']==c and t['direction']=='svd')
   rat.append((1-r['birth_rel2_selected'])/(1-best['birth_rel2_selected']))
  def ran(v):return [min(v),max(v)]
  summary.append(dict(sources=c,direction=mode,birth_retained=ran([1-r['birth_rel2_full'] for r in rr]),first_d4_retained=ran([1-r['read_rel2_full'][0] for r in rr]),fraction_of_optimal_rank1_energy=ran(rat)))
# Optimistic transfer heuristic, NOT a mathematical bound or a measured 1024 raw.
# Use the largest observed first-read retained fraction and the full-suffix gain.
alpha=max(1-e for r in local if r['sources']==1 and r['direction']=='power1' for e in r['read_rel2_full'])
gains=[1-1.245/2.262,1-1.334/2.015];proxy=[]
for i,r in enumerate(old['rows']):
 g=gains[i] if i<2 else max(gains);u=r['B']['U'];raw_ratio=1-alpha*g
 proxy.append(dict(index=i,full_suffix_gain_proxy=g,source='public mini'+str(i) if i<2 else 'optimistic best mini0 proxy; mini2/3 full-suffix exact not measured here',
  first_read_retention_proxy=alpha,estimated_raw_ratio=raw_ratio,estimated_raw=r['B']['raw']*raw_ratio,
  amplitude_optimistic_raw_ratio=1-math.sqrt(alpha)*g,threshold=u/(u+meter['delta_units']),threshold_if_delta90=u/(u+90),estimated_pass=raw_ratio<u/(u+meter['delta_units'])))
result=dict(fixed_cells=len(rows),main_k2_cells=len(local),chain_cells=sum(len(d['rows']) for d in chains),ranges=summary,meter=meter,cloud_estimates=proxy,
 cloud_run=False,cloud_spend_usd=0,decision='retain mechanism result; selected-source recovery and cost give no forecast adj pass',
 max_local_wall=max([d['wall'] for d in fixed+chains]+[meter['wall']]),max_local_rss_bytes=max([d['rss_bytes'] for d in fixed+chains]+[meter['rss_bytes']]))
(P/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
