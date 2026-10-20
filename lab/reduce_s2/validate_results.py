"""Audit the exact eight-point sweep after full cloud recovery."""
import argparse,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXPECTED={'baseline_w9','w9_n1','w9_n4','w9_n16','w6_n4_c4','w6_n8_c8','full_n4_c4','full_n8_c8'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('data',type=Path);a=parser.parse_args()
    files=list((a.data/'outputs/net0').glob('*.json'))
    assert {f.stem for f in files}==EXPECTED
    truth=np.load(ROOT/'data/mini_truth.npz')['all_layer_means'][0]
    evidence=[]
    for f in sorted(files):
        d=json.loads(f.read_text());pred=np.load(f.with_suffix('.npy'))
        assert pred.shape==truth.shape and np.isfinite(pred).all()
        raw=float(np.mean((pred[-1]-truth[-1])**2));assert math.isclose(raw,d['mse'],rel_tol=1e-9)
        life={fam:{int(b):n for b,n in bs.items()} for fam,bs in d['plan']['life'].items()}
        for row in d['layers']:
            layer=row['layer'];expected={'seed':1}
            for fam,births in life.items():
                live=sum(b<=layer and (n>=14-b or layer-b<=n) for b,n in births.items())
                if live:expected[fam]=live*(2 if fam=='tree' else 1)
            assert row['live']==expected,(f,row['layer'],row['live'],expected)
            assert row['dropped']==0,(f,row['layer'])
        comp=d['compression']
        expected_comp={(fam,b) for fam in ('nontree','ck3') if d['ranks'][fam]>0 for b in life[fam]}
        assert {(v['family'],v['birth']) for v in comp}==expected_comp
        assert len(comp)==len(expected_comp)
        assert all(0<=v['kept']<=1.00001 for v in comp)
        for event in d['replacement_events']:
            assert (event['family'],event['birth']) in expected_comp
            assert event['birth']<=event['layer']<=15
            assert math.isfinite(event['units']) and event['units']>0
        evidence.append({'point':f.stem,'raw':raw,'family_counts':'exact at all 12 suffix layers','compressed_births':len(comp)})
    print(json.dumps({'points':evidence,'status':'passed'},indent=2))
if __name__=='__main__':main()
