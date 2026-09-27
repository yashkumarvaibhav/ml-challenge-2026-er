# Final assembly step 17 (run 2026-09-27 17:59 UTC). Verbatim copy of handoff/refine17.py; only the root path R is
# parameterised (an absolute path on the build machine). Run from the package root after s16.
# Per-S1 expected-F0.5 refinement: for every S1 whose pairs were dropped between #27-11 and #27-16 by a calibrated rule
# (France kept-back house-number veto, false share 0.40; teammate-blend removals, 0.35 for p3 > 0.75 else 0.45), compute
# the exact expected change in that S1's F0.5 (binomial over which drops are false) and undo the drops with negative
# expected change (mostly drops that would leave the S1 empty), if the record is still unowned.
# Inputs : submissions/archive/sub-2026-09-27-{11,16}/matching_results.tsv, cache/test_source1.parquet,
#          cache/v6/s2/test_pred_e5base2.parquet (p2), cache/v6/s2/test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet (p3, s14a),
#          pinaksh/sub7_fr/france_numveto_dropped.parquet and pinaksh/sub11_plus/sub11_plus/matching_results.tsv
#          (teammate files; staged by reproduce_final_2719.sh from artifacts/teammate/)
# Recorded output: candidate drop S1s 4267, negative-EV S1s 436 (blend 221, kept_back 215); undo pairs 541; expected gain
#          vs #27-16 ~ 5.3e-05; #27-17 sha 3e6a274e... (official validator PASS)
"""#27-17: #27-16 with per-S1 expected-F refinement of the drops vs #27-11 (undo drops whose expected ΔF < 0)."""
import polars as pl, os
from math import comb
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'; S2=R+'cache/v6/s2/'
def read(p):
    m=pl.read_csv(p,separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
def dF_drop(k,d,f):
    r=k-d; e=0.0
    for j in range(d+1):
        pj=comb(d,j)*(1-f)**j*f**(d-j); T=r+j
        b=(1.0 if k==0 else 0.0) if T==0 else 1.25*T/(0.25*T+k)
        a=(1.0 if r==0 else 0.0) if T==0 else (1.25*r/(0.25*T+r) if r>0 else 0.0)
        e+=pj*(a-b)
    return e
m16,B16=read(R+'submissions/archive/sub-2026-09-27-16/matching_results.tsv'); c1,c2=m16.columns
_,B11=read(R+'submissions/archive/sub-2026-09-27-11/matching_results.tsv')
cty=pl.read_parquet(R+'cache/test_source1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1'})
p3=pl.read_parquet(S2+'test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet').rename({'p':'p3'})
p2=pl.read_parquet(S2+'test_pred_e5base2.parquet').rename({'p':'p2'})
kb=pl.read_parquet(R+'pinaksh/sub7_fr/france_numveto_dropped.parquet').join(p2,on=['s1','m'],how='left').filter(pl.col('p2')>0.999).select('s1','m').with_columns(pl.lit('kept_back').alias('src'))
_,P=read(R+'pinaksh/sub11_plus/sub11_plus/matching_results.tsv')
blend_rm=B11.join(P,on=['s1','m'],how='anti').join(cty.filter(pl.col('country')!='France').select('s1'),on='s1').select('s1','m').with_columns(pl.lit('blend').alias('src'))
drops=B11.join(B16,on=['s1','m'],how='anti')
drops=drops.join(pl.concat([kb,blend_rm]).unique(subset=['s1','m']),on=['s1','m'],how='left').with_columns(pl.col('src').fill_null('other')).join(p3,on=['s1','m'],how='left')
# false-share per drop: kept_back 0.40; blend by p3 band (posterior with blend evidence); others left as is
drops=drops.with_columns(pl.when(pl.col('src')=='kept_back').then(0.40)
    .when((pl.col('src')=='blend')&(pl.col('p3')>0.75)).then(0.35)
    .when(pl.col('src')=='blend').then(0.45).otherwise(None).alias('f'))
cand=drops.filter(pl.col('f').is_not_null())
# S1-level: k = pairs in #27-16 + candidate drops of that S1 (the state before these drops), d = candidate drops
k16=B16.group_by('s1').len().rename({'len':'k16'})
g=cand.group_by('s1').agg(pl.len().alias('d'),pl.col('f').mean().alias('f'),pl.col('src').first()).join(k16,on='s1',how='left').with_columns(pl.col('k16').fill_null(0))
g=g.with_columns((pl.col('k16')+pl.col('d')).alias('k'))
g=g.with_columns(pl.struct('k','d','f').map_elements(lambda r: dF_drop(r['k'],r['d'],r['f']),return_dtype=pl.Float64).alias('dF'))
bad=g.filter(pl.col('dF')<0)
print('candidate drop S1s',g.height,' negative-EV S1s',bad.height,' by src',bad.group_by('src').agg(pl.len(),pl.col('dF').sum().round(2)).to_dicts())
print('k distribution of negative-EV S1s:',bad.group_by('k').len().sort('k').to_dicts())
undo=cand.join(bad.select('s1'),on='s1').select('s1','m')
# re-add only if the record is still unowned
undo=undo.join(B16.select('m').unique(),on='m',how='anti')
new=pl.concat([B16,undo]).unique()
assert new.group_by('m').len()['len'].max()==1
N=1917670
print('undo pairs',undo.height,' expected gain vs #27-16 ≈',round(-bad['dF'].sum()/N,6))
g2=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
o=m16.select(c1).join(g2,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert o[c1].to_list()==m16[c1].to_list()
os.makedirs(R+'submissions/archive/sub-2026-09-27-17',exist_ok=True)
o.write_csv(R+'submissions/archive/sub-2026-09-27-17/matching_results.tsv',separator='\t',quote_style='never')
