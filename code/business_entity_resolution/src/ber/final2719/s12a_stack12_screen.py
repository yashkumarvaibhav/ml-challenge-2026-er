"""#27-12 components on top of scored #27-11 (France only): k0 screen of each change set.

Final assembly step 12a (run 2026-09-27 16:55 UTC). Verbatim copy of handoff/stack12.py; only the root path R is
parameterised (an absolute path on the build machine). Run from the package root.

Inputs : submissions/archive/sub-2026-09-27-{7,10,11}/matching_results.tsv, cache/test_source{1,2,3}.parquet,
         cache/v6/s2/test_pred_e5base2.parquet (standard stage-2 p2),
         cache/v6/s2/france_llm_veto_scores.parquet (Qwen2.5-7B score of France accepted pairs, ber.experiments.france_llm_veto)
Outputs: cache/v6/s2/stack12_v4adds.parquet, stack12_v4drops.parquet, stack12_llmveto.parquet
Recorded output: v4 adds 9064 (conflict 5849); v4 drops (excl rescue) 4037; LLM veto pairs still in 27-11: 567.
"""
import os
import re, sys
import polars as pl
from rapidfuzz.distance import Levenshtein
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
def pairs(tag):
    m=pl.read_csv(R+f'submissions/archive/{tag}/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
def hn(a):
    return sorted({t.lstrip('0') for t in re.findall(r'(?<![0-9])[0-9]{1,4}(?![0-9])', a or '') if t.lstrip('0')})
te=[pl.read_parquet(R+f'cache/test_source{j}.parquet') for j in (1,2,3)]
fr=te[0].filter(pl.col('country')=='France').select(pl.col('entity_id').alias('s1'),pl.col('business_address').alias('a1'))
rec=pl.concat(te[1:]).select(pl.col('entity_id').alias('m'),pl.col('business_address').alias('a2'))
b11=pairs('sub-2026-09-27-11').join(fr.select('s1'),on='s1'); b10=pairs('sub-2026-09-27-10').join(fr.select('s1'),on='s1')
b7=pairs('sub-2026-09-27-7').join(fr.select('s1'),on='s1')
rescue=pairs('sub-2026-09-27-11').join(fr.select('s1'),on='s1').join(b7,on=['s1','m'],how='anti')
p2=pl.read_parquet(R+'cache/v6/s2/test_pred_e5base2.parquet')
def conflict(d):
    d=d.join(fr,on='s1').join(rec,on='m')
    c=[bool(h1) and bool(h2) and bool(set(h2)-set(h1)) for h1,h2 in ((hn(a),hn(b)) for a,b in zip(d['a1'],d['a2']))]
    return d.with_columns(pl.Series('conf',c)).drop('a1','a2')
k=b11.group_by('s1').len().rename({'len':'k'})
def k0(d, base):  # share of pairs whose S1 has no other accepted record in base
    kk=base.group_by('s1').len().rename({'len':'k'})
    return round(d.join(kk,on='s1',how='left').with_columns(pl.col('k').fill_null(0))['k'].eq(1).mean(),4)
adds=conflict(b10.join(b11,on=['s1','m'],how='anti')); drops=b11.join(b10,on=['s1','m'],how='anti').join(rescue,on=['s1','m'],how='anti')
drops=conflict(drops).join(p2,on=['s1','m'],how='left')
print('v4 adds',adds.height,'conflict',adds['conf'].sum(),' non-conflict adds k0 (in base+adds):',k0(adds.filter(~pl.col('conf')).select('s1','m'),pl.concat([b11,adds.filter(~pl.col('conf')).select('s1','m')])))
print('v4 drops (excl rescue)',drops.height,'k0 in 27-11:',k0(drops.select('s1','m'),b11),' conflict in drops',drops['conf'].sum())
print(drops.with_columns(pl.col('p').cut([0.5,0.9,0.99,0.999]).alias('pb')).group_by('pb').agg(pl.len(),pl.col('s1').is_in(k.filter(pl.col('k')==1)['s1'].implode()).mean().round(4).alias('k0')).sort('pb'))
v=pl.read_parquet(R+'cache/v6/s2/france_llm_veto_scores.parquet').filter(pl.col('llm')<=0.02).select('s1','m')
v=v.join(b11,on=['s1','m'])
print('LLM veto pairs still in 27-11',v.height,'k0',k0(v,b11),' overlap with v4 drops',v.join(drops,on=['s1','m']).height)
adds.write_parquet(R+'cache/v6/s2/stack12_v4adds.parquet'); drops.write_parquet(R+'cache/v6/s2/stack12_v4drops.parquet'); v.write_parquet(R+'cache/v6/s2/stack12_llmveto.parquet')
