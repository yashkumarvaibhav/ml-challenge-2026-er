"""#27-12 build: France v4 selective stack on scored #27-11 (France rows only).

Final assembly step 12b (run 2026-09-27 16:57 UTC). Recovered verbatim from the build session; only the root path R is
parameterised (an absolute path on the build machine). Run from the package root after s12a.

Drops (885): the #27-10 stage-3-v4 France drops with standard stage-2 p2 <= 0.5 or p2 in (0.99, 0.999], plus the France
LLM-veto pairs (Qwen2.5-7B score <= 0.02). Adds: #27-10 France additions without a house-number conflict whose record is
not owned elsewhere after the drops.
Recorded output: drops 885 adds 3136 changed countries ['France'] pairs 5831234 -> 5833485 (sha 31d95ed2..., then
fixed by s12c and s12d).
"""
import os
import polars as pl
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
os.makedirs(R+'submissions/archive/sub-2026-09-27-12',exist_ok=True)
m=pl.read_csv(R+'submissions/archive/sub-2026-09-27-11/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
b=m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
dr=pl.read_parquet(R+'cache/v6/s2/stack12_v4drops.parquet'); v=pl.read_parquet(R+'cache/v6/s2/stack12_llmveto.parquet')
ad=pl.read_parquet(R+'cache/v6/s2/stack12_v4adds.parquet').filter(~pl.col('conf')).select('s1','m')
D=pl.concat([dr.filter(pl.col('p')<=0.5).select('s1','m'),dr.filter((pl.col('p')>0.99)&(pl.col('p')<=0.999)).select('s1','m'),v]).unique()
kept=b.join(D,on=['s1','m'],how='anti')
ad=ad.join(kept.select('m').unique(),on='m',how='anti').unique(subset='m',keep='first')
new=pl.concat([kept,ad]).unique()
assert new.group_by('m').len().filter(pl.col('len')>1).height <= 492
g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
out=m.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert out[c1].to_list()==m[c1].to_list()
out.write_csv(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv',separator='\t',quote_style='never')
cty=pl.read_parquet(R+'cache/test_source1.parquet').select(pl.col('entity_id').alias('s1'),'country')
chg=pl.concat([b.join(new,on=['s1','m'],how='anti'),new.join(b,on=['s1','m'],how='anti')]).join(cty,on='s1')
print('drops',D.height,'adds',ad.height,'changed countries',chg['country'].unique().to_list(),'pairs',b.height,'->',new.height)
