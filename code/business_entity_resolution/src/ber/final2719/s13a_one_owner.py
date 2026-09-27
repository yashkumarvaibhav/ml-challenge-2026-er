"""#27-13 part 1: one owner per record.

Train truth has 0 of 7.64M records with more than one S1, but #27-12 gave 417 records to 2+ S1s. Keep each such record
only at its highest-scored S1 (score = stage-3 v4 p where available, else standard stage-2 p2; ties by S1 id).

Final assembly step 13a (run 2026-09-27 17:25 UTC). Recovered verbatim from the build session; only the root path R is
parameterised (an absolute path on the build machine). Run from the package root after s12d.
Recorded output: dropped 456 pairs (France 351, India 94, US 11); S1s left empty by it: 43 (sha 83c41d9e..., then s13b).
"""
import polars as pl, os
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'; S=R+'cache/v6/s2/'
os.makedirs(R+'submissions/archive/sub-2026-09-27-13',exist_ok=True)
m=pl.read_csv(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
b=m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
p2=pl.read_parquet(S+'test_pred_e5base2.parquet').rename({'p':'p2'})
v4f=S+'test_pred_e5base2_llm3v4_qwen7b_r3.parquet'
d=b.join(p2,on=['s1','m'],how='left')
if os.path.exists(v4f):
    d=d.join(pl.read_parquet(v4f).rename({'p':'p4'}),on=['s1','m'],how='left').with_columns(pl.coalesce('p4','p2').alias('score'))
else:
    d=d.with_columns(pl.col('p2').alias('score'))
d=d.with_columns(pl.col('score').fill_null(0.0))
dup=d.join(d.group_by('m').len().filter(pl.col('len')>1).select('m'),on='m')
keep=dup.sort(['m','score','s1'],descending=[False,True,False]).unique(subset='m',keep='first').select('s1','m')
drop=dup.select('s1','m').join(keep,on=['s1','m'],how='anti')
new=b.join(drop,on=['s1','m'],how='anti')
assert new.group_by('m').len()['len'].max()==1
k=b.group_by('s1').len().rename({'len':'k'})
dd=drop.join(k,on='s1'); print('dropped',drop.height,'pairs; S1s left empty by it:',dd.filter(pl.col('k')==1).height)
g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
out=m.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert out[c1].to_list()==m[c1].to_list()
out.write_csv(R+'submissions/archive/sub-2026-09-27-13/matching_results.tsv',separator='\t',quote_style='never')
drop.write_parquet(S+'dedup13_drops.parquet')
cty=pl.read_parquet(R+'cache/test_source1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1'})
print(drop.join(cty,on='s1').group_by('country').len())
