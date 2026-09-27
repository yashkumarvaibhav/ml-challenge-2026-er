"""#27-13 part 2: per-S1 source caps.

Train truth never has more than 5 Source-2 or 6 Source-3 records per S1; surplus pairs (lowest score, score = stage-3 v4 p
where available, else stage-2 p2) are dropped.

Final assembly step 13b (run 2026-09-27 17:27 UTC). Recovered verbatim from the build session; only the root path R is
parameterised. Run from the package root after s13a.
Recorded output: cap-excess drops 57 pairs 5832985 -> 5832928; final #27-13 sha e59f8e99... (official validator PASS).
"""
import os
import polars as pl
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'; S=R+'cache/v6/s2/'
m=pl.read_csv(R+'submissions/archive/sub-2026-09-27-13/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
b=m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
p2=pl.read_parquet(S+'test_pred_e5base2.parquet').rename({'p':'p2'}); p4=pl.read_parquet(S+'test_pred_e5base2_llm3v4_qwen7b_r3.parquet').rename({'p':'p4'})
d=b.join(p2,on=['s1','m'],how='left').join(p4,on=['s1','m'],how='left').with_columns(pl.coalesce('p4','p2').fill_null(0.0).alias('score'),pl.col('m').str.slice(0,2).alias('src'))
d=d.with_columns(pl.col('score').rank('ordinal',descending=True).over(['s1','src']).alias('rk'))
cap=d.filter(((pl.col('src')=='S2')&(pl.col('rk')>5))|((pl.col('src')=='S3')&(pl.col('rk')>6))).select('s1','m')
new=b.join(cap,on=['s1','m'],how='anti'); print('cap-excess drops',cap.height,'pairs',b.height,'->',new.height)
g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
out=m.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert out[c1].to_list()==m[c1].to_list()
out.write_csv(R+'submissions/archive/sub-2026-09-27-13/matching_results.tsv',separator='\t',quote_style='never')
cap.write_parquet(S+'cap13_drops.parquet')
