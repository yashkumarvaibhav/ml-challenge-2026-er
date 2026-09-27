"""#27-14 candidate restriction and #27-15 France kept-back drop.

1. #27-14: keep only pairs present in candidate_pairs.tsv (the pipeline write step does the same; the A+B refit decisions
   of s14a contained 74 non-candidate pairs).
2. #27-15: drop the 2,180 France pairs of the teammate's house-number veto (record carries a 1-4 digit house number the S1
   lacks) with standard stage-2 p2 > 0.999, which #27-11 had kept back.

Final assembly step 14b/15 (run 2026-09-27 17:39 UTC). Recovered verbatim from the build session; only the root path R
is parameterised (an absolute path on the build machine). Run from the package root after s14a.
Inputs : submissions/archive/sub-2026-09-27-13/candidate_pairs.tsv, submissions/archive/sub-2026-09-27-14/matching_results.tsv,
         pinaksh/sub7_fr/france_numveto_dropped.parquet (teammate list, staged from artifacts/teammate/),
         cache/v6/s2/test_pred_e5base2.parquet (p2)
Recorded output: 27-14: removed non-candidate pairs 74 (sha 0057f29d...); 27-15: dropped kept-back 2180 (sha 58852be0...);
         official validator PASS on both.
"""
import polars as pl, os
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'; S2=R+'cache/v6/s2/'
def read(p):
    m=pl.read_csv(p,separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
_,cand=read(R+'submissions/archive/sub-2026-09-27-13/candidate_pairs.tsv')
m14,p14=read(R+'submissions/archive/sub-2026-09-27-14/matching_results.tsv'); c1,c2=m14.columns
out14=p14.join(cand,on=['s1','m'],how='semi')
print('27-14: removed non-candidate pairs',p14.height-out14.height)
def write(pairs,tag):
    g=pairs.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
    o=m14.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
    assert o[c1].to_list()==m14[c1].to_list()
    os.makedirs(R+f'submissions/archive/{tag}',exist_ok=True)
    o.write_csv(R+f'submissions/archive/{tag}/matching_results.tsv',separator='\t',quote_style='never')
write(out14,'sub-2026-09-27-14')
kb=pl.read_parquet(R+'pinaksh/sub7_fr/france_numveto_dropped.parquet').join(pl.read_parquet(S2+'test_pred_e5base2.parquet'),on=['s1','m'],how='left').filter(pl.col('p')>0.999).select('s1','m')
out15=out14.join(kb,on=['s1','m'],how='anti'); print('27-15: dropped kept-back',out14.height-out15.height)
write(out15,'sub-2026-09-27-15')
