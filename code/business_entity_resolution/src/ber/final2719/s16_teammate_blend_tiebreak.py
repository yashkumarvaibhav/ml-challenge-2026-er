"""#27-16: the teammate's US/India blend used as a tie-breaker where our stage 3 v4 (A+B refit, p3) is uncertain.

The teammate's sub11_plus US/India rows come from a blend of our standard stage-2 score and the teammate's CPU pipeline
(0.7 * logit(e5base2 stage 2) + 0.3 * logit(CPU pipeline), decision 0.6 / 0.75; src/teammate_reference/cpu_pipeline/src/er/sub11plus.py,
which reuses er.teamblend.combine; rebuilt by cpu_pipeline/run_sub11_plus.sh).
Relative to #27-11 (US/India rows):
  - remove pairs the blend rejects when our p3 <= 0.9 (holdout accepted precision at p3 0.75-0.9 is 81%, break-even ~76%);
  - add pairs the blend accepts when our p3 > 0.5, only if the pair is a candidate and the record is unowned;
  - then re-apply the per-S1 source caps (S2 <= 5, S3 <= 6).

Final assembly step 16 (run 2026-09-27 17:55 UTC). Recovered verbatim from the build session; only the root path R is
parameterised (an absolute path on the build machine). Run from the package root after s14b_s15.
Inputs : submissions/archive/sub-2026-09-27-{11,15}/, pinaksh/sub11_plus/sub11_plus/matching_results.tsv (teammate output,
         staged by reproduce_final_2719.sh), cache/test_source1.parquet, cache/v6/s2/test_pred_e5base2.parquet (p2),
         cache/v6/s2/test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet (p3, from s14a)
Recorded output: 27-16 vs 27-15: removed 2278 added 763 pairs 5830880 -> 5829363; sha 41de81e5... (official validator PASS)
"""
import polars as pl, os
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'; S2=R+'cache/v6/s2/'
def read(p):
    m=pl.read_csv(p,separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
cty=pl.read_parquet(R+'cache/test_source1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1'})
usin=cty.filter(pl.col('country')!='France').select('s1')
_,P=read(R+'pinaksh/sub11_plus/sub11_plus/matching_results.tsv'); P=P.join(usin,on='s1')
_,A=read(R+'submissions/archive/sub-2026-09-27-11/matching_results.tsv'); A=A.join(usin,on='s1')
m15,Q=read(R+'submissions/archive/sub-2026-09-27-15/matching_results.tsv'); c1,c2=m15.columns
_,cand=read(R+'submissions/archive/sub-2026-09-27-15/candidate_pairs.tsv')
p3=pl.read_parquet(S2+'test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet').rename({'p':'p3'})
rm=A.join(P,on=['s1','m'],how='anti').join(p3,on=['s1','m'],how='left').filter(pl.col('p3')<=0.9).select('s1','m')
ad=P.join(A,on=['s1','m'],how='anti').join(p3,on=['s1','m'],how='left').filter(pl.col('p3')>0.5).select('s1','m','p3')
new=Q.join(rm,on=['s1','m'],how='anti')
n_rm=Q.height-new.height
ad=ad.join(cand,on=['s1','m'],how='semi').join(new.select('m').unique(),on='m',how='anti').join(new,on=['s1','m'],how='anti')
ad=ad.sort('p3',descending=True).unique(subset='m',keep='first')
new=pl.concat([new,ad.select('s1','m')]).unique()
# caps
sc=new.join(p3,on=['s1','m'],how='left').join(pl.read_parquet(S2+'test_pred_e5base2.parquet').rename({'p':'p2'}),on=['s1','m'],how='left').with_columns(pl.coalesce('p3','p2').fill_null(0.0).alias('sc'),pl.col('m').str.slice(0,2).alias('src'))
sc=sc.with_columns(pl.col('sc').rank('ordinal',descending=True).over(['s1','src']).alias('rk'))
new=sc.filter(~(((pl.col('src')=='S2')&(pl.col('rk')>5))|((pl.col('src')=='S3')&(pl.col('rk')>6)))).select('s1','m')
assert new.group_by('m').len()['len'].max()==1
print('27-16 vs 27-15: removed',n_rm,' added',ad.height,' pairs',Q.height,'->',new.height)
g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
o=m15.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert o[c1].to_list()==m15[c1].to_list()
os.makedirs(R+'submissions/archive/sub-2026-09-27-16',exist_ok=True)
o.write_csv(R+'submissions/archive/sub-2026-09-27-16/matching_results.tsv',separator='\t',quote_style='never')
