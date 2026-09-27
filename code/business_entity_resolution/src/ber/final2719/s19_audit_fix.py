"""#27-19 (built as #27-18, then renamed): #27-17 corrected per the independent audit.

The audit (artifacts/final2719/codex_restore17.tsv, codex_negative16.tsv) found that #27-17 had restored 30 US/India
pairs that the stage-3 v4 A+B refit had rejected (p3 0.29-0.50, below t_any) rather than the teammate blend; their
holdout-analog precision is 39-41% (Wilson 95% upper bound < 0.5 break-even), so they are removed again. It also found
3 negative-expected-F drops that #27-17 had not restored; the one whose record is still unowned is restored.

Final assembly step 19 (run 2026-09-27 18:06 UTC). Recovered verbatim from the build session; only the root path R is
parameterised (an absolute path on the build machine). Run from the package root after s17.
Inputs : submissions/archive/sub-2026-09-27-17/matching_results.tsv, handoff/codex_restore17.tsv,
         handoff/codex_negative16.tsv (both staged by reproduce_final_2719.sh from artifacts/final2719/)
Recorded output: negative-EV drops not restored (excluding refit): 3; 27-18 pairs 5829875, restored extra 1;
         sha 720dce1652fc585ed3569dcddd6013e5d016f8b59dd725e13ebab8319ee301e0 (official validator PASS).
         The directory was then renamed sub-2026-09-27-18 -> sub-2026-09-27-19 (the uploaded, best-scoring submission).
"""
import polars as pl, os
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
def read(p):
    m=pl.read_csv(p,separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
m17,B17=read(R+'submissions/archive/sub-2026-09-27-17/matching_results.tsv'); c1,c2=m17.columns
r=pl.read_csv(R+'handoff/codex_restore17.tsv',separator='\t',infer_schema_length=0)
new=B17.join(r.filter(pl.col('source')=='refit_or_candidate').select('s1','m'),on=['s1','m'],how='anti')
n=pl.read_csv(R+'handoff/codex_negative16.tsv',separator='\t',infer_schema_length=0)
want=n.select('s1',pl.col('restorable_pairs').str.split(',').alias('m'),'drop_sources').explode('m').filter(pl.col('m')!='')
miss=want.join(new,on=['s1','m'],how='anti').filter(~pl.col('drop_sources').str.contains('refit'))
print('negative-EV drops not restored (excluding refit):',miss.height, miss.select('drop_sources').to_series().value_counts().to_dicts())
miss=miss.select('s1','m').join(new.select('m').unique(),on='m',how='anti')
new=pl.concat([new,miss]).unique()
assert new.group_by('m').len()['len'].max()==1
g=new.with_columns(pl.col('m').str.slice(0,2).alias('src')).group_by('s1','src').len()
assert g.filter(((pl.col('src')=='S2')&(pl.col('len')>5))|((pl.col('src')=='S3')&(pl.col('len')>6))).height==0
print('27-18 pairs',new.height,' restored extra',miss.height)
g2=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
o=m17.select(c1).join(g2,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert o[c1].to_list()==m17[c1].to_list()
os.makedirs(R+'submissions/archive/sub-2026-09-27-18',exist_ok=True)
o.write_csv(R+'submissions/archive/sub-2026-09-27-18/matching_results.tsv',separator='\t',quote_style='never')
