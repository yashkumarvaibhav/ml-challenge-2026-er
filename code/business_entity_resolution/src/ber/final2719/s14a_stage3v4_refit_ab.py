# Final assembly step 14a (run 2026-09-27 17:36 UTC). Verbatim copy of handoff/v4ab.py; only the root path R (and the
# sys.path entry derived from it) is parameterised (an absolute path on the build machine). Run from the package root
# after s13b with PYTHONPATH=code/business_entity_resolution/src.
# Inputs : cache/v6/s2/* stage-2/stage-3 band features and LLM band scores (README "Stage 3" / "Stage 3 v4"), holdout labels,
#          submissions/archive/sub-2026-09-27-{9,13}/matching_results.tsv
# Outputs: models/lgb_s3v4ab_llm_qwen7b_r3.txt (shipped as artifacts/lgb_s3v4ab_llm_qwen7b_r3.txt),
#          cache/v6/s2/test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet (p3), submissions/archive/sub-2026-09-27-14/
# Recorded output: consistency A-fit vs #27-9 US/India: + 76 - 0; A+B band rows 178193; A+B vs A-fit US/India decisions:
#          + 2023 - 1929; France rows unchanged vs #27-13: True; #27-14 vs #27-13 US/India: + 2090 - 1884.
#          (s14b_s15 then restricts #27-14 to the candidate file: sha 0057f29d...)
"""Stage-3 v4 refit on holdout halves A+B (learning curve: half->full half = +0.000078 on holdout); US/India rows for #27-14."""
import sys, os
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
sys.path.insert(0,R+'code/business_entity_resolution/src')
import polars as pl, lightgbm as lgb
from ber import llm_stage3 as S
from ber import config as C
from ber.io import load_train_pairs
from ber.model import decide
S2=S.S2
F=S.FEATS+S.EXTRA+['lpx','lllm15']
def pairs(t):
    m=pl.read_csv(R+f'submissions/archive/{t}/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
cty=pl.read_parquet(R+'cache/test_source1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1'})
usin=cty.filter(pl.col('country')!='France').select('s1')
# 1. consistency: A-fit v4 predictions + decide(0.5,0.75) == #27-9 US/India rows?
pa=pl.read_parquet(S2/'test_pred_e5base2_llm3v4_qwen7b_r3.parquet')
da=decide(pa.join(usin,on='s1'),0.5,0.75).select('s1','m')
_,p9=pairs('sub-2026-09-27-9'); p9u=p9.join(usin,on='s1')
print('consistency A-fit vs #27-9 US/India: +',da.join(p9u,on=['s1','m'],how='anti').height,' -',p9u.join(da,on=['s1','m'],how='anti').height,flush=True)
# 2. refit on A+B (world 0)
sp=pl.read_parquet(C.CACHE/'splits'/'train_s1_split.parquet')
hold=sp.filter(pl.col('is_holdout') & ~pl.col('is_ghost')).select(pl.col('entity_id').alias('s1'))
truth=load_train_pairs().select('s1','m')
sc=pl.read_parquet(S2/'holdout_pred_w0_e5base2.parquet').join(hold,on='s1')
ex=pl.read_parquet(S2/'s2feats_w0_e5base2.parquet',columns=['s1','m']+S.EXTRA)
_,band=S._v2_band(sc,'train',S2/f'llm_s3band_w0_{S.TAG}.parquet',ex)
band=S._v4_fill(S._v3_extra(band,'train',0),'holdout')
band=band.join(truth.with_columns(pl.lit(1).alias('y')),on=['s1','m'],how='left').with_columns(pl.col('y').fill_null(0))
print('A+B band rows',band.height,flush=True)
b=lgb.train(S.PARAMS,lgb.Dataset(band.select(F).to_numpy(),band['y'].to_numpy()),S.ROUNDS)
b.save_model(str(C.ROOT/'models'/f'lgb_s3v4ab_llm_{S.TAG}.txt'))
ts=pl.read_parquet(S2/'test_pred_e5base2.parquet')
d,tband=S._v2_band(ts,'test',S2/f'llm_s3band_test_{S.TAG}.parquet',S._test_extra())
tband=S._v4_fill(S._v3_extra(tband,'test'),'test')
p3=tband.select('s1','m').with_columns(pl.Series('p3',b.predict(tband.select(F).to_numpy())))
out=d.join(p3,on=['s1','m'],how='left').with_columns(pl.coalesce('p3','p').alias('p')).select('s1','m','p')
out.write_parquet(S2/f'test_pred_e5base2_llm3v4ab_{S.TAG}.parquet')
dab=decide(out.join(usin,on='s1'),0.5,0.75).select('s1','m')
print('A+B vs A-fit US/India decisions: +',dab.join(da,on=['s1','m'],how='anti').height,' -',da.join(dab,on=['s1','m'],how='anti').height,flush=True)
# 3. #27-14 = #27-13 France rows + A+B US/India decisions, then the same one-owner and per-source caps
m13,p13=pairs('sub-2026-09-27-13'); c1,c2=m13.columns
fr13=p13.join(cty.filter(pl.col('country')=='France').select('s1'),on='s1')
allp=pl.concat([fr13,dab]).unique().join(out.rename({'p':'score'}),on=['s1','m'],how='left').with_columns(pl.col('score').fill_null(0.0))
allp=allp.sort(['m','score','s1'],descending=[False,True,False]).unique(subset='m',keep='first')
allp=allp.with_columns(pl.col('m').str.slice(0,2).alias('src')).with_columns(pl.col('score').rank('ordinal',descending=True).over(['s1','src']).alias('rk'))
allp=allp.filter(~(((pl.col('src')=='S2')&(pl.col('rk')>5))|((pl.col('src')=='S3')&(pl.col('rk')>6)))).select('s1','m')
fr_after=allp.join(cty.filter(pl.col('country')=='France').select('s1'),on='s1')
print('France rows unchanged vs #27-13:',fr_after.sort(['s1','m']).equals(fr13.sort(['s1','m'])),flush=True)
g=allp.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
o=m13.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert o[c1].to_list()==m13[c1].to_list()
os.makedirs(R+'submissions/archive/sub-2026-09-27-14',exist_ok=True)
o.write_csv(R+'submissions/archive/sub-2026-09-27-14/matching_results.tsv',separator='\t',quote_style='never')
u13=p13.join(usin,on='s1'); u14=allp.join(usin,on='s1')
print('#27-14 vs #27-13 US/India: +',u14.join(u13,on=['s1','m'],how='anti').height,' -',u13.join(u14,on=['s1','m'],how='anti').height,flush=True)
