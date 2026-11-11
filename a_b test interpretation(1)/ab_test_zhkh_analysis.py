import pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
f='/mnt/user-data/uploads/dannye-dlia-keisa-po-pa-cebbefc3-5b29-4f88-b6d4-3336c88d743f__1_.xlsx'
u=pd.read_excel(f,sheet_name=0); p=pd.read_excel(f,sheet_name=1)
print(p[p.user_id.isin([9001,9002,9003,9004])][['payment_id','user_id','group','step4_success']])
u_c=u[u.group.notna()].copy()
u_c.loc[(u_c.age<14)|(u_c.age>90),'age']=np.nan
p_c=p[p.user_id.notna()&(p.payment_id!=99002)].copy(); p_c['user_id']=p_c.user_id.astype(int)
p_c=p_c.drop(columns='group').merge(u_c[['user_id','group','device_type','city','age']],on='user_id',how='inner')
p_c['y']=p_c.step4_success.notna().astype(int); p_c['s2']=p_c.step2_entered.notna().astype(int); p_c['s3']=p_c.step3_confirmed.notna().astype(int)
p_c['B']=(p_c.group=='B').astype(int)
print(len(p),'->',len(p_c),'users',len(u),'->',len(u_c))
g=p_c.groupby('group'); fun=pd.DataFrame({'opened':g.size(),'entered':g.s2.sum(),'confirmed':g.s3.sum(),'success':g.y.sum()}); print(fun)
print('1->2',(fun.entered/fun.opened).round(4).to_dict(),'2->3',(fun.confirmed/fun.entered).round(4).to_dict(),'3->4',(fun.success/fun.confirmed).round(4).to_dict(),'1->4',(fun.success/fun.opened).round(4).to_dict())

def ztest(x1,n1,x0,n0):
    p1,p0=x1/n1,x0/n0; pp=(x1+x0)/(n1+n0); se=np.sqrt(pp*(1-pp)*(1/n1+1/n0)); z=(p1-p0)/se; return p1-p0,z,2*stats.norm.sf(abs(z))
def ratio_diff(d,col,cond=None):
    """cluster-robust diff in pooled proportion (ratio estimator, delta method), clusters=user"""
    if cond: d=d[d[cond]==1]
    out={}
    for gname,dd in d.groupby('group'):
        c=dd.groupby('user_id')[col].agg(['sum','count']); N=len(c); m=c['count'].mean(); r=c['sum'].sum()/c['count'].sum()
        resid=c['sum']-r*c['count']; var=(N/(N-1))*(resid**2).sum()/(c['count'].sum()**2)
        out[gname]=(r,var,N,c['count'].sum())
    diff=out['B'][0]-out['A'][0]; se=np.sqrt(out['A'][1]+out['B'][1]); z=diff/se
    return out['A'][0],out['B'][0],diff,diff-1.96*se,diff+1.96*se,2*stats.norm.sf(abs(z)),se
n_=fun.opened
print('naive attempt z',ztest(fun.success.B,n_.B,fun.success.A,n_.A))
print('cluster-robust 1->4: A,B,diff,lo,hi,p',np.round(ratio_diff(p_c,'y')[:6],5))
for col,cond in [('s2',None),('s3','s2'),('y','s3')]:
    print('step',col,'|',cond,np.round(ratio_diff(p_c,col,cond)[:6],5))
# DEFF
uu=p_c.groupby('user_id').agg(n=('y','size'),succ=('y','sum'),group=('group','first')).reset_index(); uu['conv']=uu.succ/uu.n
a=uu[uu.group=='A']; b=uu[uu.group=='B']
print('users w/ attempts',len(a),len(b),'attempts/user',a.n.mean().round(3),b.n.mean().round(3),stats.mannwhitneyu(b.n,a.n).pvalue)
print('any success',(a.succ>0).mean(),(b.succ>0).mean(),ztest((b.succ>0).sum(),len(b),(a.succ>0).sum(),len(a)))
print('mean user conv',a.conv.mean(),b.conv.mean(),'Welch',stats.ttest_ind(b.conv,a.conv,equal_var=False),'MW',stats.mannwhitneyu(b.conv,a.conv).pvalue)
print('successes/user',a.succ.mean(),b.succ.mean(),stats.mannwhitneyu(b.succ,a.succ).pvalue)
# Users without any payments (all 1503 incl 3 w/o attempts)
allu=u_c.merge(uu[['user_id','succ']],on='user_id',how='left').fillna({'succ':0}); print('ITT any success over all assigned users',allu.groupby('group').succ.apply(lambda s:(s>0).mean()).to_dict(), allu.group.value_counts().to_dict())
# bootstrap by users
rng=np.random.default_rng(42)
ga=p_c[p_c.group=='A'].groupby('user_id').y.agg(['sum','count']).values; gb=p_c[p_c.group=='B'].groupby('user_id').y.agg(['sum','count']).values
d=[]
for _ in range(10000):
    sa=ga[rng.integers(0,len(ga),len(ga))]; sb=gb[rng.integers(0,len(gb),len(gb))]
    d.append(sb[:,0].sum()/sb[:,1].sum()-sa[:,0].sum()/sa[:,1].sum())
d=np.array(d); print('boot CI',np.percentile(d,[2.5,97.5]),'p(<=0)',(d<=0).mean())
# permutation test at user level (randomization unit)
ucv=uu.set_index('user_id'); lab=ucv.group.values; 
obs=uu[uu.group=='B'].succ.sum()/uu[uu.group=='B'].n.sum()-uu[uu.group=='A'].succ.sum()/uu[uu.group=='A'].n.sum()
S=uu.succ.values; Nn=uu.n.values; isB=(uu.group=='B').values; pr=[]
for _ in range(10000):
    pm=rng.permutation(isB); pr.append(S[pm].sum()/Nn[pm].sum()-S[~pm].sum()/Nn[~pm].sum())
pr=np.array(pr); print('perm p two-sided',(np.abs(pr)>=abs(obs)).mean())
# DEFF/ICC approx
for gname,dd in p_c.groupby('group'):
    c=dd.groupby('user_id').y.agg(['sum','count']); mbar=c['count'].mean(); r=c['sum'].sum()/c['count'].sum()
    resid=c['sum']-r*c['count']; var_cl=(len(c)/(len(c)-1))*(resid**2).sum()/(c['count'].sum()**2); var_n=r*(1-r)/c['count'].sum(); print(gname,'DEFF',var_cl/var_n)
# balance
print('SRM',u_c.group.value_counts().to_dict(),stats.chisquare(u_c.group.value_counts().values).pvalue)
for c in ['device_type','city']: print(c,pd.crosstab(u_c.group,u_c[c],normalize='index').round(3).to_dict('index'),stats.chi2_contingency(pd.crosstab(u_c.group,u_c[c]))[1])
print('age',u_c.groupby('group').age.agg(['mean','median','std']).round(2).to_dict(),stats.ttest_ind(u_c[u_c.group=='A'].age.dropna(),u_c[u_c.group=='B'].age.dropna(),equal_var=False).pvalue)
# segments
rows=[]
for c in ['device_type','city']:
    for v,dd in p_c.groupby(c):
        r=ratio_diff(dd,'y'); rows.append((c,v,*r[:6],len(dd),dd.user_id.nunique()))
seg=pd.DataFrame(rows,columns=['dim','val','A','B','diff','lo','hi','p','n','users'])
order=seg.p.argsort().values; m=len(seg); ph=np.empty(m); run=0
for k,i in enumerate(order): run=max(run,(m-k)*seg.p.iloc[i]); ph[i]=min(1,run)
seg['p_holm']=ph; print(seg.round(4)); 
# heterogeneity: diff-in-diff between segments of device
for c in ['device_type','city']:
    vals=seg[seg.dim==c]; print(c,'hetero chi2 p',stats.chi2.sf((((vals['diff']-np.average(vals['diff'],weights=1/((vals['hi']-vals['lo'])/3.92)**2))/((vals['hi']-vals['lo'])/3.92))**2).sum(),len(vals)-1))
# time
p_c['week']=((p_c.step1_opened.dt.day-1)//7+1).clip(upper=4)
wk=p_c.groupby(['week','group']).y.agg(['mean','size']).unstack(); print(wk.round(3))
p_c['day']=p_c.step1_opened.dt.day
for gname,dd in p_c.groupby('group'): 
    dm=dd.groupby('day').y.mean(); print(gname,'trend slope/day',stats.linregress(dm.index,dm.values)[:4])
# durations
for a1,b1 in [('step1_opened','step2_entered'),('step2_entered','step3_confirmed'),('step3_confirmed','step4_success')]:
    dm=(p_c[b1]-p_c[a1]).dt.total_seconds()/60
    print(a1,b1,dm.groupby(p_c.group).agg(['mean','median']).round(2).to_dict(),stats.mannwhitneyu(dm[p_c.group=='A'].dropna(),dm[p_c.group=='B'].dropna()).pvalue)
tot=(p_c.step4_success-p_c.step1_opened).dt.total_seconds()/60; print('total time',tot.groupby(p_c.group).agg(['mean','median']).round(2).to_dict())
# sensitivity
first=p_c.sort_values('step1_opened').groupby('user_id').first().reset_index(); print('first attempt',first.groupby('group').y.agg(['mean','size']).round(4).to_dict(),ztest(first[first.group=='B'].y.sum(),(first.group=='B').sum(),first[first.group=='A'].y.sum(),(first.group=='A').sum()))
print('no 99001',np.round(ratio_diff(p_c[p_c.payment_id!=99001],'y')[:6],4))
raw=p.copy(); raw['y']=raw.step4_success.notna(); print('raw',raw.groupby('group').y.mean().round(4).to_dict())
print('w/ ages kept as is: same')
# MDE (user-level, deff-adjusted)
pa=p_c[p_c.group=='A'].y.mean(); se=ratio_diff(p_c,'y')[6]; print('SE diff',se,'MDE80%',(1.96+0.84)*se)
# CI via step diff
p_c.to_pickle('pc.pkl'); u_c.to_pickle('uc.pkl'); seg.to_pickle('seg.pkl'); uu.to_pickle('uu.pkl'); wk.to_pickle('wk.pkl')
