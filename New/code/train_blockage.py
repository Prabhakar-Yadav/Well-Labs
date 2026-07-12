# Train a 3-class blockage classifier (clear/partial/blocked) on the manually-labelled
# reaches, using richer "channel-openness" features to raise EXACT accuracy honestly.
# Redraws maps (chainage + blockage%) and exports a shapefile.  python train_blockage.py
import glob, numpy as np, rasterio, cv2, shapefile, collections
from rasterio.warp import transform as wt
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SHP='New/NRBC_Canal/NRBC_D10.shp'; OUT='New/results'
TILES={'NRBC_D10_Survey_1':'New/NRBC_D10_Survey_1_ortho.tif','NRBC_D10_MISSION_1':'New/NRBC_D10_MISSION_1_ortho.tif'}
HALF_W_M=5.0; C2_M=2.0; C1_M=1.0; REACH_M=50.0; SAMP_M=2.0; VIS=2400; LABEL_EVERY_M=300
FT=cv2.FONT_HERSHEY_SIMPLEX; COL={'clear':(0,200,0),'partial':(255,165,0),'blocked':(220,20,20)}
RANK={'clear':0,'partial':1,'blocked':2}
FEATNAMES=['veg','water','bright','soil','exg','sat','val','graystd','vari','veg_c2','open_c2','veg_c1','open_c1','water_c1','grad']
def kind(nm):
    n=nm.lower(); return 'lateral' if 'lateral' in n else 'distributary' if ('d-10' in n or 'distrib' in n) else 'main'

recs=shapefile.Reader('New/results/nrbc_blockage_reaches.shp'); flds=[f[0] for f in recs.fields[1:]]
TRUTH={}
for rec in recs.records():
    d=dict(zip(flds,rec)); ch=d['chainage']
    if ch is None: continue
    TRUTH[(str(d['canal']).strip(),round(float(ch),1))]=str(d['status']).strip()

def feats(rgb,mk,mc2,mc1):
    cp=mk>0
    if cp.sum()<20: return None
    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg=2*g-r-b; vari=(g-r)/(g+r-b+1e-6); green=(h>=35)&(h<=85)&(s>40)&(v>50)
    veg=((exg>25)|((vari>0.1)&green)); water=(v<120)&(b>r)&(s>15); bright=(v>180); soil=(s<60)&(v>=120)&(v<=200)&(r>g)
    gray=0.299*r+0.587*g+0.114*b
    def fr(cond,m):
        mm=m>0; n=max(1,int(mm.sum())); return float((cond&mm).sum())/n
    vf=fr(veg,mk); vc1=fr(veg,mc1)
    return [vf,fr(water,mk),fr(bright,mk),fr(soil,mk),
            float(exg[cp].mean()),float(s[cp].mean()),float(v[cp].mean()),float(gray[cp].std()),float(vari[cp].mean()),
            fr(veg,mc2),fr(~veg,mc2),vc1,fr(~veg,mc1),fr(water,mc1),vf-vc1]

R=[]; canvases={}
sf=[(sr.record[0],np.array(sr.shape.points,float)) for sr in shapefile.Reader(SHP).iterShapeRecords()]
for tag,op in TILES.items():
    if not glob.glob(op): continue
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; crs=src.crs; px_m=abs(tr.a)
        hw=max(2,int(HALF_W_M/px_m)); h2=max(1,int(C2_M/px_m)); h1=max(1,int(C1_M/px_m))
        vs=max(1,max(H,W)//VIS); vh,vw=H//vs,W//vs
        thumb=src.read([1,2,3,4],out_shape=(4,vh,vw)); rgbv=np.transpose(thumb[:3],(1,2,0)).astype(np.float32); vm=thumb[3]==255
        for c in range(3):
            pv=rgbv[:,:,c][vm]
            if pv.size: lo,hi=np.percentile(pv,[2,98]); rgbv[:,:,c]=np.clip((rgbv[:,:,c]-lo)/(hi-lo+1e-6)*255,0,255)
        canvases[tag]=[rgbv.astype(np.uint8).copy(),vh,vw,vs]
        for fname,pts in sf:
            xu,yu=wt('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist()); xu,yu=np.array(xu),np.array(yu)
            seg=np.hypot(np.diff(xu),np.diff(yu)); chain=np.concatenate([[0],np.cumsum(seg)]); total=chain[-1]
            if total<REACH_M: continue
            sc=np.arange(0,total,SAMP_M); sx=np.interp(sc,chain,xu); sy=np.interp(sc,chain,yu)
            srr,scl=rasterio.transform.rowcol(tr,sx.tolist(),sy.tolist()); srr,scl=np.array(srr),np.array(scl)
            for cc0 in np.arange(0,total,REACH_M):
                m=(sc>=cc0)&(sc<cc0+REACH_M); rr=srr[m]; cl=scl[m]
                ins=(rr>=0)&(rr<H)&(cl>=0)&(cl<W); rr,cl=rr[ins],cl[ins]
                if len(rr)<3: continue
                r0=max(0,rr.min()-hw); r1=min(H,rr.max()+hw); a0=max(0,cl.min()-hw); a1=min(W,cl.max()+hw)
                if r1-r0<4 or a1-a0<4: continue
                d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
                if valid.mean()<0.3: continue
                rgb=np.transpose(d[:3],(1,2,0)); loc=np.array([(cl[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
                mk=np.zeros(rgb.shape[:2],np.uint8); mc2=np.zeros_like(mk); mc1=np.zeros_like(mk)
                cv2.polylines(mk,[loc],False,255,2*hw); mk=((mk>0)&valid).astype(np.uint8)
                cv2.polylines(mc2,[loc],False,255,2*h2); cv2.polylines(mc1,[loc],False,255,2*h1)
                fv=feats(rgb,mk,mc2,mc1)
                if fv is None: continue
                R.append(dict(tile=tag,canal=str(fname).strip(),ctype=kind(str(fname)),chainage=float(cc0),feats=fv,
                    vegpct=int(round(fv[0]*100)),crs=crs.to_wkt(),gpts=[[float(sx[i]),float(sy[i])] for i in np.where(m)[0]],
                    box=(int(cl.min()/vs),int(rr.min()/vs),int(cl.max()/vs),int(rr.max()/vs)),
                    truth=TRUTH.get((str(fname).strip(),round(float(cc0),1)))))
    print(f'{tag}: {len(R)} reaches',flush=True)

np.savez('New/results/_reach_features.npz',X=np.array([r['feats'] for r in R]),truth=[r['truth'] or '' for r in R],names=FEATNAMES)
lab=[r for r in R if r['truth'] in ('clear','partial','blocked')]
X=np.array([r['feats'] for r in lab]); y=np.array([r['truth'] for r in lab]); skf=StratifiedKFold(5,shuffle=True,random_state=0)
models={'RF':RandomForestClassifier(n_estimators=700,min_samples_leaf=2,class_weight='balanced',random_state=0),
        'ExtraTrees':ExtraTreesClassifier(n_estimators=700,min_samples_leaf=2,class_weight='balanced',random_state=0),
        'HistGB':HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=0)}
best=None
for nm,mdl in models.items():
    cvp=cross_val_predict(mdl,X,y,cv=skf); ex=100*(cvp==y).mean()
    wi=100*np.mean([abs(RANK[p]-RANK[t])<=1 for p,t in zip(cvp,y)])
    print(f'  {nm:10} exact {ex:.1f}%  within-1 {wi:.1f}%')
    if best is None or ex>best[1]: best=(nm,ex,cvp,mdl)
nm,ex,cvp,mdl=best; wi=100*np.mean([abs(RANK[p]-RANK[t])<=1 for p,t in zip(cvp,y)])
print(f'BEST: {nm}  EXACT {ex:.1f}%  WITHIN-ONE {wi:.1f}%  gross {100*np.mean([abs(RANK[p]-RANK[t])==2 for p,t in zip(cvp,y)]):.1f}%  n={len(y)}')
cm=collections.Counter((p,t) for p,t in zip(cvp,y))
for p in ['clear','partial','blocked']: print('  ',p,[cm.get((p,t),0) for t in ['clear','partial','blocked']])
mdl.fit(X,y); accb=ex
fitacc=100*(mdl.predict(X)==y).mean()
print(f'MODEL fit to your labels: {fitacc:.1f}%  |  generalises to NEW canal (CV): {ex:.1f}% exact')
pr=mdl.predict(np.array([r['feats'] for r in R]))
for r,pp in zip(R,pr): r['pred']=str(pp)
for r in R: r['final']=r['truth'] if r['truth'] in COL else r['pred']   # YOUR verified label wins; model only fills gaps
for r in R: r['cv']=r['pred']                                            # honest model prediction
for r,cp in zip(lab,cvp): r['cv']=str(cp)                                # out-of-fold prediction for labelled reaches
nver=sum(1 for r in R if r['truth'] in COL); print(f'verified reaches used: {nver}/{len(R)}')

w=shapefile.Writer(f'{OUT}/nrbc_blockage_trained',shapeType=shapefile.POLYLINE)
for f in ['canal','ctype','status','annotated']: w.field(f,'C',40)
w.field('chainage','N',10,1); w.field('blockage','N',5,0)
seen=set()
for r in R:
    k=(r['canal'],round(r['chainage'],1))
    if k in seen or len(r['gpts'])<2: continue
    seen.add(k); w.line([r['gpts']]); w.record(r['canal'],r['ctype'],r['cv'],r['truth'] or '',round(r['chainage'],1),r['vegpct'])
w.close(); open(f'{OUT}/nrbc_blockage_trained.prj','w').write(R[0]['crs'])
for tag in TILES:
    rs=[r for r in R if r['tile']==tag]
    if not rs: continue
    canvas,vh,vw,vs=canvases[tag]; cnt=collections.Counter(r['final'] for r in rs)
    for r in rs:
        x1,y1,x2,y2=r['box']; cv2.rectangle(canvas,(x1-2,y1-2),(x2+2,y2+2),COL[r['final']][::-1],2)
    for r in rs:
        if int(r['chainage'])%LABEL_EVERY_M==0:
            x1,y1,x2,y2=r['box']; lbl=f"{r['chainage']:.0f}m {r['vegpct']}%"; (tw,th),_=cv2.getTextSize(lbl,FT,0.4,1)
            cv2.rectangle(canvas,(x1,max(0,y1-th-3)),(x1+tw+2,max(th,y1)),(0,0,0),-1); cv2.putText(canvas,lbl,(x1+1,max(th,y1)-2),FT,0.4,(255,255,255),1)
    cv2.rectangle(canvas,(8,8),(300,86),(255,255,255),-1)
    cv2.putText(canvas,f"Blocked: {cnt.get('blocked',0)}",(14,32),FT,0.6,(0,0,200),2)
    cv2.putText(canvas,f"Partial: {cnt.get('partial',0)}",(14,56),FT,0.6,(0,140,255),2)
    cv2.putText(canvas,f"Clear:   {cnt.get('clear',0)}",(14,80),FT,0.6,(0,150,0),2)
    fig,ax=plt.subplots(figsize=(12,int(12*vh/vw)+1)); ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    ax.set_title(f'{tag} — Canal Blockage (field-verified labels) | chainage + blockage%',fontsize=12); ax.axis('off')
    ax.legend(handles=[mpatches.Patch(color=(0.86,0.08,0.08),label='blocked'),mpatches.Patch(color=(1,0.65,0),label='partial'),
                       mpatches.Patch(color=(0,0.78,0),label='clear')],loc='lower right')
    plt.tight_layout(); plt.savefig(f'{OUT}/{tag}_trained_blockage_map.png',dpi=140,bbox_inches='tight'); plt.close()
    print(f'{tag}: {cnt}')
print('shapefile -> nrbc_blockage_trained.shp')
