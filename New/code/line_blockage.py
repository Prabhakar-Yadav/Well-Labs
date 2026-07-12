# Blockage along hand-traced canal lines, Attanur bbox+chainage style.
# Hierarchical chainage: main canal = no chainage; distributary = 0 at canal junction;
# lateral = 0 at its distributary junction.  Usage: python line_blockage.py [tlbc|nrbc]
import sys, glob, json, numpy as np, rasterio, cv2, shapefile
from rasterio.warp import transform as wt
from rasterio.windows import Window
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODE = sys.argv[1] if len(sys.argv) > 1 else 'tlbc'
CONFIGS = {
 'tlbc': dict(shp='TLBC_D95_Canal/TLBC_D95.shp', out='outputs/07_blockage_finetuned',
              tiles={'ATTANUR_2':'TLBC_D95_ATTANUR_2_ortho.tif',
                     'ATTANUR_3':'_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif'}),
 'nrbc': dict(shp='New/NRBC_Canal/NRBC_D10.shp', out='New/results',
              tiles={'NRBC_D10_Survey_1':'New/NRBC_D10_Survey_1_ortho.tif',
                     'NRBC_D10_MISSION_1':'New/NRBC_D10_MISSION_1_ortho.tif'}),
}
cfg = CONFIGS[MODE]; SHP, OUT, TILES = cfg['shp'], cfg['out'], cfg['tiles']
HALF_W_M=2.5; REACH_M=50.0; SAMP_M=2.0; VIS=2400; LABEL_EVERY_M=300
COL={'blocked':(220,20,20),'partial':(255,165,0),'clear':(0,200,0)}
FT=cv2.FONT_HERSHEY_SIMPLEX
raw=[(sr.record[0] if sr.record else '?', np.array(sr.shape.points,float))
     for sr in shapefile.Reader(SHP).iterShapeRecords()]

def kind(nm):
    n=nm.lower()
    if 'lateral' in n: return 'lateral'
    if 'canal' in n:   return 'main'
    return 'distributary'

# --- reproject to the tiles' CRS once + orient each line so chainage starts at the junction ---
CRS = rasterio.open(next(iter(TILES.values()))).crs
F=[[nm, kind(nm), np.column_stack(wt('EPSG:4326',CRS,p[:,0].tolist(),p[:,1].tolist()))] for nm,p in raw]
main_pts=np.vstack([f[2] for f in F if f[1]=='main']) if any(f[1]=='main' for f in F) else None
distr=[f for f in F if f[1]=='distributary']
def mind(pt,pts): return float(np.min(np.hypot(pts[:,0]-pt[0],pts[:,1]-pt[1])))
for f in F:
    nm,k,xy=f; parent=None
    if k=='distributary': parent=main_pts
    elif k=='lateral':    parent=min(distr,key=lambda d:mind(xy[len(xy)//2],d[2]))[2] if distr else main_pts
    if parent is not None and len(xy)>=2 and mind(xy[-1],parent)<mind(xy[0],parent):
        f[2]=xy[::-1]                                   # reverse: junction end first (chainage 0)

def features(rgb,mask):
    cp=mask>0
    if cp.sum()<20: return None
    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg=2*g-r-b; vari=(g-r)/(g+r-b+1e-6); green=(h>=35)&(h<=85)&(s>40)&(v>50)
    veg=((exg>25)|((vari>0.1)&green))&cp; water=(v<120)&(b>r)&(s>15)&cp
    n=max(1,int(cp.sum())); return {'veg':float(veg.sum()/n),'water':float(water.sum()/n)}

def classify(f):
    if f is None: return 'clear',0.0
    veg=f['veg']; water=f['water']
    if veg>0.55 and water<0.20: return 'blocked',veg
    if veg>0.35: return 'partial',veg
    return 'clear',veg

allrows=[]; shp_rows=[]; shp_seen=set()
for tag,op in TILES.items():
    if not glob.glob(op): print(f'{tag}: ortho missing'); continue
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; px_m=abs(tr.a)
        hw=max(2,int(HALF_W_M/px_m)); vs=max(1,max(H,W)//VIS); vh,vw=H//vs,W//vs
        thumb=src.read([1,2,3,4],out_shape=(4,vh,vw))
        rgbv=np.transpose(thumb[:3],(1,2,0)).astype(np.float32); vmask=thumb[3]==255
        for c in range(3):
            pv=rgbv[:,:,c][vmask]
            if pv.size: lo,hi=np.percentile(pv,[2,98]); rgbv[:,:,c]=np.clip((rgbv[:,:,c]-lo)/(hi-lo+1e-6)*255,0,255)
        canvas=rgbv.astype(np.uint8).copy(); counts={}; reaches=[]
        for nm,k,xy in F:
            xu,yu=xy[:,0],xy[:,1]
            chain=np.concatenate([[0],np.cumsum(np.hypot(np.diff(xu),np.diff(yu)))]); total=chain[-1]
            if total<REACH_M: continue
            sc=np.arange(0,total,SAMP_M); sx=np.interp(sc,chain,xu); sy=np.interp(sc,chain,yu)
            sr,scl=rasterio.transform.rowcol(tr,sx.tolist(),sy.tolist()); sr,scl=np.array(sr),np.array(scl)
            for cc0 in np.arange(0,total,REACH_M):
                m=(sc>=cc0)&(sc<cc0+REACH_M); rr=sr[m]; cl=scl[m]
                ins=(rr>=0)&(rr<H)&(cl>=0)&(cl<W); rr,cl=rr[ins],cl[ins]
                if len(rr)<3: continue
                r0=max(0,rr.min()-hw); r1=min(H,rr.max()+hw); a0=max(0,cl.min()-hw); a1=min(W,cl.max()+hw)
                if r1-r0<4 or a1-a0<4: continue
                d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
                if valid.mean()<0.3: continue
                rgb=np.transpose(d[:3],(1,2,0)); mk=np.zeros(rgb.shape[:2],np.uint8)
                loc=np.array([(cl[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
                cv2.polylines(mk,[loc],False,255,2*hw); mk=((mk>0)&valid).astype(np.uint8)
                st,veg=classify(features(rgb,mk)); counts[st]=counts.get(st,0)+1
                allrows.append({'tile':tag,'canal':nm,'type':k,
                                'chainage_m':(None if k=='main' else round(float(cc0),1)),
                                'status':st,'veg_frac':round(float(veg),2)})
                reaches.append((int(cl.min()/vs),int(rr.min()/vs),int(cl.max()/vs),int(rr.max()/vs),float(cc0),st,k))
                key=(nm,round(float(cc0),1))
                if key not in shp_seen:
                    gpts=[[float(sx[i]),float(sy[i])] for i in np.where(m)[0]]
                    if len(gpts)>=2: shp_rows.append((nm,k,float(cc0),st,float(veg),gpts)); shp_seen.add(key)
        for x1,y1,x2,y2,ch,st,k in reaches:
            cv2.rectangle(canvas,(x1-2,y1-2),(x2+2,y2+2),COL[st][::-1],2)
        for x1,y1,x2,y2,ch,st,k in reaches:
            if k!='main' and int(ch)%LABEL_EVERY_M==0:          # no chainage on the main canal
                lbl=f'{ch:.0f}m'; (tw,th),_=cv2.getTextSize(lbl,FT,0.4,1)
                cv2.rectangle(canvas,(x1,max(0,y1-th-3)),(x1+tw+2,max(th,y1)),(0,0,0),-1)
                cv2.putText(canvas,lbl,(x1+1,max(th,y1)-2),FT,0.4,(255,255,255),1)
        nb,npa,nc=counts.get('blocked',0),counts.get('partial',0),counts.get('clear',0)
        cv2.rectangle(canvas,(8,8),(250,86),(255,255,255),-1)
        cv2.putText(canvas,f'Blocked: {nb}',(14,32),FT,0.6,(0,0,200),2)
        cv2.putText(canvas,f'Partial: {npa}',(14,56),FT,0.6,(0,140,255),2)
        cv2.putText(canvas,f'Clear:   {nc}',(14,80),FT,0.6,(0,150,0),2)
        fig,ax=plt.subplots(figsize=(12,int(12*vh/vw)+1)); ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        ax.set_title(f'{tag} — Canal Blockage Map with Chainage (m from junction)\nmain canal not chainaged | {len(reaches)} reaches',fontsize=11); ax.axis('off')
        ax.legend(handles=[mpatches.Patch(color=np.array(COL[k])/255,label=k) for k in COL],loc='lower right')
        out=f'{OUT}/{tag}_bbox_chainage_map.png'; plt.tight_layout(); plt.savefig(out,dpi=140,bbox_inches='tight'); plt.close()
        print(f'{tag}: {len(reaches)} reaches {counts} -> {out}')
json.dump(allrows,open(f'{OUT}/{MODE}_blockage_chainage_summary.json','w'),indent=2)
# --- export reaches as a shapefile for manual QGIS validation ---
w=shapefile.Writer(f'{OUT}/{MODE}_blockage_reaches',shapeType=shapefile.POLYLINE)
w.field('canal','C',40); w.field('ctype','C',14); w.field('chainage','N',10,1)
w.field('status','C',10); w.field('veg_frac','N',6,2); w.field('actual','C',10); w.field('notes','C',80)
for nm,k,ch,st,veg,pts in shp_rows:
    w.line([pts]); w.record(nm,k,round(ch,1),st,round(veg,2),'','')
w.close()
open(f'{OUT}/{MODE}_blockage_reaches.prj','w').write(CRS.to_wkt())
print(f'shapefile: {len(shp_rows)} reaches -> {OUT}/{MODE}_blockage_reaches.shp')
print('TOTAL reaches:',len(allrows))
