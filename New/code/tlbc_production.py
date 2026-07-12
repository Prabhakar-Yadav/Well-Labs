# Rebuild the Attanur production package from the verified tlbc_segments (4-class + damage):
# numbered zoomed images + clickable boxes.geojson + index + overview maps.  python tlbc_production.py
import glob, os, json, csv, collections, numpy as np, rasterio, cv2, shapefile
from rasterio import Affine
from rasterio.transform import xy as rxy
from rasterio.warp import transform as wt
from rasterio.windows import Window
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
ORTHOS={'ATTANUR_2':'_keep_locally/TLBC_D95_ATTANUR_2_ortho.tif','ATTANUR_3':'_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif',
        'ATTANUR_1':'_keep_locally/TLBC_D95_ATTANUR_1_ortho.tif','ATTANUR_4':'_keep_locally/TLBC_D95_ATTANUR_4_ortho.tif',
        'SHAKAPUR':'_keep_locally/TLBC_D95_SHAKAPUR_ortho.tif'}
OUT='production/attanur'; MARGIN_M=20.0; MAXPX=1100; VIS=2200; FT=cv2.FONT_HERSHEY_SIMPLEX
COL={'clear':(0,190,0),'partial':(235,205,0),'moderate':(255,140,0),'full':(220,20,20)}  # RGB
RANGES={'clear':(0,5),'partial':(6,40),'moderate':(41,70),'full':(71,100)}  # clamp blockpct into class band
os.makedirs(OUT+'/images',exist_ok=True)
r=shapefile.Reader('gis/TLBC_alignment/tlbc_segments.shp'); flds=[x[0] for x in r.fields[1:]]
segs=[]
for sr in r.shapeRecords():
    d=dict(zip(flds,sr.record)); pts=np.array(sr.shape.points,float)
    pred=(str(d['actual']).strip().lower() or str(d['predicted']).strip().lower()).replace('blocked','full')  # use VERIFIED actual marking
    if pred not in COL or len(pts)<2: continue
    segs.append(dict(rec=d,pts=pts,pred=pred,done=False))
print('segments to image:',len(segs),flush=True)
idx=[]
w=shapefile.Writer(f'{OUT}/blocks',shapeType=shapefile.POLYGON)
w.field('id','N',6); w.field('canal','C',40); w.field('ctype','C',14); w.field('chainage','N',10,1)
w.field('status','C',12); w.field('blockpct','N',5,0); w.field('damage','C',20); w.field('image','C',60)
for tag,op in ORTHOS.items():
    if not glob.glob(op) or not [s for s in segs if not s['done']]: continue
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; px_m=abs(tr.a); crs=src.crs; b=src.bounds; mg=max(4,int(MARGIN_M/px_m))
        vs=max(1,max(H,W)//VIS); vh,vw=H//vs,W//vs
        rgbv=np.transpose(src.read([1,2,3],out_shape=(3,vh,vw)),(1,2,0)).astype(np.float32)
        for c in range(3):
            p=rgbv[:,:,c]; lo,hi=np.percentile(p,[2,98]); rgbv[:,:,c]=np.clip((p-lo)/(hi-lo+1e-6)*255,0,255)
        canvas=np.ascontiguousarray(rgbv.astype(np.uint8)); tile=[]
        for s in [x for x in segs if not x['done']]:
            pts=s['pts']; cx,cy=pts[:,0].mean(),pts[:,1].mean()
            if not (b.left<=cx<=b.right and b.bottom<=cy<=b.top): continue
            rr,cc=rasterio.transform.rowcol(tr,pts[:,0].tolist(),pts[:,1].tolist()); rr,cc=np.array(rr),np.array(cc)
            ins=(rr>=0)&(rr<H)&(cc>=0)&(cc<W); rr,cc=rr[ins],cc[ins]
            if len(rr)<2: continue
            r0=max(0,rr.min()-mg); r1=min(H,rr.max()+mg); a0=max(0,cc.min()-mg); a1=min(W,cc.max()+mg)
            if r1-r0<8 or a1-a0<8: continue
            d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
            if valid.mean()<0.3: continue
            s['done']=True; rgb=np.transpose(d[:3],(1,2,0)).astype(np.float32)
            for c in range(3):
                pv=rgb[:,:,c][valid]
                if pv.size: lo,hi=np.percentile(pv,[2,98]); rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/(hi-lo+1e-6)*255,0,255)
            crop=np.ascontiguousarray(rgb.astype(np.uint8)); loc=np.array([(cc[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
            cv2.polylines(crop,[loc],False,COL[s['pred']],3)
            hh,ww=crop.shape[:2]; scl=min(1.0,MAXPX/max(hh,ww))
            if scl<1: crop=cv2.resize(crop,(int(ww*scl),int(hh*scl)))
            crop=cv2.copyMakeBorder(crop,30,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            rec=s['rec']; sid=int(rec['id']); pct=int(rec['blockpct'] or 0)
            lo,hi=RANGES[s['pred']]; pct=min(hi,max(lo,pct))  # clamp % into actual-class band
            dmg=bool(str(rec['damage']).strip())
            lab=f"#{sid} {s['pred'].upper()} {pct}%  ch {int(round(float(rec['chainage'] or 0)))}m"+("  [STRUCTURAL DAMAGE]" if dmg else "")
            cv2.putText(crop,lab,(6,21),FT,0.55,(0,220,255) if dmg else (255,255,255),1)
            img=f'images/{sid:04d}.jpg'; cv2.imwrite(f'{OUT}/{img}',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
            rows=[r0,r0,r1,r1]; cols=[a0,a1,a1,a0]; xs,ys=rxy(tr,rows,cols)
            lons,lats=wt(crs,'EPSG:4326',list(xs),list(ys)); corners=[[float(a),float(bb)] for a,bb in zip(lons,lats)]
            cxs,cys=rxy(tr,[(r0+r1)/2],[(a0+a1)/2]); clon,clat=wt(crs,'EPSG:4326',list(cxs),list(cys))
            w.poly([corners+[corners[0]]]); w.record(sid,str(rec['canal']),str(rec['ctype']),round(float(rec['chainage'] or 0),1),s['pred'],pct,'Structural damage' if dmg else '',img)
            idx.append(dict(id=sid,area=tag,canal=str(rec['canal']),type=str(rec['ctype']),chainage_m=round(float(rec['chainage'] or 0),1),
                            status=s['pred'],blockpct=pct,damage='Structural damage' if dmg else '',lon=round(float(clon[0]),7),lat=round(float(clat[0]),7),bbox_lonlat=corners,image=img))
            tile.append((int(cc.min()/vs),int(rr.min()/vs),int(cc.max()/vs),int(rr.max()/vs),s['pred'],sid,dmg))
        for x1,y1,x2,y2,pr,sid,dmg in tile:
            cv2.rectangle(canvas,(x1-2,y1-2),(x2+2,y2+2),COL[pr][::-1],2)
            if dmg: cv2.rectangle(canvas,(x1-4,y1-4),(x2+4,y2+4),(0,0,255),1)
        cnt=collections.Counter(t[4] for t in tile); nd=sum(1 for t in tile if t[6])
        fig,ax=plt.subplots(figsize=(13,int(13*vh/vw)+1)); ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        ax.set_title(f'{tag} — verified blockage {dict(cnt)} | structural damage {nd}',fontsize=11); ax.axis('off')
        ax.legend(handles=[mpatches.Patch(color=np.array(COL[k])/255,label=k) for k in COL],loc='lower right')
        plt.tight_layout(); plt.savefig(f'{OUT}/overview_{tag}.png',dpi=140,bbox_inches='tight'); plt.close()
        print(f'{tag}: imaged {len(tile)}',flush=True)
w.close(); open(f'{OUT}/blocks.prj','w').write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
fc={'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Polygon','coordinates':[x['bbox_lonlat']+[x['bbox_lonlat'][0]]]},
    'properties':{k:x[k] for k in ['id','area','canal','type','chainage_m','status','blockpct','damage','lon','lat','image']}} for x in idx]}
json.dump(fc,open(f'{OUT}/boxes.geojson','w'),indent=2); json.dump(idx,open(f'{OUT}/index.json','w'),indent=2)
with open(f'{OUT}/index.csv','w',newline='') as fcsv:
    cw=csv.writer(fcsv); cw.writerow(['id','area','canal','type','chainage_m','status','blockpct','damage','lon','lat','image'])
    for x in idx: cw.writerow([x['id'],x['area'],x['canal'],x['type'],x['chainage_m'],x['status'],x['blockpct'],x['damage'],x['lon'],x['lat'],x['image']])
print('DONE:',len(idx),'boxes ->',OUT,'| by status:',dict(collections.Counter(x['status'] for x in idx)))
