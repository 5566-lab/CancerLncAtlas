from pathlib import Path
import json, socket, sys, hashlib, textwrap, math
assert socket.gethostname().split('.')[0] == '149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
W=R/'kegg_cancer_redefinition_20261111_r1'
OLD=W/'fixed_y_remake_20261114_r1'
O=W/'ppt_y_clarification_20261114_r2'
D=O/'deliverables'; D.mkdir(parents=True,exist_ok=True)
F=O/'figures'; F.mkdir(exist_ok=True)
(O/'PPT_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'Extend evidence-based source inventory, target hierarchy and model Y presentation'},indent=2))
for p in [R/'python_deps',Path('/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps')]: sys.path.append(str(p))
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
for weight in ['Regular','Bold']:font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc')
FONT=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
NAVY='19384C'; TEAL='127F88'; ORANGE='CB6544'; BLUE='477EAB'; GRAY='5E707D'; PALE='EDF4F6'
plt.rcParams.update({'font.family':FONT,'font.size':15,'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
prs=Presentation(str(next((OLD/'deliverables').glob('*.pptx'))))
original=list(prs.slides)
def tx(sl,x,y,w,h,text,size=22,color=NAVY,bold=False):
    sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
    for i,line in enumerate(str(text).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK JP';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
    return sh
def slide(title,kicker,footer='CancerLncAtlas · 数据来源、精确主干 Y 与模型设计 · 2026-11-14'):
    s=prs.slides.add_slide(prs.slide_layouts[6]);bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.1));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string(TEAL);bar.line.fill.background()
    tx(s,.5,.25,12,.4,kicker,13,TEAL,True);tx(s,.5,.78,12.3,.9,title,29,bold=True);tx(s,.5,7.1,11.7,.25,footer,9,GRAY);tx(s,12.4,7.04,.4,.3,'0',11,GRAY)
    return s
def card(sl,x,y,w,h,title,body='',color=TEAL,ts=23,bs=20):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(PALE);sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(1.2)
    tx(sl,x+.16,y+.14,w-.32,.77,title,ts,color,True)
    if body:tx(sl,x+.16,y+.98,w-.32,h-1.03,body,bs)
def arrow(sl,x,y,w=.6):
    sh=sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,Inches(x),Inches(y),Inches(w),Inches(.3));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(GRAY);sh.line.fill.background()
def pic(sl,path,x=.75,y=1.9,w=11.8,h=4.8):
    a,b=Image.open(path).size;k=min(w/a,h/b);ww=a*k;hh=b*k;sl.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig);return F/(name+'.png')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)

# Source counts and model/condition examples are populated from host149 audit files.
exec((O/'slide_content.py').read_text(encoding='utf8'),globals())

# CONTENT_ORDER is a list of real Slide objects, retaining original readable examples.
order=CONTENT_ORDER
assert len({id(s) for s in order})==len(order)
by_id={s.slide_id:s for s in prs.slides}
sld_list=prs.slides._sldIdLst
elements={int(e.id):e for e in list(sld_list)}
ordered_ids=[s.slide_id for s in order]
for e in list(sld_list):sld_list.remove(e)
for idx,(s,slide_id) in enumerate(zip(order,ordered_ids),1):
    sld_list.append(elements[slide_id])
    for sh in s.shapes:
        if sh.has_text_frame and sh.left>Inches(12.2) and sh.top>Inches(6.9):
            sh.text_frame.paragraphs[0].text=str(idx)
            for run in sh.text_frame.paragraphs[0].runs:run.font.name='Noto Sans CJK JP';run.font.size=Pt(11);run.font.color.rgb=RGBColor.from_string(GRAY)
    s.notes_slide.notes_text_frame.text+='\n审计依据：ppt_y_clarification_20261114_r2/source_inventory、event_conditions、y_contract；固定Y原值不变。'
out=D/'CancerLncAtlas_Y来源层级与模型说明_20261114_r2.pptx'
prs.save(out)
manifest={'target_host':'149','slides':len(order),'pptx':str(out),'original_pptx':str(next((OLD/'deliverables').glob('*.pptx'))),'fixed_y_bundle':str(OLD/'Y_bundle'),'training':False,'lncrna_association':False}
(O/'PPT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('EXTENDED_PPT_COMPLETE',len(order),flush=True)
