from pathlib import Path
import sys,socket,json
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'exact_member_merge_20261114_r1';D=O/'deliverables'
for p in [R/'python_deps',Path('/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps')]:sys.path.append(str(p))
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
prs=Presentation(W/'reader_pathway_examples_20261114_r1/CancerLncAtlas_通路处理与经典实例_20261114.pptx')
for n,sl in enumerate(prs.slides,1):
    for shape in sl.shapes:
        if not shape.has_text_frame:continue
        for p in shape.text_frame.paragraphs:
            for run in p.runs:
                run.text=run.text.replace('字典 v6','字典 v7').replace('477 是主干机制目录的条目数；独立预测维度由去重后的有效观测定义。','476 个主干观测组保留 477 条来源机制；独立预测维度按唯一观测定义。').replace('477 Pathway 主干机制','476 Pathway 主干观测组')
    if n==8:
        for shape in sl.shapes:
            if shape.has_text_frame and shape.text.startswith('共享观测用于通路层分析'):
                shape.text_frame.paragraphs[0].runs[0].text='J=1 的两条记录合并为一个 29 蛋白下游模块；原始变异机制作为来源注释保留。'
sl=prs.slides.add_slide(prs.slide_layouts[6])
def text(x,y,w,h,s,size=22,color='19384C',bold=False):
    sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
    for i,line in enumerate(s.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK JP';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
bar=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.1));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string('127F88');bar.line.fill.background()
text(.5,.3,12,.4,'完全相同成员的归并',14,'127F88',True)
text(.5,.85,12,1,'477 条来源机制 → 476 个主干观测组',29,bold=True)
text(.8,2.1,11.7,1,'Hedgehog 信号：SMO/PTCH1 共同下游模块',27,'127F88',True)
text(1,3.2,11,1.1,'N00017 与 N00010：原始蛋白集合均为同样的 29 个成员\n合并为一个主条目；两份来源定义在详情中保留。',23)
text(1,4.75,11,1.1,'476 个 Pathway 观测组 + 197 个 State 身份 = 673 个目标\n共享成员观测去重；原始测量值、缺失与患者分折保留。',22)
text(.6,7.05,11.5,.25,'Functional_pathway_State_dictionary_v7.xlsx · 原始主干成员逐条比较 · BRCA / LUAD / COAD',10,'5E707D')
sl.notes_slide.notes_text_frame.text='依据 exact_member_merge_20261114_r1/GROUP_DEFINITION.json 和 VALIDATION.json。只对非空、完全相同的主干原始蛋白集合合并。变异条件仍为来源机制的独立注释。无 lncRNA 关联或模型训练。'
out=D/'CancerLncAtlas_通路字典与J1合并_20261114.pptx';prs.save(out);print('PPT_UPDATED',len(prs.slides),out,flush=True)
